import pandas as pd
import numpy as np
from nltk.corpus import wordnet
from pattern.text.en import singularize
import inflect
import re
import argparse

p = inflect.engine()

def find_sub_professions(title):
    """Segments the title phrase from the reverse and returns a list of words

    Parameters
    ----------
    title : str
        A professional title phrase

    Returns
    -------
    list
        a list of strings

    Example
    -------
    find_sub_professions("Chief Executive Officer") -> ["Officer","Executive Officer","Chief Executive Officer"]
    """

    exclude_tokens = set(['and', 'or', 'for', 'of', 'in', '-'])
    exclude_chars = set([',', '/', '(', '['])
    max_n_tokens = 5

    # remove paranthesized text
    text = re.sub("\s*\([^\)]+\)\s*", " ", title)

    # consolidate whitespace into a single whitespace
    text = re.sub("\s+", " ", text)

    # strip leading and trailing whitespace
    text = text.strip()

    # title is not segmented if it contains any exclude_token
    if set(text.lower().split()).intersection(exclude_tokens):
        return [title]
    
    # title is not segmented if it contains any exclude_char
    if set(text.lower()).intersection(exclude_chars):
        return [title]

    # title is not segmented if it contains more than five tokens
    if len(text.split()) > max_n_tokens:
        return [title]

    # title is not segmented if it is uppercase
    if re.sub("[^a-zA-Z0-9]", "", text).isupper():
        return [title]

    # title is not segmented if the last token has length less than 2
    if len(text.split()[-1]) < 2:
        return [title]

    tokens = text.split()
    sub_professions = [title]

    # backward segmentation
    for i in range(1, len(tokens)):
        sub_professions.append(" ".join(tokens[i:]))

    return sub_professions

def backward_segmentation_expansion(soc_csv_filepath, soc_txt_filepath, bseg_filepath):
    """Find new job titles using backward segmentation of SOC job titles

    Parameters
    ----------
    soc_csv_filepath : str
        csv filepath of SOC job titles

    soc_txt_filepath : str
        txt filepath to which we will write the SOC titles

    bseg_filepath : str
        txt filepath to which we will write the new job titles
        created by backward segmentation of SOC job titles
    """

    # read SOC titles
    soc_df = pd.read_csv(soc_csv_filepath, index_col=None)
    soc_titles = soc_df["2018 SOC Direct Match Title"].dropna().unique()
    print(f"{'SOC':30s} = {len(soc_titles)} SOC titles")

    # create new titles using backward segmentation
    expanded_titles = []
    for title in soc_titles:
        expanded_titles.extend(find_sub_professions(title))
    expanded_titles = np.array(list(set(expanded_titles).difference(set(soc_titles))))
    print(f"{'BSEG':30s} = {len(expanded_titles)} new job titles added by backward segmentation")

    # write SOC titles
    open(soc_txt_filepath, "w").write("\n".join(sorted(soc_titles)))

    # write new job titles
    open(bseg_filepath, "w").write("\n".join(sorted(expanded_titles)))

def find_wordnet_synsets_from_word(word):
    """Return noun synsets of word and their hyponyms

    Parameters
    ----------
    word : str
        The word for which we need to find the wordnet synsets

    Returns
    -------
    list
        A list of wordnet synsets
    """

    # replace whitespace by underscore
    word = re.sub("\s+", "_", word.lower())
    synsets = []

    # search noun synsets
    for synset in wordnet.synsets(word, pos = "n"):

        # search person and group semantic category
        if synset.lexname() in ["noun.person", "noun.group"]:
            synsets.append(synset)

            # search noun person and noun group hyponyms
            for hyponym_synset in synset.hyponyms():
                if hyponym_synset.lexname() in ["noun.person", "noun.group"]:
                    synsets.append(hyponym_synset)

    return synsets

def find_wordnet_synsets(soc_txt_filepath, bseg_filtered_filepath, wn_syn_filepath):
    """Find the wordnet synsets of the SOC titles and the new titles expanded from SOC
    titles using backward segmentation

    Parameters
    ----------
    soc_txt_filepath : str
        txt filepath containing SOC titles

    bseg_filtered_filepath : str
        txt filepath containing the manually filtered job titles created from SOC job titles 
        using backward segmentation

    wn_syn_filepath : str
        txt filepath to which we will write the wordnet synsets and their definitions 
    """

    # read SOC and job titles created from backward segmentation
    soc_titles = open(soc_txt_filepath).read().strip().split("\n")
    bseg_titles = open(bseg_filtered_filepath).read().strip().split("\n")
    print(f"{'BSEG FILTERED':30s} = {len(bseg_titles)} new job titles added by backward segmentation (after manually filtering)")

    # search wordnet synsets
    synsets = []
    for word in list(soc_titles) + list(bseg_titles):
        synsets.extend(find_wordnet_synsets_from_word(word))

    synsets = np.unique(synsets)
    print(f"{'WORDNET SYNSETS':30s} = {len(synsets)} wordnet synsets found from SOC and BSEG titles")

    # write synsets to file
    with open(wn_syn_filepath,"w") as fw:
        for synset in synsets:
            fw.write(f"{synset.name():25s}\t{synset.definition()}\n")
        
def wordnet_expansion(soc_txt_filepath, bseg_filtered_filepath, wn_syn_filtered_filepath, wn_title_filepath):
    """Find the phrases from the manually filtered wordnet synsets

    Parameters
    ----------
    soc_txt_filepath : str
        txt filepath containing SOC titles

    bseg_filtered_filepath : str
        txt filepath containing the manually filtered job titles created from SOC job titles 
        using backward segmentation

    wn_syn_filtered_filepath : str
        txt filepath containing manually filtered wordnet synsets

    wn_title_filepath : str
        txt filepath containing the new titles found from the filtered wordnet synsets
    """

    # read SOC and job titles created from SOC using backward segmentation
    soc_titles = open(soc_txt_filepath).read().strip().split("\n")
    soc_titles = set([x.lower() for x in soc_titles])
    bseg_titles = open(bseg_filtered_filepath).read().strip().split("\n")
    bseg_titles = set([x.lower() for x in bseg_titles])
    lines = open(wn_syn_filtered_filepath).read().strip().split("\n")
    print(f"{'WORDNET SYNSETS FILTERED':30s} = {len(lines)} wordnet synsets found from SOC and BSEG titles (after manual filtering)")

    # find lemma names for each wordnet synset
    wn_titles = []
    for line in lines:
        synset = wordnet.synset(line.split()[0])
        wn_titles.extend([re.sub("_"," ", x) for x in synset.lemma_names()])

    wn_titles = set(wn_titles).difference(soc_titles.union(bseg_titles))
    wn_titles = sorted(wn_titles)
    print(f"{'WORDNET TITLES':30s} = {len(wn_titles)} new wordnet titles found from wordnet synsets")

    open(wn_title_filepath,"w").write("\n".join(wn_titles))

def merge_expansions(soc_txt_filepath, bseg_filtered_filepath, wn_title_filtered_txt_filepath, final_filepath):
    """Merge the list of job titles from SOC, Backward Segmentation and WordNet synonyms

    Parameters
    ----------
    soc_txt_filepath : str
        txt filepath containing SOC titles

    bseg_filtered_filepath : str
        txt filepath containing the manually filtered job titles created from SOC job titles 
        using backward segmentation

    wn_title_filtered_txt_filepath : str
        txt filepath containing manually filtered job titles created from wordnet synonyms of
        SOC job titles, and job titles created from SOC job titles using backward segmentation

    final_filepath : str
        txt filepath to which the SOC job titles, manually filtered backward segmented job titles, 
        and wordnet job titles will be written
    """
    soc_titles = set(open(soc_txt_filepath).read().lower().strip().split("\n"))
    bseg_titles = set(open(bseg_filtered_filepath).read().lower().strip().split("\n"))
    wn_titles = set(open(wn_title_filtered_txt_filepath).read().lower().strip().split("\n"))
    print(f"{'WORDNET TITLES FILTERED':30s} = {len(wn_titles)} new wordnet titles found from wordnet synsets (after manual filtering)")

    final_list = sorted(soc_titles.union(bseg_titles).union(wn_titles))
    print(f"{'FINAL TITLES':30s} = {len(final_list)} job titles (SOC + BSEG FILTERED + WORDNET TITLES FILTERED)")
    open(final_filepath, "w").write("\n".join(final_list))

def find_singular_form(word):
    """Find the singular form of the word
    """
    singular = p.singular_noun(word)
    if isinstance(singular, str) and "False" not in singular and not word.endswith("ss"):
        singular = singular.replace("polouse","police")
        return singular
    return word

def find_plural_form(word):
    """Find the plural form of the word
    """
    plural = p.plural_noun(word)
    if isinstance(plural, str) and "False" not in plural:
        plural = plural.replace("polices","police")
        plural = plural.replace("tradesmans","tradesmen")
        return plural
    return word

def find_inflections(final_filepath, inflection_filepath):
    """Find inflected forms of the job titles in the final list and save them

    Parameters
    ----------
    final_filepath : str
        txt filepath containing the final (SOC + BSEG + WORDNET) list of job titles

    inflection_filepath : str
        csv filepath containing the inflected forms of the final list of job titles.

    INFLECTION CSV
    --------------
    It contains five columns -

    word : str
        job title

    singular : str
        singular form of the word
        If we fail to find a singular form, the text from the word column is copied

    plural : str
        plural form of the word
        If we fail to find a plural form, the text from the word column is copied

    singular_key : str
        text used to search in Whoosh search index for singular form of the word

    plural_key : str
        text used to search in Whoosh search index for plural form of the word
    """
    job_titles = set(open(final_filepath).read().strip().split("\n"))
    ending_tokens = ["system", "systems", "operation", "operations", "intelligence", "skill", "skills", "business", "sale", "sales", "application", "product", "weapon","affair","tactic"]
    false_positives = ["academic", "21 dealer","acid dipper"]
    word_col = []
    singular_col = []
    plural_col = []

    for job_title in sorted(job_titles):
        word = job_title
        singular = find_singular_form(word)
        word = singular
        plural = find_plural_form(word)

        # if word ends in any of the ending_tokens or equals any false positives,
        # do not add it to
        if not any(word.endswith(x) for x in ending_tokens) and not word in false_positives:
            word_col.append(word)
            singular_col.append(singular)
            plural_col.append(plural)

    inflection_df = pd.DataFrame()
    inflection_df["word"] = word_col
    inflection_df["singular"] = singular_col
    inflection_df["plural"] = plural_col
    inflection_df["singular_key"] = inflection_df["singular"].apply(lambda x: singularize(x))
    inflection_df["plural_key"] = inflection_df["plural"].apply(lambda x: singularize(x))

    inflection_df.drop_duplicates(inplace=True)

    print(f"{'INFLECTIONS':30s} = {inflection_df.shape[0]} job titles")
    inflection_df.to_csv(inflection_filepath, index=False)

def manually_add_professions(inflection_filepath):
    """
    manually add some professions
    """
    professions = ["soldier", "secretary", "umpire", "general secretary", "cricketer", "ambassador", "general", "pawnbroker", "gymnast", "spokesman", "spokeswoman", "hacker", "astronaut", "bellman", "bellboy", "anchorman", "anchorwoman", "contortionist", "choreographer", "cosmonaut", "cupbearer", "farmhand", "general secretary", "huntsman", "major", "marine", "masterchef", "nba player", "nfl player", "swimmer", "babysitter", "endocrinologist", "farmer", "home secretary", "infantry man", "neurologist", "ophthalmologist", "optometrist", "personal assistant", "pornographer", "porn star", "station master", "tradesman", "tradeswoman"]

    inflection_df = pd.read_csv(inflection_filepath, index_col=None)
    records = [[profession, profession, find_plural_form(profession), singularize(profession), singularize(find_plural_form(profession))] for profession in professions]
    df = pd.DataFrame(records, columns=["word", "singular", "plural", "singular_key", "plural_key"])

    inflection_df = pd.concat([inflection_df, df])
    inflection_df = inflection_df.drop_duplicates()
    inflection_df = inflection_df.sort_values("word")
    inflection_df.to_csv(inflection_filepath, index=False)
    print(f"{'PROFESSIONS':30s} = {inflection_df.shape[0]} job titles")

def create_gazetteer():
    """Create gazetteer of job titles. Use --help to read the documentation for each argument.
    Don't change bseg_filtered, wn_syn_filtered or wn_title_filtered
    """
    parser = argparse.ArgumentParser(description="Gazetteer construction of job titles", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--soc_csv", type=str, default="data/gazetteer/soc.csv", help="CSV filepath containing SOC job titles", dest="soc_csv")
    parser.add_argument("--soc_txt", type=str, default="data/gazetteer/soc.txt", help="TXT filepath to which we will write the SOC job titles", dest="soc_txt")
    parser.add_argument("--bseg", type=str, default="data/gazetteer/bseg.txt", help="TXT filepath to which we will write the BSEG job titles (new job titles created from SOC job titles using backward segmentation)", dest="bseg")
    parser.add_argument("--bseg_filtered", type=str, default="data/gazetteer/bseg.filtered.txt", help="TXT filepath containing BSEG FILTERED (manually filtered BSEG job titles)", dest="bseg_filtered")
    parser.add_argument("--wn_syn", type=str, default="data/gazetteer/wn.syn.txt", help="TXT filepath to which we will write WORDNET SYNSETS (WordNet synsets found from SOC + BSEG job titles)", dest="wn_syn")
    parser.add_argument("--wn_syn_filtered", type=str, default="data/gazetteer/wn.syn.filtered.txt", help="TXT filepath containing WORDNET SYNSETS FILTERED (manually filtered WORDNET SYNSETS)", dest="wn_syn_filtered")
    parser.add_argument("--wn_title", type=str, default="data/gazetteer/wn.title.txt", help="TXT filepath to which we will write WORDNET TITLES (WordNet titles found from WORDNET SYNSETS FILTERED)", dest="wn_title")
    parser.add_argument("--wn_title_filtered", type=str, default="data/gazetteer/wn.title.filtered.txt", help="TXT filepath containing WORDNET TITLES FILTERES (manually filtered WORDNET TITLES)", dest="wn_title_filtered")
    parser.add_argument("--final", type=str, default="data/gazetteer/final.txt", help="TXT filepath to which the final list of job titles will be written", dest="final")
    parser.add_argument("--inflection", type=str, default="data/gazetteer/inflection.csv", help="CSV filepath to which the inflected forms of the job titles in the final list will be written", dest="inflection")

    args = parser.parse_args()

    soc_csv_filepath = args.soc_csv
    soc_txt_filepath = args.soc_txt
    bseg_filepath = args.bseg
    bseg_filtered_filepath = args.bseg_filtered
    wn_syn_filepath = args.wn_syn
    wn_syn_filtered_filepath = args.wn_syn_filtered
    wn_title_filepath = args.wn_title
    wn_title_filtered_filepath = args.wn_title_filtered
    final_filepath = args.final
    inflection_filepath = args.inflection

    backward_segmentation_expansion(soc_csv_filepath, soc_txt_filepath, bseg_filepath)

    find_wordnet_synsets(soc_txt_filepath, bseg_filtered_filepath, wn_syn_filepath)

    wordnet_expansion(soc_txt_filepath, bseg_filtered_filepath, wn_syn_filtered_filepath, wn_title_filepath)

    merge_expansions(soc_txt_filepath, bseg_filtered_filepath, wn_title_filtered_filepath, final_filepath)

    find_inflections(final_filepath, inflection_filepath)

    manually_add_professions(inflection_filepath)

if __name__ == "__main__":
    create_gazetteer()
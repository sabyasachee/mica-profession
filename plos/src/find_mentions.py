import json
import jsonlines
import pandas as pd
import os
from tqdm import tqdm
import nltk
import argparse

def contains_open_left_bracket(text):
    """Returns true if the text contains an open left bracket (parantheses, curly braces, square brackets)
    when read in reverse.

    Parameters
    ----------
    text : str
        The string in which we will check for open left bracket

    Returns
    -------
    True/False
    """
    c = 0
    i = len(text) - 1
    while i >= 0:
        if text[i] in ["(", "{", "["]:
            c += 1
        elif text[i] in [")", "}", "]"]:
            c -= 1
        if c > 0:
            return True
        i -= 1
    return False

def contains_open_right_bracket(text):
    """Returns false if the text contains an open right bracket (parantheses, curly braces, square brackets)
    when read from start.

    Parameters
    ----------
    text : str
        The string in which we will check for open right bracket

    Returns
    -------
    True/False
    """
    c = 0
    i = 0
    while i < len(text):
        if text[i] in ["(", "{", "["]:
            c -= 1
        elif text[i] in [")", "}", "]"]:
            c += 1
        if c > 0:
            return True
        i += 1
    return False

def is_speaker(left_text, mention, right_text):
    """Returns true if the mention is a speaker name.
    The heuristic used is left_text should be empty and right_text should start with `:`

    Parameters
    ----------
    left_text : str
        text on the left side of the mention

    right_text : str
        text on the right side of the mention

    Returns
    -------
    True/False
    """
    return right_text.strip().startswith(":")

def is_song(text):
    """Returns true if text is part of a song.
    The heuristic used is text is song if it contains `♪`

    Parameters
    ----------
    text : str
        text to check if it is part of a song

    Returns
    -------
    True/False
    """
    return "♪" in text

def find_mentions(key_to_si_filepath, si_to_rsi_filepath, sentences_filepath, processed_filepath, profession_filepath, prof_to_si_filepath, mentions_directory, max_n_words):
    """Find mentions of professions in subtitle sentences and save it as a csv. The profession to file sentence index 
    dictionary is also found and saved.

    Parameters
    ----------
    key_to_si_filepath : str \\
        JSON filepath containing key to file sentence index dictionary.
        It is of the form `{key:[(imdb, sent)]}`

    si_to_rsi_filepath : str \\
        JSON filepath containing file sentence index to flattened sentence
        index dictionary. It is of the form `{imdb:{key:rsi}}`

    sentences_filepath : str \\
        TEXT filepath containing subtitle sentences which contain some
        professional mention.

    processed_filepath : str \\
        JSONLINES filepath containing NLP processed sentences which contain
        POS and NER tags.

    prof_to_si_filepath : str \\
        JSON filepath to which the profession to file sentence index
        dictionary will be saved.

    max_n_words : int \\
        subtitle sentences containing more than `max_n_words` are
        ignored.

    mentions_directory : str \\
        Directory to which the mention csv files will be saved.
        Six csv files will be saved:

        1. mentions_directory/mentions_long.csv: sentences which have more than `max_n_words` words.
        2. mentions_directory/mentions_nomatch.csv: mentions that we couldn't match (the subtitle 
            sentence tokens did not match with professional title tokens)
        3. mentions_directory/mentions_bracketed.csv: mentions that are enclosed in brackets
        4. mentions_directory/mentions_speaker.csv: mentions that are speaker names
        5. mentions_directory/mentions_song.csv: mentions that are part of some song.
        6. mentions_directory/mentions.csv: mentions
    """

    # read the dictionaries, profession gazetteer, sentences and NLP processed docs
    print(f"reading dictionaries")
    key_to_si = json.load(open(key_to_si_filepath))
    si_to_rsi = json.load(open(si_to_rsi_filepath))

    print(f"reading profession gazetteer")
    profession_df = pd.read_csv(profession_filepath, index_col = None)

    print(f"reading sentences")
    sentences = open(sentences_filepath).read().strip().split("\n")

    processed = [doc for doc in tqdm(jsonlines.open(processed_filepath), desc="reading processed", total=len(sentences))]

    prof_to_si = {}
    long_sentences = []
    no_match_sentences = []
    bracketed_sentences = []
    speaker_sentences = []
    song_sentences = []
    mentions = []

    # for each profession in the gazetteer, get the singular and plural keys
    # find the list of file sentence indices from these keys using key_to_si
    # for each file sentence index, find the flattened sentence index (rsi)
    # using si_to_rsi
    # find the sentence using rsi
    # find the match using the processed doc tokens
    for _, row in tqdm(profession_df.iterrows(), total=len(profession_df), desc="mentions"):
        prof = row["word"]
        si_list = set([(imdb, sent) for key in [row["singular_key"], row["plural_key"]] for imdb, sent in key_to_si[key]])
        prof_to_si[prof] = []

        for imdb, sent in si_list:
            rsi = si_to_rsi[imdb][str(sent)]
            sentence = sentences[rsi]
            
            if len(sentence.split()) <= max_n_words:
                doc = processed[rsi]
                profession_toks = set([tuple(nltk.wordpunct_tokenize(row["singular"])), tuple(row["singular"].split()), tuple(nltk.wordpunct_tokenize(row["plural"])), tuple(row["plural"].split())])
                text_tok = [word["word"] for word in doc]
                match = set()

                for tok in profession_toks:
                    for i in range(len(text_tok) - len(tok) + 1):
                        text_subtok = [token.lower() for token in text_tok[i:i + len(tok)]]
                        if text_subtok == list(tok):
                            match.add((i, i + len(tok)))

                if match:
                    for i, j in match:
                        left = " ".join(text_tok[:i]).strip()
                        target = " ".join(text_tok[i:j]).strip()
                        right = " ".join(text_tok[j:]).strip()

                        if contains_open_left_bracket(left) and contains_open_right_bracket(right):
                            bracketed_sentences.append([prof, imdb, sent, rsi, left, target, right])
                        elif is_speaker(left, target, right):
                            speaker_sentences.append([prof, imdb, sent, rsi, left, target, right])
                        elif is_song(left) or is_song(right):
                            song_sentences.append([prof, imdb, sent, rsi, sentence])
                        else:
                            prof_to_si[prof].append([imdb, sent])
                            mentions.append([prof, imdb, sent, rsi, left, target, right, i, j])
                else:
                    no_match_sentences.append([prof, imdb, sent, rsi, sentence])
            else:
                long_sentences.append([prof, imdb, sent, rsi, sentence])

    # save profession to file sentence index
    print("saving profession to file sentence index dictionary")
    json.dump(prof_to_si, open(prof_to_si_filepath, "w"))

    # create dataframes
    long_sentences_df = pd.DataFrame(long_sentences, columns=["profession", "imdb", "sent", "rsi", "sentence"])
    no_match_sentences_df = pd.DataFrame(no_match_sentences, columns=["profession", "imdb", "sent", "rsi", "sentence"])
    bracketed_sentences_df = pd.DataFrame(bracketed_sentences, columns=["profession", "imdb", "sent", "rsi", "left", "mention", "right"])
    speaker_sentences_df = pd.DataFrame(speaker_sentences, columns=["profession", "imdb", "sent", "rsi", "left", "mention", "right"])
    song_sentences_df = pd.DataFrame(song_sentences, columns=["profession", "imdb", "sent", "rsi", "sentence"])
    mentions_df = pd.DataFrame(mentions, columns=["profession", "imdb", "sent", "rsi", "left", "mention", "right", "start", "end"])

    # print lengths
    S = len(long_sentences_df) + len(no_match_sentences_df) + len(bracketed_sentences_df) + len(speaker_sentences_df) + len(song_sentences_df) + len(mentions_df)
    print(f"#mentions longer than {max_n_words} words = {len(long_sentences_df)} ({100*len(long_sentences_df)/S:.3f}%)")
    print(f"#mentions not containing profession = {len(no_match_sentences_df)} ({100*len(no_match_sentences_df)/S:.3f}%)")
    print(f"#mentions which are bracketed = {len(bracketed_sentences_df)} ({100*len(bracketed_sentences_df)/S:.3f}%)")
    print(f"#mentions which are speakers = {len(speaker_sentences_df)} ({100*len(speaker_sentences_df)/S:.3f}%)")
    print(f"#mentions which are songs = {len(song_sentences_df)} ({100*len(song_sentences_df)/S:.3f}%)")
    print(f"#mentions = {len(mentions_df)} ({100*len(mentions_df)/S:.3f}%)")

    print("saving mentions")
    long_sentences_df.to_csv(os.path.join(mentions_directory, "mentions_long.csv"), index=False)
    no_match_sentences_df.to_csv(os.path.join(mentions_directory, "mentions_nomatch.csv"), index=False)
    bracketed_sentences_df.to_csv(os.path.join(mentions_directory, "mentions_bracketed.csv"), index=False)
    speaker_sentences_df.to_csv(os.path.join(mentions_directory, "mentions_speaker.csv"), index=False)
    song_sentences_df.to_csv(os.path.join(mentions_directory, "mentions_song.csv"), index=False)
    mentions_df.to_csv(os.path.join(mentions_directory, "mentions.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find mentions", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--k2s", type=str, dest="k2s", help="json file of key to sentence index", default="/proj/sbaruah/subtitle/profession/csl/data/mentions/key_to_si.json")
    parser.add_argument("--s2r", type=str, help="json file of file sentence index to flattened sentence index", default="/proj/sbaruah/subtitle/profession/csl/data/mentions/si_to_rsi.json", dest="s2r")
    parser.add_argument("--sent", type=str, help="txt file of subtitle sentences which have some professiona word", default="/proj/sbaruah/subtitle/profession/csl/data/mentions/sentences.txt", dest="sent")
    parser.add_argument("--proc", type=str, help="json file of processed sentences (tokens, POS, NER)", default="/proj/sbaruah/subtitle/profession/csl/data/mentions/processed_finegrained.jsonl", dest="proc")
    parser.add_argument("--gzt", type=str, help="csv file of profession gazetteer", default="/proj/sbaruah/subtitle/profession/csl/data/gazetteer/inflection.csv", dest="gzt")
    parser.add_argument("--p2s", type=str, help="json file to which profession to file sentence index will be saved", default="/proj/sbaruah/subtitle/profession/csl/data/mentions/prof_to_si.json", dest="p2s")
    parser.add_argument("--mentions", type=str, help="directory to which the mention csv files will be saved", default="/proj/sbaruah/subtitle/profession/csl/data/mentions/", dest="mentions")
    parser.add_argument("--max_words", type=int, help="Maximum number of words allowed in the subtitle sentence", default=100, dest="max")

    args = parser.parse_args()
    key_to_si_filepath = args.k2s
    si_to_rsi_filepath = args.s2r
    sentences_filepath = args.sent
    processed_filepath = args.proc
    profession_filepath = args.gzt
    prof_to_si_filepath = args.p2s
    mentions_directory = args.mentions
    max_n_words = args.max

    find_mentions(key_to_si_filepath, si_to_rsi_filepath, sentences_filepath, processed_filepath, profession_filepath, prof_to_si_filepath, mentions_directory, max_n_words)
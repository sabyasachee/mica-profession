import pandas as pd
import jsonlines
from nltk.corpus import wordnet as wn
from tqdm import tqdm
import argparse

def filter_mentions(mentions_file, professions_file, filtered_mentions_file, filtered_professions_file, \
    zero_sense_professions_file, professional_senses_file, pos_ner_file, wsd_file, nopos_wsd_file):
    print("reading mentions")
    mentions_df = pd.read_csv(mentions_file, index_col=None)

    print("reading professions")
    professions_df = pd.read_csv(professions_file, index_col=None)

    print("reading profession words with zero senses")
    zero_sense_professions = set(open(zero_sense_professions_file).read().strip().split("\n"))

    print("reading professional senses")
    professional_senses = set([wn.synset(line.split()[0]) for line in open(professional_senses_file)])

    print("removing profession words that have zero sense and are not professions")
    all_zero_sense_professions = set(professions_df[professions_df.n_synsets == 0].profession)
    delete_zero_sense_professions = all_zero_sense_professions.difference(zero_sense_professions)
    mentions_df = mentions_df[~mentions_df.profession.isin(delete_zero_sense_professions)]

    print("removing profession words that have some sense but no professional senses")
    nonzero_sense_professions = set(professions_df[professions_df.n_synsets > 0].profession)
    delete_zero_professional_sense_professions = set()
    for profession in nonzero_sense_professions:
        senses = set(wn.synsets(profession.lower().replace(" ","_")))
        if len(senses.intersection(professional_senses)) == 0:
            delete_zero_professional_sense_professions.add(profession)
    mentions_df = mentions_df[~mentions_df.profession.isin(delete_zero_professional_sense_professions)]

    print("creating mention professions file")
    records = []

    S = 0
    for profession, df in mentions_df.groupby("profession"):
        n_words = len(profession.split())
        synsets = wn.synsets(profession.lower().replace(" ", "_"))
        noun_synsets = wn.synsets(profession.lower().replace(" ", "_"), pos="n")
        professional_synsets = set(synsets).intersection(professional_senses)
        n_synsets = len(synsets)
        n_noun_synsets = len(noun_synsets)
        n_mentions = len(df)
        n_professional_synsets = len(professional_synsets)
        n_non_professional_synsets = len(synsets) - len(professional_synsets)
        S += n_mentions

        records.append([profession, n_mentions, n_words, n_synsets, n_noun_synsets, n_professional_synsets, \
            n_non_professional_synsets])

    records = sorted(records, key=lambda record: record[1], reverse=True)
    cumul_frac_mentions = []
    s = 0

    for record in records:
        s += record[1]
        cumul_frac_mentions.append(s/S)

    print(f"\nsome statistics =>")
    print(f"top 100 professions cover {100*cumul_frac_mentions[99]:.2f}% mentions")
    print(f"top 500 professions cover {100*cumul_frac_mentions[499]:.2f}% mentions")
    print(f"top 1000 professions cover {100*cumul_frac_mentions[999]:.2f}% mentions")
    print(f"top 1500 professions cover {100*cumul_frac_mentions[1499]:.2f}% mentions\n")

    print("saving mention professions")
    filtered_professions_df = pd.DataFrame(records, columns=["profession", "n_mentions", "n_words", "n_senses",\
        "n_noun_senses", "n_professional_senses", "n_non_professional_senses"])
    filtered_professions_df.to_csv(filtered_professions_file, index=False)

    print("reading pos and ner")
    pos_ner_docs = list(tqdm(jsonlines.open(pos_ner_file), total=3577313))

    print("reading mention senses")
    wsd_docs = list(tqdm(jsonlines.open(wsd_file), total=3577313))

    print("reading no pos mention senses")
    nopos_wsd_docs = list(tqdm(jsonlines.open(nopos_wsd_file), total=3577313))

    print("adding pos, ner, sense and no pos sense columns for unigram mentions")
    start_list, end_list, rsi_list = mentions_df.start.values, mentions_df.end.values, mentions_df.rsi.values
    pos_list, ner_list, sense_list, nopos_sense_list = [], [], [], []
    
    for start, end, rsi in tqdm(zip(start_list, end_list, rsi_list), total=len(start_list)):
        pos, ner, sense, nopos_sense = None, None, None, None
        if end - start == 1:
            token = pos_ner_docs[rsi][start]
            pos, ner = token["pos"], token["ner"]
            sense_str = wsd_docs[rsi][start]
            if sense_str:
                sense = sense_str
            nopos_sense_str = nopos_wsd_docs[rsi][start]
            if nopos_sense_str:
                nopos_sense = nopos_sense_str
        pos_list.append(pos)
        ner_list.append(ner)
        sense_list.append(sense)
        nopos_sense_list.append(nopos_sense)

    mentions_df["pos"] = pos_list
    mentions_df["ner"] = ner_list
    mentions_df["sense"] = sense_list
    mentions_df["no_pos_sense"] = nopos_sense_list

    print("saving mentions")
    mentions_df.to_csv(filtered_mentions_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="filter mentions and professions according to profession word, and \
        create new mentions and professions file", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--in_mentions", default="data/mentions/mentions.csv", help="mentions file")
    parser.add_argument("--in_professions", default="data/mentions/professions.csv", help="professions file")
    parser.add_argument("--out_mentions", default="data/mentions/mentions.word_filtered.csv", help="filtered mentions \
        file")
    parser.add_argument("--out_professions", default="data/mentions/professions.word_filtered.csv", help="filtered \
        professions file")
    parser.add_argument("--zero_sense_professions", default="data/gazetteer/title.zerosense.filtered.txt", help="\
        file containing words that have zero wordnet sense and are professions")
    parser.add_argument("--professional_senses", default="data/gazetteer/wn.syn.mention.txt", help="file containing \
        professional senses")
    parser.add_argument("--pos_ner_docs", default="data/mentions/pos.ner.jsonl", help="pos and ner docs")
    parser.add_argument("--wsd_docs", default="data/mentions/wsd.jsonl", help="wsd docs")
    parser.add_argument("--wsd_no_pos_docs", default="data/mentions/wsd.nopos.jsonl", help="wsd without pos tagging \
        docs")

    args = parser.parse_args()
    mentions_file = args.in_mentions
    professions_file = args.in_professions
    filtered_mentions_file = args.out_mentions
    filtered_professions_file = args.out_professions
    zero_sense_professions_file = args.zero_sense_professions
    professional_senses_file = args.professional_senses
    pos_ner_file = args.pos_ner_docs
    wsd_file = args.wsd_docs
    nopos_wsd_file = args.wsd_no_pos_docs

    filter_mentions(mentions_file, professions_file, filtered_mentions_file, filtered_professions_file, \
    zero_sense_professions_file, professional_senses_file, pos_ner_file, wsd_file, nopos_wsd_file)
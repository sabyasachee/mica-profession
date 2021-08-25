import pandas as pd
from nltk.corpus import wordnet
import argparse
from tqdm import tqdm

def list_professions(mentions_filepath, professions_filepath):
    print("reading mentions")
    mentions = pd.read_csv(mentions_filepath, index_col=None)

    records = []

    print("finding profession in mentions")
    for profession, df in mentions.groupby("profession"):
        n_words = len(profession.split())
        synsets = wordnet.synsets(profession.lower().replace(" ", "_"))
        noun_synsets = wordnet.synsets(profession.lower().replace(" ", "_"), pos="n")
        n_synsets = len(synsets)
        n_noun_synsets = len(noun_synsets)
        n_mentions = len(df)

        records.append([profession, n_mentions, n_words, n_synsets, n_noun_synsets])
    
    records = sorted(records, key=lambda record: record[1], reverse=True)
    cumul_frac_mentions = []
    s = 0
    S = len(mentions)

    for record in records:
        s += record[1]
        cumul_frac_mentions.append(s/S)

    print(f"\nsome statistics =>")
    print(f"top 100 professions cover {100*cumul_frac_mentions[99]:.2f}% mentions")
    print(f"top 500 professions cover {100*cumul_frac_mentions[499]:.2f}% mentions")
    print(f"top 1000 professions cover {100*cumul_frac_mentions[999]:.2f}% mentions")
    print(f"top 1500 professions cover {100*cumul_frac_mentions[1499]:.2f}% mentions\n")

    print("saving professions data")
    professions_df = pd.DataFrame(records, columns=["profession","n_mentions","n_words","n_synsets","n_noun_synsets"])
    professions_df.to_csv(professions_filepath, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List the professions found in the subtitles", \
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mentions", help="csv file of mentions", \
        default="/proj/sbaruah/subtitle/profession/csl/data/mentions/mentions.csv", dest="mentions")
    parser.add_argument("--profession", help="csv file to which professions will be saved", \
        default="/proj/sbaruah/subtitle/profession/csl/data/mentions/professions.csv", dest="profession")

    args = parser.parse_args()
    mentions_filepath = args.mentions
    professions_filepath = args.profession

    list_professions(mentions_filepath, professions_filepath)
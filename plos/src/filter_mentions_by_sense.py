import pandas as pd
import argparse

def filter_mentions_by_sense(in_mentions_file, in_professions_file, out_mentions_file, out_professions_file):
    print("reading mentions")
    mentions_df = pd.read_csv(in_mentions_file, index_col=None)

    print("reading professions")
    professions_df = pd.read_csv(in_professions_file, index_col=None)

    print("filtering by is_nopos_profession")
    mentions_df = mentions_df[mentions_df.is_nopos_profession == 1]

    print("updating n_mentins in professions")
    profession_counts = dict([(profession, 0) for profession in professions_df.profession])
    
    for profession, df in mentions_df.groupby("profession"):
        profession_counts[profession] = len(df)

    n_mentions_list = [profession_counts[profession] for profession in professions_df.profession]
    professions_df.n_mentions = n_mentions_list

    professions_df.sort_values(by="n_mentions", ascending=False, inplace=True)

    print("dropping professions with zero n_mentions")
    professions_df = professions_df[professions_df.n_mentions > 0]

    s, S = 0, professions_df.n_mentions.sum()
    cumul_frac_mentions = []

    for n_mentions in professions_df.n_mentions:
        s += n_mentions
        cumul_frac_mentions.append(s/S)

    print(f"\nsome statistics =>")
    print(f"top 100 professions cover {100*cumul_frac_mentions[99]:.2f}% mentions")
    print(f"top 500 professions cover {100*cumul_frac_mentions[499]:.2f}% mentions")
    print(f"top 1000 professions cover {100*cumul_frac_mentions[999]:.2f}% mentions")
    print(f"top 1500 professions cover {100*cumul_frac_mentions[1499]:.2f}% mentions\n")

    print("saving mentions")
    mentions_df.to_csv(out_mentions_file, index=False)

    print("saving professions")
    professions_df.to_csv(out_professions_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="filter by sense", \
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--in_mentions", default="data/mentions/mentions.word_filtered.prediction_added.csv", \
        help="mentions file")
    parser.add_argument("--in_professions", default="data/mentions/professions.word_filtered.prediction_added.csv", \
        help="professions file")
    parser.add_argument("--out_mentions", default="data/mentions/mentions.word_filtered.sense_filtered.csv", \
        help="mentions file filtered by sense")
    parser.add_argument("--out_professions", default="data/mentions/professions.word_filtered.sense_filtered.csv", \
        help="professions file containing updated n_mentions after sense filtering")

    args = parser.parse_args()
    in_mentions_file = args.in_mentions
    in_professions_file = args.in_professions
    out_mentions_file = args.out_mentions
    out_professions_file = args.out_professions

    filter_mentions_by_sense(in_mentions_file, in_professions_file, out_mentions_file, out_professions_file)
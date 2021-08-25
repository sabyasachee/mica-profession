import pandas as pd

def find_frequency_by_merged_profession(mentions_file, frequency_file, professions_file, \
    merged_profession_frequency_file):
    print("reading mentions")
    mentions_df = pd.read_csv(mentions_file, index_col=None, dtype={"soc_code":str, "soc_name":str})
    frequency_df = pd.read_csv(frequency_file, index_col=None)
    professions_df = pd.read_csv(professions_file, index_col=None)

    merged_professions = professions_df.profession_merge.unique()[:500]
    print(f"top {len(merged_professions)} merged professions considered")

    n_mentions_covered = professions_df[professions_df.profession_merge.isin(merged_professions)].n_mentions.sum()
    n_mentions_total = professions_df.n_mentions.sum()
    percent = 100*n_mentions_covered/n_mentions_total
    print(f"{percent:.2f}% mentions covered")

    print("finding merged profession frequency")
    map_df = mentions_df.loc[mentions_df.profession_merge.isin(merged_professions), ["profession","profession_merge"]]\
        .drop_duplicates()
    merged_frequency_df = frequency_df.merge(map_df, on=["profession"]).drop(columns=["profession","no_pos_sense"])
    merged_frequency_df = merged_frequency_df.groupby("profession_merge").agg(sum)

    print("saving merged frequency data")
    merged_frequency_df.to_csv(merged_profession_frequency_file)

if __name__ == "__main__":
    find_frequency_by_merged_profession("data/mentions/mentions.word_filtered.sense_filtered.soc_mapped.merged.csv", \
        "data/mentions/frequency.csv", "data/mentions/professions.word_filtered.sense_filtered.merged.csv",\
             "data/analysis_data/top500_merged_profession_frequency.csv")
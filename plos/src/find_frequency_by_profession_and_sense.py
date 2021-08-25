import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def find_frequency_by_profession_and_sense(mentions_file, imdb_file, frequency_file, max_ngram, max_year, min_year, sample=False, cutoff=False):
    print("reading imdb data")
    imdb_df = pd.read_csv(imdb_file, index_col=None)

    if sample:
        print("sampling imdb data")
        min_n_imdb_titles_in_year = imdb_df[(imdb_df.year >= min_year) & (imdb_df.year <= max_year)].groupby("year").agg(len).imdb_ID.min()
        imdb_year_dfs = []

        for year, imdb_year_df in imdb_df.groupby("year"):
            if year >= min_year and year <= max_year:
                sampled_df = imdb_year_df.sample(n=min_n_imdb_titles_in_year, replace=False)
                sampled_df["year"] = year
                imdb_year_dfs.append(sampled_df)

        imdb_df = pd.concat(imdb_year_dfs)

    if cutoff:
        print("cutoff imdb data")
        quant = np.quantile(imdb_df["1_gram_count"], 0.99)
        imdb_df = imdb_df[imdb_df["1_gram_count"] < quant]

    print("reading mentions data")
    mentions_df = pd.read_csv(mentions_file, index_col=None, dtype={"soc_code":str, "soc_name":str})

    if sample or cutoff:
        mentions_df = mentions_df[mentions_df.imdb.isin(imdb_df.imdb_ID)]
    
    imdb_year_data = {}
    year_ngram_count_data = defaultdict(lambda: defaultdict(int))

    print("finding imdb year and year ngram count")
    for _, row in tqdm(imdb_df.iterrows(), total=len(imdb_df)):
        year, imdb = row.year, row.imdb_ID
        ngram_count_arr = [row[f"{i + 1}_gram_count"] for i in range(max_ngram)]
        imdb_year_data[imdb] = year
        for i in range(max_ngram):
            year_ngram_count_data[year][i] += ngram_count_arr[i]

    imdb_col = mentions_df.imdb.values
    year_col = np.zeros(len(imdb_col), dtype=int)

    print("adding year to mentions file")
    for imdb, year in tqdm(imdb_year_data.items()):
        year_col[imdb_col == imdb] = year
    mentions_df["year"] = year_col

    profession_sense_frequency_data = defaultdict(lambda: defaultdict(int))
    mentions_df.no_pos_sense.fillna("none", inplace=True)
    n_groups = len(mentions_df[["profession", "no_pos_sense", "year"]].drop_duplicates())

    print("finding profession and sense counts by year")
    for (profession, sense, year), df in tqdm(mentions_df.groupby(["profession", "no_pos_sense", "year"]), total=n_groups):
        profession_ngram_size = len(profession.split())
        year_ngram_count = year_ngram_count_data[year][profession_ngram_size-1]
        profession_sense_frequency_data[(profession, sense)][year] = len(df)/year_ngram_count

    records = []

    print("creating frequency data")
    for (profession, sense), frequency_data in tqdm(profession_sense_frequency_data.items()):
        record = np.zeros(max_year - min_year + 1)
        for year, frequency in frequency_data.items():
            if year >= min_year and year <= max_year:
                record[year - min_year] = frequency
        records.append([profession, sense] + record.tolist())

    frequency_df = pd.DataFrame(records, columns=["profession", "no_pos_sense"] + [str(year) for year in range(min_year, max_year + 1)])
    frequency_df.no_pos_sense.replace("none", np.nan, inplace=True)

    print("saving frequency data")
    frequency_df.to_csv(frequency_file, index=False)

if __name__ == "__main__":
    # find_frequency_by_profession_and_sense("data/mentions/mentions.word_filtered.sense_filtered.soc_mapped.merged.csv", "data/imdb/imdb.ngram.csv", "data/mentions/frequency.csv", 5, 2017, 1950)
    find_frequency_by_profession_and_sense("data/mentions/mentions.word_filtered.sense_filtered.soc_mapped.merged.csv", "data/imdb/imdb.ngram.csv", "data/mentions/frequency.year.csv", 5, 2017, 1980)
    # find_frequency_by_profession_and_sense("data/mentions/mentions.word_filtered.sense_filtered.soc_mapped.merged.csv", "data/imdb/imdb.ngram.csv", "data/mentions/frequency.sample.csv", 5, 2017, 1950, sample=True)
    # find_frequency_by_profession_and_sense("data/mentions/mentions.word_filtered.sense_filtered.soc_mapped.merged.csv", "data/imdb/imdb.ngram.csv", "data/mentions/frequency.cutoff.csv", 5, 2017, 1950, cutoff=True)
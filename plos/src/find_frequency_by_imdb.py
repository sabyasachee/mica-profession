from typing import Counter
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import argparse

def find_frequency_by_imdb(mentions_file, imdb_file, professions_file, imdb_frequency_file, min_year, max_year):
    print("reading mentions data")
    mentions_df = pd.read_csv(mentions_file, index_col=None, dtype={"soc_code":str, "soc_name":str})

    print("reading imdb data")
    imdb_attr_df = pd.read_csv(imdb_file, index_col=None)

    print("finding imdb ngram count")
    imdb_ngram_dict = {}
    for _, row in imdb_attr_df.iterrows():
        imdb_ngram_dict[row["imdb_ID"]] = [row["1_gram_count"], row["2_gram_count"], row["3_gram_count"], row["4_gram_count"], row["5_gram_count"]]

    print("reading professions data")
    professions_df = pd.read_csv(professions_file, index_col=None)
    professions = professions_df.profession_merge.unique()[:500]

    print("finding frequency by imdb")
    imdb_profession_frequency_dict = defaultdict(lambda: defaultdict(int))
    for imdb, imdb_df in tqdm(mentions_df.groupby("imdb"), total=mentions_df["imdb"].unique().size, desc="imdb"):
        for profession, profession_df in imdb_df.groupby("profession_merge"):
            if profession in professions:
                ngram = len(profession.split())
                frequency = len(profession_df)/imdb_ngram_dict[imdb][ngram - 1]
                imdb_profession_frequency_dict[imdb][profession] = frequency

    print("creating imdb profession frequency data")
    imdb_profession_frequency_data = []
    for imdb, profession_frequency_dict in imdb_profession_frequency_dict.items():
        record = [imdb]
        for profession in professions:
            record.append(profession_frequency_dict[profession])
        imdb_profession_frequency_data.append(record)

    imdb_frequency_df = pd.DataFrame(data=imdb_profession_frequency_data, columns=["imdb_ID"] + professions.tolist())
    imdb_frequency_df = imdb_attr_df[["imdb_ID", "imdb_kind", "year", "imdb_genres", "imdb_countries", "box_office_in_dollars"]].merge(imdb_frequency_df, on="imdb_ID", how="left")
    imdb_frequency_df = imdb_frequency_df[(imdb_frequency_df["year"] >= min_year) & (imdb_frequency_df["year"] <= max_year)]
    imdb_frequency_df = imdb_frequency_df.sort_values(by=["year", "imdb_ID"])
    imdb_frequency_df[professions] = imdb_frequency_df[professions].fillna(0)

    print("converting categorical variables to boolean columns")
    
    imdb_kind_set = imdb_frequency_df["imdb_kind"].dropna().unique().tolist()
    imdb_kind_data = np.zeros((len(imdb_frequency_df), len(imdb_kind_set)), dtype=np.int)
    for i, imdb_kind in tqdm(enumerate(imdb_frequency_df["imdb_kind"]), desc="imdb_kind"):
        if pd.notna(imdb_kind):
            imdb_kind_data[i, imdb_kind_set.index(imdb_kind)] = 1
    imdb_kind_header = [f"kind_{kind.replace(' ','_')}" for kind in imdb_kind_set]
    
    imdb_genre_set = list(set([genre for genre_list in imdb_frequency_df["imdb_genres"].dropna().str.split(";") for genre in genre_list]))
    imdb_genre_data = np.zeros((len(imdb_frequency_df), len(imdb_genre_set)), dtype=np.int)
    for i, imdb_genres in tqdm(enumerate(imdb_frequency_df["imdb_genres"]), desc="imdb_genre"):
        if pd.notna(imdb_genres):
            for genre in imdb_genres.split(";"):
                imdb_genre_data[i, imdb_genre_set.index(genre)] = 1
    imdb_genre_header = [f"genre_{genre.replace('-','_')}" for genre in imdb_genre_set]

    imdb_country_list = [country for country_list in imdb_frequency_df["imdb_countries"].dropna().str.split(";") for country in country_list]
    imdb_country_items = sorted(Counter(imdb_country_list).items(), key = lambda x: x[1], reverse=True)
    imdb_country_set = [country for country, _ in imdb_country_items[:25]]
    imdb_country_data = np.zeros((len(imdb_frequency_df), len(imdb_country_set)), dtype=np.int)
    for i, imdb_countries in tqdm(enumerate(imdb_frequency_df["imdb_countries"]), desc="imdb_country"):
        if pd.notna(imdb_countries):
            for country in imdb_countries.split(";"):
                if country in imdb_country_set:
                    imdb_country_data[i, imdb_country_set.index(country)] = 1
    imdb_country_header = [f"country_{country.replace(' ','_')}" for country in imdb_country_set]

    imdb_frequency_df = imdb_frequency_df.drop(columns=["imdb_kind", "imdb_genres", "imdb_countries"])
    imdb_frequency_df = pd.concat([imdb_frequency_df, pd.DataFrame(data=imdb_kind_data, columns=imdb_kind_header)], axis=1)
    imdb_frequency_df = pd.concat([imdb_frequency_df, pd.DataFrame(data=imdb_genre_data, columns=imdb_genre_header)], axis=1)
    imdb_frequency_df = pd.concat([imdb_frequency_df, pd.DataFrame(data=imdb_country_data, columns=imdb_country_header)], axis=1)
    imdb_frequency_df = imdb_frequency_df[["imdb_ID", "year"] + imdb_kind_header + imdb_genre_header + imdb_country_header + ["box_office_in_dollars"] + professions.tolist()]

    imdb_frequency_df.to_csv(imdb_frequency_file, index=False)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mentions", type=str, default="data/mentions/mentions.word_filtered.sense_filtered.soc_mapped.merged.csv")
    parser.add_argument("--imdb", type=str, default="data/imdb/imdb.ngram.box_office.csv")
    parser.add_argument("--professions", type=str, default="data/mentions/professions.word_filtered.sense_filtered.merged.csv")
    parser.add_argument("--out_imdb_frequency", type=str, default="data/analysis_data/imdb_frequency.csv")
    parser.add_argument("--min_year", type=int, default=1950)
    parser.add_argument("--max_year", type=int, default=2017)
    args = parser.parse_args()
    find_frequency_by_imdb(args.mentions, args.imdb, args.professions, args.out_imdb_frequency, args.min_year, args.max_year)

if __name__ == "__main__":
    main()
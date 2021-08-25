import os
import pandas as pd
import numpy as np
from tqdm import tqdm, trange

def collect(x):
    return set(val for val in x)

def create_data(mentions_file, professions_file, imdb_file, profession_media_dir, soc_media_dir):

    mentions_df = pd.read_csv(mentions_file, index_col=None, dtype={"soc_code":str, "soc_name":str})
    professions_df = pd.read_csv(professions_file, index_col=None)
    imdb_df = pd.read_csv(imdb_file, index_col=None)

    genres = set()
    countries = set()

    for genre_str in imdb_df["imdb_genres"]:
        if pd.notna(genre_str):
            genres.update(genre_str.split(";"))
    
    for country_str in imdb_df["imdb_countries"]:
        if pd.notna(country_str):
            countries.update(country_str.split(";"))

    genres = sorted(genres)
    countries = sorted(countries)

    genre_data = np.zeros((len(imdb_df), len(genres)), dtype=int)
    country_data = np.zeros((len(imdb_df), len(countries)), dtype=int)

    for i, genre_str in enumerate(imdb_df["imdb_genres"]):
        if pd.notna(genre_str):
            for genre in genre_str.split(";"):
                j = genres.index(genre)
                genre_data[i, j] = 1

    for i, country_str in enumerate(imdb_df["imdb_countries"]):
        if pd.notna(country_str):
            for country in country_str.split(";"):
                j = countries.index(country)
                country_data[i, j] = 1

    genre_columns = ["Genre:" + genre for genre in genres]
    country_columns = ["Country:" + country for country in countries]

    genre_df = pd.DataFrame(genre_data, columns=genre_columns)
    country_df = pd.DataFrame(country_data, columns=country_columns)

    imdb_df = imdb_df.drop(columns=["imdb_genres", "imdb_countries"])
    imdb_df = pd.concat([imdb_df, genre_df, country_df], axis=1)
    imdb_df = imdb_df[imdb_df["imdb_kind"].notna()]
    imdb_df = imdb_df[["imdb_ID", "year", "imdb_kind"] + genre_columns + country_columns + ["1_gram_count", "2_gram_count", "3_gram_count", "4_gram_count", "5_gram_count"]]
    imdb_df = imdb_df.rename(columns={"imdb_ID": "imdb", "imdb_kind": "kind"})
    imdb_df = imdb_df.groupby(["year", "kind"] + genre_columns + country_columns).agg({"1_gram_count": sum, "2_gram_count": sum, "3_gram_count": sum, "4_gram_count": sum, "5_gram_count": sum, "imdb": collect})
    imdb_df["n_titles"] = imdb_df["imdb"].apply(lambda x: len(x))

    imdb_to_rowindex = {}
    for i, imdb_set in enumerate(imdb_df["imdb"]):
        for imdb in imdb_set:
            imdb_to_rowindex[imdb] = i
    imdb_df = imdb_df.drop(columns=["imdb"])

    professions = professions_df["profession_merge"].unique()[:500]

    for profession in tqdm(professions, desc="profession"):
        
        profession_mentions_df = mentions_df[mentions_df["profession_merge"] == profession]
        profession_media_df = imdb_df.copy()
        n = len(profession.split())
        profession_media_df["n_total_mentions"] = imdb_df[f"{n}_gram_count"]
        profession_media_df = profession_media_df.drop(columns=["1_gram_count", "2_gram_count", "3_gram_count", "4_gram_count", "5_gram_count"])

        n_mentions = np.zeros(len(profession_media_df), dtype=int)
        n_pos_mentions = np.zeros(len(profession_media_df), dtype=int)
        n_neg_mentions = np.zeros(len(profession_media_df), dtype=int)

        for _, row in tqdm(profession_mentions_df.iterrows(), total=len(profession_mentions_df), desc=profession):
            imdb = row["imdb"]
            sentiment = row["sentiment_label"]
            
            if imdb in imdb_to_rowindex:
                i = imdb_to_rowindex[imdb]
                n_mentions[i] += 1
                if sentiment == 1:
                    n_pos_mentions[i] += 1
                elif sentiment == -1:
                    n_neg_mentions[i] += 1

        profession_media_df["n_mentions"] = n_mentions
        profession_media_df["n_pos_mentions"] = n_pos_mentions
        profession_media_df["n_neg_mentions"] = n_neg_mentions
        profession_media_df["freq"] = n_mentions/profession_media_df["n_total_mentions"]
        profession_media_df["sentiment"] = n_pos_mentions/(n_pos_mentions + n_neg_mentions + 1e-23)
    
        file = os.path.join(profession_media_dir, f"{profession}.csv")
        profession_media_df.to_csv(file, index=True)

    for x in trange(23, desc="soc"):

        soc_code = 11 + 2*x
        soc_mentions_df = mentions_df[mentions_df['soc_code'].fillna('').str.contains(str(soc_code))].copy()
        soc_mentions_df["n"] = soc_mentions_df["profession_merge"].apply(lambda x: len(x.split()))
        ns = soc_mentions_df["n"].unique()

        for n in tqdm(ns, desc=str(soc_code)):
            socn_mentions_df = soc_mentions_df[soc_mentions_df["n"] == n]
            socn_media_df = imdb_df.copy()
            socn_media_df["n_total_mentions"] = imdb_df[f"{n}_gram_count"]
            socn_media_df = socn_media_df.drop(columns=["1_gram_count", "2_gram_count", "3_gram_count", "4_gram_count", "5_gram_count"])

            n_mentions = np.zeros(len(socn_media_df), dtype=int)
            n_pos_mentions = np.zeros(len(socn_media_df), dtype=int)
            n_neg_mentions = np.zeros(len(socn_media_df), dtype=int)

            for _, row in tqdm(socn_mentions_df.iterrows(), desc=str(n), total=len(socn_mentions_df)):
                imdb = row["imdb"]
                sentiment = row["sentiment_label"]
                
                if imdb in imdb_to_rowindex:
                    i = imdb_to_rowindex[imdb]
                    n_mentions[i] += 1
                    if sentiment == 1:
                        n_pos_mentions[i] += 1
                    elif sentiment == -1:
                        n_neg_mentions[i] += 1

            socn_media_df["n_mentions"] = n_mentions
            socn_media_df["n_pos_mentions"] = n_pos_mentions
            socn_media_df["n_neg_mentions"] = n_neg_mentions
            socn_media_df["freq"] = n_mentions/socn_media_df["n_total_mentions"]
            socn_media_df["sentiment"] = n_pos_mentions/(n_pos_mentions + n_neg_mentions + 1e-23)
        
            file = os.path.join(soc_media_dir, f"{soc_code}.{n}.csv")
            socn_media_df.to_csv(file, index=True)

if __name__ == "__main__":
    mentions_file = "/proj/sbaruah/subtitle/profession/csl/data/mentions/mentions.word_filtered.sense_filtered.soc_mapped.merged.sentiment_added.csv"
    professions_file = "/proj/sbaruah/subtitle/profession/csl/data/mentions/professions.word_filtered.sense_filtered.merged.csv"
    imdb_file = "/proj/sbaruah/subtitle/profession/csl/data/imdb/imdb.ngram.box_office.csv"
    profession_media_dir = "/proj/sbaruah/subtitle/profession/csl/data/analysis_data/media_data/profession"
    soc_media_dir = "/proj/sbaruah/subtitle/profession/csl/data/analysis_data/media_data/soc"
    create_data(mentions_file, professions_file, imdb_file, profession_media_dir, soc_media_dir)
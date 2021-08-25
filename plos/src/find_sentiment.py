import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def find_sentiment(mentions_file, imdb_file, profession_file, out_profession_file, out_soc_file):
    print('reading mentions, imdb, professions')
    mentions_df = pd.read_csv(mentions_file, index_col=None, dtype={'soc_code':str, 'soc_name':str})
    imdb_df = pd.read_csv(imdb_file, index_col=None)
    profession_df = pd.read_csv(profession_file, index_col=None)
    
    print('adding year col to mentions')
    mentions_df = mentions_df.merge(imdb_df, left_on='imdb', right_on='imdb_ID', how='left')
    professions = profession_df.profession_merge.unique()[:500]

    print('finding sentiment by profession')
    gdf = mentions_df[mentions_df.profession_merge.isin(professions)].groupby(['profession','year','sentiment_label']).agg({'imdb':len}).imdb
    gdf = gdf.reindex(index=pd.MultiIndex.from_product([professions, np.arange(1950, 2018), [-1,0,1]], names=['profession','year','sentiment']), fill_value=0)
    gdf.to_csv(out_profession_file, header=True)

    print('finding sentiment by soc')
    name_year_sentiment_dict = defaultdict(lambda: {'imdb':0})
    df = mentions_df[mentions_df.soc_name.notna()]
    name_col = df.soc_name.values
    year_col = df.year.values
    label_col = df.sentiment_label.values
    names = set()

    for name_str, year, label in tqdm(zip(name_col, year_col, label_col), total=len(name_col)):
        for name in name_str.split(';'):
            names.add(name)
            name_year_sentiment_dict[(name, year, label)]['imdb'] += 1

    hdf = pd.DataFrame.from_dict(name_year_sentiment_dict, orient='index')
    hdf = hdf.reindex(index=pd.MultiIndex.from_product([sorted(names), np.arange(1950, 2018), [-1,0,1]], names=['soc','year','sentiment']), fill_value=0)
    hdf.to_csv(out_soc_file)

if __name__=='__main__':
    find_sentiment('data/mentions/mentions.word_filtered.sense_filtered.soc_mapped.merged.sentiment_added.csv', 'data/imdb/imdb.csv', 'data/mentions/professions.word_filtered.sense_filtered.merged.csv', 'data/analysis_data/top500_merged_profession_sentiment.csv', 'data/analysis_data/soc_sentiment.csv')
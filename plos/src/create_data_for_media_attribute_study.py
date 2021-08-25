import pandas as pd
import tqdm
import argparse

def create_data_for_media_attribute_study(mentions_file, professions_file, imdb_file, soc_media_attribute_file, profession_media_attribute_file):
    print('reading mentions, professions, imdb')
    mentions_df = pd.read_csv(mentions_file, index_col=None, dtype={'soc_code':str, 'soc_name':str})
    professions_df = pd.read_csv(professions_file, index_col=None)
    imdb_df = pd.read_csv(imdb_file, index_col=None)

    print('merging mentions and imdb')
    mentions_df = mentions_df.merge(imdb_df, left_on='imdb', right_on='imdb_ID', how='inner')[['soc_code','profession_merge','imdb','sentiment_label','year','imdb_kind','imdb_genres','imdb_countries','1_gram_count','2_gram_count','3_gram_count','4_gram_count','5_gram_count']].copy()
    print()

    soc_records = []
    print('creating media attribute study data for soc groups')
    for i in tqdm.trange(23):
        soc_code = 11 + 2 * i
        soc_mentions_df = mentions_df[mentions_df['soc_code'].fillna('').str.contains(str(soc_code))]
        groups = soc_mentions_df.groupby(['profession_merge','year','imdb_kind','imdb_genres','imdb_countries'], dropna=False)

        for (profession, year, kind, genres, countries), group_df in tqdm.tqdm(groups, total=groups.ngroups):
            n = len(profession.split())
            n_total_mentions = group_df.drop_duplicates(subset=['imdb'])[f'{n}_gram_count'].sum()
            n_mentions = len(group_df)
            n_pos_mentions = (group_df.sentiment_label == 1).sum()
            n_neg_mentions = (group_df.sentiment_label == -1).sum()
            n_titles = group_df.imdb.unique().size
            soc_records.append([soc_code, profession, year, kind, genres, countries, n_titles, n_mentions, n_total_mentions, n_pos_mentions, n_neg_mentions])

    print('creating freq and sentiment var')
    soc_media_attribute_df = pd.DataFrame(soc_records, columns=['soc_code','profession','year','kind','genres','countries','n_titles','n_mentions','n_total_mentions','n_pos_mentions','n_neg_mentions'])
    soc_media_attribute_df['freq'] = soc_media_attribute_df['n_mentions']/soc_media_attribute_df['n_total_mentions']
    soc_media_attribute_df['sentiment'] = soc_media_attribute_df['n_pos_mentions']/(soc_media_attribute_df['n_pos_mentions'] + soc_media_attribute_df['n_neg_mentions'] + 1e-23)
    soc_media_attribute_df.to_csv(soc_media_attribute_file, index=False)
    print()

    professions = professions_df['profession_merge'].unique()[:500]
    groups = mentions_df[mentions_df['profession_merge'].isin(professions)].groupby(['profession_merge','year','imdb_kind','imdb_genres','imdb_countries'], dropna=False)
    profession_records = []
    print('creating media attribute study data for professions')
        
    for (profession, year, kind, genres, countries), group_df in tqdm.tqdm(groups, total=groups.ngroups):
        n = len(profession.split())
        n_total_mentions = group_df.drop_duplicates(subset=['imdb'])[f'{n}_gram_count'].sum()
        n_mentions = len(group_df)
        n_pos_mentions = (group_df.sentiment_label == 1).sum()
        n_neg_mentions = (group_df.sentiment_label == -1).sum()
        n_titles = group_df.imdb.unique().size
        profession_records.append([profession, year, kind, genres, countries, n_titles, n_mentions, n_total_mentions, n_pos_mentions, n_neg_mentions])

    profession_media_attribute_df = pd.DataFrame(profession_records, columns=['profession','year','kind','genres','countries','n_titles','n_mentions','n_total_mentions','n_pos_mentions','n_neg_mentions'])
    profession_media_attribute_df.to_csv(profession_media_attribute_file, index=False)

if __name__=='__main__':
    arg_parser = argparse.ArgumentParser(description='create data for media attribute study for soc groups and professions', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument('-m', default='data/mentions/mentions.word_filtered.sense_filtered.soc_mapped.merged.sentiment_added.csv', type=str, help='csv file of mentions')
    arg_parser.add_argument('-p', default='data/mentions/professions.word_filtered.sense_filtered.merged.csv', type=str, help='csv file of professions')
    arg_parser.add_argument('-i', default='data/imdb/imdb.ngram.box_office.csv', type=str, help='csv file of media attributes')
    arg_parser.add_argument('-ms', default='data/analysis_data/soc.media_attribute.csv', type=str, help='output: csv file of media attribute data for soc groups')
    arg_parser.add_argument('-mp', default='data/analysis_data/profession.media_attribute.csv', type=str, help='output: csv file of media attribute data for professions')
    args = arg_parser.parse_args()

    mentions_file = args.m
    professions_file = args.p
    imdb_file = args.i
    soc_media_attribute_file = args.ms
    profession_media_attribute_file = args.mp
    create_data_for_media_attribute_study(mentions_file, professions_file, imdb_file, soc_media_attribute_file, profession_media_attribute_file)
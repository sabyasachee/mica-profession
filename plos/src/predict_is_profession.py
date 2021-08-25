import pandas as pd
from tqdm import tqdm
import argparse
import json
import re

def is_profession(mentions_file, professions_file, professional_senses_file, characters_file, out_mentions_file, \
    out_professions_file):
    print("reading mentions")
    mentions_df = pd.read_csv(mentions_file, index_col=None)

    print("reading professions")
    professions_df = pd.read_csv(professions_file, index_col=None)

    print("reading professional senses")
    professional_sense_str_set = [line.split()[0] for line in open(professional_senses_file)]

    print("reading characters file")
    characters_data = json.load(open(characters_file))

    mentions_df["is_profession"] = 0
    mentions_df["is_nopos_profession"] = 0

    print("is_profession, is_nopos_profession = 1 if profession.n_senses = 0")
    zero_sense_professions = set(professions_df[professions_df.n_senses == 0].profession)
    mentions_df.loc[mentions_df.profession.isin(zero_sense_professions), ["is_profession", "is_nopos_profession"]] = 1

    print("is_profession, is_nopos_profession = 1 if profession.n_words > 1")
    polygram_professions = set(professions_df[professions_df.n_words > 1].profession)
    mentions_df.loc[mentions_df.profession.isin(polygram_professions), ["is_profession", "is_nopos_profession"]] = 1

    print("is_profession = 1 if mention.sense ∈ professional_senses")
    print("is_nopos_profession = 1 if mention.no_pos_sense ∈ professional_senses")
    mentions_df.loc[mentions_df.sense.notna() & (mentions_df.sense.isin(professional_sense_str_set)), \
        "is_profession"] = 1
    mentions_df.loc[mentions_df.no_pos_sense.notna() & (mentions_df.no_pos_sense.isin(professional_sense_str_set)), \
        "is_nopos_profession"] = 1

    print("is_profession, is_nopos_profession = 0 if mention.ner = PERSON and character.name contains \
        mention.mention for some character in the movie")
    mentions_df.imdb = mentions_df.imdb.astype(str).str.zfill(7)
    imdb_list, mention_list = mentions_df.imdb.values, mentions_df.mention.values
    is_person_list = []
    
    for imdb, mention in tqdm(zip(imdb_list, mention_list), total=len(imdb_list)):
        is_person = False
        if imdb in characters_data:
            for character in characters_data[imdb]:
                if mention.strip().lower() in character.strip().lower().split():
                    is_person = True
                    break
        is_person_list.append(is_person)

    mentions_df["is_person"] = is_person_list
    mentions_df.loc[mentions_df.is_person & (mentions_df.ner == "PERSON"), ["is_profession", "is_nopos_profession"]] = 0

    print("is_profession, is_nopos_profession = 0 if mention.ner = ORGANIZATION")
    mentions_df.loc[mentions_df.ner == "ORGANIZATION", ["is_profession", "is_nopos_profession"]] = 0

    print("updating n_mentions in professions file")
    profession_counts = dict([(profession, 0) for profession in professions_df.profession])
    
    for profession, df in mentions_df[mentions_df.is_profession == 1].groupby("profession"):
        profession_counts[profession] = len(df)

    n_mentions_list = [profession_counts[profession] for profession in professions_df.profession]
    professions_df.n_mentions = n_mentions_list

    professions_df.sort_values(by="n_mentions", ascending=False, inplace=True)
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

    print("saving professions")
    professions_df.to_csv(out_professions_file, index=False)

    print("saving mentions")
    mentions_df.to_csv(out_mentions_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="add is_person and is_profession column", \
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--in_mentions", default="data/mentions/mentions.word_filtered.csv", help="mentions file")
    parser.add_argument("--in_professions", default="data/mentions/professions.word_filtered.csv", help="professions \
        file")
    parser.add_argument("--out_mentions", default="data/mentions/mentions.word_filtered.prediction_added.csv", \
        help="mentions file containing is_person and is_profession columns")
    parser.add_argument("--out_professions", default="data/mentions/professions.word_filtered.prediction_added.csv", \
        help="professions file containing updated n_mentions")
    parser.add_argument("--professional_senses", default="data/gazetteer/wn.syn.mention.txt", help="contains \
        professional senses")
    parser.add_argument("--characters", default="data/imdb/characters.json", help="contains character data")

    args = parser.parse_args()
    mentions_file = args.in_mentions
    professions_file = args.in_professions
    out_mentions_file = args.out_mentions
    out_professions_file = args.out_professions
    professional_senses_file = args.professional_senses
    characters_file = args.characters

    is_profession(mentions_file, professions_file, professional_senses_file, characters_file, out_mentions_file, \
    out_professions_file)
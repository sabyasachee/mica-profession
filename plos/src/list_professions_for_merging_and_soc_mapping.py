from typing import DefaultDict
import re
from numpy.core.records import record
import pandas as pd
from collections import defaultdict
from nltk.corpus import wordnet as wn
import json

def list_professions_for_merging_and_mapping(professions_file, senses_file, soc_titles_file, soc_definitions_file, \
    out_map_file, out_soc_data_file):
    professions_df = pd.read_csv(professions_file, index_col=None)
    soc_professions_df = pd.read_csv(soc_titles_file, index_col=None)
    soc_definitions_df = pd.read_csv(soc_definitions_file, index_col=None)
    professional_senses = [wn.synset(line.split()[0]) for line in open(senses_file)]

    professions_df.sort_values("n_mentions", inplace=True, ascending=False)
    media_professions = professions_df.profession.values[:500]
    fraction = 100*professions_df.n_mentions[:500].sum()/professions_df.n_mentions.sum()
    print(f"top 500 professions cover {fraction:.2f}% of all professional mentions")

    soc_profession_data = defaultdict(set)
    soc_name_data = dict()
    media_profession_sense_data = defaultdict(set)
    media_profession_soc_data = defaultdict(lambda: defaultdict(list))

    print("finding professions of soc major code")
    for _, row in soc_professions_df.iterrows():
        soc = row["2018 SOC Code"][:2]
        profession = row["2018 SOC Direct Match Title"]
        soc_profession_data[soc].add(profession)

    print("finding name of soc major code")
    for soc in soc_profession_data:
        code = f"{soc}-0000"
        name = soc_definitions_df[soc_definitions_df["SOC Code"] == code]["SOC Title"].values[0]
        soc_name_data[soc] = f"{soc}:{name}"

    print("finding professional senses of media professions")
    for profession in media_professions:
        senses = set(wn.synsets(profession.replace(" ", "_").lower().strip()))
        media_profession_sense_data[profession] = senses.intersection(professional_senses)

    print("finding soc major code data of media professions")
    for profession in sorted(media_professions):
        pattern = "(^|\W)" + re.escape(profession.lower().strip()) + "($|\W)"
        for soc, soc_professions in soc_profession_data.items():
            for soc_profession in soc_professions:
                if re.search(pattern, soc_profession.lower()):
                    media_profession_soc_data[profession][soc_name_data[soc]].append(soc_profession)

    print("listing professions for mapping and merging")
    records = []

    for profession in media_professions:
        soc_data = media_profession_soc_data[profession]
        professional_senses = media_profession_sense_data[profession]

        if len(soc_data) == 1:
            soc_name = list(soc_data.keys())[0]

            if professional_senses:
                for sense in professional_senses:
                    records.append([profession, "", sense.name(), sense.definition(), 1, 1, soc_name])
            else:
                records.append([profession, "", "", "", 1, 1, soc_name])
        else:
            if soc_data:
                soc_data_count = [(soc, len(soc_professions)) for soc, soc_professions in soc_data.items()]
                soc_data_count = sorted(soc_data_count, key = lambda x: x[1], reverse=True)
                total_count = sum(x[1] for x in soc_data_count)
                soc_name = soc_data_count[0][0]
                fraction = soc_data_count[0][1]/total_count
            else:
                soc_name = ""
                fraction = 0
            
            if professional_senses:
                for sense in professional_senses:
                    records.append([profession, "", sense.name(), sense.definition(), len(soc_data), fraction, soc_name])
            else:
                records.append([profession, "", "", "", len(soc_data), fraction, soc_name])

    map_df = pd.DataFrame(records, columns=["profession", "profession_merge", "sense_name", "sense_definition", \
        "n_soc", "percent_max_soc", "max_soc"])
    map_df.to_csv(out_map_file, index=False)
    json.dump(media_profession_soc_data, open(out_soc_data_file, "w"), indent=2)

if __name__ == "__main__":
    list_professions_for_merging_and_mapping("data/mentions/professions.word_filtered.sense_filtered.csv", \
        "data/gazetteer/wn.syn.mention.txt", "data/gazetteer/soc.csv", "data/gazetteer/soc.definition.csv", \
            "data/mentions/professions.soc.map.csv", "data/mentions/professions.soc.data.json")
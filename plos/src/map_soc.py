import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

soc_names = [
"Management",
"Business and Financial Operations",
"Computer and Mathematical",
"Architecture and Engineering",
"Life, Physical, and Social Science",
"Community and Social Service",
"Legal",
"Educational Instruction and Library",
"Arts, Design, Entertainment, Sports, and Media",
"Healthcare Practitioners and Technical Occupations",
"Healthcare Support",
"Protective Service",
"Food Preparation and Serving Related Occupations",
"Building and Grounds Cleaning and Maintenance",
"Personal Care and Service",
"Sales and Related Occupations",
"Office and Administrative Support",
"Farming, Fishing, and Forestry",
"Construction and Extraction",
"Installation, Maintenance, and Repair",
"Production",
"Transportation and Material Moving",
"Military Specific Occupations"
]

def map_profession_to_soc(map_file, soc_file, in_mentions_file, in_professions_file, out_mentions_file, \
    out_professions_file):
    print("reading map file")
    map_df = pd.read_csv(map_file, index_col=None)

    print("reading soc file")
    soc_df = pd.read_csv(soc_file, index_col=None)
    soc_df["major_code"] = soc_df["2018 SOC Code"].str[:2].astype(int)
    soc_df["profession"] = soc_df["2018 SOC Direct Match Title"].str.lower().str.strip()

    print("finding soc profession to soc major code")
    soc_dict = defaultdict(set)
    for _, row in soc_df.iterrows():
        soc_dict[row.profession].add(row.major_code)

    print("reading mentions file")
    mentions_df = pd.read_csv(in_mentions_file, index_col=None)

    print("reading professions file")
    professions_df = pd.read_csv(in_professions_file, index_col=None)

    profession_arr = mentions_df.profession.values
    profession_arr_2 = professions_df.profession.values
    sense_arr = mentions_df.no_pos_sense.values
    
    map_soc_code_arr = np.empty(len(mentions_df), dtype="<U100")
    map_soc_name_arr = np.empty(len(mentions_df), dtype="<U1000")
    merge_profession_arr = profession_arr[:]
    merge_profession_arr_2 = profession_arr_2[:]

    print("mapping and merging")
    for _, row in tqdm(map_df.iterrows(), total=len(map_df)):
        profession, profession_merge, sense, primary_soc, secondary_soc = row.profession, row.profession_merge, \
            row.sense_name, row.primary_soc, row.secondary_soc
        merge_profession_arr[profession_arr == profession] = profession_merge
        merge_profession_arr_2[profession_arr_2 == profession] = profession_merge
        soc_codes = set()

        if pd.notna(primary_soc) and primary_soc != "Can't Say" and primary_soc != "General":
            soc_codes.add(int(primary_soc[:2]))
        
        if pd.notna(secondary_soc) and secondary_soc != "Can't Say" and secondary_soc != "General":
            soc_codes.add(int(secondary_soc[:2]))

        for major_code in soc_dict[profession]:
            soc_codes.add(major_code)
        
        soc_codes = list(soc_codes)
        soc_codes_str = ";".join([str(x) for x in soc_codes])
        soc_names_str = ";".join([soc_names[(x - 11)//2] for x in soc_codes])

        if pd.notna(sense):
            map_soc_code_arr[(profession_arr == profession) & (sense_arr == sense)] = soc_codes_str
            map_soc_name_arr[(profession_arr == profession) & (sense_arr == sense)] = soc_names_str
        else:
            map_soc_code_arr[profession_arr == profession] = soc_codes_str
            map_soc_name_arr[profession_arr == profession] = soc_names_str

    print("saving mentions")
    mentions_df["profession_merge"] = merge_profession_arr
    mentions_df["soc_code"] = map_soc_code_arr
    mentions_df["soc_name"] = map_soc_name_arr
    mentions_df.to_csv(out_mentions_file, index=False)

    print("saving professions")
    professions_df["profession_merge"] = merge_profession_arr_2
    professions_df.to_csv(out_professions_file, index=False)

if __name__ == "__main__":
    map_profession_to_soc("data/mentions/professions.soc.map.filled.csv", "data/gazetteer/soc.csv", \
        "data/mentions/mentions.word_filtered.sense_filtered.csv", \
        "data/mentions/professions.word_filtered.sense_filtered.csv", \
        "data/mentions/mentions.word_filtered.sense_filtered.soc_mapped.merged.csv", \
        "data/mentions/professions.word_filtered.sense_filtered.merged.csv")
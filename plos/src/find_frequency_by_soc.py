import pandas as pd
import numpy as np
from map_soc import soc_names

def find_frequency_by_soc(mentions_file, frequency_file, soc_frequency_file):
    print("reading mentions")
    mentions_df = pd.read_csv(mentions_file, index_col=None, dtype={"soc_code":str, "soc_name":str})
    frequency_df = pd.read_csv(frequency_file, index_col=None)
    
    print("finding soc frequency")
    map_df = mentions_df[["profession","no_pos_sense","soc_code","soc_name"]].dropna(subset=["soc_code"]).drop_duplicates()
    merge_frequency_df = frequency_df.merge(map_df, on=["profession","no_pos_sense"])
    
    records = []
    years = frequency_df.columns[2:]

    for i in range(23):
        soc_code = str(11 + 2*i)
        soc_name = soc_names[i]
        record = [soc_code, soc_name] + merge_frequency_df.loc[merge_frequency_df.soc_code.str.contains(soc_code), years].sum().values.tolist()
        records.append(record)

    record = ["100", "STEM"] + merge_frequency_df.loc[merge_frequency_df.soc_code.str.match("(15)|(17)|(19)|(29)") | merge_frequency_df.no_pos_sense.str.match("(doctor.n.04)"), years].sum().values.tolist()
    records.append(record)

    soc_frequency_df = pd.DataFrame(records, columns=["soc_code","soc_name"] + years.tolist())

    print("saving soc frequency file")
    soc_frequency_df.to_csv(soc_frequency_file, index=False)

if __name__ == "__main__":
    # find_frequency_by_soc("data/mentions/mentions.word_filtered.sense_filtered.soc_mapped.merged.csv", "data/mentions/frequency.csv", "data/analysis_data/soc_frequency.csv")
    # find_frequency_by_soc("data/mentions/mentions.word_filtered.sense_filtered.soc_mapped.merged.csv", "data/mentions/frequency.sample.csv", "data/analysis_data/soc_frequency.sample.csv")
    # find_frequency_by_soc("data/mentions/mentions.word_filtered.sense_filtered.soc_mapped.merged.csv", "data/mentions/frequency.cutoff.csv", "data/analysis_data/soc_frequency.cutoff.csv")
    find_frequency_by_soc("data/mentions/mentions.word_filtered.sense_filtered.soc_mapped.merged.csv", "data/mentions/frequency.year.csv", "data/analysis_data/soc_frequency.year.csv")
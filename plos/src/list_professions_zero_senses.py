import pandas as pd

def list_professions_with_zero_sense(professions_file, zero_sense_professions_file):
    professions_df = pd.read_csv(professions_file, index_col=None)
    professions_zero_sense = sorted(professions_df[professions_df.n_synsets == 0].profession)
    open(zero_sense_professions_file, "w").write("\n".join(professions_zero_sense))

if __name__ == "__main__":
    list_professions_with_zero_sense("data/mentions/professions.csv", "data/gazetteer/title.zerosense.txt")
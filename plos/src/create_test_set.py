import pandas as pd
import numpy as np
from tqdm import tqdm

def create_test_set(mentions_file, test_file):
    mentions_df = pd.read_csv(mentions_file, index_col=None)
    np.random.seed(0)

    pos_df = mentions_df[mentions_df.is_profession == 1].sample(n=100, replace=False)
    neg_df = mentions_df[mentions_df.is_profession == 0].sample(n=100, replace=False)
    test_df = pd.concat([pos_df, neg_df])
    test_df["sentence"] = test_df.left.fillna("") + " <<" + test_df.mention + ">> " + test_df.right.fillna("")
    test_df = test_df.sample(frac=1)

    test_df.to_csv(test_file, index=False)

if __name__ == "__main__":
    create_test_set("data/mentions/mentions.word_filtered.prediction_added.csv", "data/mentions/mentions.test.csv")
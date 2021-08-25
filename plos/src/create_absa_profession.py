import pandas as pd
import numpy as np
import inflect
import math
from collections import Counter
import argparse

def split_by_profession(dataset_df, test_len):
    test_ratio = test_len / len(dataset_df)
    professions = dataset_df["profession"].unique()
    n_test_professions = math.ceil(test_ratio * len(professions))
    test_professions_list = [np.random.choice(professions, n_test_professions, replace=False) for _ in range(10)]
    test_size_list = [abs(dataset_df["profession"].isin(test_professions).sum() - test_len) for test_professions in test_professions_list]
    best_test_index = np.argmin(test_size_list)
    test_professions = sorted(test_professions_list[best_test_index])
    train_professions = sorted([profession for profession in professions if profession not in test_professions])
    train_dataset_df = dataset_df[dataset_df["profession"].isin(train_professions)].copy()
    test_dataset_df = dataset_df[dataset_df["profession"].isin(test_professions)].copy()
    return train_dataset_df, test_dataset_df, train_professions, test_professions

def write_absa_raw(dataset_df, data_file):
    with open(data_file, "w") as fw:
        for _, row in dataset_df.iterrows():
            sentence = f"{row.left.strip()} $T$ {row.right.strip()}".strip()
            if row["sentiment"] == "positive":
                label = 1
            elif row["sentiment"] == "negative":
                label = -1
            else:
                label = 0
            fw.write(f"{sentence}\n{row['target'].strip()}\n{label}\n")

def create_profession_absa_data(mturk_annotated_file, dataset_file, train_file, test_file, val_file=None, val_ratio=0.1, test_ratio=0.1, allow_profession_overlap_between_train_and_test=False, allow_profession_overlap_between_train_and_val=False, seed=0):
    mturk_df = pd.read_csv(mturk_annotated_file, index_col=None)
    mturk_df["Input.text_left"] = mturk_df["Input.text_left"].fillna("")
    mturk_df["Input.text_right"] = mturk_df["Input.text_right"].fillna("")
    inflector = inflect.engine()

    records = []

    for (left, target, right), df in mturk_df.groupby(["Input.text_left", "Input.target", "Input.text_right"]):
        labels = df["Answer.sentiment"].values
        labelset = set(labels)
        if len(labelset) == 1 and len(labels) > 1:
            sentiment = labels[0]
            normalized_target = target.lower().strip()
            profession = inflector.singular_noun(normalized_target)
            if not isinstance(profession, str) or ("False" in profession) or (profession in ["police", "waitress"]):
                profession = normalized_target
            records.append([profession, ";".join(df["WorkerId"]), left, target, right, sentiment])

    dataset_df = pd.DataFrame(records, columns=["profession", "workers", "left", "target", "right", "sentiment"])
    val_len = math.ceil(val_ratio*len(dataset_df))
    test_len = math.ceil(test_ratio*len(dataset_df))

    np.random.seed(seed)
    
    if allow_profession_overlap_between_train_and_test:
        test_dataset_df = dataset_df.sample(n=test_len, replace=False).copy()
        train_dataset_df = dataset_df.loc[~dataset_df.index.isin(test_dataset_df.index)].copy()
    else:
        train_dataset_df, test_dataset_df, train_professions, test_professions = split_by_profession(dataset_df, test_len)

    if val_file is not None:
        if allow_profession_overlap_between_train_and_val:
            val_dataset_df = train_dataset_df.sample(frac=val_ratio, replace=False).copy()
            train_dataset_df = train_dataset_df.loc[~train_dataset_df.index.isin(val_dataset_df.index)].copy()
        else:
            train_dataset_df, val_dataset_df, train_professions, val_professions = split_by_profession(train_dataset_df, val_len)
        
        train_dataset_df["partition"] = "train"
        val_dataset_df["partition"] = "val"
        test_dataset_df["partition"] = "test"
        dataset_df = pd.concat([train_dataset_df, val_dataset_df, test_dataset_df])

        write_absa_raw(train_dataset_df, train_file)
        write_absa_raw(val_dataset_df, val_file)
        write_absa_raw(test_dataset_df, test_file)

        print(f"train==>\n{len(train_professions)} professions\n{train_professions}\n{len(train_dataset_df)} examples\nlabel distribution = {Counter(train_dataset_df['sentiment'])}\n")
        print(f"val==>\n{len(val_professions)} professions\n{val_professions}\n{len(val_dataset_df)} examples\nlabel distribution = {Counter(val_dataset_df['sentiment'])}\n")
        print(f"test==>\n{len(test_professions)} professions\n{test_professions}\n{len(test_dataset_df)} examples\nlabel distribution = {Counter(test_dataset_df['sentiment'])}\n")
    
    else:
        train_dataset_df["partition"] = "train"
        test_dataset_df["partition"] = "test"
        dataset_df = pd.concat([train_dataset_df, test_dataset_df])

        write_absa_raw(train_dataset_df, train_file)
        write_absa_raw(test_dataset_df, test_file)

        print(f"train==>\n{len(train_professions)} professions\n{train_professions}\n{len(train_dataset_df)} examples\nlabel distribution = {Counter(train_dataset_df['sentiment'])}\n")
        print(f"test==>\n{len(test_professions)} professions\n{test_professions}\n{len(test_dataset_df)} examples\nlabel distribution = {Counter(test_dataset_df['sentiment'])}\n")

    dataset_df.to_csv(dataset_file, index=False)

def main():
    parser = argparse.ArgumentParser(description="create profession ABSA dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mturk", type=str, default="data/sentiment/Batch_4195910_batch_results.csv", help="Amazon MTurk annotated csv file for profession-aspect sentiment")
    parser.add_argument("--out_dataset", type=str, default="data/sentiment/sentiment.csv", help="output dataset csv file")
    parser.add_argument("--out_absa_train", type=str, default="data/sentiment/train.raw", help="output ABSA train file")
    parser.add_argument("--out_absa_val", type=str, default="data/sentiment/val.raw", help="output ABSA val file")
    parser.add_argument("--out_absa_test", type=str, default="data/sentiment/test.raw", help="output ABSA test file")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="val size in fraction")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="test size in fraction")
    parser.add_argument("--allow_profession_overlap_between_train_and_val", action="store_true", default=False, help="set to allow professions to appear as aspect targets in both train and val set")
    parser.add_argument("--allow_profession_overlap_between_train_and_test", action="store_true", default=False, help="set to allow professions to appear as aspect targets in both train and test set")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    create_profession_absa_data(args.mturk, args.out_dataset, args.out_absa_train, args.out_absa_test, val_file=args.out_absa_val, val_ratio=args.val_ratio, test_ratio=args.test_ratio, allow_profession_overlap_between_train_and_test=args.allow_profession_overlap_between_train_and_test, allow_profession_overlap_between_train_and_val=args.allow_profession_overlap_between_train_and_val, seed=args.seed)

if __name__ == "__main__":
    main()
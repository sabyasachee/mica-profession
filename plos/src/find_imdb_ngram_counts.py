import os
import pandas as pd
from tqdm import tqdm

def find_ngram_count_for_subtitle_files(imdb_file, out_imdb_file, data_dir, max_ngram):
    imdb_df = pd.read_csv(imdb_file, index_col=None)
    n_ngram_cols = [[] for _ in range(max_ngram)]

    print("finding ngram counts")
    for _, row in tqdm(imdb_df.iterrows(), total=len(imdb_df)):
        imdb = row.imdb_ID
        lines = open(os.path.join(data_dir, f"{imdb}.txt")).read().strip().split("\n")
        n_ngram = [0 for _ in range(max_ngram)]

        for line in lines:
            n_words = len(line.split())
            for i, ngram_size in enumerate(range(max_ngram)):
                n_ngram[i] += max(n_words - ngram_size, 0)

        for i in range(max_ngram):
            n_ngram_cols[i].append(n_ngram[i])

    print("saving imdb with ngram counts")
    for i in range(max_ngram):
        imdb_df[f"{i + 1}_gram_count"] = n_ngram_cols[i]
    imdb_df.to_csv(out_imdb_file, index=False)

if __name__ == "__main__":
    find_ngram_count_for_subtitle_files("data/imdb/imdb.csv", "data/imdb/imdb.ngram.csv", \
        "/proj/sbaruah/subtitle/profession/data/text", 5)
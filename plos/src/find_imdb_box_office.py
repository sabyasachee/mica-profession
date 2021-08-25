import pandas as pd
from utils.imdbmovie import get_movies
import argparse

def find_imdb_box_office(imdb_file, imdb_box_office_file):
    imdb_df = pd.read_csv(imdb_file, index_col=None)
    imdb_dict = get_movies(imdb_df["imdb_ID"].values, verbose=True)
    box_office_col = []

    for imdb in imdb_df["imdb_ID"]:
        try:
            box_office = int(imdb_dict[str(imdb).zfill(7)]["box office"]["Opening Weekend United States"].split()[0].lstrip("$").replace(",", ""))
        except Exception:
            box_office = None
        box_office_col.append(box_office)

    imdb_df["box_office_in_dollars"] = box_office_col
    imdb_df.to_csv(imdb_box_office_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--imdb", default="data/imdb/imdb.ngram.csv", type=str)
    parser.add_argument("--out_imdb", default="data/imdb/imdb.ngram.box_office.csv", type=str)
    args = parser.parse_args()
    find_imdb_box_office(args.imdb, args.out_imdb)
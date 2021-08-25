import json
import argparse
import pandas as pd
from pattern.text.en import singularize

from whoosh.qparser import QueryParser
from whoosh.fields import Schema, ID, TEXT
from whoosh.index import open_dir
from whoosh.analysis import LowercaseFilter, RegexTokenizer, Filter

class SingularizeFilter(Filter):
    """
    Whoosh custom filter for singularizing text.
    Required to instantiate the schema.
    """
    def __call__(_, tokens):
        for t in tokens:
            t.text = singularize(t.text)
            yield t

def search(index, profession_filepath, key_to_si_filepath, search_index_dir):
    """Search key words (from the profession gazetteer) in subtitle sentences.

    Parameters
    ----------

    index : int
        The search index is divided into 10 indices for faster query.
        `index` (between 0 and 1) chooses one of these search indices
        to search from.

    profession_filepath : str
        CSV file containing the profession gazetteer. 
        It contains `singular_key` and `plural_key` columns, 
        from which the list of keys is selected.

    key_to_si_filepath : str
        JSON filepath to which the search output results are saved.
        The keys are the keys from the profession gazetteer.
        The values are list of sentence indexes which are 2-element lists
        of IMDb id (also the filename) and sentence index.
        IMDb id is a string and sentence index is an integer.

    search_index_dir: str
        The folder path which store the 10 search indices.
    """

    # read the profession gazetteer
    profession_df = pd.read_csv(profession_filepath, index_col = None)

    # retrieve the list of keys
    keys = set(profession_df["singular_key"].dropna().tolist() + profession_df["plural_key"].dropna().tolist())

    key_to_si = {}

    for key in keys:
        key_to_si[key] = []

    # open the Whoosh search index
    print(f"index {index:2d}: opening search index...")
    custom_analyzer = RegexTokenizer(expression="\w+|[^\w\s]+") | LowercaseFilter() | SingularizeFilter()
    schema = Schema(imdb_ID=ID(stored=True), sent_ID=ID(stored=True), content=TEXT(analyzer=custom_analyzer))
    # search_index = open_dir("/proj/sbaruah/subtitle/profession/data/search_index/", indexname=f"index{index}", schema=schema)
    search_index = open_dir(search_index_dir, indexname=f"index{index}", schema=schema)

    engine = search_index.searcher()
    query_parser = QueryParser("content", schema)

    # search the keys
    for i, key in enumerate(keys):
        query = query_parser.parse(f'"{key}"')
        results = engine.search(query, limit=None)
        for hit in results:
            imdb_ID = str(hit["imdb_ID"])
            sent_ID = int(hit["sent_ID"])
            key_to_si[key].append([imdb_ID, sent_ID])
        print(f"index {index:2d}: {i + 1:5d}/{len(keys)} searched")

    # save the search results
    print(f"index {index:2d}: saving key to sentence index dictionary")
    # json.dump(key_to_si, open(f"key_to_si_{index}.json","w"))
    json.dump(key_to_si, open(key_to_si_filepath,"w"))

    engine.close()

def create_key():
    parser = argparse.ArgumentParser(description="Create key to sentence index dictionary", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--index", type=int, choices=list(range(10)), help="search index number", default=0, dest="index")
    parser.add_argument("--gzt", type=str, help="CSV filepath of profession gazetteer", default="/proj/sbaruah/subtitle/profession/csl/data/gazetteer/inflection.csv", dest="gzt")
    parser.add_argument("--out", type=str, help="JSON filepath to which the search results are saved", default="/proj/sbaruah/subtitle/profession/csl/data/mentions/key_to_si_0.json", dest="out")
    parser.add_argument("--index_dir", type=str, help="Directory of the search index", default="/proj/sbaruah/subtitle/profession/data/search_index/", dest="dir")

    args = parser.parse_args()
    index = args.index
    profession_filepath = args.gzt
    key_to_si_filepath = args.out
    search_index_dir = args.dir

    search(index, profession_filepath, key_to_si_filepath, search_index_dir)

if __name__ == "__main__":
    create_key()
import json
import argparse

def merge_search_outputs(key_to_si_filepaths, merged_key_to_si_filepath, si_to_key_filepath):
    """Merge the search results from the 10 different key_to_si json files

    Parameters
    ----------

    key_to_si_filepaths : list
        List of json filepaths.
        Each json contains the search result of keys (from profession gazetteer) 
        in subtitle sentences.

    merged_key_to_si_filepath : str
        JSON filepath to which the final merged key to sentence index 
        dictionary will be saved.
        It will be of the form {`key`: List[[`imdb`, `sent`]]}

    si_to_key_filepath : str
        JSON filepath to which the reverse mapping, sentence index to key, will be
        saved.
        It will be of the form {`imdb`: {`sent`: List[`key`]}}
    """

    # read the list of key to sentence index dictionaries
    key_to_si_dictionaries = []
    for filepath in key_to_si_filepaths:
        print(f"opening {filepath}")
        key_to_si_dictionaries.append(json.load(open(filepath)))

    key_to_si = {}
    si_to_key = {}
    keys = key_to_si_dictionaries[0].keys()

    # merge the dictionaries
    print(f"merging/appending key_to_si dictionaries")
    for key in keys:
        si_list = []
        for dictionary in key_to_si_dictionaries:
            for imdb, sent in dictionary[key]:
                si_list.append([str(imdb), int(sent)])
        
        if key not in key_to_si:
            key_to_si[key] = si_list
        else:
            key_to_si[key].extend(si_list)

    # ensure the list of sentence indices contain no duplicates
    for key in key_to_si:
        si_list = key_to_si[key]
        si_list = [(imdb, sent) for imdb, sent in si_list]
        si_list = list(set(si_list))
        key_to_si[key] = si_list

    # create the reverse mapping: sentence index to key
    print(f"creating si_to_key dictionary")
    for key, si_list in key_to_si.items():
        for imdb, sent in si_list:
            if imdb not in si_to_key:
                si_to_key[imdb] = {}
            if sent not in si_to_key[imdb]:
                si_to_key[imdb][sent] = []
            si_to_key[imdb][sent].append(key)

    # save the key to sentence index and sentence index to key dictionaries
    print(f"saving dictionaries")
    json.dump(key_to_si, open(merged_key_to_si_filepath, "w"))
    json.dump(si_to_key, open(si_to_key_filepath, "w"))

def merge():
    parser = argparse.ArgumentParser(description="Merge key to sentence index dictionaries", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--files", nargs="+", default=[f"/proj/sbaruah/subtitle/profession/csl/data/mentions/key_to_si_{i}.json" for i in range(10)], dest="files", help="list of JSON key to sentence dictionary files")
    parser.add_argument("--k2s", type=str, default="/proj/sbaruah/subtitle/profession/csl/data/mentions/key_to_si.json", dest="k2s", help="JSON filepath to which the final merged key to sentence dictionary will be saved")
    parser.add_argument("--s2k", type=str, default="/proj/sbaruah/subtitle/profession/csl/data/mentions/si_to_key.json", dest="s2k", help="JSON filepath to which the reverse dictionary, sentence index to key, will be saved")
    
    args = parser.parse_args()
    key_to_si_filepaths = args.files
    merged_key_to_si_filepath = args.k2s
    si_to_key_filepath = args.s2k

    merge_search_outputs(key_to_si_filepaths, merged_key_to_si_filepath, si_to_key_filepath)

if __name__ == "__main__":
    merge()
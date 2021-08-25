import os
import json
from tqdm import tqdm
import argparse

def find_sentences(key_to_si_filepath, subtitle_directory, sentences_filepath, si_to_rsi_filepath, rsi_to_si_filepath):
    """
    Save subtitle sentences that contain some key (from `key_to_si_filepath` json) in `sentences_filepath` text file.
    Find the mapping between sentence index in the text file and file sentence index and save them.

    Parameters
    ----------

    key_to_si_filepath : str
        JSON filepath containing key to file sentence index.
        It is of the form {`key`: [[`imdb`,`sent`]]}

    subtitle_directory : str
        Directory containing the subtitle text files.

    sentences_filepath : str
        TEXT filepath to which the subtitle sentences that contain
        some professional word will be saved.
        The index of the sentence in the text file is called the
        flattened sentence index (rsi).

    si_to_rsi_filepath : str
        JSON filepath containing file sentence index to 
        flattened sentence index map.
        It is of the form {`imdb`:{`sent`:`rsi`}}

    rsi_to_si_filepath : str
        JSON filepath containing the flattened sentence index to
        file sentence index map.
        It is of the form {`rsi`:[[`imdb`,`sent`]]}
    """

    # read key to sentence index dictionary
    print("opening key to sentence index dictionary")
    key_to_si = json.load(open(key_to_si_filepath))

    # find imdb to file sentence index
    print("finding dictionary: imdb -> file sentence index")
    imdb_to_fsi = {}
    for si_list in key_to_si.values():
        for imdb, fsi in si_list:
            if imdb not in imdb_to_fsi:
                imdb_to_fsi[imdb] = set()
            imdb_to_fsi[imdb].add(fsi)

    # find sentence to flattened sentence index (sentence_to_rsi)
    # find file sentence index to flattened sentence index (si_to_rsi)
    # find flattened sentence index to file sentence index (rsi_to_si)
    print("finding dictionary: sentence index <-> flattened sentence index")
    sentence_to_rsi = {}
    si_to_rsi = {}
    rsi_to_si = {}

    for imdb, fsi_list in tqdm(imdb_to_fsi.items(), desc = "creating maps"):
        # text = open(f"/proj/sbaruah/subtitle/profession/data/text/{imdb}.txt").read().strip().split("\n")
        text = open(os.path.join(subtitle_directory, f"{imdb}.txt")).read().strip().split("\n")
        si_to_rsi[imdb] = {}

        for fsi in fsi_list:
            sentence = text[fsi]
            
            if sentence not in sentence_to_rsi:
                rsi = len(sentence_to_rsi)
                sentence_to_rsi[sentence] = rsi
                rsi_to_si[rsi] = []
            
            rsi = sentence_to_rsi[sentence]
            rsi_to_si[rsi].append((imdb, fsi))
            si_to_rsi[imdb][fsi] = rsi

    # save dictionaries
    print("saving dictionaries: sentence index <-> flattened sentence index")
    json.dump(si_to_rsi, open(si_to_rsi_filepath, "w"))
    json.dump(rsi_to_si, open(rsi_to_si_filepath, "w"))

    # save text
    print("saving sentences")
    sentences = sorted(sentence_to_rsi.keys(), key = lambda x: sentence_to_rsi[x])
    open(sentences_filepath, "w").write("\n".join(sentences))

def flatten_sentences():
    parser = argparse.ArgumentParser(description="Create text file containing all sentences that include some professional word, and create dictionaries to map index of the sentence in the text file to the file sentence index", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--k2s", type=str, dest="k2s", help="key to sentence index filepath", default="/proj/sbaruah/subtitle/profession/csl/data/mentions/key_to_si.json")
    parser.add_argument("--data", type=str, help="subtitle text directory", default="/proj/sbaruah/subtitle/profession/data/text/", dest="data")
    parser.add_argument("--out", type=str, help="txt filepath containing subtitle sentences that contain some professional word", default="/proj/sbaruah/subtitle/profession/csl/data/mentions/sentences.txt", dest="out")
    parser.add_argument("--s2r", type=str, help="json filepath containing file sentence index to flattened sentence index", default="/proj/sbaruah/subtitle/profession/csl/data/mentions/si_to_rsi.json", dest="s2r")
    parser.add_argument("--r2s", type=str, help="json filepath containing flattened sentence index to file sentence index", default="/proj/sbaruah/subtitle/profession/csl/data/mentions/rsi_to_si.json", dest="r2s")

    args = parser.parse_args()
    key_to_si_filepath = args.k2s
    subtitle_directory = args.data
    sentences_filepath = args.out
    si_to_rsi_filepath = args.s2r
    rsi_to_si_filepath = args.r2s

    find_sentences(key_to_si_filepath, subtitle_directory, sentences_filepath, si_to_rsi_filepath, rsi_to_si_filepath)

if __name__ == "__main__":
    flatten_sentences()
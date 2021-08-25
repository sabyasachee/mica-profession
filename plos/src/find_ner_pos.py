from stanza.server import CoreNLPClient
import json
import numpy as np
import argparse
import os

def process_sentences(sentences_filepath, n_batches, batch_index, max_n_words, processed_dir):
    """Tokenize and find the POS and NER tags of the subtitle sentences.

    Parameters
    ----------
        
    sentences_filepath : str \\
        TEXT filepath containing subtitle sentences that contain some professional title
    
    max_n_words : int \\
        If sentences contain more than `max_n_words`, ignore them.
        Their processed docs are empty lists

    processed_filepath : str \\
        JSON filepath to which the processed sentence docs will be saved.
        It is of the form `[[{token:str, ner:str, xpos:str}]]`
    """

    # read the sentences
    sentences = open(sentences_filepath).read().strip().split("\n")
    n_batch_sentences = int(np.ceil(len(sentences)/n_batches))
    sentences = sentences[batch_index * n_batch_sentences: (batch_index + 1) * n_batch_sentences]
    port = 12345 + batch_index

    # create the NLP processing pipeline
    docs = []
    with CoreNLPClient(threads=8, annotators=["tokenize","pos","ner"], output_format="json", memory="8G", \
        be_quiet=True, timeout=60000, stderr=open(f"error_{batch_index}.log", "w"), \
            endpoint=f"http://localhost:{port}") as client:

        for i, text in enumerate(sentences):
            doc = []
            if len(text.split()) <= max_n_words:
                stanford_doc = client.annotate(text)
                for sentence in stanford_doc["sentences"]:
                    for token in sentence["tokens"]:
                        doc.append(dict(lemma=token["lemma"], word=token["originalText"], \
                            start=token["characterOffsetBegin"], end=token["characterOffsetEnd"], pos=token["pos"], \
                                ner=token["ner"]))
            docs.append(doc)
            print(f"batch {batch_index:2d}. {100*(i + 1)//len(sentences):3d}% sentences processed")
            docs.append(doc)

    # save the processed sentences
    print(f"batch {batch_index:2d}. saving processed batch")
    processed_filepath = os.path.join(processed_dir, f"processed_finegrained_{batch_index}.json")
    json.dump(docs, open(processed_filepath, "w"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize and find POS and named entities using Stanford CoreNLP", \
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--sent", type=str, help="TEXT filepath containing subtitle sentences", \
        default="/proj/sbaruah/subtitle/profession/csl/data/mentions/sentences.txt", dest="sent")
    parser.add_argument("--n_batches", type=int, help="number of batches", default=10, dest="n_batches")
    parser.add_argument("--batch_index", type=int, help="batch index. 0 <= batch_index < number of batches", default=0, \
        dest="batch_index")
    parser.add_argument("--max_words", type=int, help="Maximum number of words allowed in the subtitle sentence", \
        default=100, dest="max")
    parser.add_argument("--out", type=str, help="directory to which the processed json files will be saved", \
        default="/proj/sbaruah/subtitle/profession/csl/data/mentions/", dest="out")

    args = parser.parse_args()
    sentences_filepath = args.sent
    n_batches = args.n_batches
    batch_index = args.batch_index
    max_n_words = args.max
    processed_dir = args.out

    process_sentences(sentences_filepath, n_batches, batch_index, max_n_words, processed_dir)
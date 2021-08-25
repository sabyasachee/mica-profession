import pandas as pd
import jsonlines
from stanza.server import CoreNLPClient
from tqdm import tqdm

def find_mturk_ner_pos(mturk_file, mturk_processed_file):
    mturk = pd.read_csv(mturk_file, index_col=None)
    sentences = mturk.left.fillna("").str.strip() + " " + mturk.mention.str.strip() + " " + mturk.right.fillna("").str.strip()
    client = CoreNLPClient(threads=8, annotators=["tokenize","pos","ner"], output_format="json", memory="8G", be_quiet=False, timeout=60000, stderr=open("error.log", "w"))

    docs = []
    print("finding ner and pos of mturk sentences")

    for sentence in tqdm(sentences):
        stanford_doc = client.annotate(sentence)
        doc = []
        
        for sent in stanford_doc["sentences"]:
            for token in sent["tokens"]:
                doc.append(dict(lemma=token["lemma"], word=token["originalText"], start=token["characterOffsetBegin"], end=token["characterOffsetEnd"], pos=token["pos"], ner=token["ner"]))
                
        docs.append(doc)

    print("writing output")
    with jsonlines.open(mturk_processed_file, "w") as writer:
        for doc in tqdm(docs):
            writer.write(doc)

if __name__ == "__main__":
    find_mturk_ner_pos("data/mturk/mturk.csv", "data/mturk/mturk.pos.ner.jsonl")
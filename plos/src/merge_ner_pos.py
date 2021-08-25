import json
import jsonlines
from tqdm import tqdm

def merge_finegrained_processed(processed_filepaths, merged_filepath):
    docs = []

    for filepath in tqdm(processed_filepaths, desc="reading docs"):
        doc = json.load(open(filepath))
        docs.extend(doc)

    with jsonlines.open(merged_filepath, "w") as writer:
        for doc in tqdm(docs, desc="writing docs"):
            writer.write(doc)

if __name__ == "__main__":
    processed_filepaths = [f"data/mentions/processed_finegrained_{i}.json" for i in range(10)]
    merged_filepath = "data/mentions/processed_finegrained.jsonl"
    merge_finegrained_processed(processed_filepaths, merged_filepath)
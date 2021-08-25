import math

from collections import namedtuple
from nltk.corpus import wordnet as wn
import numpy as np
import jsonlines
import torch
from tqdm import tqdm, trange
import argparse

from ewiser.fairseq_ext.data.dictionaries import Dictionary, ResourceManager, DEFAULT_DICTIONARY
from ewiser.fairseq_ext.data.utils import make_offset
from ewiser.fairseq_ext.models.sequence_tagging import LinearTaggerModel

def make_offset(synset):
    return "wn:" + str(synset.offset()).zfill(8) + synset.pos()

def find_wsd(processed_filepath, wsd_checkpoint_filepath, wsd_filepath, device, batch_size, use_pos=True):
    dictionary = Dictionary.load(DEFAULT_DICTIONARY)
    output_dictionary = ResourceManager.get_offsets_dictionary()

    wnpos = {"NN":"n", "VB":"v", "JJ":"a", "RB":"r"}
    _FakeTask = namedtuple("_FakeTask", ("dictionary", "output_dictionary", "kind"))
    task = _FakeTask(dictionary, output_dictionary, "wsd")

    print("loading wsd model")
    data = torch.load(wsd_checkpoint_filepath, map_location="cpu")
    model = LinearTaggerModel.build_model(data["args"], task).eval()
    model.load_state_dict(data["model"])
    model.to(device)

    maxlen = -1
    docs = []
    for doc in tqdm(jsonlines.open(processed_filepath), total=3577313):
        docs.append(doc)
        maxlen = max(maxlen, len(doc))
    print(f"{len(docs)} subtitle docs read. maxlen = {maxlen} tokens")
    indices_list, tokens_list, synsets_list, output_indices_list = [], [], [], []
    
    for doc in tqdm(docs, desc="preparing for WSD"):
        indices = []
        tokens = []
        synsets = []
        output_indices = []
        
        for word in doc:
            tokens.append(word["word"])
            indices.append(dictionary.index(word["word"]))

            lemma, pos = word["lemma"], word["pos"]
            pos = wnpos.get(pos[:2])
            if pos and use_pos:
                word_synsets = wn.synsets(lemma.lower(), pos)
            else:
                word_synsets = wn.synsets(lemma.lower())
            
            offsets = [make_offset(synset) for synset in word_synsets]
            word_output_indices = np.array([output_dictionary.index(offset) for offset in offsets])
            
            synsets.append(word_synsets)
            output_indices.append(word_output_indices)
        
        indices_list.append(indices)
        tokens_list.append(tokens)
        synsets_list.append(synsets)
        output_indices_list.append(output_indices)

    n_batches = math.ceil(len(docs)/batch_size)
    disambiguated_synsets_list = []
    print(f"processing {n_batches} batches for WSD")

    for i in trange(n_batches, desc="WSD"):
        batch_tokens, batch_indices, batch_synsets, batch_output_indices = tokens_list[i * batch_size: (i + 1) * batch_size], \
            indices_list[i * batch_size: (i + 1) * batch_size], synsets_list[i * batch_size: (i + 1) * batch_size], \
                output_indices_list[i * batch_size: (i + 1) * batch_size]

        batch_maxlen = max(len(x) for x in batch_indices)
        batch_indices = [indices + [dictionary.pad_index] * (batch_maxlen - len(indices)) for indices in batch_indices]
        batch_indices = torch.LongTensor(batch_indices).to(device)

        with torch.no_grad():
            logits, _ = model(src_tokens=batch_indices, src_tokens_str=batch_tokens)
        
        logits = logits.detach().cpu()
        logits[:,:,0:2] = -1e7

        batch_disambiguated_synsets = []
        for i, (example_synsets, example_output_indices) in enumerate(zip(batch_synsets, batch_output_indices)):
            example_disambiguated_synsets = []
            for j, (word_synsets, word_output_indices) in enumerate(zip(example_synsets, example_output_indices)):
                disambiguated_synset = ""

                if len(word_output_indices):
                    logits_token = logits[i, j]
                    logits_synsets = logits_token[word_output_indices]
                    index = torch.max(logits_synsets, -1).indices.item()
                    disambiguated_synset = word_synsets[index].name()
                
                example_disambiguated_synsets.append(disambiguated_synset)
            batch_disambiguated_synsets.append(example_disambiguated_synsets)
        disambiguated_synsets_list.extend(batch_disambiguated_synsets)
    
    print("saving output")
    with jsonlines.open(wsd_filepath, "w") as writer:
        for synsets_list in tqdm(disambiguated_synsets_list, "writing WSD output"):
            writer.write(synsets_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find Word Sense", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", default="data/mentions/pos.ner.jsonl", \
        help="jsonlines filepath containing output of Stanford CoreNLP preprocessing on subtitles")
    parser.add_argument("--checkpoint", default="data/wsd_models/ewiser.semcor+wngt.pt", \
        help="torch checkpoint filepath of EWISER English model")
    parser.add_argument("--device", default="cpu", help="CUDA device or CPU")
    parser.add_argument("--batch_size", default=20, help="number of subtitles in a batch", type=int)
    parser.add_argument("--out", default="data/mentions/wsd.jsonl", help="jsonlines filepath containing WSD output")
    parser.add_argument("--disable_pos", action="store_true", help="set if you choose not use POS to find wordnet synsets")

    args = parser.parse_args()
    processed_filepath = args.data
    wsd_checkpoint_filepath = args.checkpoint
    wsd_filepath = args.out
    device = args.device
    batch_size = args.batch_size
    use_pos = not args.disable_pos

    find_wsd(processed_filepath, wsd_checkpoint_filepath, wsd_filepath, device, batch_size, use_pos=use_pos)
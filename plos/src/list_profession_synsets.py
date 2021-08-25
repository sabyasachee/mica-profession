import pandas as pd
from nltk.corpus import wordnet as wn

def list_synsets_not_found_in_gazetteer(professions_file, gazetteer_wordnet_file, found_synsets_file, notfound_synsets_file):
    professions = pd.read_csv(professions_file, index_col=None)
    synsets_set = set()

    for profession in professions.profession:
        synsets = wn.synsets(profession.lower().replace(" ", "_"), pos="n")
        for synset in synsets:
            if synset.lexname() in ["noun.person", "noun.group"]:
                synsets_set.add(synset)
    
    gazetteer_synsets_set = set()
    for line in open(gazetteer_wordnet_file):
        gazetteer_synsets_set.add(wn.synset(line.split()[0]))
    
    found_synsets = synsets_set.intersection(gazetteer_synsets_set)
    print(f"{len(found_synsets)} wordnet synsets of mentions present in gazetteer")

    notfound_synsets = synsets_set.difference(gazetteer_synsets_set)
    print(f"{len(notfound_synsets)} wordnet synsets of mentions not present in gazetteer")

    with open(found_synsets_file, "w") as fw:
        for synset in found_synsets:
            fw.write(f"{synset.name():30s}\t{synset.definition()}\n")

    with open(notfound_synsets_file, "w") as fw:
        for synset in notfound_synsets:
            fw.write(f"{synset.name():30s}\t{synset.definition()}\n")

if __name__ == "__main__":
    list_synsets_not_found_in_gazetteer("data/mentions/professions.csv", "data/gazetteer/wn.syn.filtered.txt", "data/gazetteer/wn.syn.mention.found.txt", "data/gazetteer/wn.syn.mention.notfound.txt")
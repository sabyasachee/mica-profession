import pandas as pd
from utils.imdbmovie import get_movies
from tqdm import tqdm
import json

def find_imdb_character_names(mentions_file, characters_file):
    print("reading mentions")
    mentions_df = pd.read_csv(mentions_file, index_col=None)
    imdb_set = set(mentions_df.imdb)

    print("finding imdb movie data")
    movie_data = get_movies(imdb_set, verbose=True)

    character_data = {}

    print("finding character names")
    for imdb, data in tqdm(movie_data.items()):
        character_names = []
        try:
            for person in data["cast"]:
                character_names.append(person.currentRole["name"])
        except Exception:
            pass
        character_data[imdb] = character_names

    print("saving character names data")
    json.dump(character_data, open(characters_file, "w"), indent=2)

if __name__ == "__main__":
    find_imdb_character_names("data/mentions/mentions.csv", "data/imdb/characters.json")
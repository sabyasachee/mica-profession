import pandas as pd
from matplotlib import pyplot as plt
import tikzplotlib
from collections import Counter
import numpy as np

def kind_barplot(imdb_df, kind_texfile):
    year_ticks = ["1950-1959", "1960-1969", "1970-1979", "1980-1989", "1990-1999", "2000-2009", "2010-2017"]
    year_ticks = [f"{x:>10s}" for x in year_ticks]
    year_axis = np.arange(7)

    movie_count = []
    tv_count = []

    subtitle_year = imdb_df.year.values
    imdb_kind = imdb_df.imdb_kind.values

    for i in range(7):
        min_year = 1950 + 10*i
        max_year = min(min_year + 9, 2017)
        
        movie_count.append(((subtitle_year >= min_year) & (subtitle_year <= max_year) & (imdb_kind == "movie")).sum())
        tv_count.append(((subtitle_year >= min_year) & (subtitle_year <= max_year) & (imdb_kind == "episode")).sum())
    
    width = 0.4

    plt.figure(figsize=(6, 4))
    plt.rcParams.update({"font.size":18})

    plt.bar(year_axis-width/2, movie_count, width=width, label="movie", color="cornflowerblue")
    plt.bar(year_axis+width/2, tv_count, width=width, label="tv", color="lightsalmon")
    plt.xticks(ticks=year_axis, labels=year_ticks, rotation=90, fontsize=8)
    plt.legend(prop={"size":24})
    plt.ylabel("Count", fontsize=24)
    tikzplotlib.save(kind_texfile)

def genre_barplot(imdb_df, genre_texfile):
    imdb_genre = [genre for imdb_genre_str in imdb_df.imdb_genres.dropna() for genre in imdb_genre_str.split(";")]
    genre_count, genre_ticks = [], []

    for genre, count in sorted(Counter(imdb_genre).items(), key=lambda x: x[1], reverse=True):
        genre_count.append(count)
        genre_ticks.append(genre)

    genre_ticks = [f"{x:>10s}" for x in genre_ticks]
    genre_axis = np.arange(len(genre_ticks))

    topk = 10

    plt.figure(figsize=(6, 4))
    plt.rcParams.update({"font.size":18})

    plt.bar(genre_axis[:topk], genre_count[:topk], width=0.6, color="seagreen", label="genre")
    plt.xticks(ticks=genre_axis[:topk], labels=genre_ticks[:topk], rotation=90)
    plt.legend(prop={"size":24})
    plt.ylabel("Count")
    tikzplotlib.save(genre_texfile)

def country_barplot(imdb_df, country_texfile):
    imdb_country = [country for imdb_country_str in imdb_df.imdb_countries.dropna() for country in \
        imdb_country_str.split(";")]
    country_count, country_ticks = [], []

    for country, count in sorted(Counter(imdb_country).items(), key=lambda x: x[1], reverse=True):
        country_count.append(count)
        country = "U.S." if country == "United States" else country
        country = "U.K." if country == "United Kingdom" else country
        country_ticks.append(country)

    country_ticks = [f"{x:>10s}" for x in country_ticks]
    country_axis = np.arange(len(country_ticks))

    topk = 10

    plt.figure(figsize=(6, 4))
    plt.rcParams.update({"font.size":18})

    plt.bar(country_axis[:topk], country_count[:topk], width=0.6, color="mediumpurple", label="country")
    plt.xticks(ticks=country_axis[:topk], labels=country_ticks[:topk], rotation=90)
    plt.legend(prop={"size":24})
    plt.ylabel("Count")
    tikzplotlib.save(country_texfile)

def language_barplot(imdb_df, language_texfile):
    imdb_language = [language for imdb_language_str in imdb_df.imdb_languages.dropna() for language in \
        imdb_language_str.split(";")]
    lang_count, lang_ticks = [], []

    for lang, count in sorted(Counter(imdb_language).items(), key=lambda x: x[1], reverse=True):
        lang_count.append(count)
        lang_ticks.append(lang)

    lang_ticks = [f"{x:>10s}" for x in lang_ticks]
    lang_axis = np.arange(len(lang_ticks))

    topk = 10

    plt.figure(figsize=(6, 4))
    plt.rcParams.update({"font.size":18})

    plt.bar(lang_axis[:topk], lang_count[:topk], width=0.6, color="coral", label="language")
    plt.xticks(ticks=lang_axis[:topk], labels=lang_ticks[:topk], rotation=90)
    plt.legend(prop={"size":24})
    plt.ylabel("Count")
    tikzplotlib.save(language_texfile)

def barplot_subtitle_data(imdb_file, kind_texfile, genre_texfile, country_texfile, language_texfile):
    imdb_df = pd.read_csv(imdb_file, index_col=None)

    kind_barplot(imdb_df, kind_texfile)
    genre_barplot(imdb_df, genre_texfile)
    country_barplot(imdb_df, country_texfile)
    language_barplot(imdb_df, language_texfile)
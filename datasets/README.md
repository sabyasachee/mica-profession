## Profession Taxonomy and Mentions

The professional taxonomy and mentions corpus are available for download at [Google-Drive/Profession](https://drive.google.com/drive/folders/1U1NOkg_hSzfjFqp-Z3a3xbPc_c2JJlwU?usp=sharing).

The folder contains five files:

1. __soc-taxonomy.csv__
It contains the Standard Occupational Classification (2018) taxonomy. It is a four-tiered taxonomy broken down into major, minor, broad and detailed groups. Detailed groups contain a set of closely related professions. There are 23 major SOC groups. This taxonomy is not searchable because the job titles are very detailed.

2. __expanded-taxonomy-professions.csv__ and __expanded-taxonomy-synsets.txt__
Expanded taxonomy extends the SOC taxonomy by including more unigram job titles and professional synsets, which makes it searchable. The first file contains the list of professions. The second file contains the synsets.

3. __soc-mapped-expanded-taxonomy.csv__
It contains a subset of the expanded taxonomy which has been mapped to SOC groups. It contains 500 professions and 562 synsets, and covers more than 94% of the professional mentions.

4. __professional-mentions-corpus.csv__
It contains the professional mentions in the OpenSubtitles sentences from IMDb titles between the years 1950-2017. It contains 10 columns:

    i. _profession_ column is the name of the profession.

    ii. _soc-code_ is the two-letter SOC major group code the profession is mapped to. Sometimes the profession can be mapped to two SOC groups (_secondary-SOC_ column in __soc-mapped-expanded-taxonomy.csv__). The letter codes are separated by semicolon then. If the profession and sense pair is absent in __soc-mapped-expanded-taxonomy.csv__, then this column is empty.

    iii. _soc-name_ is the name of the SOC major group the profession is mapped to. Similar to _soc-code_ column, there can be two names separated by semicolon, or the column can be empty.

    iv. _imdb-id_ is the IMDb identifier of the title to which the subtitle sentence belongs.

    v. _sentence_ is the subtitle sentence containing the professional mention.

    vi. _sentence-left-context_ is the substring of the subtitle sentence present to the left of the professional mention.

    vii. _sentence-mention_ is the professional mention.

    viii. _sentence-right-context_ is the substring of the subtitle sentence present to the right of the professional mention.

    ix. _sense_ is the corresponding WordNet sense of the professional mention. It is also present in __expanded-taxonomy-synsets.txt__.
    
    x. _sentiment_ is the targeted sentiment label of the professional mention. It is 0 (neutral), 1 (positive), or -1 (negative).
#!/bin/bash

python create_profession_gazetteer.py

for index in {0..9}
do
    python key_to_si.py --index $index --out /proj/sbaruah/subtitle/profession/csl/data/mentions/key_to_si_$index.json &
    declare bgpid[$((index + 1))]=$!
done

wait ${bgpid[@]}

python key_to_si_merge.py

for index in {0..9}
do
    rm /proj/sbaruah/subtitle/profession/csl/data/mentions/key_to_si_$index.json
done

python si_to_rsi.py
python process_sentence.py
python find_mentions.py
python annotation.py
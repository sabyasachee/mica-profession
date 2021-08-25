#!/bin/bash

for batch in {0..9}
do
    python process_sentence_finegrained.py --n_batches 10 --batch_index $batch &
    declare bgpid[$((batch + 1))]=$!
done

wait ${bgpid[@]}

python merge_finegrained_processed.py

for index in {0..9}
do
    rm error_$index.log
    rm data/mentions/processed_finegrained_$index.json
done
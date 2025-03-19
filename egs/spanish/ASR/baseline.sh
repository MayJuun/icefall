#!/bin/bash

./zipformer/decode_spanish.py \
  --epoch 30 \
  --avg 1 \
  --exp-dir ~/asr-projects/models/icefall-asr-commonvoice-fr-pruned-transducer-stateless7-streaming-2023-04-02/exp \
  --use-averaged-model 0 \
  --max-duration 1000 \
  --decoding-method greedy_search \
  --bpe-model ~/asr-projects/models/icefall-asr-commonvoice-fr-pruned-transducer-stateless7-streaming-2023-04-02/data/lang_bpe_500/bpe.model \

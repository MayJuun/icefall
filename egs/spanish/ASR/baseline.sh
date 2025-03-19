#!/bin/bash

./zipformer/decode_spanish.py \
  --epoch 99 \
  --avg 1 \
  --use-averaged-model false \
  --exp-dir /home/grey_faulkenberry_mayjuun_com/asr-projects/models/icefall-asr-commonvoice-fr-pruned-transducer-stateless7-streaming-2023-04-02/exp \
  --num-encoder-layers "2,4,3,2,4" \
  --downsampling-factor "1,2,4,8,2" \
  --feedforward-dim "1024,1024,2048,2048,1024" \
  --num-heads "8,8,8,8,8" \
  --encoder-dim "384,384,384,384,384" \
  --encoder-unmasked-dim "256,256,256,256,256" \
  --cnn-module-kernel "31,31,31,31,31" \
  --decoder-dim 512 \
  --joiner-dim 512

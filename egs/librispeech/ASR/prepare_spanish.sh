#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

# Number of parallel jobs for Lhotse if needed
nj=15

# run step 0 to step 5 by default
stage=0
stop_stage=5

# Directory where your Spanish data resides.
# E.g., if you have /path/to/es-datasets with train/dev/test subfolders, unify CSV, etc.
es_data_dir=/absolute/path/to/es-datasets

# A CSV containing info about every clip (audio_path, transcript, split, etc.).
# That file is your "unified_manifest.csv" with 1 row per segment.
es_csv=$es_data_dir/unified_manifest.csv

# Where we store all downloaded or external data: for musan or other.
dl_dir=$PWD/download

# We’ll store Lhotse manifests, FBank features, BPE models, etc. in `data/`.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "Running prepare_spanish.sh"

# SentencePiece vocabulary sizes to train
vocab_sizes=(500 2000 5000)  # Adjust as you like

# If you want to use MUSAN for data augmentation:
use_musan=true

# --- STAGE -1 [Optional]: Download MUSAN (if you want to do augmentation)
if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
  log "Stage -1: Download musan if needed"
  if $use_musan; then
    if [ ! -d $dl_dir/musan ]; then
      lhotse download musan $dl_dir
    fi
  fi
fi

# --- STAGE 0: Prepare Spanish Lhotse manifests from your CSV
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Prepare Spanish manifest (cuts) from unified_manifest.csv"
  mkdir -p data/manifests

  # We'll call a Python script that reads your CSV, splits into train/dev/test, and writes:
  #  data/manifests/spa_cuts_train.jsonl.gz
  #  data/manifests/spa_cuts_dev.jsonl.gz
  #  data/manifests/spa_cuts_test.jsonl.gz
  if [ ! -f data/manifests/spa_cuts_train.jsonl.gz ]; then
    # The script below is an example name; you'll need to create it in local/prepare_spanish.py
    python3 local/prepare_spanish.py \
      --csv-path $es_csv \
      --data-dir $es_data_dir \
      --manifest-dir data/manifests \
      --nj $nj
  fi
fi

# --- STAGE 1: Prepare MUSAN manifest (optional)
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  if $use_musan; then
    log "Stage 1: Prepare musan manifest"
    mkdir -p data/manifests
    if [ ! -f data/manifests/musan_cuts.jsonl.gz ]; then
      lhotse prepare musan $dl_dir/musan data/manifests
      # This will produce something like data/manifests/musan_recordings_all.jsonl.gz
      # Then you'd turn that into cuts with some standard script, or a similar approach:
      python3 local/compute_fbank_musan.py \
        --recordings-json data/manifests/musan_recordings_all.jsonl.gz \
        --output-json data/manifests/musan_cuts.jsonl.gz
    fi
  else
    log "Skipping MUSAN as use_musan=false"
  fi
fi

# --- STAGE 2: Compute FBank for Spanish
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Compute FBank for Spanish"
  mkdir -p data/fbank

  if [ ! -f data/fbank/.spa.done ]; then
    # You'd write a small script local/compute_fbank_spanish.py
    # that loads data/manifests/spa_cuts_{train,dev,test}.jsonl.gz,
    # uses Lhotse to compute_fbank, and writes out new CutSets with feature info.
    python3 local/compute_fbank_spanish.py \
      --in-dir data/manifests \
      --out-dir data/fbank
    touch data/fbank/.spa.done
  fi
fi

# --- STAGE 3: (Optional) Validate your Spanish FBank data
# Similar to local/validate_manifest.py in the LibriSpeech recipe
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Validate Spanish fbank data"
  # e.g., python local/validate_manifest.py data/fbank/spa_cuts_train.jsonl.gz
  # ...
fi

# --- STAGE 4: Combine or shuffle training cuts if desired
# (like the example that merges train-clean-100 + train-clean-360 + train-other-500 for LibriSpeech)
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: (Optional) Shuffle train cuts"
  # If you want a big random shuffle for training, do something like:
  if [ ! -f data/fbank/spa_cuts_train-shuf.jsonl.gz ]; then
    gunzip -c data/fbank/spa_cuts_train.jsonl.gz | shuf | gzip -c \
      > data/fbank/spa_cuts_train-shuf.jsonl.gz
  fi
fi

# --- STAGE 5: Prepare BPE-based lang for Spanish
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Prepare BPE based lang for Spanish"
  # We'll create data/lang_bpe_{vocab_size} for each vocabulary size
  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}
    mkdir -p $lang_dir

    if [ ! -f $lang_dir/transcript_words.txt ]; then
      log "Generate transcripts for BPE training"
      # We can extract text from the train supervision/cuts
      # or you can gather them from your CSV again
      python3 local/extract_texts_for_bpe.py \
        --manifest data/fbank/spa_cuts_train.jsonl.gz \
        --output $lang_dir/transcript_words.txt
    fi

    if [ ! -f $lang_dir/bpe.model ]; then
      log "Train SentencePiece BPE model (vocab_size=$vocab_size)"
      python3 local/train_bpe_model.py \
        --lang-dir $lang_dir \
        --vocab-size $vocab_size \
        --transcript $lang_dir/transcript_words.txt
    fi
  done
fi

log "All done. Spanish data is ready for Zipformer training!"

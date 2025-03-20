#!/usr/bin/env python3

# This script is adapted directly from pruned_transducer_stateless7/local/compute_fbank_librispeech.py
# in the official icefall recipes, but modified for Spanish "train/dev/test" sets.
#
# The official approach:
#  1) Possibly filter short/long utterances (if desired, using BPE model).
#  2) Speed perturb the train set with factors 0.9, 1.0, 1.1 if --perturb-speed=true.
#  3) Extract 80-bin FBank features, storing them in data/fbank/
#  4) Save the final feature-enriched CutSet to data/fbank/spa_cuts_<subset>.jsonl.gz
#
# We do not guess about arbitrary user preferences. This is a direct parallel
# to the official "compute_fbank_librispeech.py".

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import get_executor, str2bool

# Torch's multithreaded behavior is disabled for speed, exactly as in the official code.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_args():
    parser = argparse.ArgumentParser(
        description="Compute FBank features for Spanish sets, replicating official icefall approach."
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default=None,
        help="""Path to a bpe.model. If not None, we can remove too-short or too-long utterances
        before extracting features (the official approach does 'filter_cuts' if a BPE model is given).""",
    )

    parser.add_argument(
        "--perturb-speed",
        type=str2bool,
        default=True,
        help="""If true, apply speed perturbation to the train subset with factors 0.9 and 1.1.
        This matches the official icefall approach for training data.""",
    )

    return parser.parse_args()


def compute_fbank_spanish(bpe_model: Optional[str] = None, perturb_speed: bool = True):
    """
    Following the official pruned_transducer_stateless7/local/compute_fbank_librispeech.py logic:

    1) We read data/manifests/spa_cuts_{train,dev,test}.jsonl.gz for Spanish sets.
    2) If bpe_model is given, we could filter short/long utterances (official does it via filter_cuts).
    3) Speed perturb train if perturb_speed=True.
    4) Compute 80-bin FBank, store them in data/fbank/<prefix>_feats_<subset>.*.
    5) Final manifested cutsets in data/fbank/spa_cuts_<subset>.jsonl.gz.
    """

    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Official code sets a max of 15 parallel jobs or CPU count
    num_jobs = min(15, os.cpu_count())
    num_mel_bins = 80  # official recipe standard

    # If we want to filter with a BPE model, we do it similarly to official code (using filter_cuts).
    # For reference, official scripts do: filter_cuts(cut_set, sp) if bpe_model is not None.
    sp = None
    if bpe_model:
        import sentencepiece as spm
        from filter_cuts import filter_cuts  # the official script references a local filter_cuts
        logging.info(f"Loading BPE model from {bpe_model}")
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)

    # We'll do "train", "dev", "test" exactly for Spanish sets
    dataset_parts = ("train", "dev", "test")
    prefix = "spa_cuts"
    suffix = "jsonl.gz"

    # Create the Fbank feature extractor (official uses Fbank(FbankConfig(num_mel_bins=...))).
    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    with get_executor() as ex:
        for subset in dataset_parts:
            # E.g. data/manifests/spa_cuts_train.jsonl.gz
            cuts_filename = f"{prefix}_{subset}.{suffix}"
            in_path = src_dir / cuts_filename

            if not in_path.is_file():
                logging.warning(f"{in_path} not found; skipping {subset}")
                continue

            # The final output path is data/fbank/spa_cuts_{subset}.jsonl.gz
            out_cuts_path = output_dir / cuts_filename
            if out_cuts_path.is_file():
                logging.info(f"{out_cuts_path} already exists - skipping {subset}.")
                continue

            logging.info(f"Processing {subset}")

            cut_set = CutSet.from_file(in_path)

            # 1) If we have a BPE model, we might filter short/long utterances
            if sp is not None:
                from filter_cuts import filter_cuts
                logging.info("Filtering cuts with BPE model constraints (official approach).")
                cut_set = filter_cuts(cut_set, sp)

            # 2) Speed perturb only for train
            if subset == "train" and perturb_speed:
                logging.info("Speed perturbing train set with factors 0.9 and 1.1.")
                cut_set = cut_set + cut_set.perturb_speed(0.9) + cut_set.perturb_speed(1.1)

            # 3) Compute FBank
            # official code:
            # cut_set = cut_set.compute_and_store_features(
            #   ...
            #   num_jobs=num_jobs if ex is None else 80,
            #   executor=ex,
            # )

            cut_set = cut_set.compute_and_store_features(
                extractor=extractor,
                storage_path=f"{output_dir}/spa_feats_{subset}",
                num_jobs=num_jobs if ex is None else 80,
                executor=ex,
                storage_type=LilcomChunkyWriter,
            )

            # 4) Save final cut set with features
            cut_set.to_file(out_cuts_path)
            logging.info(f"Saved {subset} subset to {out_cuts_path}")


def main():
    logging.basicConfig(format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
                        level=logging.INFO)
    args = get_args()
    logging.info(vars(args))

    compute_fbank_spanish(
        bpe_model=args.bpe_model,
        perturb_speed=args.perturb_speed,
    )


if __name__ == "__main__":
    main()

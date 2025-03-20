#!/usr/bin/env python3

import argparse
import csv
import os
import logging
from pathlib import Path

from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet, CutSet, fix_manifests

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a Spanish CSV manifest to Lhotse cuts for train/dev/test."
    )
    parser.add_argument("--csv-path", required=True, help="Path to your unified_manifest.csv")
    parser.add_argument("--data-dir", required=True, help="Root directory containing the WAV files")
    parser.add_argument("--manifest-dir", required=True, help="Output directory for Lhotse manifests")
    parser.add_argument("--nj", type=int, default=15, help="(Unused) Number of parallel jobs if needed")
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
                        level=logging.INFO)

    Path(args.manifest_dir).mkdir(parents=True, exist_ok=True)

    train_recordings, train_supers = [], []
    dev_recordings, dev_supers = [], []
    test_recordings, test_supers = [], []

    with open(args.csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_rel_path = row["audio_path"]  # e.g. "train/common_voice/..."
            text = row["text"] or ""
            split = row["split"]
            duration_str = row["duration"] or "0.0"
            duration = float(duration_str)
            speaker = row["speaker"] if row["speaker"] else "unknown"
            start_str = row["start_time"] or "0.0"
            start_time = float(start_str)

            # We'll rely on the CSV for sample_rate/num_samples
            # - If missing, default to 16k or compute from duration
            sr_str = row["sample_rate"] or "16000"
            sr = float(sr_str)
            ns_str = row["num_samples"]
            if ns_str:
                num_samples = int(float(ns_str))
            else:
                # fallback: compute from duration * sr if not in CSV
                num_samples = int(duration * sr + 0.5)

            # Create a unique ID for this recording
            recording_id = audio_rel_path.replace("/", "_").replace(".wav", "")

            # Full absolute path to the WAV
            audio_path = os.path.join(args.data_dir, audio_rel_path)

            # Manually construct a Lhotse Recording without reading the file
            rec = Recording(
                id=recording_id,
                sampling_rate=int(sr),
                num_samples=num_samples,
                duration=duration,
                sources=[
                    {
                        "type": "file",
                        "channels": [0],
                        "source": audio_path
                    }
                ],
            )

            # Create a SupervisionSegment
            sup = SupervisionSegment(
                id=recording_id,
                recording_id=recording_id,
                start=start_time,
                duration=duration,
                text=text,
                language="Spanish",
                speaker=speaker,
            )

            # Distribute into train/dev/test lists
            if split == "train":
                train_recordings.append(rec)
                train_supers.append(sup)
            elif split == "validation":
                dev_recordings.append(rec)
                dev_supers.append(sup)
            elif split == "test":
                test_recordings.append(rec)
                test_supers.append(sup)
            else:
                # If there's an unknown split, skip or log
                logging.warning(f"Skipping unknown split: {split}")

    # Build RecordingSet & SupervisionSet for each split
    logging.info("Creating RecordingSet and SupervisionSet for train...")
    train_rec_set = RecordingSet.from_recordings(train_recordings)
    train_sup_set = SupervisionSet.from_segments(train_supers)
    train_rec_set, train_sup_set = fix_manifests(train_rec_set, train_sup_set)
    train_cuts = CutSet.from_manifests(recordings=train_rec_set, supervisions=train_sup_set)

    logging.info("Creating RecordingSet and SupervisionSet for dev...")
    dev_rec_set = RecordingSet.from_recordings(dev_recordings)
    dev_sup_set = SupervisionSet.from_segments(dev_supers)
    dev_rec_set, dev_sup_set = fix_manifests(dev_rec_set, dev_sup_set)
    dev_cuts = CutSet.from_manifests(recordings=dev_rec_set, supervisions=dev_sup_set)

    logging.info("Creating RecordingSet and SupervisionSet for test...")
    test_rec_set = RecordingSet.from_recordings(test_recordings)
    test_sup_set = SupervisionSet.from_segments(test_supers)
    test_rec_set, test_sup_set = fix_manifests(test_rec_set, test_sup_set)
    test_cuts = CutSet.from_manifests(recordings=test_rec_set, supervisions=test_sup_set)

    # Save to JSONL.GZ
    train_cuts.to_file(f"{args.manifest_dir}/spa_cuts_train.jsonl.gz")
    dev_cuts.to_file(f"{args.manifest_dir}/spa_cuts_dev.jsonl.gz")
    test_cuts.to_file(f"{args.manifest_dir}/spa_cuts_test.jsonl.gz")

    logging.info(f"Finished! Wrote train/dev/test cuts to {args.manifest_dir}")

if __name__ == "__main__":
    main()

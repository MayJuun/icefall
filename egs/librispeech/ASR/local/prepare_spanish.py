#!/usr/bin/env python3

import argparse
import csv
import os
from pathlib import Path
import logging

from lhotse import RecordingSet, Recording, SupervisionSet, SupervisionSegment, CutSet, fix_manifests

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--nj", type=int, default=15)
    return parser.parse_args()

def main():
    args = parse_args()
    Path(args.manifest_dir).mkdir(parents=True, exist_ok=True)

    train_recordings, train_supers = [], []
    dev_recordings, dev_supers = [], []
    test_recordings, test_supers = [], []

    with open(args.csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_rel_path = row["audio_path"]
            text = row["text"]
            split = row["split"]
            duration = float(row["duration"]) if row["duration"] else 0.0
            speaker = row["speaker"] if row["speaker"] else "unknown"
            start_time = float(row["start_time"]) if row["start_time"] else 0.0

            # create a unique ID
            recording_id = audio_rel_path.replace("/", "_")
            # full audio path
            audio_path = os.path.join(args.data_dir, audio_rel_path)

            rec = Recording(
                id=recording_id,
                sampling_rate=16000,  # or read from CSV if it differs
                num_channels=1,
                duration=duration,
                sources=[{"type": "file", "channels": [0], "source": audio_path}],
            )
            sup = SupervisionSegment(
                id=recording_id,
                recording_id=recording_id,
                start=start_time,
                duration=duration,
                text=text,
                language="Spanish",
                speaker=speaker,
            )

            # Put them into train/dev/test based on CSV
            if split == "train":
                train_recordings.append(rec)
                train_supers.append(sup)
            elif split == "validation":
                dev_recordings.append(rec)
                dev_supers.append(sup)
            elif split == "test":
                test_recordings.append(rec)
                test_supers.append(sup)

    # Convert to RecordingSet, SupervisionSet
    train_rec_set = RecordingSet.from_recordings(train_recordings)
    train_sup_set = SupervisionSet.from_segments(train_supers)
    train_rec_set, train_sup_set = fix_manifests(train_rec_set, train_sup_set)
    train_cuts = CutSet.from_manifests(recordings=train_rec_set, supervisions=train_sup_set)

    dev_rec_set = RecordingSet.from_recordings(dev_recordings)
    dev_sup_set = SupervisionSet.from_segments(dev_supers)
    dev_rec_set, dev_sup_set = fix_manifests(dev_rec_set, dev_sup_set)
    dev_cuts = CutSet.from_manifests(recordings=dev_rec_set, supervisions=dev_sup_set)

    test_rec_set = RecordingSet.from_recordings(test_recordings)
    test_sup_set = SupervisionSet.from_segments(test_supers)
    test_rec_set, test_sup_set = fix_manifests(test_rec_set, test_sup_set)
    test_cuts = CutSet.from_manifests(recordings=test_rec_set, supervisions=test_sup_set)

    train_cuts.to_file(f"{args.manifest_dir}/spa_cuts_train.jsonl.gz")
    dev_cuts.to_file(f"{args.manifest_dir}/spa_cuts_dev.jsonl.gz")
    test_cuts.to_file(f"{args.manifest_dir}/spa_cuts_test.jsonl.gz")

    logging.info("Spanish manifests created successfully!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert a local Mozilla Common Voice release to a Hugging Face DatasetDict."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
from datasets import Audio, Dataset, DatasetDict, concatenate_datasets


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--common-voice-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--num-proc", type=int, default=1)
    return parser.parse_args()


def read_tsv(path: Path) -> Dataset:
    frame = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
    )
    return Dataset.from_pandas(frame, preserve_index=False)


def main() -> None:
    args = arguments()
    root = args.common_voice_dir.expanduser().resolve()
    clips = root / "clips"
    splits = {
        "train": read_tsv(root / "train.tsv"),
        "validation": read_tsv(root / "dev.tsv"),
        "test": read_tsv(root / "test.tsv"),
    }
    train = concatenate_datasets([splits["train"], splits["validation"]])
    result = DatasetDict({"train": train, "test": splits["test"]})

    def resolve_audio(batch: dict) -> dict:
        batch["path"] = [str((clips / name).resolve()) for name in batch["path"]]
        return batch

    result = result.map(resolve_audio, batched=True, num_proc=args.num_proc)
    removable = [
        column
        for column in result["train"].column_names
        if column not in {"path", "sentence"}
    ]
    result = result.remove_columns(removable).rename_column("path", "audio")
    result = result.cast_column(
        "audio", Audio(sampling_rate=args.sample_rate, decode=False)
    )
    result.save_to_disk(str(args.output_dir.expanduser().resolve()))
    print(result)


if __name__ == "__main__":
    main()

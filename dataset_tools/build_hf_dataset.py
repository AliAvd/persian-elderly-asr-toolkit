#!/usr/bin/env python3
"""Build, save and optionally upload a Hugging Face ASR DatasetDict."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-tsv", type=Path)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--max-alignment-wer", type=float, default=0.55)
    parser.add_argument("--include-review", action="store_true")
    parser.add_argument(
        "--group-by", choices=("speaker_id", "source_id"), default="speaker_id"
    )
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hub-repo")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            audio = Path(row["audio"]).expanduser().resolve()
            if not audio.is_file():
                raise FileNotFoundError(f"manifest line {line_number}: missing {audio}")
            row["audio"] = str(audio)
            rows.append(row)
    return rows


def apply_human_reviews(rows: list[dict], review_path: Path) -> int:
    by_chunk_id = {str(row["chunk_id"]): row for row in rows}
    approved_values = {"1", "true", "yes", "y", "بله", "تایید", "تأیید"}
    applied = 0
    with review_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"chunk_id", "sentence", "approved"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"review TSV columns missing: {sorted(missing)}")
        for line, review in enumerate(reader, 2):
            if review["approved"].strip().lower() not in approved_values:
                continue
            chunk_id = review["chunk_id"].strip()
            if chunk_id not in by_chunk_id:
                raise ValueError(f"review TSV line {line}: unknown chunk_id {chunk_id}")
            sentence = " ".join(review["sentence"].split())
            if not sentence:
                raise ValueError(f"review TSV line {line}: approved text is empty")
            row = by_chunk_id[chunk_id]
            row["sentence"] = sentence
            row["needs_review"] = False
            row["human_verified"] = True
            applied += 1
    return applied


def stable_score(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def split_by_group(
    rows: list[dict],
    group_by: str,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict]]:
    if validation_ratio < 0 or test_ratio < 0 or validation_ratio + test_ratio >= 1:
        raise ValueError(
            "validation/test ratios must be non-negative and sum to less than 1"
        )

    if group_by == "speaker_id":
        speakers = {str(row.get("speaker_id", "")).strip() for row in rows}
        if not speakers or speakers <= {"", "unknown"}:
            print("WARNING: unknown speaker IDs; falling back to source_id grouping")
            group_by = "source_id"

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_by])].append(row)

    names = sorted(groups, key=lambda name: stable_score(name, seed))
    count = len(names)
    test_count = round(count * test_ratio)
    validation_count = round(count * validation_ratio)
    if count >= 3 and test_ratio > 0:
        test_count = max(1, test_count)
    if count - test_count >= 2 and validation_ratio > 0:
        validation_count = max(1, validation_count)
    while test_count + validation_count >= count and validation_count:
        validation_count -= 1

    test_groups = set(names[:test_count])
    validation_groups = set(names[test_count : test_count + validation_count])
    result = {"train": [], "validation": [], "test": []}
    for name, group_rows in groups.items():
        split = (
            "test"
            if name in test_groups
            else "validation"
            if name in validation_groups
            else "train"
        )
        result[split].extend(group_rows)
    return {split: values for split, values in result.items() if values}


def write_qwen_jsonl(output_dir: Path, splits: dict[str, list[dict]]) -> None:
    target_dir = output_dir / "qwen_jsonl"
    target_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        with (target_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                item = {
                    "audio": row["audio"],
                    "text": f"language Persian<asr_text>{row['sentence']}",
                }
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    args = arguments()
    rows = read_manifest(args.manifest.expanduser().resolve())
    if args.review_tsv:
        applied = apply_human_reviews(rows, args.review_tsv.expanduser().resolve())
        print(f"Applied {applied} approved human reviews")
    accepted = []
    rejected = []
    for row in rows:
        score = row.get("alignment_wer")
        reasons = []
        if not args.min_duration <= float(row["duration"]) <= args.max_duration:
            reasons.append("duration")
        human_verified = bool(row.get("human_verified", False))
        if not human_verified:
            if score is None or float(score) > args.max_alignment_wer:
                reasons.append("alignment_wer")
            if row.get("needs_review") and not args.include_review:
                reasons.append("needs_review")
        if not row.get("sentence", "").strip():
            reasons.append("empty_text")
        (rejected if reasons else accepted).append((row, reasons))

    if not accepted:
        raise ValueError(
            "No rows passed filters. Review manifest rows first, or deliberately use "
            "--include-review with a suitable --max-alignment-wer."
        )

    split_rows = split_by_group(
        [row for row, _ in accepted],
        args.group_by,
        args.validation_ratio,
        args.test_ratio,
        args.seed,
    )

    from datasets import Audio, Dataset, DatasetDict

    columns = [
        "audio",
        "sentence",
        "speaker_id",
        "source_id",
        "chunk_id",
        "start",
        "end",
        "duration",
        "asr_prediction",
        "alignment_wer",
        "alignment_method",
        "human_verified",
    ]
    datasets = {}
    for split, values in split_rows.items():
        records = [{column: row.get(column) for column in columns} for row in values]
        dataset = Dataset.from_list(records)
        datasets[split] = dataset.cast_column(
            "audio", Audio(sampling_rate=args.sample_rate)
        )
    dataset_dict = DatasetDict(datasets)

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"{output_dir} is not empty; choose a new output directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))
    write_qwen_jsonl(output_dir, split_rows)

    with (output_dir / "rejected.jsonl").open("w", encoding="utf-8") as handle:
        for row, reasons in rejected:
            handle.write(
                json.dumps({**row, "rejection_reasons": reasons}, ensure_ascii=False)
                + "\n"
            )

    print(dataset_dict)
    print(f"Accepted chunks: {len(accepted)}; rejected chunks: {len(rejected)}")
    print(f"Saved dataset: {output_dir}")
    print(f"Qwen JSONL: {output_dir / 'qwen_jsonl'}")

    if args.hub_repo:
        dataset_dict.push_to_hub(args.hub_repo, private=args.private)
        print(f"Uploaded: https://huggingface.co/datasets/{args.hub_repo}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge an incremental chunking run into an existing chunk directory safely."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--increment-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_tsv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def write_tsv(path: Path, fields: list[str], rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = arguments()
    base = args.base_dir.resolve()
    increment = args.increment_dir.resolve()

    base_rows = read_jsonl(base / "manifest.jsonl")
    new_rows = read_jsonl(increment / "manifest.jsonl")
    known = {row["chunk_id"] for row in base_rows}
    duplicates = sorted(known.intersection(row["chunk_id"] for row in new_rows))
    if duplicates:
        raise ValueError(f"Duplicate chunk IDs: {duplicates[:10]}")

    for folder in ("chunks", "texts"):
        destination = base / folder
        destination.mkdir(parents=True, exist_ok=True)
        for source in sorted((increment / folder).iterdir()):
            target = destination / source.name
            if target.exists():
                raise FileExistsError(target)
            shutil.copy2(source, target)

    for row in new_rows:
        row["audio"] = str((base / "chunks" / Path(row["audio"]).name).resolve())
    write_jsonl(base / "manifest.jsonl", base_rows + new_rows)
    write_jsonl(
        base / "qwen_all.jsonl",
        [
            {
                "audio": row["audio"],
                "text": f"language Persian<asr_text>{row['sentence']}",
            }
            for row in base_rows + new_rows
        ],
    )

    for filename in ("all_chunks.tsv", "review.tsv"):
        base_fields, old = read_tsv(base / filename)
        new_fields, added = read_tsv(increment / filename)
        if base_fields != new_fields:
            raise ValueError(f"TSV schemas differ for {filename}")
        for row in added:
            if row.get("audio"):
                row["audio"] = str(
                    (base / "chunks" / Path(row["audio"]).name).resolve()
                )
            if row.get("text_file"):
                row["text_file"] = str(
                    (base / "texts" / Path(row["text_file"]).name).resolve()
                )
        write_tsv(base / filename, base_fields, old + added)

    print(f"Merged {len(new_rows)} chunks; total: {len(base_rows) + len(new_rows)}")


if __name__ == "__main__":
    main()

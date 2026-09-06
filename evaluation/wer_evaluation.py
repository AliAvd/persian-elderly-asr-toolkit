#!/usr/bin/env python3
"""Evaluate one or more Qwen3-ASR checkpoints on a JSONL test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from jiwer import cer, wer
from qwen_asr import Qwen3ASRModel


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_outputs"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--high-wer-threshold", type=float, default=0.4)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def load_samples(path: Path) -> list[dict[str, str]]:
    samples = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            text = str(item.get("text", "")).split("<asr_text>", 1)[-1].strip()
            audio = Path(item["audio"]).expanduser()
            if not audio.is_file():
                raise FileNotFoundError(f"line {number}: {audio}")
            if text:
                samples.append({"audio": str(audio.resolve()), "reference": text})
    return samples


def evaluate_checkpoint(
    checkpoint: str,
    samples: list[dict[str, str]],
    output_dir: Path,
    batch_size: int,
    threshold: float,
    device: str,
) -> dict[str, float | str | int]:
    use_cuda = device.startswith("cuda")
    model = Qwen3ASRModel.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16 if use_cuda else torch.float32,
        device_map=device,
    )
    rows = []
    for offset in range(0, len(samples), batch_size):
        batch = samples[offset : offset + batch_size]
        results = model.transcribe(
            audio=[item["audio"] for item in batch], language="Persian"
        )
        if len(results) != len(batch):
            raise RuntimeError("ASR output count differs from input count")
        for sample, result in zip(batch, results):
            prediction = result.text.strip()
            rows.append(
                {
                    **sample,
                    "prediction": prediction,
                    "sample_wer": wer(sample["reference"], prediction),
                }
            )

    references = [row["reference"] for row in rows]
    predictions = [row["prediction"] for row in rows]
    output_dir.mkdir(parents=True, exist_ok=True)
    name = checkpoint.rstrip("/").replace("/", "_")
    with (output_dir / f"{name}_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / f"{name}_high_wer.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(row for row in rows if row["sample_wer"] > threshold)

    return {
        "checkpoint": checkpoint,
        "samples": len(rows),
        "wer": wer(references, predictions),
        "cer": cer(references, predictions),
    }


def main() -> None:
    args = arguments()
    samples = load_samples(args.test_jsonl)
    if not samples:
        raise ValueError(f"No usable samples found in {args.test_jsonl}")
    summaries = [
        evaluate_checkpoint(
            checkpoint,
            samples,
            args.output_dir,
            args.batch_size,
            args.high_wer_threshold,
            args.device,
        )
        for checkpoint in args.checkpoint
    ]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

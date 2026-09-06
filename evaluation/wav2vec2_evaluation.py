#!/usr/bin/env python3
"""Evaluate a Wav2Vec2 CTC checkpoint on a Qwen-style JSONL test split."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import librosa
import torch
from jiwer import cer, wer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

PUNCTUATION = re.compile(r"[,?؛:!؟._–«»#&()…=\"'،ـ]+")
DIACRITICS = re.compile(r"[\u064b-\u0652]")
CHAR_MAP = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه"})


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument(
        "--output-csv", type=Path, default=Path("wav2vec2_predictions.csv")
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--sample-rate", type=int, default=16_000)
    return parser.parse_args()


def normalize(text: str) -> str:
    text = text.split("<asr_text>", 1)[-1].translate(CHAR_MAP)
    text = DIACRITICS.sub("", PUNCTUATION.sub(" ", text)).replace("\u200c", " ")
    return " ".join(text.lower().split())


def load_rows(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            reference = normalize(str(item.get("text", "")))
            audio = Path(item["audio"]).expanduser()
            if not audio.is_file():
                raise FileNotFoundError(f"line {number}: {audio}")
            if reference:
                rows.append({"audio": str(audio.resolve()), "reference": reference})
    return rows


def main() -> None:
    args = arguments()
    rows = load_rows(args.test_jsonl)
    processor = Wav2Vec2Processor.from_pretrained(args.checkpoint)
    model = Wav2Vec2ForCTC.from_pretrained(args.checkpoint).to(args.device).eval()

    predictions = []
    with torch.inference_mode():
        for row in rows:
            audio, _ = librosa.load(row["audio"], sr=args.sample_rate, mono=True)
            values = processor(
                audio, sampling_rate=args.sample_rate, return_tensors="pt"
            ).input_values.to(args.device)
            token_ids = model(values).logits.argmax(dim=-1)
            prediction = normalize(processor.batch_decode(token_ids)[0])
            predictions.append({**row, "prediction": prediction})

    references = [row["reference"] for row in predictions]
    hypotheses = [row["prediction"] for row in predictions]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)

    print(f"samples={len(predictions)}")
    print(f"wer={wer(references, hypotheses):.6f}")
    print(f"cer={cer(references, hypotheses):.6f}")


if __name__ == "__main__":
    main()

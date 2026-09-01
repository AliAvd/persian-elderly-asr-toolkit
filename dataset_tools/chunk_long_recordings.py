#!/usr/bin/env python3
"""Split long recordings with Silero VAD and align chunks to reference text."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

AUDIO_SUFFIXES = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".aac"}
TEXT_SUFFIXES = {".txt", ".text"}
PREFIX_RE = re.compile(
    r"^(?:voice|audio|recording|text|transcript)[-_ ]*", re.IGNORECASE
)
NON_WORD_RE = re.compile(r"[^0-9A-Za-z\u0600-\u06ff\s]+")
CHAR_MAP = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه"})


@dataclass(frozen=True)
class SourcePair:
    audio: Path
    text: Path
    speaker_id: str
    source_id: str


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def seconds(self, sample_rate: int) -> float:
        return (self.end - self.start) / sample_rate


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--asr-checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--target-duration", type=float, default=8.0)
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--max-duration", type=float, default=15.0)
    parser.add_argument("--max-internal-silence", type=float, default=1.2)
    parser.add_argument("--speech-pad-ms", type=int, default=120)
    parser.add_argument("--min-speech-ms", type=int, default=180)
    parser.add_argument("--min-silence-ms", type=int, default=250)
    parser.add_argument("--review-wer", type=float, default=0.55)
    parser.add_argument("--speaker-id", default="unknown")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(CHAR_MAP)
    return " ".join(NON_WORD_RE.sub(" ", text).split())


def key_for(path: Path) -> str:
    value = PREFIX_RE.sub("", path.stem.lower())
    return re.sub(r"[^0-9a-z\u0600-\u06ff]+", "", value) or path.stem.lower()


def safe_id(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z\u0600-\u06ff_-]+", "_", value.strip())
    return value.strip("_") or "recording"


def resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def read_pairs(
    input_dir: Path, metadata: Path | None, speaker: str
) -> list[SourcePair]:
    if metadata:
        pairs = []
        with metadata.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"audio", "text", "speaker_id"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"metadata columns missing: {sorted(missing)}")
            for line, row in enumerate(reader, 2):
                audio, text = (
                    resolve(input_dir, row["audio"]),
                    resolve(input_dir, row["text"]),
                )
                if not audio.is_file() or not text.is_file():
                    raise FileNotFoundError(f"metadata line {line} has a missing file")
                pairs.append(
                    SourcePair(
                        audio,
                        text,
                        row["speaker_id"].strip() or speaker,
                        safe_id(row.get("source_id") or key_for(audio)),
                    )
                )
        return pairs

    files = [item for item in input_dir.rglob("*") if item.is_file()]
    audios = [item for item in files if item.suffix.lower() in AUDIO_SUFFIXES]
    texts = [
        item
        for item in files
        if item.suffix.lower() in TEXT_SUFFIXES
        or (not item.suffix and item.name.lower().startswith(("text", "transcript")))
    ]
    text_map: dict[str, list[Path]] = {}
    for item in texts:
        text_map.setdefault(key_for(item), []).append(item)

    pairs, errors = [], []
    for audio in sorted(audios):
        exact = audio.with_suffix(".txt")
        candidates = [exact] if exact.is_file() else text_map.get(key_for(audio), [])
        if len(candidates) != 1:
            errors.append(f"{audio}: {len(candidates)} matching texts")
            continue
        inferred_speaker = audio.parent.name if audio.parent != input_dir else speaker
        pairs.append(
            SourcePair(audio, candidates[0], inferred_speaker, safe_id(key_for(audio)))
        )
    if errors:
        raise ValueError(
            "Ambiguous audio/text pairs; use metadata.csv:\n  " + "\n  ".join(errors)
        )
    if not pairs:
        raise ValueError(f"No audio/text pairs found in {input_dir}")
    return pairs


def split_regions(
    timestamps: Sequence[dict],
    total_samples: int,
    sample_rate: int,
    target_seconds: float,
    min_seconds: float,
    max_seconds: float,
    max_silence_seconds: float,
    pad_ms: int,
) -> list[Span]:
    if not timestamps:
        return []
    maximum = round(max_seconds * sample_rate)
    target = round(target_seconds * sample_rate)
    minimum = round(min_seconds * sample_rate)
    max_gap = round(max_silence_seconds * sample_rate)
    pad = round(pad_ms * sample_rate / 1000)

    regions = []
    for item in timestamps:
        cursor, end = int(item["start"]), int(item["end"])
        while end - cursor > maximum:
            regions.append((cursor, cursor + maximum))
            cursor += maximum
        if end > cursor:
            regions.append((cursor, end))

    grouped = []
    current = [regions[0][0], regions[0][1]]
    for start, end in regions[1:]:
        gap, projected = start - current[1], end - current[0]
        if gap > max_gap or projected > maximum or current[1] - current[0] >= target:
            grouped.append(current)
            current = [start, end]
        else:
            current[1] = end
    grouped.append(current)

    merged = []
    for item in grouped:
        if (
            merged
            and item[1] - item[0] < minimum
            and item[1] - merged[-1][0] <= maximum
            and item[0] - merged[-1][1] <= max_gap
        ):
            merged[-1][1] = item[1]
        else:
            merged.append(item)
    return [
        Span(max(0, start - pad), min(total_samples, end + pad))
        for start, end in merged
        if end > start
    ]


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, ref_word in enumerate(reference, 1):
        current = [row]
        for column, hyp_word in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ref_word != hyp_word),
                )
            )
        previous = current
    return previous[-1]


def proportional(words: list[str], durations: list[float]) -> list[list[str]]:
    total, cumulative, bounds = sum(durations), 0.0, [0]
    for duration in durations[:-1]:
        cumulative += duration
        bounds.append(round(len(words) * cumulative / total))
    bounds.append(len(words))
    return [words[start:end] for start, end in itertools.pairwise(bounds)]


def align(
    reference: list[str],
    predictions: list[list[str]],
    durations: list[float],
    beam_width: int = 64,
) -> list[list[str]]:
    """Find a minimum-cost contiguous reference partition for ASR predictions."""
    if not predictions:
        return []
    if not reference:
        return [[] for _ in predictions]
    total_duration = sum(durations)
    states: dict[int, tuple[float, list[tuple[int, int]]]] = {0: (0.0, [])}
    for index, (prediction, duration) in enumerate(zip(predictions, durations)):
        left_chunks = len(predictions) - index - 1
        next_states = {}
        for start, (base_cost, path) in states.items():
            remaining = len(reference) - start
            if index == len(predictions) - 1:
                ends = [len(reference)]
            else:
                duration_guess = len(reference) * duration / total_duration
                center = (
                    0.75 * (len(prediction) or duration_guess) + 0.25 * duration_guess
                )
                low = max(1, math.floor(center * 0.45) - 2)
                high = max(low, math.ceil(center * 1.8) + 3)
                high = min(high, remaining - left_chunks)
                low = min(low, high)
                ends = range(start + low, start + high + 1)
            for end in ends:
                if end < start or len(reference) - end < left_chunks:
                    continue
                ref_span = reference[start:end]
                distance = edit_distance(ref_span, prediction)
                wer_cost = distance / max(len(ref_span), len(prediction), 1)
                expected = len(reference) * duration / total_duration
                length_cost = abs(len(ref_span) - expected) / max(expected, 1)
                cost = base_cost + wer_cost + 0.12 * length_cost
                old = next_states.get(end)
                if old is None or cost < old[0]:
                    next_states[end] = (cost, path + [(start, end)])
        if not next_states:
            raise RuntimeError("Transcript alignment failed")
        states = dict(
            sorted(next_states.items(), key=lambda item: item[1][0])[:beam_width]
        )
    if len(reference) not in states:
        raise RuntimeError("Alignment did not consume the complete transcript")
    return [reference[start:end] for start, end in states[len(reference)][1]]


def get_qwen(checkpoint: str, device: str):
    import torch
    from qwen_asr import Qwen3ASRModel

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return Qwen3ASRModel.from_pretrained(checkpoint, dtype=dtype, device_map=device)


def transcribe(model, paths: list[Path], batch_size: int) -> list[str]:
    texts = []
    for offset in range(0, len(paths), batch_size):
        batch = [str(path) for path in paths[offset : offset + batch_size]]
        texts.extend(
            normalize(result.text)
            for result in model.transcribe(audio=batch, language="Persian")
        )
    if len(texts) != len(paths):
        raise RuntimeError("ASR output count differs from chunk count")
    return texts


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = arguments()
    input_dir, output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    manifest_path = output_dir / "manifest.jsonl"
    chunks_dir = output_dir / "chunks"
    texts_dir = output_dir / "texts"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} exists; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    texts_dir.mkdir(parents=True, exist_ok=True)

    pairs = read_pairs(
        input_dir, args.metadata.resolve() if args.metadata else None, args.speaker_id
    )
    source_ids = [pair.source_id for pair in pairs]
    duplicates = sorted({item for item in source_ids if source_ids.count(item) > 1})
    if duplicates:
        raise ValueError(
            f"Duplicate source_id values would overwrite chunks: {duplicates}"
        )

    import librosa
    import numpy as np
    import soundfile as sf
    from silero_vad import get_speech_timestamps, load_silero_vad

    vad = load_silero_vad()
    qwen = get_qwen(args.asr_checkpoint, args.device) if args.asr_checkpoint else None
    rows = []

    for pair_number, pair in enumerate(pairs, 1):
        print(f"[{pair_number}/{len(pairs)}] {pair.audio.name}")
        waveform, _ = librosa.load(
            pair.audio, sr=args.sample_rate, mono=True, dtype=np.float32
        )
        timestamps = get_speech_timestamps(
            waveform,
            vad,
            sampling_rate=args.sample_rate,
            min_speech_duration_ms=args.min_speech_ms,
            min_silence_duration_ms=args.min_silence_ms,
            speech_pad_ms=0,
        )
        spans = split_regions(
            timestamps,
            len(waveform),
            args.sample_rate,
            args.target_duration,
            args.min_duration,
            args.max_duration,
            args.max_internal_silence,
            args.speech_pad_ms,
        )
        if not spans:
            print("  skipped: VAD found no speech")
            continue

        chunk_paths = []
        for index, span in enumerate(spans):
            path = chunks_dir / f"{pair.source_id}_chunk_{index:04d}.wav"
            sf.write(
                path,
                waveform[span.start : span.end],
                args.sample_rate,
                subtype="PCM_16",
            )
            chunk_paths.append(path)

        reference_words = normalize(pair.text.read_text(encoding="utf-8-sig")).split()
        durations = [span.seconds(args.sample_rate) for span in spans]
        predictions = (
            transcribe(qwen, chunk_paths, args.batch_size)
            if qwen
            else [""] * len(spans)
        )
        targets = (
            align(reference_words, [item.split() for item in predictions], durations)
            if qwen
            else proportional(reference_words, durations)
        )

        for index, (span, path, target_words, prediction) in enumerate(
            zip(spans, chunk_paths, targets, predictions)
        ):
            target = " ".join(target_words)
            score = (
                edit_distance(target_words, prediction.split())
                / max(len(target_words), 1)
                if qwen
                else None
            )
            rows.append(
                {
                    "audio": str(path.resolve()),
                    "sentence": target,
                    "speaker_id": pair.speaker_id,
                    "source_id": pair.source_id,
                    "chunk_id": f"{pair.source_id}_{index:04d}",
                    "source_audio": str(pair.audio.resolve()),
                    "source_text": str(pair.text.resolve()),
                    "start": round(span.start / args.sample_rate, 3),
                    "end": round(span.end / args.sample_rate, 3),
                    "duration": round(span.seconds(args.sample_rate), 3),
                    "asr_prediction": prediction,
                    "alignment_wer": round(score, 4) if score is not None else None,
                    "alignment_method": "qwen_dynamic_partition"
                    if qwen
                    else "duration_proportional",
                    "needs_review": not target
                    or score is None
                    or score > args.review_wer,
                }
            )

    for row in rows:
        text_path = texts_dir / f"{row['chunk_id']}.txt"
        text_path.write_text(row["sentence"] + "\n", encoding="utf-8")

    with (output_dir / "all_chunks.tsv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = [
            "chunk_id",
            "audio",
            "text_file",
            "sentence",
            "asr_prediction",
            "alignment_wer",
            "needs_review",
            "speaker_id",
            "source_id",
            "start",
            "end",
            "duration",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            inspection_row = {field: row.get(field, "") for field in fields}
            inspection_row["text_file"] = str(
                (texts_dir / f"{row['chunk_id']}.txt").resolve()
            )
            writer.writerow(inspection_row)

    write_jsonl(manifest_path, rows)
    write_jsonl(
        output_dir / "qwen_all.jsonl",
        [
            {
                "audio": row["audio"],
                "text": f"language Persian<asr_text>{row['sentence']}",
            }
            for row in rows
        ],
    )
    with (output_dir / "review.tsv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = [
            "chunk_id",
            "audio",
            "sentence",
            "asr_prediction",
            "alignment_wer",
            "speaker_id",
            "source_id",
            "approved",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            if row["needs_review"]:
                review_row = {field: row.get(field, "") for field in fields}
                review_row["approved"] = ""
                writer.writerow(review_row)

    print(f"Saved {len(rows)} chunks and {manifest_path}")
    print(f"Manual review rows: {sum(row['needs_review'] for row in rows)}")
    if qwen is None:
        print(
            "WARNING: no ASR checkpoint; manually review every proportional alignment."
        )


if __name__ == "__main__":
    main()

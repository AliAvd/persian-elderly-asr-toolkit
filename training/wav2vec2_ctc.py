#!/usr/bin/env python3
"""Fine-tune Wav2Vec2-base or XLSR with CTC on an ASR JSONL dataset."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Audio, load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

PUNCTUATION = re.compile(r"[,?؛:!؟._–«»#&()…=\"'،ـ]+")
DIACRITICS = re.compile(r"[\u064b-\u0652]")
CHAR_MAP = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه"})


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/wav2vec2-base")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--epochs", type=float, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--save-steps", type=int, default=500)
    return parser.parse_args()


def normalize(text: str) -> str:
    text = text.split("<asr_text>", 1)[-1].translate(CHAR_MAP)
    text = DIACRITICS.sub("", PUNCTUATION.sub(" ", text)).replace("\u200c", " ")
    return " ".join(text.lower().split())


def build_processor(
    dataset: Any, output_dir: Path, sample_rate: int
) -> Wav2Vec2Processor:
    characters = sorted(set(" ".join(dataset["sentence"])) | {" "})
    vocabulary = {character: index for index, character in enumerate(characters)}
    vocabulary["|"] = vocabulary.pop(" ")
    vocabulary["[UNK]"] = len(vocabulary)
    vocabulary["[PAD]"] = len(vocabulary)
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "vocab.json"
    vocab_path.write_text(json.dumps(vocabulary, ensure_ascii=False), encoding="utf-8")

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=sample_rate,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    return Wav2Vec2Processor(feature_extractor=extractor, tokenizer=tokenizer)


@dataclass
class CTCDataCollator:
    processor: Wav2Vec2Processor

    def __call__(self, features: list[dict]) -> dict:
        inputs = [{"input_values": item["input_values"]} for item in features]
        labels = [{"input_ids": item["labels"]} for item in features]
        batch = self.processor.pad(inputs, padding=True, return_tensors="pt")
        label_batch = self.processor.pad(
            labels=labels, padding=True, return_tensors="pt"
        )
        batch["labels"] = label_batch["input_ids"].masked_fill(
            label_batch.attention_mask.ne(1), -100
        )
        return batch


def main() -> None:
    args = arguments()
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(args.train_jsonl),
            "validation": str(args.validation_jsonl),
        },
    )

    def prepare_text(row: dict) -> dict:
        return {"sentence": normalize(row["text"])}

    dataset = dataset.map(prepare_text)
    dataset = dataset.filter(lambda row: bool(row["sentence"]))
    dataset = dataset.cast_column("audio", Audio(sampling_rate=args.sample_rate))
    processor = build_processor(dataset["train"], args.output_dir, args.sample_rate)

    def prepare_audio(row: dict) -> dict:
        audio = row["audio"]
        row["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        row["labels"] = processor(text=row["sentence"]).input_ids
        return row

    dataset = dataset.map(
        prepare_audio,
        remove_columns=dataset["train"].column_names,
        num_proc=1,
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
    )
    model.freeze_feature_encoder()

    training = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.save_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        group_by_length=True,
        remove_unused_columns=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=CTCDataCollator(processor),
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

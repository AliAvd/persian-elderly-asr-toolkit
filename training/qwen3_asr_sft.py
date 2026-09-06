# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import random
import re
import shutil
from dataclasses import dataclass
from typing import Any

import audiomentations as auds
import librosa
import numpy as np
import torch
from audiomentations.core.transforms_interface import BaseWaveformTransform
from datasets import load_dataset
from qwen_asr import Qwen3ASRModel
from transformers import GenerationConfig, Trainer, TrainerCallback, TrainingArguments

AUG_PROB: float = 0.5


class InsertPause(BaseWaveformTransform):
    """Extend likely word boundaries by inserting silence into mono audio."""

    supports_multichannel = False

    def __init__(
        self,
        min_pause_duration: float = 0.15,
        max_pause_duration: float = 0.80,
        min_pauses: int = 1,
        max_pauses: int = 3,
        energy_quantile: float = 0.35,
        min_spacing: float = 0.30,
        p: float = 0.5,
    ):
        super().__init__(p)
        if not 0.0 <= min_pause_duration <= max_pause_duration:
            raise ValueError("pause durations must satisfy 0 <= min <= max")
        if not 1 <= min_pauses <= max_pauses:
            raise ValueError("pause counts must satisfy 1 <= min <= max")
        if not 0.0 < energy_quantile <= 1.0:
            raise ValueError("energy_quantile must be in (0, 1]")
        if min_spacing < 0.0:
            raise ValueError("min_spacing must be non-negative")

        self.min_pause_duration = min_pause_duration
        self.max_pause_duration = max_pause_duration
        self.min_pauses = min_pauses
        self.max_pauses = max_pauses
        self.energy_quantile = energy_quantile
        self.min_spacing = min_spacing

    def randomize_parameters(self, samples: np.ndarray, sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if not self.parameters["should_apply"]:
            return

        num_samples = samples.shape[-1]
        # Very short utterances do not have enough context for a safe pause.
        if num_samples < int(0.5 * sample_rate):
            self.parameters["should_apply"] = False
            return

        frame_length = max(1, round(0.020 * sample_rate))
        hop_length = max(1, round(0.010 * sample_rate))
        margin = max(frame_length, round(0.05 * num_samples))

        positions = np.arange(margin, num_samples - margin, hop_length)
        if positions.size == 0:
            self.parameters["should_apply"] = False
            return

        energies = np.asarray(
            [
                np.mean(np.square(samples[pos : pos + frame_length], dtype=np.float64))
                for pos in positions
            ]
        )
        requested_pauses = random.randint(self.min_pauses, self.max_pauses)
        candidate_indices = list(range(len(positions)))
        random.shuffle(candidate_indices)
        candidate_indices.sort(key=lambda index: energies[index])
        candidate_limit = max(
            requested_pauses,
            int(np.ceil(self.energy_quantile * len(candidate_indices))),
        )
        candidates = [
            int(positions[index]) for index in candidate_indices[:candidate_limit]
        ]

        min_spacing_samples = round(self.min_spacing * sample_rate)
        selected_positions = []
        for position in candidates:
            if all(
                abs(position - other) >= min_spacing_samples
                for other in selected_positions
            ):
                selected_positions.append(int(position))
                if len(selected_positions) == requested_pauses:
                    break

        if not selected_positions:
            self.parameters["should_apply"] = False
            return

        selected_positions.sort()
        self.parameters["pause_positions"] = selected_positions
        self.parameters["pause_lengths"] = [
            round(
                random.uniform(self.min_pause_duration, self.max_pause_duration)
                * sample_rate
            )
            for _ in selected_positions
        ]

    def apply(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        chunks = []
        cursor = 0
        for position, pause_length in zip(
            self.parameters["pause_positions"],
            self.parameters["pause_lengths"],
        ):
            chunks.append(samples[cursor:position])
            chunks.append(np.zeros(pause_length, dtype=samples.dtype))
            cursor = position
        chunks.append(samples[cursor:])
        return np.concatenate(chunks)


## Augmentor
augmentor = auds.Compose(
    [
        auds.AddBackgroundNoise("./backgrounds", p=0.75),
        auds.OneOf(
            [
                auds.Gain(p=1.0),
                auds.GainTransition(
                    p=1.0, duration_unit="fraction", min_duration=0.2, max_duration=0.9
                ),
            ],
            p=0.8,
        ),
        auds.TimeStretch(
            min_rate=0.8, max_rate=2.0, leave_length_unchanged=False, p=0.9
        ),
        InsertPause(
            min_pause_duration=0.15,
            max_pause_duration=0.80,
            min_pauses=1,
            max_pauses=3,
            p=0.7,
        ),
        auds.ApplyImpulseResponse(
            ir_path="./Impulses", leave_length_unchanged=False, p=0.5
        ),
        auds.OneOf(
            [
                auds.BitCrush(p=1.0),
                auds.AddGaussianSNR(p=1.0),
                auds.TanhDistortion(p=1.0),
                auds.Mp3Compression(quality=5, p=1.0),
            ],
            p=0.8,
        ),
        auds.OneOf(
            [
                auds.HighPassFilter(p=1.0),
                auds.BandPassFilter(p=1.0),
            ],
            p=0.8,
        ),
    ],
    p=AUG_PROB,
)


def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no `.thinker.forward`. "
            "Your qwen3_asr model may be incompatible."
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> str | None:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step = None
    best_path = None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step = step
            best_path = path
    return best_path


def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn_prefix_only(processor):
    def _preprocess(ex: dict[str, Any]) -> dict[str, Any]:
        prompt = ex.get("prompt", "")
        dummy_audio = None
        prefix_msgs = build_prefix_messages(prompt, dummy_audio)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]
        return {
            "prompt": prompt,
            "audio": ex["audio"],
            "target": ex["text"],
            "prefix_text": prefix_text,
        }

    return _preprocess


@dataclass
class DataCollatorForQwen3ASRFinetuning:
    processor: Any
    sampling_rate: int = 16000
    augmentor: auds.Compose | None = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]
        augment_flags = [f.get("apply_augmentation", True) for f in features]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]
        audios = [load_audio(p, sr=self.sampling_rate) for p in audio_paths]

        # Apply augmentation if provided
        if self.augmentor is not None:
            augmented_audios = []
            for audio, should_augment in zip(audios, augment_flags):
                # audiomentations expects numpy arrays
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()

                if should_augment:
                    audio = self.augmentor(
                        samples=audio, sample_rate=self.sampling_rate
                    )

                augmented_audios.append(audio)

            audios = augmented_audios

        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        return full_inputs


class CastFloatInputsTrainer(Trainer):
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs


def copy_required_hf_files_for_qwen_asr(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    required = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.json",
        "merges.txt",
        "vocab.json",
    ]
    for fn in required:
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))


class MakeEveryCheckpointInferableCallback(TrainerCallback):
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)

        copy_required_hf_files_for_qwen_asr(self.base_model_path, ckpt_dir)
        return control


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR Finetuning")

    # Paths
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--train_file", type=str, default="train.jsonl")
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./qwen3-asr-finetuning-out")

    # Audio
    p.add_argument("--sr", type=int, default=16000)

    # Train hyper-params
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_acc", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--warmup_ratio", type=float, default=0.02)

    # DataLoader
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)

    # Save
    p.add_argument("--save_strategy", type=str, default="steps")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=5)

    # Resume
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)

    return p.parse_args()


def main():
    args_cli = parse_args()

    if not args_cli.train_file:
        raise ValueError(
            "TRAIN_FILE is required (json/jsonl). Needs fields: audio, text, optional prompt"
        )

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args_cli.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    raw_ds = load_dataset(
        "json",
        data_files={
            "train": args_cli.train_file,
            **({"validation": args_cli.eval_file} if args_cli.eval_file else {}),
        },
    )
    ds = raw_ds.map(make_preprocess_fn_prefix_only(processor), num_proc=1)

    for split in ds:
        ds[split] = ds[split].add_column(
            "apply_augmentation", [split == "train"] * len(ds[split])
        )

    keep = {"prompt", "audio", "target", "prefix_text", "apply_augmentation"}
    for split in ds:
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    collator = DataCollatorForQwen3ASRFinetuning(
        processor=processor,
        sampling_rate=args_cli.sr,
        augmentor=augmentor,
    )

    training_args = TrainingArguments(
        output_dir=args_cli.output_dir,
        per_device_train_batch_size=args_cli.batch_size,
        gradient_accumulation_steps=args_cli.grad_acc,
        per_device_eval_batch_size=args_cli.batch_size,
        eval_accumulation_steps=args_cli.grad_acc,
        learning_rate=args_cli.lr,
        num_train_epochs=args_cli.epochs,
        logging_steps=args_cli.log_steps,
        lr_scheduler_type=args_cli.lr_scheduler_type,
        warmup_ratio=args_cli.warmup_ratio,
        dataloader_num_workers=args_cli.num_workers,
        dataloader_pin_memory=(args_cli.pin_memory == 1),
        dataloader_persistent_workers=(args_cli.persistent_workers == 1),
        dataloader_prefetch_factor=args_cli.prefetch_factor
        if args_cli.num_workers > 0
        else None,
        save_strategy=args_cli.save_strategy,
        save_steps=args_cli.save_steps,
        save_total_limit=args_cli.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps",
        eval_steps=args_cli.save_steps,
        do_eval=bool(args_cli.eval_file),
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
        prediction_loss_only=False,
    )
    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[
            MakeEveryCheckpointInferableCallback(base_model_path=args_cli.model_path)
        ],
    )

    resume_from = (args_cli.resume_from or "").strip()
    if not resume_from and args_cli.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    if resume_from:
        if trainer.args.process_index == 0:
            print(f"[resume] resume_from_checkpoint = {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()


if __name__ == "__main__":
    main()

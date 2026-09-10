# coding=utf-8
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
"""Reusable historical Qwen components used by the experiment adapter; not a CLI."""
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import librosa
import torch
import audiomentations as auds
from audiomentations.core.transforms_interface import BaseWaveformTransform
from transformers import Trainer

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

        frame_length = max(1, int(round(0.020 * sample_rate)))
        hop_length = max(1, int(round(0.010 * sample_rate)))
        margin = max(frame_length, int(round(0.05 * num_samples)))

        positions = np.arange(margin, num_samples - margin, hop_length)
        if positions.size == 0:
            self.parameters["should_apply"] = False
            return

        energies = np.asarray([
            np.mean(np.square(samples[pos:pos + frame_length], dtype=np.float64))
            for pos in positions
        ])
        requested_pauses = random.randint(self.min_pauses, self.max_pauses)
        candidate_indices = list(range(len(positions)))
        random.shuffle(candidate_indices)
        candidate_indices.sort(key=lambda index: energies[index])
        candidate_limit = max(
            requested_pauses,
            int(np.ceil(self.energy_quantile * len(candidate_indices))),
        )
        candidates = [
            int(positions[index])
            for index in candidate_indices[:candidate_limit]
        ]

        min_spacing_samples = int(round(self.min_spacing * sample_rate))
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
            int(round(random.uniform(
                self.min_pause_duration, self.max_pause_duration
            ) * sample_rate))
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

def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav

def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]

def make_preprocess_fn_prefix_only(processor):
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
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
    augmentor: Optional[auds.Compose] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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
                        samples=audio,
                        sample_rate=self.sampling_rate
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

"""Project augmentation with explicit assets and named ablation groups."""
from pathlib import Path
import ast

def legacy_components():
    # Compile only audited reusable definitions; never execute legacy module globals
    # (the legacy global augmentor requires relative asset paths at import time).
    import random, re, shutil, os
    import numpy as np
    import torch, librosa
    import audiomentations as auds
    from audiomentations.core.transforms_interface import BaseWaveformTransform
    from dataclasses import dataclass
    from typing import Any, Dict, List, Optional
    from transformers import Trainer
    source = Path(__file__).resolve().parents[2] / 'training' / 'qwen3_asr_legacy.py'
    names = {'InsertPause', 'patch_outer_forward', 'load_audio', 'build_prefix_messages',
             'make_preprocess_fn_prefix_only', 'DataCollatorForQwen3ASRFinetuning', 'CastFloatInputsTrainer'}
    tree = ast.parse(source.read_text(encoding='utf-8'))
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    namespace = dict(locals())
    exec(compile(tree, str(source), 'exec'), namespace)
    return namespace

def build_augmentor(config, components=None):
    if not config.get('enabled', False): return None
    import audiomentations as a
    parts = components or legacy_components()
    exclude = set(config.get('exclude', []))
    aliases = {'TimeStretch':'speed', 'InsertPause':'pause', 'AddBackgroundNoise':'background',
               'ApplyImpulseResponse':'rir', 'Gain':'gain', 'quality':'distortion', 'filters':'frequency'}
    exclude = {aliases.get(x, x) for x in exclude}
    valid = {'speed', 'pause', 'background', 'rir', 'gain', 'distortion', 'frequency'}
    if exclude - valid: raise ValueError(f'Unknown augmentation exclusions: {exclude - valid}')
    transforms = []
    if 'background' not in exclude:
        path = config.get('background_dir')
        if not path or not Path(path).is_dir(): raise FileNotFoundError('Set augmentation.background_dir to training-only noise assets')
        transforms.append(a.AddBackgroundNoise(path, p=.75))
    if 'gain' not in exclude: transforms.append(a.OneOf([a.Gain(p=1), a.GainTransition(p=1, duration_unit='fraction', min_duration=.2, max_duration=.9)], p=.8))
    if 'speed' not in exclude: transforms.append(a.TimeStretch(min_rate=.8, max_rate=2., leave_length_unchanged=False, p=.9))
    if 'pause' not in exclude: transforms.append(parts['InsertPause'](p=.7))
    if 'rir' not in exclude:
        path = config.get('rir_dir')
        if not path or not Path(path).is_dir(): raise FileNotFoundError('Set augmentation.rir_dir to training-only impulse responses')
        transforms.append(a.ApplyImpulseResponse(ir_path=path, leave_length_unchanged=False, p=.5))
    if 'distortion' not in exclude: transforms.append(a.OneOf([a.BitCrush(p=1), a.AddGaussianSNR(p=1), a.TanhDistortion(p=1), a.Mp3Compression(quality=5,p=1)],p=.8))
    if 'frequency' not in exclude: transforms.append(a.OneOf([a.HighPassFilter(p=1),a.BandPassFilter(p=1)],p=.8))
    return a.Compose(transforms, p=config.get('probability', .5))

#!/usr/bin/env python3
"""Validate and run a catalog experiment; paths are relative to experiments/."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def resolve_config(config_path, checkpoint=None, seed=None):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    def absolute(value):
        p = Path(value).expanduser()
        return str(p if p.is_absolute() else ROOT / p)
    data = config['data']
    for key in ('train_manifest', 'validation_manifest'):
        if data.get(key):
            data[key] = absolute(data[key])
    data['tests'] = {name: absolute(path) for name, path in data.get('tests', {}).items()}
    for key in ('background_dir', 'rir_dir'):
        if config.get('augmentation', {}).get(key):
            config['augmentation'][key] = absolute(config['augmentation'][key])
    config['training']['output_dir'] = absolute(config['training']['output_dir'])
    model = config['model']
    env_checkpoint = os.environ.get(model.get('checkpoint_env', ''))
    if checkpoint or env_checkpoint:
        model['init_checkpoint'] = absolute(checkpoint or env_checkpoint)
    elif model.get('init_checkpoint'):
        model['init_checkpoint'] = absolute(model['init_checkpoint'])
    model['vocabulary_manifests']=[absolute(p) for p in model.get('vocabulary_manifests',[])]
    if seed is not None:
        config['training']['seed'] = seed
        config['training']['output_dir'] += f'_seed{seed}'
    return config

def validate(config):
    errors = []
    data = config['data']
    paths = dict(data.get('tests', {}))
    if config['mode'] == 'train':
        paths.update({key: data.get(key) for key in ('train_manifest', 'validation_manifest')})
    for name, path in paths.items():
        if not path or not Path(path).is_file():
            errors.append(f'Missing {name}: {path}')
        elif Path(path).stat().st_size == 0:
            errors.append(f'Empty {name}: {path}')
    model = config['model']
    if model.get('requires_checkpoint') and not model.get('init_checkpoint'):
        errors.append('This baseline needs a Persian ASR checkpoint: pass --checkpoint PATH or set ' + model['checkpoint_env'])
    if model.get('init_checkpoint') and not Path(model['init_checkpoint']).is_dir():
        errors.append('Missing initialization checkpoint: ' + model['init_checkpoint'])
    if config.get('blocked_reason'):
        errors.append(config['blocked_reason'])
    return errors

def main(default_config=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=default_config)
    parser.add_argument('--checkpoint', help='Local model checkpoint, overriding initialization')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output-dir', help='Separate output directory for this run')
    parser.add_argument('--max-steps', type=int, help='Fix training update budget for controlled comparisons')
    parser.add_argument('--resume', help='Trainer checkpoint directory to resume')
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs without training')
    parser.add_argument('--print-config', action='store_true')
    args = parser.parse_args()
    if not args.config:
        parser.error('--config is required')
    config = resolve_config(args.config, args.checkpoint, args.seed)
    def local_path(value):
        p=Path(value).expanduser()
        return str(p if p.is_absolute() else ROOT/p)
    if args.output_dir:config['training']['output_dir']=local_path(args.output_dir)
    if args.max_steps is not None:
        if args.max_steps<=0:parser.error('--max-steps must be positive')
        config['training']['max_steps']=args.max_steps
    if args.resume:
        resume=local_path(args.resume)
        if not Path(resume,'trainer_state.json').is_file():parser.error('--resume must contain trainer_state.json')
        config['training']['resume_from_checkpoint']=resume
    if args.print_config:
        print(json.dumps(config, ensure_ascii=False, indent=2))
    errors = validate(config)
    if errors:
        parser.exit(2, '\n'.join(errors) + '\n')
    from shared.audit import audit_config
    report=audit_config(config)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    from shared.training import run
    result=run(config, dry_run=args.dry_run)
    if args.dry_run: print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Generate reproducible, explicitly named experiment configurations."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
MODELS = {
    'qwen': 'Qwen/Qwen3-ASR-1.7B',
    'wav2vec2_base': 'facebook/wav2vec2-base',
    'wav2vec2_xlsr': 'facebook/wav2vec2-large-xlsr-53',
}
rows = []

def add(number, name, composition, families=None, aug=False, exclude=None, mode='train', test_names=None, adaptation=False, blocked=None):
    scenario = f'{number:02d}_{name}'
    for family in families or MODELS:
        relative = f'catalog/{scenario}/{family}'
        directory = ROOT / relative
        directory.mkdir(parents=True, exist_ok=True)
        model = {'family': family, 'pretrained': MODELS[family]}
        if number in (1, 2) and family != 'qwen':
            model.update(requires_checkpoint=True, checkpoint_env=f'ASR_{family.upper()}_CHECKPOINT')
            baseline='wav2vec2_base/output_ver_1/checkpoint-547500' if family=='wav2vec2_base' else 'wav2vec2_xlsr/output_dir_2/checkpoint-21900'
            model['init_checkpoint']='../'+baseline
        if adaptation and family!='qwen': model['expand_vocabulary']=True
        if adaptation:
            model['init_checkpoint'] = f'outputs/03_public/{family}/final'
        tests = test_names or ['public', 'elderly', 'ganjoor', 'filimo', 'mozilla']
        if not test_names and ('senior' in composition or composition=='seniortalk'): tests=tests+['seniortalk']
        manifest_dir=composition+('/qwen' if family=='qwen' else '')
        config = {
            'experiment': f'{scenario}/{family}', 'mode': mode, 'model': model,
            'data': {'train_manifest': f'data/manifests/{manifest_dir}/train.jsonl' if mode == 'train' else None,
                     'validation_manifest': f'data/manifests/{manifest_dir}/validation.jsonl' if mode == 'train' else None,
                     'tests': {s: f'data/manifests/{s}/test.jsonl' for s in tests}},
            'augmentation': {'enabled': aug, 'exclude': exclude or [], 'background_dir': 'data/augmentation/background_train', 'rir_dir': 'data/augmentation/rir_train'},
            'training': {'output_dir': f'outputs/{scenario}/{family}', 'epochs': 10,
                         'batch_size': 1 if family == 'qwen' else 4, 'gradient_accumulation': 8,
                         'learning_rate': 2e-5 if family == 'qwen' else 3e-4, 'seed': 42},
        }
        if blocked:
            config['blocked_reason'] = blocked
        (directory / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
        (directory / 'run.py').write_text("#!/usr/bin/env python3\nfrom pathlib import Path\nimport sys\nROOT = Path(__file__).resolve().parents[3]\nsys.path.insert(0, str(ROOT))\nfrom run_experiment import main\nif __name__ == '__main__':\n    main(str(Path(__file__).with_name('config.json')))\n")
        rows.append({'experiment': config['experiment'], 'config': relative + '/config.json', 'mode': mode, 'composition': composition, 'augmentation': aug})

def main():
    rows.clear()
    add(1, 'baseline_public', 'public', mode='evaluate', test_names=['public', 'ganjoor', 'filimo', 'mozilla'])
    add(2, 'baseline_elderly', 'elderly', mode='evaluate', test_names=['elderly'])
    add(3, 'public', 'public')
    add(4, 'elderly_only_adaptation', 'elderly', adaptation=True)
    add(5, 'public_elderly', 'public_elderly')
    add(6, 'public_aug', 'public', aug=True)
    add(7, 'public_elderly_aug', 'public_elderly', aug=True)
    add(8, 'seniortalk_only_adaptation', 'seniortalk', adaptation=True)
    number = 9
    for suffix in ['senior', 'elderly_senior']:
        for aug in (False, True):
            add(number, 'public_' + suffix + ('_aug' if aug else ''), 'public_' + suffix, aug=aug)
            number += 1
    for component in ['speed', 'pause', 'background', 'rir', 'gain', 'distortion', 'frequency']:
        add(number, 'ablation_without_' + component.lower(), 'public_elderly', ['qwen'], aug=True, exclude=[component])
        number += 1
    for composition in ['public_unfiltered', 'public_filtered', 'public_unfiltered_matched']:
        add(number, 'filter_' + composition, composition, ['qwen'])
        number += 1
    for fraction in ['025', '050', '075']:
        add(number, 'elderly_volume_' + fraction, 'public_elderly_' + fraction, ['qwen'], aug=True)
        number += 1
    add(number, 'public_without_ganjoor', 'public_without_ganjoor', ['qwen'])
    number += 1
    add(number,'qwen_original_elderly_only','elderly',['qwen'])
    for number, scenario, vad in [(28,'long_recording_direct',False),(29,'long_recording_vad',True)]:
        add(number,scenario,'elderly',mode='evaluate',test_names=['elderly_long'])
        for family in MODELS:
            path=ROOT/f'catalog/{number:02d}_{scenario}/{family}/config.json'
            c=json.loads(path.read_text());c['model']['init_checkpoint']=f'outputs/07_public_elderly_aug/{family}/final';c['evaluation']={'vad':vad,**({} if vad else {'window_seconds':20})};path.write_text(json.dumps(c,indent=2))
    add(30,'noise_evaluation','public',mode='evaluate',test_names=[source+'_noisy_'+condition for source in ['public','elderly'] for condition in ['snr20','snr10','snr5','rir','channel','combined']]+['public','elderly'])
    for family in MODELS:
        path=ROOT/f'catalog/30_noise_evaluation/{family}/config.json';c=json.loads(path.read_text());c['model']['init_checkpoint']=f'outputs/07_public_elderly_aug/{family}/final';path.write_text(json.dumps(c,indent=2))
    (ROOT / 'catalog/index.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(f'Generated {len(rows)} model configurations.')

if __name__ == '__main__':
    main()

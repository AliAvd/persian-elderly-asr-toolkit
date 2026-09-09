"""Shared corpus metrics; empty hypotheses remain errors, failures are explicit."""
import json
import time
import re
import unicodedata
from pathlib import Path

LANGUAGES = {'fa': 'Persian', 'zh': 'Chinese', 'zh-cn': 'Chinese', 'ja': 'Japanese', 'en': 'English'}

def plain_text(text):
    return re.sub(r'^language\s+[^<]+<asr_text>', '', str(text)).strip()

def normalize(text):
    text = unicodedata.normalize('NFKC', plain_text(text)).translate(str.maketrans('يك', 'یک')).replace('\u200c', ' ')
    text = ''.join(str(unicodedata.digit(c)) if c.isdigit() and unicodedata.category(c) == 'Nd' else c for c in text)
    return ' '.join(''.join(' ' if unicodedata.category(c).startswith('P') else c for c in text).split())

def distance(a, b):
    try:
        from rapidfuzz.distance.Levenshtein import distance as fast_distance
        return fast_distance(a,b)
    except ImportError: pass
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (x != y)))
        previous = current
    return previous[-1]

def metrics(rows, normalized=False):
    errors_w = errors_c = words = chars = 0
    for row in rows:
        transform = normalize if normalized else plain_text
        ref, hyp = transform(row['reference']), transform(row['prediction'])
        # WER uses whitespace segmentation for every language; Chinese CER is primary.
        errors_w += distance(ref.split(), hyp.split()); words += len(ref.split())
        errors_c += distance(list(ref), list(hyp)); chars += len(ref)
    return {'wer': errors_w / words if words else None, 'cer': errors_c / chars if chars else None,
            'word_errors': errors_w, 'reference_words': words, 'character_errors': errors_c, 'reference_characters': chars}

def read_manifest(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]

def evaluate_manifest(path, predict, output_dir, name, failure_policy='abort'):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []; started=time.perf_counter(); initial_cache_hits=getattr(predict,'cache_hits',0)
    with (out / f'{name}.predictions.jsonl').open('w', encoding='utf-8') as handle:
        for index, item in enumerate(read_manifest(path)):
            row = dict(item, sample_index=index, reference=plain_text(item['text']))
            try:
                row['prediction'] = predict(item)
                row['error'] = None
            except Exception as exc:
                row.update(prediction='', error=f'{type(exc).__name__}: {exc}')
                handle.write(json.dumps(row, ensure_ascii=False) + '\n'); handle.flush()
                if failure_policy != 'count_as_empty':
                    raise RuntimeError(f'Evaluation failed at {path}:{index+1}; partial predictions saved') from exc
                rows.append(row)
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + '\n'); handle.flush(); rows.append(row)
            if (index+1)%100==0:print(f'{name}: evaluated {index+1} samples',flush=True)
    elapsed=time.perf_counter()-started
    result = {'inference_seconds':elapsed,'real_time_factor':elapsed/sum(r.get('duration',0) for r in rows) if sum(r.get('duration',0) for r in rows) else None, 'samples': len(rows), 'failures': sum(r['error'] is not None for r in rows),
              'raw': metrics(rows), 'normalized': metrics(rows, True),
              'wer_segmentation': 'whitespace; use CER as primary for Chinese/Japanese',
              'units': 'fractions (multiply by 100 for percent)', 'failure_policy': failure_policy}
    result['reused_predictions']=getattr(predict,'cache_hits',0)-initial_cache_hits
    if result['reused_predictions']:
        result['real_time_factor']=None
        result['timing_note']='Contains reused predictions; not an independent speed measurement'
    target=out/f'{name}.metrics.json';temporary=target.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(target)
    return result

"""Preflight identities, file availability and split-leakage checks."""
from pathlib import Path
import json

def identities(path):
    ids=set();pcm=set();groups=set();count=0
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():continue
            r=json.loads(line);count+=1
            if not Path(r['audio']).is_file():raise FileNotFoundError(r['audio'])
            if not str(r.get('text','')).strip():raise ValueError(f'Empty reference {path}:{count}')
            ids.add(str(Path(r['audio']).resolve()))
            if r.get('pcm_sha256'):pcm.add(r['pcm_sha256'])
            if r.get('source_id'):groups.add(r['source_id'])
    if not count:raise ValueError(f'Empty manifest: {path}')
    return {'count':count,'paths':ids,'pcm':pcm,'groups':groups}

def audit_config(config):
    data=config['data'];cache={}
    def get(p):
        if p not in cache:cache[p]=identities(p)
        return cache[p]
    tests={n:get(p) for n,p in data.get('tests',{}).items()}
    if config['mode']=='train':
        tr=get(data['train_manifest']);va=get(data['validation_manifest'])
        checks=[('train','validation',tr,va)]+[('train',n,tr,x) for n,x in tests.items()]+[('validation',n,va,x) for n,x in tests.items()]
        for left,right,a,b in checks:
            for key in ['paths','pcm','groups']:
                overlap=a[key]&b[key]
                if overlap:raise ValueError(f'Data leakage {left}/{right}: {len(overlap)} overlapping {key}; prepare splits again')
    return {'status':'passed','manifests':{p:x['count'] for p,x in cache.items()},'checks':['existing audio','nonempty references']+(['path, canonical PCM and source recording separation'] if config['mode']=='train' else []),'speaker_independent':False,'limitation':'shared speakers allowed; perceptually similar re-encodings beyond exact canonical PCM are not guaranteed absent'}

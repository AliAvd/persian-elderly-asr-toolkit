#!/usr/bin/env python3
"""Download pinned official corpora; never substitutes Common Voice versions."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
import requests

ASR = Path(os.environ.get('ASR_DATA_ROOT', str(Path(__file__).resolve().parents[1]))).expanduser().resolve()
SENIOR_REPO = "BAAI/SeniorTalk"
SENIOR_REVISION = "d1ddfa691c4b9f434a5e6abb5103e68e1c7280cb"
API = "https://mozilladatacollective.com/api"

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def fetch(url, target, expected_sha=None, expected_size=None):
    """Resume only verified ranges; atomically publish after size/checksum checks."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if (expected_size is None or target.stat().st_size == expected_size) and (expected_sha is None or sha256(target) == expected_sha):
            print(f"Verified existing {target.name}", flush=True)
            return target
        raise RuntimeError(f"Existing file does not match upstream metadata: {target}")
    partial = target.with_name(target.name + '.part')
    for attempt in range(6):
        offset = partial.stat().st_size if partial.exists() else 0
        if expected_size and offset == expected_size:
            break
        try:
            with requests.get(url, headers={'Range': f'bytes={offset}-'} if offset else {}, stream=True, timeout=(30,120)) as response:
                if response.status_code not in (200,206):
                    raise RuntimeError(f"Download HTTP {response.status_code}: {target.name}")
                if response.status_code == 206 and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                    raise RuntimeError('Unexpected server resume range')
                mode = 'ab' if response.status_code == 206 else 'wb'
                with partial.open(mode) as out:
                    for block in response.iter_content(4*1024*1024):
                        if block: out.write(block)
            break
        except (requests.RequestException, RuntimeError) as exc:
            if attempt == 5: raise RuntimeError(f"Download failed for {target.name}: {type(exc).__name__}") from None
            time.sleep(min(2**attempt,16))
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise RuntimeError(f'Incomplete download: {target.name}')
    if expected_sha and sha256(partial) != expected_sha:
        raise RuntimeError(f'Checksum mismatch: {target.name}; remove .part before retrying')
    partial.replace(target)
    print(f"Downloaded and verified {target.name}", flush=True)
    return target

def common_voice(dataset_id=None, version=26, prompt_key=False):
    expected_name=f"Common Voice Scripted Speech {version}.0 - Persian"
    if not dataset_id and version==26: dataset_id="cmqinhw5100v8nr07gyg5gi4v"
    if dataset_id:
        response=requests.get(f'{API}/datasets/{dataset_id}', timeout=60)
        response.raise_for_status(); metadata=response.json()
    else:
        response=requests.get(f'{API}/datasets',params={'q':'Persian','limit':100},timeout=60)
        response.raise_for_status()
        candidates=[d for d in response.json()['items'] if d['name'].strip()==expected_name]
        if len(candidates)!=1: raise RuntimeError(f'Cannot uniquely locate official CV{version} Persian. Supply --mozilla-dataset-id.')
        metadata=candidates[0]
    if metadata['name'].strip() != expected_name:
        raise RuntimeError(f'Dataset is not the requested Common Voice {version} Persian release')
    dest=ASR/f'datasets/raw/common_voice_{version}'
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'upstream_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    key=os.environ.get('MDC_API_KEY')
    if not key and prompt_key:
        import getpass
        key=getpass.getpass('Mozilla API key (not saved): ')
    if not key:
        raise RuntimeError(f"MDC_API_KEY is not configured. Accept access terms at https://mozilladatacollective.com/datasets/{metadata['id']} and set the key locally.")
    response=requests.post(f"{API}/datasets/{metadata['id']}/download",headers={'Authorization':f'Bearer {key}'},timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f'Mozilla download HTTP {response.status_code}; check account access/accepted terms. No alternate source substituted.')
    download=response.json()
    archive=fetch(download['downloadUrl'],dest/Path(download['filename']).name,download.get('checksum'),int(download['sizeBytes']))
    output=ASR/'commonvoice_fa'
    with tarfile.open(archive,'r:*') as tar:
        tar.extractall(output,filter='data')
    roots=[p.parent for p in output.rglob('train.tsv') if p.parent.name=='fa' and f'{version}.0' in str(p)]
    if len(roots)!=1:raise RuntimeError(f'Archive extracted but CV{version} fa train.tsv root is ambiguous')
    for name in ['train.tsv','dev.tsv','test.tsv']:
        if not (roots[0]/name).is_file():raise RuntimeError(f'Missing extracted {name}')
    status={'status':'downloaded_extracted','root':str(roots[0]),'archive_sha256':sha256(archive),'source':metadata['datasetUrl']}
    (dest/'download_status.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status,indent=2))

def seniortalk(include_dialogues=False, workers=3):
    dest=ASR/'datasets/raw/seniortalk'
    response=requests.get(f'https://huggingface.co/api/datasets/{SENIOR_REPO}/tree/{SENIOR_REVISION}',params={'recursive':'true','limit':1000},timeout=60)
    response.raise_for_status()
    entries=[x for x in response.json() if x['type']=='file' and (x['path']=='README.md' or x['path'].startswith('sentence_data/') or include_dialogues and x['path'].startswith('dialogue_data/'))]
    if not any(x['path'].endswith('.parquet') for x in entries):raise RuntimeError('No upstream parquet shards')
    dest.mkdir(parents=True,exist_ok=True)
    provenance={'repository':SENIOR_REPO,'revision':SENIOR_REVISION,'files':entries,'configuration':'sentence_data','license':'cc-by-nc-sa-4.0'}
    (dest/'upstream_metadata.json').write_text(json.dumps(provenance,indent=2))
    print(f"SeniorTalk: {len(entries)} files, {sum(x['size'] for x in entries)/1e9:.2f} GB",flush=True)
    def get(entry):
        return fetch(f"https://huggingface.co/datasets/{SENIOR_REPO}/resolve/{SENIOR_REVISION}/{entry['path']}?download=true",dest/entry['path'],entry.get('lfs',{}).get('oid'),entry['size'])
    with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(get,entries))
    (dest/'download_status.json').write_text(json.dumps({'status':'downloaded_verified','revision':SENIOR_REVISION,'files':len(entries),'next_step':'prepare_data.py seniortalk exports embedded audio and manifests'},indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',choices=['common_voice_25','common_voice_26','seniortalk','all'])
    p.add_argument('--mozilla-dataset-id')
    p.add_argument('--prompt-key',action='store_true')
    p.add_argument('--include-dialogues',action='store_true')
    p.add_argument('--workers',type=int,default=3)
    a=p.parse_args()
    if a.dataset in ('common_voice_25','common_voice_26','all'):common_voice(a.mozilla_dataset_id,25 if a.dataset=='common_voice_25' else 26,a.prompt_key)
    if a.dataset in ('seniortalk','all'):seniortalk(a.include_dialogues,a.workers)

if __name__=='__main__':main()

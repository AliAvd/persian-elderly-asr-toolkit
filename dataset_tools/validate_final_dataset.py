#!/usr/bin/env python3
"""Validate actual final audio/label pairs, provenance, splits, and both loader formats."""
import os
from pathlib import Path
import json,hashlib,sys
import soundfile as sf
import numpy as np
ROOT=Path(os.environ.get('ASR_DATA_ROOT', str(Path(__file__).resolve().parents[1]))).expanduser().resolve();HF=ROOT/'Final_Gathered_Dataset_HF';CH=ROOT/'Final_Gatheres_Dataset_Chunks'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from shared.evaluation import normalize
from shared.dataset_loading import load_training_data
from datasets import load_from_disk,Audio

def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def main():
 rows=read(CH/'manifest.jsonl');inventory=json.loads((CH/'source_inventory.json').read_text());texts={r['source_id']:Path(r['text']).read_text(encoding='utf-8-sig') for r in inventory};by={}
 for r in rows:
  by.setdefault(r['source_id'],[]).append(r)
  assert texts[r['source_id']][r['text_start_char']:r['text_end_char']]==r['sentence']
  pcm,sr=sf.read(r['audio'],dtype='int16');assert sr==16000 and pcm.ndim==1 and len(pcm)>0
  assert len(pcm)/sr==r['duration']
  assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==r['pcm_sha256']
  assert abs(r['end']-r['start']-r['duration'])<1e-6
 assert len(by)==len(inventory)
 for sid,values in by.items():
  assert normalize(' '.join(r['sentence'] for r in values))==normalize(texts[sid]),sid
  assert all(a['end']<=b['start']+1e-6 for a,b in zip(values,values[1:])),sid
 splits={s:read(HF/'manifests'/(s+'.jsonl')) for s in ['train','validation','test']}
 for key in ['source_id','source_audio_sha256','pcm_sha256','split_group']:
  values=[{r[key] for r in v} for v in splits.values()];assert all(not a&b for i,a in enumerate(values) for b in values[i+1:]),key
 for split,values in splits.items():
  assert values
  assert all(not r['needs_review'] and normalize(r['sentence']) and 1<=r['duration']<=16 for r in values)
  q=read(HF/'qwen_jsonl'/(split+'.jsonl'));assert len(q)==len(values)
  assert all(r['audio']==v['audio'] and r['text']=='language Persian<asr_text>'+v['sentence'] for r,v in zip(q,values))
 ds=load_from_disk(str(HF))
 for split in ds:
  assert len(ds[split])==len(splits[split])
  record=ds[split].cast_column('audio',Audio(decode=False))[0]['audio'];assert record['bytes']
  decoded=ds[split][0]['audio'];assert decoded['sampling_rate']==16000 and np.isfinite(decoded['array']).all()
 for folder in ['manifests','qwen_jsonl']:
  loaded=load_training_data({s+'_manifest':str(HF/folder/(s+'.jsonl')) for s in ['train','validation']})
  assert len(loaded['train'])==len(splits['train']) and len(loaded['validation'])==len(splits['validation'])
 result={'passed':True,'recordings':len(by),'chunks_checked':len(rows),'accepted':sum(map(len,splits.values())),'checks':['every audio decoded, mono16k PCM checksum and duration','every label exact source substring','all normalized reference words assigned once across accepted + review chunks','nonoverlapping chunk intervals within recordings','disjoint source/audio/PCM/transcript groups across splits','no empty accepted labels','DatasetDict load and embedded audio decoding','CTC and Qwen actual experiment training-data loaders'],'human_alignment_review':False}
 (HF/'validation_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()

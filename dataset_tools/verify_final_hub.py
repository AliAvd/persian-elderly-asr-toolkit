#!/usr/bin/env python3
"""Verify the explicit Hub revision, loader, audio, and all portable manifest paths."""
import hashlib,json
from pathlib import Path
import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi,HfFileSystem,hf_hub_download
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1];HF=ROOT/'Final_Gathered_Dataset_HF';REPO='AliAvd/persian-elderly-asr'
api=HfApi();info=api.repo_info(REPO,repo_type='dataset');assert info.private;revision=info.sha
files={r.rfilename for r in info.siblings};expected={'train':1980,'validation':294,'test':329}
summary=json.loads(Path(hf_hub_download(REPO,'processing_summary.json',repo_type='dataset',revision=revision)).read_text())
for split,n in expected.items():
 assert summary['splits'][split]['samples']==n
 for folder,prefix in [('qwen_jsonl',''),('portable/qwen_jsonl','portable/'),('portable/manifests','portable/')]:
  p=hf_hub_download(REPO,folder+'/'+split+'.jsonl',repo_type='dataset',revision=revision)
  rows=[json.loads(l) for l in Path(p).read_text().splitlines()];assert len(rows)==n
  assert all(prefix+r['audio'] in files for r in rows),(folder,split)
fs=HfFileSystem()
actual_counts={}
for split,n in expected.items():
 paths=sorted(p for p in files if p.startswith('data/'+split+'-') and p.endswith('.parquet'))
 count=0
 for path in paths:
  with fs.open('datasets/'+REPO+'@'+revision+'/'+path,'rb') as f:count+=pq.ParquetFile(f).metadata.num_rows
 assert count==n,(split,count,n)
 actual_counts[split]=count
print('Remote Parquet row counts:',actual_counts,flush=True)
print('Remote manifests and every referenced audio file verified',flush=True)
# Default loader must ignore review/metadata JSONL and expose only the three Parquet splits.
ds=load_dataset(REPO,revision=revision,streaming=True)
assert set(ds)==set(expected),list(ds)
checks={}
for split in expected:
 row=next(iter(ds[split]));audio=row['audio'];assert audio['sampling_rate']==16000
 pcm=np.rint(np.clip(audio['array'],-1,1-1/32768)*32768).astype('<i2')
 assert hashlib.sha256(pcm.tobytes()).hexdigest()==row['pcm_sha256']
 assert row['sentence'].strip() and row['split']==split
 checks[split]={'expected_rows':expected[split],'first_audio_checksum_matches':True,'first_chunk_id':row['chunk_id']}
 print(split,'default loader and audio decode passed',flush=True)
result={'passed':True,'repo':REPO,'private':info.private,'revision':revision,'splits':checks,'actual_parquet_rows':actual_counts,'portable_audio_paths_checked':sum(expected.values()),'default_loader':'datasets.load_dataset(repo, revision=revision, streaming=True)','review_audio_files':sum(p.startswith('portable/review/audio/') for p in files)}
(HF/'hub_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

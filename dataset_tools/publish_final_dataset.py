#!/usr/bin/env python3
"""Publish only the validated final dataset to its own private repository."""
from pathlib import Path
import argparse,json,shutil
from datasets import load_from_disk
from huggingface_hub import HfApi
ROOT=Path(__file__).resolve().parents[1];HF=ROOT/'Final_Gathered_Dataset_HF'
REPO='AliAvd/persian-elderly-asr'

def prepare_portable():
 ds=load_from_disk(str(HF))
 portable=HF/'portable';portable.mkdir(exist_ok=True)
 for split in ds:
  rows=[json.loads(l) for l in (HF/'manifests'/(split+'.jsonl')).read_text().splitlines()]
  qwen=[];manifest=[]
  for r in rows:
   source=Path(r['audio']);relative=Path('audio')/r['speaker_id']/source.name;target=portable/relative;target.parent.mkdir(parents=True,exist_ok=True)
   if not target.exists():target.hardlink_to(source)
   qwen.append(dict(audio=relative.as_posix(),text='language Persian<asr_text>'+r['sentence'],speaker_id=r['speaker_id'],source_id=r['source_id'],duration=r['duration']))
   manifest.append({**r,'audio':relative.as_posix()})
  for folder,values in [('qwen_jsonl',qwen),('manifests',manifest)]:
   path=portable/folder/(split+'.jsonl');path.parent.mkdir(exist_ok=True);path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values))
 (portable/'materialize_qwen.py').write_text('''from pathlib import Path
import json
root=Path(__file__).resolve().parent
for source in (root/'qwen_jsonl').glob('*.jsonl'):
 rows=[json.loads(l) for l in source.read_text().splitlines() if l.strip()]
 for r in rows:
  r['audio']=str((root/r['audio']).resolve())
  assert Path(r['audio']).is_file(),r['audio']
 out=root/'qwen_local'/source.name;out.parent.mkdir(exist_ok=True)
 out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\\n' for r in rows))
 print(out,len(rows))
''')
 # Review audio remains outside the default training configuration.
 review=[]
 for line in (HF/'rejected.jsonl').read_text().splitlines():
  r=json.loads(line);source=Path(r['audio']);relative=Path('review/audio')/r['speaker_id']/source.name
  target=portable/relative;target.parent.mkdir(parents=True,exist_ok=True)
  if not target.exists():target.hardlink_to(source)
  review.append({**r,'audio':relative.as_posix()})
 (portable/'review/manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in review))
 return portable

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
 audit=json.loads((HF/'validation_report.json').read_text());assert audit['passed']
 portable=prepare_portable()
 if args.prepare_only:
  print('Portable upload package prepared locally:',portable);return
 api=HfApi();assert api.whoami()['name']=='AliAvd'
 api.create_repo(REPO,repo_type='dataset',private=True,exist_ok=True)
 assert api.repo_info(REPO,repo_type='dataset').private,'Refusing to upload to public repository'
 ds=load_from_disk(str(HF));ds.push_to_hub(REPO,private=True,max_shard_size='300MB')
 api.upload_folder(repo_id=REPO,repo_type='dataset',folder_path=str(portable),path_in_repo='portable',ignore_patterns=['qwen_local/**'],commit_message='Add portable paired audio and Qwen manifests')
 # Replace the previous corpus's root Qwen manifests as well.
 for split in ('train','validation','test'):
  rows=[json.loads(l) for l in (portable/'qwen_jsonl'/(split+'.jsonl')).read_text().splitlines()]
  for row in rows:row['audio']='portable/'+row['audio']
  payload=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows).encode()
  api.upload_file(repo_id=REPO,repo_type='dataset',path_or_fileobj=payload,path_in_repo='qwen_jsonl/'+split+'.jsonl',commit_message='Replace legacy Qwen split with final corpus')
 for name in ['README.md','processing_summary.json','validation_report.json','experiment_integration_report.json','rejected.jsonl']:
  api.upload_file(repo_id=REPO,repo_type='dataset',path_or_fileobj=str(HF/name),path_in_repo=name)
 info=api.repo_info(REPO,repo_type='dataset');(HF/'hub_publish.json').write_text(json.dumps({'repo':REPO,'private':info.private,'revision':info.sha,'url':'https://huggingface.co/datasets/'+REPO},indent=2))
 (HF/'UPLOAD_STATUS.json').write_text(json.dumps({'status':'published','repo':REPO,'private':info.private,'revision':info.sha},indent=2))
 print('Published',REPO,info.sha,flush=True)
if __name__=='__main__':main()

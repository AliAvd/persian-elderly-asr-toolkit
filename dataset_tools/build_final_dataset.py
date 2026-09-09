#!/usr/bin/env python3
"""Reference-based CTC alignment and reproducible exports of the final elderly corpus.
No model training. Uncertain automatic alignment is quarantined, never human-verified.
"""
import argparse, collections, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from shared.evaluation import normalize, distance
from shared.training import restore_ctc_special_tokens
CHUNKS=ROOT/'Final_Gatheres_Dataset_Chunks'
HF=ROOT/'Final_Gathered_Dataset_HF'
CHECKPOINT=ROOT/'wav2vec2_base/output_ver_1/checkpoint-547500'

def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)
def jsonl(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('w') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 tmp.replace(path)
def read(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def inventory():
 rows=[]
 for a in sorted((ROOT/'Final_Gathered_Dataset').rglob('*')):
  if a.suffix.lower()!='.m4a':continue
  ts=[t for t in a.parent.iterdir() if t.stem.lower()==a.stem.lower() and t.suffix.lower()=='.txt']
  if len(ts)!=1:raise ValueError(f'Expected one paired transcript: {a}')
  rows.append(dict(audio=str(a),text=str(ts[0]),speaker_id=a.parent.name,source_id=a.parent.name+'_'+a.stem.lower(),audio_sha256=sha(a),text_sha256=sha(ts[0])))
 assert len({r['source_id'] for r in rows})==len(rows)
 return rows

def emissions(wave,processor,model,device):
 import torch
 # 20-second cores, 2-second context on both sides. Global 20-ms frame grid.
 stride=320;kernel=400;total=max(0,(len(wave)-kernel)//stride+1);parts=[]
 for first in range(0,total,1000):
  last=min(first+1000,total);left=max(0,first-100);right=min(total,last+100)
  x=wave[left*stride:(right-1)*stride+kernel]
  inputs=processor(x,sampling_rate=16000,return_tensors='pt')
  with torch.inference_mode(): logits=model(**{k:v.to(device) for k,v in inputs.items()}).logits[0].float().cpu()
  parts.append(logits[first-left:last-left].log_softmax(-1))
 result=torch.cat(parts);assert len(result)==total
 return result

def align_source(pair,processor,model,device):
 import torch, torchaudio
 sid=pair['source_id'];done=CHUNKS/'sources'/(sid+'.jsonl');meta=CHUNKS/'sources'/(sid+'.meta.json')
 if done.exists() and meta.exists():
  previous=json.loads(meta.read_text())
  if previous['audio_sha256']!=pair['audio_sha256'] or previous['text_sha256']!=pair['text_sha256']:raise ValueError('Source changed: '+sid)
  return read(done)
 decoded=subprocess.run(['ffmpeg','-v','error','-i',pair['audio'],'-f','f32le','-ac','1','-ar','16000','pipe:1'],capture_output=True,check=True)
 wave=np.frombuffer(decoded.stdout,dtype='<f4').copy();assert len(wave)>400 and np.isfinite(wave).all()
 lp=emissions(wave,processor,model,device)
 # Keep original transcript words/character offsets while aligning their normalized forms.
 original=Path(pair['text']).read_text(encoding='utf-8-sig');words=[];ids=[]
 for m in re.finditer(r'\S+',original):
  clean=normalize(m.group());tokens=processor.tokenizer(clean,add_special_tokens=False).input_ids
  if not tokens:continue
  if ids:ids.append(processor.tokenizer.word_delimiter_token_id)
  lo=len(ids);ids.extend(tokens);words.append(dict(raw=m.group(),start_char=m.start(),end_char=m.end(),token_start=lo,token_end=len(ids),unknown=processor.tokenizer.unk_token_id in tokens))
 if not ids:raise ValueError('Empty normalized transcript '+sid)
 target=torch.tensor([ids],dtype=torch.int64)
 path,scores=torchaudio.functional.forced_align(lp.unsqueeze(0),target,blank=model.config.pad_token_id)
 spans=torchaudio.functional.merge_tokens(path[0],scores[0].exp(),blank=model.config.pad_token_id)
 assert len(spans)==len(ids),(len(spans),len(ids))
 assert [s.token for s in spans]==ids
 for w in words:
  ts=spans[w['token_start']:w['token_end']];w.update(start=ts[0].start*.02,end=ts[-1].end*.02+.025,confidence=float(np.mean([float(t.score) for t in ts])))
 # Partition at aligned word boundaries; every original lexical word assigned exactly once.
 groups=[];current=[]
 for w in words:
  if current and (w['end']-current[0]['start']>15 or (current[-1]['end']-current[0]['start']>=8) or w['start']-current[-1]['end']>1.2):groups.append(current);current=[]
  current.append(w)
 if current:groups.append(current)
 rows=[];last_end=0
 for index,g in enumerate(groups):
  next_start=groups[index+1][0]['start'] if index+1<len(groups) else len(wave)/16000
  start=max(last_end,g[0]['start']-.08);end=min(len(wave)/16000,g[-1]['end']+.08,(g[-1]['end']+next_start)/2)
  start_sample=max(0,round(start*16000));end_sample=min(len(wave),round(end*16000));last_end=end_sample/16000
  pcm=np.rint(np.clip(wave[start_sample:end_sample],-1,1-1/32768)*32768).astype('<i2')
  chunk_id=f'{sid}_chunk_{index:04d}';audio=CHUNKS/'chunks'/pair['speaker_id']/(chunk_id+'.wav');audio.parent.mkdir(parents=True,exist_ok=True);sf.write(audio,pcm,16000,subtype='PCM_16')
  # Exact substring of supplied transcript, not ASR-generated training labels.
  text=original[g[0]['start_char']:g[-1]['end_char']];normalized=normalize(text)
  pred=processor.decode(lp[max(0,int(start/.02)):min(len(lp),int(end/.02))].argmax(-1).tolist())
  ref=normalize(text);hyp=normalize(pred);wer=distance(ref.split(),hyp.split())/max(1,len(ref.split()));cer=distance(ref,hyp)/max(1,len(ref))
  duration=len(pcm)/16000;reasons=[]
  if not 1<=duration<=16:reasons.append('duration_outside_1_16_seconds')
  if not ref:reasons.append('empty_normalized_text')
  if any(w['unknown'] for w in g):reasons.append('unsupported_alignment_character')
  if cer>.45:reasons.append('alignment_cer_above_0.45')
  if not len(pcm) or not np.any(pcm):reasons.append('silent_audio')
  row=dict(audio=str(audio),sentence=text,text=text,speaker_id=pair['speaker_id'],source_id=sid,chunk_id=chunk_id,source_audio=pair['audio'],source_text=pair['text'],source_audio_sha256=pair['audio_sha256'],source_text_sha256=pair['text_sha256'],start=start_sample/16000,end=end_sample/16000,duration=duration,text_start_char=g[0]['start_char'],text_end_char=g[-1]['end_char'],asr_prediction=pred,alignment_wer=wer,alignment_cer=cer,alignment_confidence=float(np.mean([w['confidence'] for w in g])),alignment_method='reference_ctc_forced_alignment',alignment_checkpoint=str(CHECKPOINT),human_verified=False,needs_review=bool(reasons),review_reasons=reasons,pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest())
  rows.append(row);t=CHUNKS/'texts'/pair['speaker_id']/(chunk_id+'.txt');t.parent.mkdir(parents=True,exist_ok=True);t.write_text(text+'\n')
 jsonl(done,rows);write(meta,{**pair,'duration':len(wave)/16000,'chunks':len(rows),'accepted':sum(not r['needs_review'] for r in rows),'alignment_checkpoint':str(CHECKPOINT)})
 print(sid,'chunks',len(rows),'accepted',sum(not r['needs_review'] for r in rows),flush=True)
 return rows

def build(limit=None):
 import torch
 from transformers import Wav2Vec2Processor,Wav2Vec2ForCTC
 torch.set_num_threads(2)
 device='cuda:0' if torch.cuda.is_available() else 'cpu'
 if device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(0.30)
 processor=Wav2Vec2Processor.from_pretrained(CHECKPOINT);model=Wav2Vec2ForCTC.from_pretrained(CHECKPOINT).to(device).eval();restore_ctc_special_tokens(processor,model)
 pairs=inventory();write(CHUNKS/'source_inventory.json',pairs);rows=[]
 for i,pair in enumerate(pairs[:limit] if limit else pairs):
  print('SOURCE',i+1,'of',len(pairs),flush=True);rows.extend(align_source(pair,processor,model,device))
 jsonl(CHUNKS/'manifest.jsonl',rows);jsonl(CHUNKS/'review.jsonl',[r for r in rows if r['needs_review']])
 if not limit:export(rows)

def export(rows):
 from datasets import Dataset,DatasetDict,Audio
 accepted=[];rejected=[];seen=set()
 for r in rows:
  if r['needs_review']:rejected.append(r);continue
  if r['pcm_sha256'] in seen:rejected.append({**r,'review_reasons':['duplicate_pcm']});continue
  seen.add(r['pcm_sha256']);accepted.append(dict(r))
 # Exact repeated source audio/transcripts grouped together, never across splits.
 pairs=inventory();parent={p['source_id']:p['source_id'] for p in pairs}
 def find(x):
  while parent[x]!=x:x=parent[x]
  return x
 seen_keys={}
 for p in pairs:
  keys=[('audio',p['audio_sha256']),('text',hashlib.sha256(normalize(Path(p['text']).read_text()).encode()).hexdigest())]
  for key in keys:
   if key in seen_keys:parent[find(p['source_id'])]=find(seen_keys[key])
   else:seen_keys[key]=p['source_id']
 groups=sorted({find(r['source_id']) for r in accepted},key=lambda s:hashlib.sha256(('42:'+s).encode()).hexdigest())
 assert len(groups)>=3
 strata=collections.defaultdict(list)
 for group in groups:
  speakers=tuple(sorted({r['speaker_id'] for r in accepted if find(r['source_id'])==group}))
  strata[speakers].append(group)
 test=set();val=set()
 for speakers,names in strata.items():
  # A one-recording speaker stays in train; never split their recording into held-out chunks.
  if len(names)<3:continue
  n=max(1,round(.1*len(names)));test.update(names[:n]);val.update(names[n:2*n])
 splits={s:[] for s in ['train','validation','test']}
 for r in accepted:
  group=find(r['source_id']);split='test' if group in test else 'validation' if group in val else 'train';r.update(split=split,split_group=group);splits[split].append(r)
 assert all(splits.values())
 for key in ['source_id','source_audio_sha256','pcm_sha256','split_group']:
  sets=[{r[key] for r in v} for v in splits.values()];assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:]),key
 # Embed audio bytes in Arrow so save_to_disk output is independently portable.
 ds=DatasetDict({s:Dataset.from_list([{**r,'audio':{'bytes':Path(r['audio']).read_bytes(),'path':Path(r['audio']).name}} for r in values]).cast_column('audio',Audio(sampling_rate=16000)) for s,values in splits.items()})
 if (HF/'dataset_dict.json').exists():raise FileExistsError('Existing HF export; refuse overwriting without explicit archive')
 ds.save_to_disk(str(HF))
 for split,values in splits.items():
  jsonl(HF/'manifests'/(split+'.jsonl'),values)
  jsonl(HF/'qwen_jsonl'/(split+'.jsonl'),[dict(audio=r['audio'],text='language Persian<asr_text>'+r['sentence'],speaker_id=r['speaker_id'],source_id=r['source_id'],duration=r['duration']) for r in values])
 jsonl(HF/'manifest.jsonl',[r for values in splits.values() for r in values]);jsonl(HF/'rejected.jsonl',rejected)
 allq=[dict(audio=r['audio'],text='language Persian<asr_text>'+r['sentence']) for values in splits.values() for r in values]
 for name in ['qwen_all.jsonl','qwen_al.jsonl']:jsonl(HF/name,allq);jsonl(CHUNKS/name,allq)
 summary={'source_recordings':len(pairs),'source_hours':sum(json.loads(p.read_text())['duration'] for p in (CHUNKS/'sources').glob('*.meta.json'))/3600,'all_chunks':len(rows),'accepted':len(accepted),'rejected':len(rejected),'splits':{s:{'samples':len(v),'hours':sum(r['duration'] for r in v)/3600,'sources':len({r['source_id'] for r in v}),'speakers':sorted({r['speaker_id'] for r in v})} for s,v in splits.items()},'seed':42,'split_policy':'source recording + identical source transcript/audio groups; shared speakers allowed; approximately 80/10/10 by group count within speaker strata; one-recording speaker remains train','speaker_independent':False,'human_verified':False,'alignment_quality_filter':'normalized greedy CTC CER <= 0.45; nonempty supported text; 1–16 seconds; nonzero PCM','alignment_checkpoint':str(CHECKPOINT),'leakage_audit':'disjoint source IDs, source audio hashes, PCM hashes, and duplicate-transcript groups across all splits'}
 write(HF/'processing_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
 (HF/'README.md').write_text('''---\nlanguage:\n- fa\ntask_categories:\n- automatic-speech-recognition\nconfigs:\n- config_name: default\n  data_files:\n  - split: train\n    path: data/train-*.parquet\n  - split: validation\n    path: data/validation-*.parquet\n  - split: test\n    path: data/test-*.parquet\n---\n# Final gathered Persian elderly speech\n\n80 paired recordings from four speaker folders. Reference transcripts were aligned with a historical Persian Wav2Vec2-base checkpoint, then cut at word boundaries into approximately 8-second chunks (maximum accepted duration 16 seconds). Labels are substrings of the supplied transcripts, not generated ASR labels. Automatic alignments are not human-verified. Uncertain chunks are excluded and listed in rejected.jsonl. Filtering using an ASR model can select easier speech; report this limitation when using this test set.\n\nSplits are deterministic by source recording and identical transcript/audio groups, stratified by speaker where at least three recordings exist. A speaker with only one recording remains in training. Speakers may overlap. These are not speaker-independent results. Full provenance and counts are in processing_summary.json. No license or public redistribution permission is inferred; this repository is private.\n\nUse `datasets.load_dataset("AliAvd/persian-elderly-asr")` for the default embedded-audio Parquet configuration. Local Arrow export: `datasets.load_from_disk("ASR/Final_Gathered_Dataset_HF")`. Qwen requires `language Persian<asr_text>` labels; local files are in qwen_jsonl/. Portable Hub manifests and audio are under portable/; run portable/materialize_qwen.py after snapshot_download to generate absolute local audio paths.\n''')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--limit',type=int);a=p.parse_args();build(a.limit)

#!/usr/bin/env python3
"""Prepare real audio, explicit splits and Qwen/CTC manifests for the catalog."""
from __future__ import annotations
import argparse, collections, csv, hashlib, io, json, math, re, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

ROOT=Path(__file__).resolve().parents[1]
ASR=ROOT.parent
DATA=ROOT/'data'
MAN=DATA/'manifests'
SPLITS=('train','validation','test')

def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)

def read_rows(path):
    with Path(path).open(encoding='utf-8') as f:return [json.loads(l) for l in f if l.strip()]

def write_rows(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    tmp.replace(path)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def group_split(key):
    x=int(hashlib.sha256(('42:'+str(key)).encode()).hexdigest()[:12],16)%100
    return 'test' if x<10 else 'validation' if x<20 else 'train'

def pcm_audio(value):
    if isinstance(value,dict):value=io.BytesIO(value['bytes']) if value.get('bytes') else value['path']
    if isinstance(value,(str,Path)) and Path(value).suffix.lower() in {'.m4a','.aac','.mp4'}:
        import subprocess
        decoded=subprocess.run(['ffmpeg','-v','error','-i',str(value),'-f','f32le','-ac','1','-ar','16000','pipe:1'],check=True,capture_output=True)
        audio=np.frombuffer(decoded.stdout,dtype='<f4');sr=16000
    else:
        audio,sr=sf.read(value,dtype='float32',always_2d=True)
        audio=audio.mean(axis=1)
    if sr!=16000:
        g=math.gcd(sr,16000);audio=resample_poly(audio,16000//g,sr//g)
    if not np.isfinite(audio).all():raise ValueError('non-finite audio')
    # Canonical PCM16 hashes detect identical decoded audio despite differing containers.
    return np.rint(np.clip(audio,-1,1-1/32768)*32768).astype('<i2')

def materialize(job):
    row,value,min_seconds,max_seconds=job
    try:
        pcm=pcm_audio(value);duration=len(pcm)/16000
        if not min_seconds<=duration<=max_seconds:raise ValueError(f'duration outside [{min_seconds}, {max_seconds}]: {duration:.3f}')
        if not row['text'].strip():raise ValueError('empty reference')
        if not np.any(pcm):raise ValueError('silent audio')
        row['duration']=duration;row['pcm_sha256']=hashlib.sha256(pcm.tobytes()).hexdigest()
        path=DATA/'audio'/row['source']/(row['id']+'.wav');path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            existing=sf.read(path,dtype='int16')[0]
            if hashlib.sha256(existing.astype('<i2').tobytes()).hexdigest()!=row['pcm_sha256']:
                raise ValueError(f'Audio ID collision or damaged prepared file: {path}')
        else:sf.write(path,pcm,16000,subtype='PCM_16')
        row['audio']=str(path.resolve());return row,None
    except Exception as e:return None,{'id':row['id'],'reason':str(e),'source':row['source']}

def arrow_rows(paths):
    for p in paths:
        reader=pa.ipc.open_stream(str(p))
        for batch in reader:
            yield from batch.to_pylist()

def source_jobs(source):
    if source=='elderly':
        # Existing source recording split is preserved; user explicitly allows shared speakers.
        splitmap={}
        for split in SPLITS:
            for r in read_rows(ASR/f'Final_Gathered_Dataset_HF/qwen_jsonl/{split}.jsonl'):splitmap[Path(r['audio']).stem]=split
        for r in read_rows(ASR/'Final_Gatheres_Dataset_Chunks/manifest.jsonl'):
            key=Path(r['audio']).stem
            if key not in splitmap:continue
            row=dict(id='elderly_'+key,source=source,source_id='elderly:'+r['source_id'],speaker_id=r.get('speaker_id','unknown'),language='Persian',text=r['sentence'],split=splitmap[key],human_verified=False,alignment_method=r.get('alignment_method'),original_start=r['start'],original_end=r['end'])
            yield row,r['audio'],.25,30
    elif source=='ganjoor':
        for i,r in enumerate(arrow_rows(sorted((ASR/'asr_dataset/train').glob('*.arrow')))):
            old=Path(r['audio']['path']); group='ganjoor:'+old.stem.split('_chunk_',1)[0]
            row=dict(id=f'ganjoor_{i:06}',source=source,source_id=group,speaker_id=' '.join(r.get('speaker_id','unknown').split()),language='Persian',text=r['sentence'],split=group_split(group),original_audio=str(old))
            yield row,r['audio'],2,20
    elif source=='filimo':
        cache=Path('/home/shared/huggingface/PerSets___filimo-persian-asr/default/1.0.0/69c7d3c82fbcfd8d2bbfda13abf5ba96b2ed6f1d8ff7236402d22528f8210de3')
        files=sorted(cache.glob('filimo-persian-asr-unvalidated-*.arrow'))
        if len(files)!=10:raise FileNotFoundError('Original PerSets Filimo cache shards missing; no other dataset is substituted')
        for i,r in enumerate(arrow_rows(files)):
            name=Path(r.get('file_name') or r['audio']['path']).stem
            # Filimo filenames contain a five-digit recording prefix then segment number.
            group='filimo:'+name[:5] if re.fullmatch(r'\d{10}',name) else 'filimo:'+Path(r['audio']['path']).parent.name
            row=dict(id=f'filimo_{name}',source=source,source_id=group,speaker_id='unknown',language='Persian',text=r['text'],split=group_split(group),original_audio=r['audio']['path'],grouping='filename-prefix inferred recording; speaker overlap not excluded')
            yield row,r['audio'],2,20
    elif source=='mozilla':
        roots=[p.parent for p in (ASR/'commonvoice_fa').rglob('train.tsv') if p.parent.name=='fa' and '26.0' in str(p)]
        if len(roots)!=1:raise FileNotFoundError('Download/extract approved Common Voice 26 Persian first')
        root=roots[0]
        for split,filename in [('train','train.tsv'),('validation','dev.tsv'),('test','test.tsv')]:
            with (root/filename).open(encoding='utf-8') as f:
                for r in csv.DictReader(f,delimiter='\t',quoting=csv.QUOTE_NONE):
                    name=Path(r['path']).stem
                    row=dict(id='mozilla_'+name,source=source,source_id='mozilla:'+name,speaker_id=r.get('client_id','unknown'),language='Persian',text=r['sentence'],split=split,age=r.get('age',''),gender=r.get('gender',''),version=root.parent.name,original_audio=str(root/'clips'/r['path']))
                    yield row,str(root/'clips'/r['path']),2,20
    elif source=='seniortalk':
        root=ASR/'datasets/raw/seniortalk/sentence_data'
        metadata=ASR/'datasets/raw/seniortalk/download_status.json'
        if not metadata.is_file():raise FileNotFoundError('Complete checksum-verified SeniorTalk download first')
        for split,stem in [('train','train'),('validation','dev'),('test','test')]:
            files=sorted(root.glob(stem+'-*.parquet'))
            if not files:raise FileNotFoundError(f'SeniorTalk {stem} shards missing')
            for p in files:
                for b in pq.ParquetFile(p).iter_batches(batch_size=32):
                    for r in b.to_pylist():
                        name=Path(r['path']['path']).stem
                        m=re.match(r'(Elderly\d+)(S\d+)',name)
                        group='seniortalk:'+(m.group(1) if m else name)
                        row=dict(id='seniortalk_'+name,source=source,source_id=group,speaker_id=(m.group(2) if m else 'unknown'),language='Chinese',text=r['text'],split=split,original_audio=r['path']['path'],grouping='Elderly recording prefix; official split retained unless source conflict')
                        yield row,r['path'],.25,30
    else:
        raise ValueError(source)

def remove_leakage(rows):
    # Entire source recordings stay together; test takes precedence over validation/train.
    priority={'train':0,'validation':1,'test':2}
    targets={}
    for r in rows:
        g=r['source_id'];targets[g]=max(targets.get(g,'train'),r['split'],key=priority.get)
    for r in rows:
        r.setdefault('original_split',r['split'])
        r['split']=targets[r['source_id']]
    # Content duplicates removed globally, preserving evaluation rows before training.
    seen=set();clean=[];removed=[]
    for r in sorted(rows,key=lambda x:-priority[x['split']]):
        if r['pcm_sha256'] in seen:removed.append({'id':r['id'],'reason':'duplicate canonical PCM audio','split':r['split']});continue
        seen.add(r['pcm_sha256']);clean.append(r)
    return clean,removed

def publish(source,rows,rejected):
    rows,duplicates=remove_leakage(rows);rejected+=duplicates
    stats={}
    for split in SPLITS:
        values=sorted((r for r in rows if r['split']==split),key=lambda r:r['id'])
        if not values:raise ValueError(f'{source} {split} empty')
        path=MAN/source/(split+'.jsonl');write_rows(path,values)
        qwen=[dict(r,text=f'language {r["language"]}<asr_text>{r["text"]}') for r in values]
        write_rows(MAN/source/'qwen'/path.name,qwen)
        stats[split]={'samples':len(values),'hours':sum(r['duration'] for r in values)/3600,'sha256':digest(path),'source_recordings':len(set(r['source_id'] for r in values))}
    write_rows(MAN/source/'rejected.jsonl',rejected)
    write_json(MAN/source/'summary.json',{'source':source,'splits':stats,'rejected':len(rejected),'sample_rate':16000,'channels':1,'speaker_independent':False,'split_policy':'official/source-recording groups; canonical PCM duplicates removed; shared speakers allowed by user','senior_transcript_review':'automatic alignment not human-verified' if source=='elderly' else None,'upstream':'PerSets/filimo-persian-asr' if source=='filimo' else 'BAAI/SeniorTalk' if source=='seniortalk' else source})
    print(source,json.dumps(stats),flush=True)

def prepare(source,workers=8):
    if (MAN/source/'summary.json').exists():
        print(f'{source} already prepared; delete its generated summary to rebuild',flush=True);return
    rows=[];reject=[]
    # Keep executor queue bounded: Arrow audio bytes can otherwise exhaust memory.
    jobs=iter(source_jobs(source));done=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while True:
            import itertools
            batch=list(itertools.islice(jobs,workers*4))
            if not batch:break
            for row,error in pool.map(materialize,batch):
                if row:rows.append(row)
                else:reject.append(error)
                done+=1
            if done%1024==0:print(source,'processed',done,'accepted',len(rows),flush=True)
    publish(source,rows,reject)

def compose():
    sources={}
    for name in ['mozilla','filimo','ganjoor','elderly','seniortalk']:
        if (MAN/name/'summary.json').is_file():sources[name]={s:read_rows(MAN/name/(s+'.jsonl')) for s in SPLITS}
    # Cross-source exact audio duplication is prohibited, same test may intentionally be reported both individually and pooled.
    compositions={'public':['mozilla','filimo','ganjoor'],'public_without_ganjoor':['mozilla','filimo']}
    for suffix in ['elderly','senior','elderly_senior']:
        comps=['seniortalk' if x=='senior' else x for x in suffix.split('_')]
        compositions['public_'+suffix]=['mozilla','filimo','ganjoor']+comps
    status={}
    for name,members in compositions.items():
        missing=[x for x in members if x not in sources]
        if missing:status[name]={'ready':False,'missing_sources':missing};continue
        merged=[dict(r) for m in members for s in SPLITS for r in sources[m][s]]
        publish(name,merged,[]);status[name]={'ready':True,'sources':members}
    if status.get('public_elderly',{}).get('ready'):
        base={s:read_rows(MAN/'public'/(s+'.jsonl')) for s in SPLITS}
        elderly=sources['elderly']['train']
        ordered=sorted(elderly,key=lambda r:hashlib.sha256(('volume42:'+r['id']).encode()).hexdigest())
        for label,fraction in [('025',.25),('050',.5),('075',.75)]:
            members=[dict(r) for s in SPLITS for r in base[s]]+[dict(r) for r in ordered[:max(1,round(len(ordered)*fraction))]]+[dict(r) for s in ['validation','test'] for r in sources['elderly'][s]]
            publish('public_elderly_'+label,members,[])
        # No teacher-filtered test: unfiltered is the shared deterministic-quality public base.
        publish('public_unfiltered',[dict(r) for s in SPLITS for r in base[s]],[])
    write_json(DATA/'preparation_status.json',status)
    print('Composition readiness:',json.dumps(status),flush=True)

def augmentation_assets():
    report={}
    for kind,folder in [('background',ASR/'qwen_asr/backgrounds'),('rir',ASR/'qwen_asr/Impulses')]:
        files=sorted(folder.glob('*.wav'),key=lambda p:hashlib.sha256(p.name.encode()).hexdigest())
        if not files:raise FileNotFoundError(folder)
        test_count=max(1,round(len(files)*.2));report[kind]={}
        for split,values in [('test',files[:test_count]),('train',files[test_count:])]:
            out=DATA/'augmentation'/(kind+'_'+split);out.mkdir(parents=True,exist_ok=True)
            for p in values:
                q=out/p.name
                if not q.exists():shutil.copy2(p,q)
            report[kind][split]=[{'name':p.name,'sha256':digest(p)} for p in values]
    write_json(DATA/'augmentation/asset_split.json',report)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',choices=['mozilla','filimo','ganjoor','elderly','seniortalk','all','compose','augmentation']);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if a.source=='all':
        for s in ['mozilla','seniortalk','ganjoor','filimo','elderly']:prepare(s,a.workers)
        augmentation_assets();compose()
    elif a.source=='compose':compose()
    elif a.source=='augmentation':augmentation_assets()
    else:prepare(a.source,a.workers)
if __name__=='__main__':main()

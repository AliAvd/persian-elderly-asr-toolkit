#!/usr/bin/env python3
"""Prepare fixed noisy tests, teacher-filtered TRAIN data, and long-recording tests."""
import argparse,hashlib,json,random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve,butter,sosfilt
from shared.prepare_data import ROOT,ASR,DATA,MAN,SPLITS,read_rows,write_rows,write_json,publish,materialize,pcm_audio
from shared.evaluation import normalize,plain_text,distance

@lru_cache(maxsize=128)
def asset_audio(path):
    return pcm_audio(path).astype(np.float32)/32768

def noise_one(job):
    row,condition,backgrounds,rirs=job
    seed=int(hashlib.sha256((row['id']+':'+condition+':42').encode()).hexdigest()[:8],16)
    rng=np.random.default_rng(seed);pcm=pcm_audio(row['audio']);audio=pcm.astype(np.float32)/32768
    info={'condition':condition,'seed':seed,'original_pcm_sha256':row['pcm_sha256']}
    if condition.startswith('snr') or condition=='combined':
        noise_path=backgrounds[seed%len(backgrounds)];noise=asset_audio(str(noise_path))
        if len(noise)<len(audio):noise=np.tile(noise,int(np.ceil(len(audio)/len(noise))))
        start=int(rng.integers(0,len(noise)-len(audio)+1));noise=noise[start:start+len(audio)].copy()
        snr=int(condition[3:]) if condition.startswith('snr') else 10
        noise*=np.sqrt((np.mean(audio**2)+1e-12)/(np.mean(noise**2)+1e-12))*10**(-snr/20)
        audio=audio+noise;info.update(snr_db=snr,noise_file=noise_path.name,noise_offset=start)
    if condition in ('rir','combined'):
        rir_path=rirs[seed%len(rirs)];rir=asset_audio(str(rir_path))
        rir=rir/(np.sqrt(np.sum(rir**2))+1e-12);audio=fftconvolve(audio,rir)[:len(audio)];info['rir_file']=rir_path.name
    if condition in ('channel','combined'):
        audio=sosfilt(butter(4,[300,3400],btype='bandpass',fs=16000,output='sos'),audio)
        audio=np.round(audio*127)/127;info['channel']='300-3400Hz bandpass + 8bit quantization'
    peak=float(np.max(np.abs(audio)))
    if peak>1:audio=audio/peak
    path=DATA/'audio'/('noise_'+condition)/(row['id']+'.wav');path.parent.mkdir(parents=True,exist_ok=True)
    sf.write(path,audio,16000,subtype='PCM_16')
    return dict(row,audio=str(path),augmentation=info,pcm_sha256=hashlib.sha256(pcm_audio(str(path)).tobytes()).hexdigest())

def noisy(workers=8, sources=('elderly','public')):
    backgrounds=sorted((DATA/'augmentation/background_test').glob('*.wav'));rirs=sorted((DATA/'augmentation/rir_test').glob('*.wav'))
    if not backgrounds or not rirs:raise FileNotFoundError('Run shared/prepare_data.py augmentation first')
    for source in sources:
        rows=read_rows(MAN/source/'test.jsonl')
        for condition in ['snr20','snr10','snr5','rir','channel','combined']:
            name=source+'_noisy_'+condition;target=MAN/name/'test.jsonl'
            if target.exists():continue
            with ThreadPoolExecutor(max_workers=workers) as pool:
                # Bounded batches avoid retaining all PCM arrays.
                results=[]
                for start in range(0,len(rows),workers*4):
                    results.extend(pool.map(noise_one,[(r,condition,backgrounds,rirs) for r in rows[start:start+workers*4]]))
            write_rows(target,results);write_json(MAN/name/'summary.json',{'samples':len(results),'condition':condition,'source_test':str(MAN/source/'test.jsonl'),'heldout_augmentation_assets':True})
            print('Prepared',name,len(results),flush=True)

def teacher_filter(checkpoint,threshold=.2):
    import torch
    from qwen_asr import Qwen3ASRModel
    model=Qwen3ASRModel.from_pretrained(checkpoint,dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,device_map='cuda:0' if torch.cuda.is_available() else 'cpu')
    rows=read_rows(MAN/'public/train.jsonl');selected=[];decisions=[]
    for i,r in enumerate(rows):
        prediction=model.transcribe(audio=r['audio'],language='Persian')[0].text
        ref=normalize(r['text']).split();score=distance(ref,normalize(prediction).split())/max(1,len(ref))
        decisions.append({'id':r['id'],'prediction':prediction,'normalized_wer':score,'accepted':score<threshold})
        if score<threshold:selected.append(r)
        if (i+1)%100==0:print('Teacher scored',i+1,flush=True)
    if not selected:raise RuntimeError('Teacher retained no training data')
    heldout=[r for s in ['validation','test'] for r in read_rows(MAN/'public'/(s+'.jsonl'))]
    publish('public_filtered',[dict(r) for r in selected+heldout],[])
    by_source={}
    for r in selected:by_source[r['source']]=by_source.get(r['source'],0)+1
    rng=random.Random(42);matched=[]
    for source,n in by_source.items():matched+=rng.sample([r for r in rows if r['source']==source],n)
    publish('public_unfiltered_matched',[dict(r) for r in matched+heldout],[])
    write_rows(DATA/'teacher_filter_decisions.jsonl',decisions)
    write_json(DATA/'teacher_filter_metadata.json',{'checkpoint':checkpoint,'threshold':threshold,'filtered_split':'train only','control':'same sample counts within each source; hours must also be reported'})

def long_recordings():
    # Resolve held-out original pairs from the final source inventory (case-sensitive paths).
    inventory={r['source_id']:r for r in json.loads((ASR/'Final_Gatheres_Dataset_Chunks/source_inventory.json').read_text())}
    rows=read_rows(MAN/'elderly/test.jsonl')
    ids=sorted({r['source_id'].split(':',1)[1] for r in rows});out=[]
    for source_id in ids:
        source=inventory[source_id];audio=Path(source['audio']);text=Path(source['text'])
        pcm=pcm_audio(str(audio));wav=DATA/'audio/elderly_long'/(source_id+'.wav');wav.parent.mkdir(parents=True,exist_ok=True);sf.write(wav,pcm,16000,subtype='PCM_16')
        out.append({'id':'elderly_long_'+source_id,'audio':str(wav),'text':text.read_text(encoding='utf-8-sig').strip(),'language':'Persian','source':'elderly_long','source_id':'elderly:'+source_id,'speaker_id':source['speaker_id'],'duration':len(pcm)/16000,'pcm_sha256':hashlib.sha256(pcm.tobytes()).hexdigest(),'split':'test'})
    write_rows(MAN/'elderly_long/test.jsonl',out)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('task',choices=['noisy','filter','long']);p.add_argument('--checkpoint');p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if a.task=='noisy':noisy(a.workers)
    elif a.task=='long':long_recordings()
    elif a.task=='filter':
        if not a.checkpoint:p.error('--checkpoint required')
        teacher_filter(a.checkpoint)
if __name__=='__main__':main()

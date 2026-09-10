#!/usr/bin/env python3
"""Persistent, resource-aware ten-epoch campaign. Never terminates other workloads."""
import argparse,csv,datetime,fcntl,json,os,shutil,subprocess,sys,time
from pathlib import Path
from run_experiment import ROOT,resolve_config,validate

PYTHON=Path(sys.executable)
DEFAULT=ROOT/'runs/20260908_10epochs'

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)

def prepare(directory):
 directory.mkdir(parents=True,exist_ok=True)
 if (directory/'jobs.json').exists():return
 jobs=[]
 for entry in read(ROOT/'catalog/index.json'):
  c=resolve_config(ROOT/entry['config']);name=c['experiment'];family=c['model']['family'];t=c['training']
  t.update(epochs=10,max_steps=-1,output_dir=str(directory/'outputs'/name),save_total_limit=1,save_only_model=True,gradient_checkpointing=True)
  # One item per microbatch preserves the original effective batch through accumulation.
  t['gradient_accumulation']=t.get('gradient_accumulation',8)*t.get('batch_size',1);t['batch_size']=1
  checkpoint=c['model'].get('init_checkpoint')
  if checkpoint:
   # Only checkpoints produced by this campaign are relocated. External baselines stay put.
   try:relative=Path(checkpoint).relative_to(ROOT/'outputs')
   except ValueError:pass
   else:c['model']['init_checkpoint']=str(directory/'outputs'/relative)
  cfg=directory/'configs'/(name+'.json');write(cfg,c)
  dependency=[]
  if name.startswith(('04_','08_')):dependency=['03_public/'+family]
  if name.startswith(('28_','29_','30_')):dependency=['07_public_elderly_aug/'+family]
  if name.startswith(('21_','22_')):dependency=['prepare_teacher_filter']
  # Baselines use the exact selected checkpoints, independently of the new training.
  if name.startswith('02_'):priority=0
  elif name.startswith('01_'):priority=1
  else:priority=int(name.split('_')[0])+2
  minimum=18000 if family=='qwen' and c['mode']=='train' else 8500 if family=='wav2vec2_xlsr' and c['mode']=='train' else 6000 if c['mode']=='train' or family=='qwen' else 3000
  jobs.append({'id':name,'kind':c['mode'],'config':str(cfg),'dependencies':dependency,'status':'pending','minimum_free_mb':minimum,'priority':priority,'attempts':0})
 jobs.append({'id':'prepare_teacher_filter','kind':'prepare','dependencies':[],'status':'pending','minimum_free_mb':6000,'priority':40,'attempts':0})
 write(directory/'jobs.json',jobs)
 write(directory/'campaign.json',{'created':stamp(),'epochs':10,'seed':42,'baselines':{j['id']:read(j['config'])['model'].get('init_checkpoint') for j in jobs if j['id'].startswith(('01_','02_'))},'gpu_policy':'Only currently free memory; never terminate other users; one campaign job per device','checkpoint_policy':'Final and best model weights saved; optimizer checkpoints omitted to fit campaign on available disk; failed training restarts rather than silently resuming with a fresh optimizer','scope':'60 catalog configurations plus teacher-data preparation; EARS excluded'})
 export(directory,jobs)

def export(directory,jobs):
 rows=[];lines=['# Ten-epoch experiment results','',f'Updated: {stamp()}','', 'WER/CER are percentages. Raw and normalized values are separate. Pending and failed runs have no invented results. Baseline checkpoints are user-selected; their historical training/test overlap is not certified.','', '| Experiment | Status | Evaluation | Normalization | Samples | WER % | CER % |','|---|---|---|---|---:|---:|---:|']
 for job in sorted(jobs,key=lambda j:j['id']):
  if job['kind']=='prepare':continue
  out=directory/'outputs'/job['id'];paths=sorted((out/'evaluation').glob('*.metrics.json'))
  if not paths:lines.append(f"| {job['id']} | {job['status']} | — | — | — | — | — |")
  for path in paths:
   m=read(path);suite=path.name.removesuffix('.metrics.json');cfg=read(job['config'])
   for norm in ['raw','normalized']:
    wer=m[norm]['wer'];cer=m[norm]['cer'];w=wer*100 if wer is not None else None;c=cer*100 if cer is not None else None
    row={'experiment':job['id'],'status':job['status'],'mode':job['kind'],'epochs_requested':10 if job['kind']=='train' else 0,'model':cfg['model']['family'],'initial_checkpoint':cfg['model'].get('init_checkpoint') or cfg['model']['pretrained'],'seed':42,'evaluation':suite,'normalization':norm,'samples':m['samples'],'failures':m['failures'],'wer_percent':w,'cer_percent':c,'rtf':m.get('real_time_factor'),'reference_words':m[norm]['reference_words'],'reference_characters':m[norm]['reference_characters'],'metrics_file':str(path),'predictions_file':str(path.with_name(suite+'.predictions.jsonl'))}
    rows.append(row);ws='—' if w is None else f'{w:.4f}';cs='—' if c is None else f'{c:.4f}'
    lines.append(f"| {job['id']} | {job['status']} | {suite} | {norm} | {m['samples']} | {ws} | {cs} |")
 (directory/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 if rows:
  tmp=directory/'results.csv.tmp'
  with tmp.open('w',encoding='utf-8-sig',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
  tmp.replace(directory/'results.csv')
 write(directory/'status.json',{'updated':stamp(),'counts':{s:sum(j['status']==s for j in jobs) for s in ['pending','running','completed','failed','waiting_gpu','waiting_dependency','blocked']},'jobs':jobs})

def available_gpus(allowed):
 r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 return [(int(a),int(b)) for line in r.stdout.splitlines() for a,b in [line.split(',')] if allowed is None or int(a) in allowed]

def supervise(directory,allowed=None,poll=30):
 lock=(directory/'supervisor.lock').open('w')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit('Campaign supervisor already running')
 lock.write(str(os.getpid()));lock.flush();jobs=read(directory/'jobs.json');running={}
 # Do not duplicate children of an interrupted supervisor.
 for job in jobs:
  if job['status']=='running':
   try:os.kill(job['pid'],0)
   except ProcessLookupError:job.update(status='failed',reason='Supervisor interrupted; inspect log before retry')
   else:raise SystemExit(f"Prior worker {job['pid']} is still alive; no duplicate launched")
 while True:
  for name,(process,log,job) in list(running.items()):
   code=process.poll()
   if code is None:continue
   log.close();job['returncode']=code;job['finished']=stamp()
   if job['kind']=='prepare':complete=(ROOT/'data/manifests/public_filtered/summary.json').exists() and (ROOT/'data/manifests/public_unfiltered_matched/summary.json').exists()
   else:complete=(directory/'outputs'/name/'metrics.json').exists()
   job['status']='completed' if code==0 and complete else 'failed'
   if job['status']=='failed':job['reason']='See '+job['log']
   print(stamp(),name,job['status'],flush=True);del running[name]
  states={j['id']:j['status'] for j in jobs};free=available_gpus(allowed);busy={j['gpu'] for _,_,j in running.values()}
  for job in sorted(jobs,key=lambda j:(j['priority'],j['id'])):
   if job['status'] in ['completed','failed','running']:continue
   if any(states.get(dep)!='completed' for dep in job['dependencies']):job.update(status='waiting_dependency',reason='Requires '+', '.join(job['dependencies']));continue
   if job['kind']!='prepare':
    errors=validate(read(job['config']))
    if errors:job.update(status='blocked',reason='; '.join(errors));continue
   if shutil.disk_usage(directory).free<30*1024**3:job.update(status='blocked',reason='Less than 30 GiB free disk');continue
   candidates=[(gpu,mb) for gpu,mb in free if gpu not in busy and mb>=job['minimum_free_mb']]
   if not candidates:job.update(status='waiting_gpu',reason=f"Waiting for {job['minimum_free_mb']} MiB free; other workloads untouched");continue
   gpu,mb=max(candidates,key=lambda x:x[1]);env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',ASR_CUDA_MEMORY_LIMIT_MB=str(max(1024,mb-1200)))
   logfile=directory/'logs'/(job['id'].replace('/','__')+'.log');logfile.parent.mkdir(parents=True,exist_ok=True);log=logfile.open('a')
   if job['kind']=='prepare':cmd=[str(PYTHON),'-B','-u',str(ROOT/'prepare_extra.py'),'filter','--checkpoint','Qwen/Qwen3-ASR-1.7B']
   else:cmd=[str(PYTHON),'-B','-u',str(ROOT/'run_experiment.py'),'--config',job['config']]
   process=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   job.update(status='running',pid=process.pid,gpu=gpu,started=stamp(),log=str(logfile),attempts=job['attempts']+1);job.pop('reason',None);running[job['id']]=(process,log,job);busy.add(gpu)
   print(stamp(),'started',job['id'],'GPU',gpu,'PID',process.pid,flush=True)
  write(directory/'jobs.json',jobs);export(directory,jobs)
  if all(j['status'] in ['completed','failed'] for j in jobs):break
  if not running and all(j['status'] in ['completed','failed','blocked','waiting_dependency'] for j in jobs):
   print('No runnable jobs; inspect status.json',flush=True);break
  time.sleep(poll)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','report','retry-failed']);p.add_argument('--directory',type=Path,default=DEFAULT);p.add_argument('--gpus',help='Comma-separated allowed GPU indices; otherwise any device with enough free memory');a=p.parse_args();directory=a.directory.resolve();prepare(directory)
 if a.action=='run':supervise(directory,{int(x) for x in a.gpus.split(',')} if a.gpus else None)
 elif a.action=='report':export(directory,read(directory/'jobs.json'))
 elif a.action=='retry-failed':
  jobs=read(directory/'jobs.json')
  for j in jobs:
   if j['status']=='failed':j['status']='pending'
  write(directory/'jobs.json',jobs)
 print(directory,flush=True)
if __name__=='__main__':main()

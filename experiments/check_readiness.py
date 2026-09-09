#!/usr/bin/env python3
"""Report actual data/checkpoint prerequisites without downloading or training."""
import argparse,json
from pathlib import Path
from run_experiment import ROOT, resolve_config, validate

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verbose',action='store_true');args=parser.parse_args()
    rows=[]
    for entry in json.loads((ROOT/'catalog/index.json').read_text()):
        config=resolve_config(ROOT/entry['config'])
        problems=validate(config)
        data=config['data'];paths=list(data['tests'].values())+([data.get('train_manifest'),data.get('validation_manifest')] if config['mode']=='train' else [])
        data_ready=all(p and Path(p).is_file() and Path(p).stat().st_size for p in paths)
        rows.append({'experiment':entry['experiment'],'data_ready':data_ready,'ready_to_launch':not problems,'prerequisites':problems})
    summaries={p.parent.name:json.loads(p.read_text()) for p in (ROOT/'data/manifests').glob('*/summary.json')}
    result={'data_ready':sum(r['data_ready'] for r in rows),'ready_to_launch':sum(r['ready_to_launch'] for r in rows),'total':len(rows),'note':'Presence checks only; each run also audits split separation. Training-dependent checkpoints are created by earlier experiments.','datasets':summaries,'experiments':rows}
    target=ROOT/'readiness.json';target.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(f"Data ready: {result['data_ready']}/{len(rows)}; launch prerequisites present: {result['ready_to_launch']}/{len(rows)}; details: {target}")
    for r in rows:
        if args.verbose and not r['ready_to_launch']: print(r['experiment']+': '+ '; '.join(r['prerequisites']))
if __name__=='__main__':main()

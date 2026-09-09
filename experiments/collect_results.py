#!/usr/bin/env python3
"""Collect completed metrics only; never invent values for unexecuted experiments."""
import argparse,csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--outputs',type=Path,default=ROOT/'outputs');p.add_argument('--csv',type=Path,default=ROOT/'results.csv');a=p.parse_args()
    rows=[]
    for path in sorted(a.outputs.rglob('metrics.json')):
        configpath=path.parent/'resolved_config.json'
        if not configpath.exists():continue
        config=json.loads(configpath.read_text())
        for split,metric in json.loads(path.read_text()).items():
            for normalization in ['raw','normalized']:
                m=metric[normalization]
                rows.append({'experiment':config['experiment'],'seed':config['training'].get('seed',42),'suite':split,'normalization':normalization,'samples':metric['samples'],'failures':metric['failures'],'wer_percent':None if m['wer'] is None else m['wer']*100,'cer_percent':None if m['cer'] is None else m['cer']*100,'reference_words':m['reference_words'],'reference_characters':m['reference_characters'],'real_time_factor':metric.get('real_time_factor'),'metrics_file':str(path)})
    if not rows: print('No completed experimental evaluations. No result table generated.');return
    a.csv.parent.mkdir(parents=True,exist_ok=True)
    with a.csv.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'Collected {len(rows)} metric rows into {a.csv}')
if __name__=='__main__':main()

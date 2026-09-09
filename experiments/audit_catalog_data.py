#!/usr/bin/env python3
"""Audit each unique available dataset configuration, independent of checkpoint readiness."""
import json
from pathlib import Path
from run_experiment import ROOT,resolve_config
from shared.audit import audit_config

def main():
    checked={};rows=[]
    for entry in json.loads((ROOT/'catalog/index.json').read_text()):
        config=resolve_config(ROOT/entry['config']);data=config['data']
        paths=list(data['tests'].values())+([data['train_manifest'],data['validation_manifest']] if config['mode']=='train' else [])
        missing=[p for p in paths if not p or not Path(p).is_file()]
        if missing:rows.append({'experiment':entry['experiment'],'status':'missing_data','missing':missing});continue
        key=json.dumps([config['mode'],data],sort_keys=True)
        if key not in checked:
            try:checked[key]=audit_config(config)
            except Exception as e:checked[key]={'status':'failed','error':str(e)}
            print(entry['experiment'],checked[key]['status'],flush=True)
        rows.append({'experiment':entry['experiment'],**checked[key]})
    (ROOT/'data/catalog_audit.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    failures=[r for r in rows if r['status']=='failed']
    print('Passed',sum(r['status']=='passed' for r in rows),'Missing',sum(r['status']=='missing_data' for r in rows),'Failed',len(failures))
    if failures:raise SystemExit(1)
if __name__=='__main__':main()

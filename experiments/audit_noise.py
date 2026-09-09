#!/usr/bin/env python3
"""Check that noisy tests preserve references and use held-out augmentation assets."""
import json
from pathlib import Path
from shared.prepare_data import DATA,MAN,read_rows,write_json

def main():
    assets=json.loads((DATA/'augmentation/asset_split.json').read_text())
    for kind in ['background','rir']:
        assert not {x['sha256'] for x in assets[kind]['train']}&{x['sha256'] for x in assets[kind]['test']},'Repeated augmentation asset across splits'
    names={kind:{x['name'] for x in assets[kind]['test']} for kind in assets}
    result={}
    for source in ['elderly','public']:
        originals={r['id']:r for r in read_rows(MAN/source/'test.jsonl')}
        for condition in ['snr20','snr10','snr5','rir','channel','combined']:
            name=source+'_noisy_'+condition;rows=read_rows(MAN/name/'test.jsonl')
            assert len(rows)==len(originals) and {r['id'] for r in rows}==set(originals),name
            for r in rows:
                old=originals[r['id']];info=r['augmentation']
                assert r['text']==old['text'] and r['source_id']==old['source_id'],r['id']
                assert info['original_pcm_sha256']==old['pcm_sha256'],r['id']
                assert r['audio']!=old['audio'] and Path(r['audio']).is_file(),r['id']
                if 'noise_file' in info:assert info['noise_file'] in names['background']
                if 'rir_file' in info:assert info['rir_file'] in names['rir']
            result[name]={'samples':len(rows),'status':'passed'}
    write_json(DATA/'noise_audit.json',result);print('All 12 fixed noisy test suites passed')
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Download ONLY original Q2RL Lift/Can/Square state datasets and optional assets.

Run from the project root in a network-enabled shell. Files are streamed to .part,
verified against Hugging Face LFS sha256, then atomically renamed. Existing files
are verified, never silently replaced. No task/data generation or conversions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def api(url):
    with urllib.request.urlopen(url,timeout=30) as r: return json.load(r)


def download(repo,kind,filename,destination):
    plural='datasets' if kind=='dataset' else 'models'
    metadata=api(f'https://huggingface.co/api/{plural}/{repo}')
    revision=metadata['sha']
    folder=filename.rsplit('/',1)[0] if '/' in filename else ''
    entries=api(f'https://huggingface.co/api/{plural}/{repo}/tree/{revision}'+('/'+folder if folder else ''))
    entry=next(x for x in entries if x['path']==filename)
    expected=entry.get('lfs',{}).get('oid')
    prefix='datasets/' if kind=='dataset' else ''
    url=f'https://huggingface.co/{prefix}{repo}/resolve/{revision}/{filename}'
    destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():
        partial=destination.with_suffix(destination.suffix+'.part')
        with urllib.request.urlopen(url,timeout=120) as r, partial.open('wb') as f:
            for chunk in iter(lambda:r.read(2**20),b''): f.write(chunk)
        candidate=partial
    else: candidate=destination
    h=hashlib.sha256()
    with candidate.open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''): h.update(chunk)
    actual=h.hexdigest()
    if expected and actual!=expected: raise RuntimeError(f'Checksum mismatch: {candidate}')
    if candidate!=destination: candidate.replace(destination)
    provenance=dict(repo=repo,repo_type=kind,revision=revision,file=filename,url=url,
                    sha256=actual,lfs_sha256=expected,size=destination.stat().st_size)
    destination.with_suffix(destination.suffix+'.source.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance),flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--tasks',nargs='+',choices=['lift','can','square'],default=['lift','can','square'])
    p.add_argument('--checkpoints',action='store_true'); args=p.parse_args()
    for task in args.tasks:
        file=f'{task}/ph/low_dim_v141.hdf5'
        download('theaiinstitute/q2rl_robomimic_datasets','dataset',file,'data/datasets/'+file)
        if args.checkpoints:
            download(f'theaiinstitute/gmm_robomimic_{task}_state','model','gmm.pth',f'data/reference/{task}/gmm.pth')


if __name__=='__main__': main()

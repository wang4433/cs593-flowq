"""Bounded nonlinear density diagnostics on held-out original demo states.

64-step RK4 is a numerical reference, not an analytic ground truth. This script
uses no simulator interactions and never updates the policy.
"""
import argparse
import json
import resource
import time
from pathlib import Path
import numpy as np
import torch
from flowq.data import load_demos, sha256
from flowq.flow import fingerprint, tensor
from flowq.runner import load_flow, seed, write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--flow',required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    if Path(args.out).exists(): raise FileExistsError(args.out)
    cfg=json.loads(Path(args.config).read_text()); seed(71)
    start=time.perf_counter(); flow,ck=load_flow(args.flow,cfg)
    before=fingerprint(flow)
    _,(vs,_)=load_demos(cfg,ck['manifest'])
    s=tensor(vs[np.linspace(0,len(vs)-1,32,dtype=int)])
    a=flow.sample(s,steps=64,generator=torch.Generator().manual_seed(12))
    reference=flow.log_prob(s,a,steps=64)
    convergence=[]
    for n in [2,4,8,16,32]:
        t=time.perf_counter(); lp=flow.log_prob(s,a,steps=n)
        convergence.append(dict(steps=n,max_error_vs_64=(lp-reference).abs().max().item(),
            mean_error_vs_64=(lp-reference).abs().mean().item(),batch32_seconds=time.perf_counter()-t))
    entropy=[]
    for n in [2,8,32]:
        estimates=[]; ses=[]; t=time.perf_counter()
        for k in range(32):
            h,se=flow.entropy(s[:1],samples=n,steps=16,
                            generator=torch.Generator().manual_seed(200+k))
            estimates.append(h.item()); ses.append(se.item())
        entropy.append(dict(samples=n,replicates=32,mean=float(np.mean(estimates)),
            variance=float(np.var(estimates,ddof=1)),mean_reported_se=float(np.mean(ses)),
            seconds=time.perf_counter()-t))
    traces=[]
    for n in [1,4,16]:
        estimates=[]; t=time.perf_counter()
        for k in range(32):
            estimates.append(flow.log_prob(s[:1],a[:1],steps=16,trace='hutchinson',trace_samples=n,
                generator=torch.Generator().manual_seed(900+k)).item())
        traces.append(dict(samples=n,replicates=32,bias_vs_exact64=float(np.mean(estimates)-reference[0]),
            variance=float(np.var(estimates,ddof=1)),seconds=time.perf_counter()-t))
    assert fingerprint(flow)==before and all(p.grad is None for p in flow.parameters())
    assert convergence[-1]['max_error_vs_64']<convergence[0]['max_error_vs_64']
    write_json(args.out,dict(flow_sha256=sha256(args.flow),state_source='32 held-out validation states',
        numerical_reference='RK4 64 exact; no analytic nonlinear oracle',convergence=convergence,
        entropy=entropy,hutchinson=traces,frozen_unchanged=True,
        max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        elapsed_seconds=time.perf_counter()-start))


if __name__=='__main__': main()

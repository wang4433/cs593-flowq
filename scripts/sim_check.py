"""REAL simulator integration check with UNTRAINED flow; not a research pilot.

No demonstration training, no synthetic demonstration files, no publishable BC
checkpoint. Tests the gate, value fitting, replay, learning, and evaluation.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from flowq.env import Task
from flowq.flow import Flow, freeze, fingerprint, mlp, tensor
from flowq.runner import Gate, evaluate, features, returns, seed, write_json
from flowq.rl import Learner, Replay


def main():
    p=argparse.ArgumentParser(); p.add_argument('--task',choices=['lift','square'],default='lift')
    args=p.parse_args(); cfg=json.loads(Path(f'configs/{args.task}_full.json').read_text())
    cfg.update(hidden=32,flow_hidden=32,batch_size=32,eval_episodes=1,entropy_samples=2,
               density={'steps':4,'trace':'exact','trace_samples':1})
    seed(0); start=time.perf_counter(); env=Task(cfg); first=env.reset(0)
    flow=freeze(Flow(len(first)-1,hidden=32)); original=fingerprint(flow)
    value=mlp(len(first),1,32); initial=Learner(len(first),7,cfg)
    rows=[]; rewards=[]; s=first; done=False
    g=Gate(flow,None,None,cfg,'A')
    while not done:
        a,log=g.act(s); ns,r,done,info=env.step(a)
        rows.append((s,a,r,ns,done)); rewards.append(r); s=ns
    warmup_success=info['success']
    x=features(tensor(np.stack([r[0] for r in rows])),flow)
    target=tensor(returns(rewards,cfg['gamma']))
    opt=torch.optim.Adam(value.parameters(),lr=.003)
    for _ in range(100):
        loss=((value(x).squeeze(-1)-target)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    freeze(value); value_hash=fingerprint(value)
    actions=tensor(np.stack([r[1] for r in rows])); raw=tensor(np.stack([r[0] for r in rows]))
    # Use the same estimator exercised by C, including independent entropy draws.
    lp=flow.log_prob(raw,actions,**cfg['density'])
    ent,_=flow.entropy(raw,samples=2,**cfg['density'])
    with torch.no_grad(): q=value(x).squeeze(-1)+cfg['q_alpha']*(lp+ent)
    initial.initialize(x,actions,q,20,32)
    results={}; decisions=[]
    try:
        for method in ['B','C']:
            seed(0); learner=copy.deepcopy(initial); before=fingerprint(learner.actor)
            replay=Replay(); gate=Gate(flow,value,learner,cfg,method)
            for ss,a,r,ns,d in rows:
                replay.add(features(tensor(ss),flow).numpy(),a,r,features(tensor(ns),flow).numpy(),d)
            s=env.reset(2000); done=False; n=0; counts={'flow':0,'rl':0}; decision=0; density=0
            while not done:
                a,log=gate.act(s); ns,r,done,info=env.step(a)
                replay.add(features(tensor(s),flow).numpy(),a,r,features(tensor(ns),flow).numpy(),done)
                losses=learner.update(replay.batch(32)); n+=1; s=ns; counts[log['selected']]+=1
                decision+=log['decision_seconds']; density+=log['density_seconds']
                decisions.append(dict(method=method,step=len(rows)+n,**log,**losses))
            assert fingerprint(learner.actor)!=before
            assert len(replay.rows)==len(rows)+n
            ev=evaluate(env,flow,value,learner,cfg,method)
            results[method]=dict(online_steps=n,success=info['success'],selection_counts=counts,
                updates=learner.updates,replay_size=len(replay.rows),evaluation=ev,
                mean_decision_ms=decision/n*1000,mean_density_ms=density/n*1000)
        assert results['B']['evaluation']['episodes'][0]['initial_observation_sha256']==results['C']['evaluation']['episodes'][0]['initial_observation_sha256']
        assert fingerprint(flow)==original and fingerprint(value)==value_hash
    finally: env.close()
    report=dict(label='UNTRAINED_FLOW_SIMULATOR_INTEGRATION_ONLY',task=args.task,seed=0,
                horizon=cfg['horizon'],warmup_steps=len(rows),warmup_success=bool(warmup_success),
                results=results,flow_hash=original,value_hash=value_hash,frozen_unchanged=True,
                elapsed_seconds=time.perf_counter()-start,config=cfg)
    write_json(f'artifacts/checks/{args.task}_sim.json',report)
    with Path(f'artifacts/checks/{args.task}_decisions.jsonl').open('w') as f:
        for row in decisions: f.write(json.dumps(row,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()

"""Load and freeze a trained Flow BC; evaluate original Robomimic closed loop.

No value model, critic, replay, likelihood estimator, or optimizer is required.
Saved trajectories are evaluation records and are never added to demonstrations.
"""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
import numpy as np
import torch
from flowq.data import sha256
from flowq.env import Task
from flowq.flow import fingerprint, tensor
from flowq.runner import load_flow, seed, write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--flow',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--seed',type=int,default=30000)
    args=p.parse_args()
    if args.episodes<1: p.error('--episodes must be positive')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(Path(args.config).read_text()); seed(cfg['seed'])
    flow,ck=load_flow(args.flow,cfg)
    before=fingerprint(flow); start=time.perf_counter()
    env=Task(cfg,ck['manifest']['env']); episodes=[]
    try:
        for ep in range(args.episodes):
            env_seed=args.seed+ep
            generator=torch.Generator().manual_seed(env_seed+100000)
            s=env.reset(env_seed); initial=hashlib.sha256(s.tobytes()).hexdigest()
            states=[s]; actions=[]; rewards=[]; latencies=[]; done=False
            while not done:
                tick=time.perf_counter()
                action=flow.sample(tensor(s)[None],steps=cfg['density']['steps'],generator=generator)[0].numpy()
                latencies.append(time.perf_counter()-tick)
                s,r,done,info=env.step(action)
                states.append(s); actions.append(action); rewards.append(info['raw_reward'])
            episode=dict(seed=env_seed,noise_seed=env_seed+100000,success=info['success'],
                timeout=info['timeout'],steps=len(actions),raw_reward_sum=float(sum(rewards)),
                initial_observation_sha256=initial,decision_seconds=sum(latencies),
                mean_decision_ms=float(np.mean(latencies)*1000),p95_decision_ms=float(np.percentile(latencies,95)*1000))
            episodes.append(episode)
            np.savez_compressed(out/f'episode_{ep:03d}.npz',states=np.stack(states),
                actions=np.stack(actions),raw_rewards=np.array(rewards),decision_seconds=latencies)
            write_json(out/'episodes.json',episodes)
            print(json.dumps(episode),flush=True)
        assert fingerprint(flow)==before
        assert not flow.training and all(not p.requires_grad and p.grad is None for p in flow.parameters())
    finally: env.close()
    n=len(episodes); successes=sum(e['success'] for e in episodes)
    # Wilson interval describes only these episode outcomes; one training seed.
    z=1.96; phat=successes/n; denom=1+z*z/n
    center=(phat+z*z/(2*n))/denom
    half=z*np.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/denom
    report=dict(task=cfg['task'],label='FROZEN_TRAINED_FLOW_BC_CLOSED_LOOP',
        checkpoint=str(Path(args.flow).resolve()),checkpoint_sha256=sha256(args.flow),
        flow_fingerprint=before,successes=successes,episodes_count=n,success_rate=phat,
        wilson_95_interval=[center-half,center+half],
        total_environment_steps=sum(e['steps'] for e in episodes),
        horizon=cfg['horizon'],integration_steps=cfg['density']['steps'],
        frozen_unchanged=True,evaluation_did_not_train=True,episodes=episodes,
        elapsed_seconds=time.perf_counter()-start,
        max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,config=cfg)
    write_json(out/'summary.json',report)
    print(json.dumps({k:report[k] for k in ['successes','episodes_count','success_rate','total_environment_steps','elapsed_seconds','frozen_unchanged']}),flush=True)


if __name__=='__main__': main()

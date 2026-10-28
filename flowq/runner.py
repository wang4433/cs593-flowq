"""Matched frozen-flow baselines A/B/C; all experiments require a trained BC file."""
import argparse
import copy
import hashlib
import json
import resource
import time
from pathlib import Path
import numpy as np
import torch
from .flow import Flow, tensor, freeze, fingerprint, finite, mlp
from .data import audit, load_demos, sha256
from .env import Task
from .rl import Learner, Replay


def write_json(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def seed(n):
    np.random.seed(n); torch.manual_seed(n); torch.set_num_threads(1)


def validate_config(cfg):
    if cfg['task'] not in ('lift','can','square'):
        raise ValueError('Only original Lift, Can, Square tasks are permitted')
    if cfg['entropy_samples']<2:
        raise ValueError('Use at least two entropy samples to report standard error')
    if cfg['initial_rollouts']*cfg['horizon']>cfg['total_steps']:
        raise ValueError('Worst-case complete initial rollouts exceed the interaction budget')
    for k in ('initial_rollouts','horizon','batch_size','bc_updates','value_updates','critic_init_updates',
              'eval_episodes','eval_interval','log_stride','density_batch_size'):
        if cfg[k]<1: raise ValueError(f'{k} must be positive')
    if not 0<cfg['gamma']<=1 or cfg['q_alpha']<0 or cfg['sac_alpha']<0:
        raise ValueError('Invalid discount or temperature')
    if cfg.get('replay_demonstrations',False):
        raise ValueError('This runner implements the no-offline-data online replay condition only')


def features(s,flow):
    return torch.cat([(s[...,:flow.obs_dim]-flow.mean)/flow.std,s[...,-1:]],-1)


def contract(cfg):
    return {k:cfg[k] for k in ('task','seed','horizon','gamma','reward_scale','reward_bias','q_alpha',
                               'sac_alpha','versions','density','entropy_samples','hidden','lr',
                               'train_mask','valid_mask','debug_demos')}


def load_flow(path,cfg):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if not ck.get('trained_on_original_demonstrations'):
        raise ValueError('Experiments require a checkpoint trained on original demonstrations')
    if ck['task']!=cfg['task']: raise ValueError('Flow task mismatch')
    for key in ('train_mask','valid_mask','debug_demos','boundary_epsilon'):
        if ck['config'][key]!=cfg[key]: raise ValueError(f'Flow preprocessing/data mismatch: {key}')
    flow=Flow(ck['obs_dim'],hidden=ck['hidden'],max_velocity=ck['config'].get('flow_max_velocity'))
    flow.load_state_dict(ck['flow']); freeze(flow)
    return flow,ck


def train_bc(cfg,out):
    start=time.perf_counter(); seed(cfg['seed'])
    if Path(out).exists(): raise FileExistsError(out)
    manifest=audit(cfg)
    (s,a),(vs,va)=load_demos(cfg,manifest)
    s,a,vs,va=map(tensor,(s,a,vs,va))
    flow=Flow(s.shape[-1],hidden=cfg['flow_hidden'],max_velocity=cfg.get('flow_max_velocity'))
    flow.mean.copy_(s.mean(0)); flow.std.copy_(s.std(0).clamp_min(cfg.get('obs_std_floor',1e-3)))
    opt=torch.optim.Adam(flow.parameters(),lr=cfg['lr'])
    logs=[]
    for step in range(cfg['bc_updates']):
        i=torch.randint(len(s),(cfg['batch_size'],))
        loss=flow.loss(s[i],a[i],cfg['boundary_epsilon'])
        finite(loss,'BC loss'); opt.zero_grad(); loss.backward(); opt.step()
        if step%cfg.get('bc_log_interval',100)==0 or step==cfg['bc_updates']-1:
            with torch.no_grad():
                j=torch.randint(len(vs),(min(1024,len(vs)),))
                vl=flow.loss(vs[j],va[j],cfg['boundary_epsilon'])
            logs.append(dict(step=step+1,loss=loss.item(),validation_fm_loss=vl.item()))
            print(json.dumps(logs[-1]),flush=True)
    Path(out).mkdir(parents=True,exist_ok=False)
    ck=dict(flow=flow.state_dict(),obs_dim=s.shape[-1],hidden=cfg['flow_hidden'],
            task=cfg['task'],manifest=manifest,config=cfg,trained_on_original_demonstrations=True)
    torch.save(ck,Path(out)/'flow.pt')
    write_json(Path(out)/'bc.json',dict(updates=logs,seconds=time.perf_counter()-start,
                                     flow_hash=fingerprint(flow),manifest=manifest,
                                     max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))


class Gate:
    def __init__(self,flow,value,learner,cfg,method,seed_offset=0):
        self.flow,self.value,self.learner,self.cfg,self.method=flow,value,learner,cfg,method
        self.proposal_rng=torch.Generator().manual_seed(cfg['seed']+100+seed_offset)
        self.density_rng=torch.Generator().manual_seed(cfg['seed']+200+seed_offset)

    def act(self,raw):
        start=time.perf_counter(); s=tensor(raw)[None]
        af=self.flow.sample(s,steps=self.cfg['density']['steps'],generator=self.proposal_rng)
        density_sec=0.0; lp=h=se=qf=qr=None
        selected='flow'
        if self.method!='A':
            with torch.no_grad():
                x=features(s,self.flow)
                ar,_=self.learner.actor.sample(x,deterministic=True)
                qr=self.learner.critic(x,ar).min(0).values.item()
            if self.method=='B':
                with torch.no_grad(): qf=self.learner.critic(x,af).min(0).values.item()
            else:
                ds=time.perf_counter()
                lp=self.flow.log_prob(s,af,generator=self.density_rng,**self.cfg['density']).item()
                ent,err=self.flow.entropy(s,samples=self.cfg['entropy_samples'],generator=self.density_rng,**self.cfg['density'])
                h,se=ent.item(),err.item()
                with torch.no_grad(): qf=self.value(x).item()+self.cfg['q_alpha']*(lp+h)
                density_sec=time.perf_counter()-ds
            finite([qf,qr],'candidate Q')
            # Exact upstream hard comparison. Equality chooses flow.
            if qr>qf: selected='rl'
            action=ar if selected=='rl' else af
        else: action=af
        return action.detach().numpy()[0],dict(selected=selected,q_flow=qf,q_rl=qr,
            q_difference=None if qr is None else qr-qf,log_prob=lp,entropy=h,entropy_se=se,
            density_seconds=density_sec,decision_seconds=time.perf_counter()-start)


def returns(rewards,gamma):
    g=0; result=[]
    for r in reversed(rewards): g=r+gamma*g; result.append(g)
    return result[::-1]


def qest(cfg,flow_path,out):
    seed(cfg['seed']); start=time.perf_counter(); flow,ck=load_flow(flow_path,cfg)
    Path(out).mkdir(parents=True,exist_ok=False)
    env=Task(cfg,ck['manifest']['env']); gate=Gate(flow,None,None,cfg,'A')
    states=[]; actions=[]; next_states=[]; rewards=[]; terminals=[]; targets=[]; episodes=[]
    initial_hash=fingerprint(flow)
    try:
        for ep in range(cfg['initial_rollouts']):
            s=env.reset(cfg['seed']+1000+ep); rs=[]; done=False
            while not done:
                a,_=gate.act(s); ns,r,done,info=env.step(a)
                states.append(s); actions.append(a); next_states.append(ns); rewards.append(r); terminals.append(done)
                rs.append(r); s=ns
            targets.extend(returns(rs,cfg['gamma']))
            episodes.append(dict(length=len(rs),success=info['success']))
            print(json.dumps(dict(stage='warmup',episode=ep,**episodes[-1])),flush=True)
    finally: env.close()
    if len(states)>cfg['total_steps']: raise ValueError('Warmup exceeds total training interaction budget')
    s,a,ns,g=map(tensor,(np.stack(states),np.stack(actions),np.stack(next_states),targets))
    x=features(s,flow); value=mlp(x.shape[-1],1,cfg['hidden'])
    opt=torch.optim.Adam(value.parameters(),lr=cfg['lr'])
    for _ in range(cfg['value_updates']):
        i=torch.randint(len(x),(cfg['batch_size'],)); loss=((value(x[i]).squeeze(-1)-g[i])**2).mean()
        finite(loss,'value loss')
        opt.zero_grad(); loss.backward(); opt.step()
    freeze(value)
    # Cache reference estimates once. Both B/C receive identical initial critic,
    # actor, warmup replay, and estimates. Only the online gate differs.
    q=[]; density_start=time.perf_counter()
    gen=torch.Generator().manual_seed(cfg['seed']+200)
    for offset in range(0,len(s),cfg['density_batch_size']):
        ss=s[offset:offset+cfg['density_batch_size']]; aa=a[offset:offset+cfg['density_batch_size']]
        lp=flow.log_prob(ss,aa,generator=gen,**cfg['density'])
        h,_=flow.entropy(ss,samples=cfg['entropy_samples'],generator=gen,**cfg['density'])
        with torch.no_grad(): q.append(value(features(ss,flow)).squeeze(-1)+cfg['q_alpha']*(lp+h))
    q=torch.cat(q); finite(q,'warmup reference estimates')
    density_seconds=time.perf_counter()-density_start
    learner=Learner(x.shape[-1],7,cfg)
    learner.initialize(x,a,q,cfg['critic_init_updates'],cfg['batch_size'])
    assert fingerprint(flow)==initial_hash
    torch.save(dict(value=value.state_dict(),actor=learner.actor.state_dict(),critic=learner.critic.state_dict(),
                    states=s,actions=a,next_states=ns,rewards=tensor(rewards),terminals=tensor(terminals),
                    mc_returns=g,q_reference=q,flow_sha256=sha256(flow_path),contract=contract(cfg),
                    config=cfg,episodes=episodes),Path(out)/'warmup.pt')
    with torch.no_grad():
        v = value(x).squeeze(-1)
        value_mse=((v-g)**2).mean().item()
        qs=learner.critic(x,a)
        critic_mse=((qs-q)**2).mean().item()
    write_json(Path(out)/'qest.json',dict(warmup_steps=len(s),episodes=episodes,value_fit_mse=value_mse,
        elapsed_seconds=time.perf_counter()-start,density_seconds=density_seconds,
        flow_unchanged=True,value_hash=fingerprint(value),flow_sha256=sha256(flow_path),
        q_mean=q.mean().item(),q_std=q.std().item(),critic_fit_mse=critic_mse,
        calibration_scope='training warmup states only; not held-out calibration',
        mc_return_mean=g.mean().item(),value_mean=v.mean().item(),
        critic_mean=qs.mean().item(),q_min=q.min().item(),q_max=q.max().item(),
        max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))


def load_warmup(cfg,flow_path,warmup_path):
    flow,ck=load_flow(flow_path,cfg)
    w=torch.load(warmup_path,map_location='cpu',weights_only=False)
    if w['contract']!=contract(cfg) or w['flow_sha256']!=sha256(flow_path):
        raise ValueError('Warmup/flow/config mismatch; matched comparison requires identical artifacts')
    learner=Learner(flow.obs_dim+1,7,cfg)
    learner.actor.load_state_dict(w['actor']); learner.critic.load_state_dict(w['critic'])
    learner.target.load_state_dict(w['critic'])
    value=mlp(flow.obs_dim+1,1,cfg['hidden']); value.load_state_dict(w['value']); freeze(value)
    return flow,ck,w,value,learner


def evaluate(env,flow,value,learner,cfg,method):
    before=[fingerprint(m) for m in (flow,value,learner.actor,learner.critic,learner.target)]
    modes=[m.training for m in (flow,value,learner.actor,learner.critic,learner.target)]
    updates=learner.updates; np_rng=np.random.get_state(); episodes=[]
    for m in (flow,value,learner.actor,learner.critic,learner.target): m.eval()
    try:
        with torch.random.fork_rng():
            gate=Gate(flow,value,learner,cfg,method,seed_offset=10000)
            for ep in range(cfg['eval_episodes']):
                s=env.reset(cfg['eval_seed']+ep); done=False; steps=0; flow_count=0; reward=0
                rs=[]; predicted_values=[]; decision_seconds=0.; density_seconds=0.
                init_hash=hashlib.sha256(s.tobytes()).hexdigest()
                while not done:
                    if method=='A':
                        with torch.no_grad():
                            predicted_values.append(value(features(tensor(s)[None],flow)).item())
                    a,log=gate.act(s); s,r,done,info=env.step(a)
                    reward+=r; steps+=1; flow_count+=log['selected']=='flow'
                    rs.append(r); decision_seconds+=log['decision_seconds']; density_seconds+=log['density_seconds']
                calibration={}
                if method=='A':
                    errors=np.array(predicted_values)-np.array(returns(rs,cfg['gamma']))
                    calibration=dict(frozen_value_return_mse=float(np.mean(errors**2)),
                                     frozen_value_return_bias=float(np.mean(errors)))
                episodes.append(dict(seed=cfg['eval_seed']+ep,success=info['success'],steps=steps,
                                     return_sum=reward,flow_fraction=flow_count/steps,
                                     initial_observation_sha256=init_hash,
                                     decision_seconds=decision_seconds,density_seconds=density_seconds,
                                     **calibration))
    finally:
        np.random.set_state(np_rng)
        for m,mode in zip((flow,value,learner.actor,learner.critic,learner.target),modes): m.train(mode)
    assert before==[fingerprint(m) for m in (flow,value,learner.actor,learner.critic,learner.target)]
    assert updates==learner.updates
    assert modes==[m.training for m in (flow,value,learner.actor,learner.critic,learner.target)]
    return dict(success_rate=float(np.mean([e['success'] for e in episodes])),episodes=episodes,
                evaluation_steps=sum(e['steps'] for e in episodes),evaluation_did_not_train=True)


def run(cfg,flow_path,warmup_path,method,out):
    seed(cfg['seed']); start=time.perf_counter()
    flow,ck,w,value,learner=load_warmup(cfg,flow_path,warmup_path)
    Path(out).mkdir(parents=True,exist_ok=False)
    env=Task(cfg,ck['manifest']['env']); eval_env=Task(cfg,ck['manifest']['env'])
    frozen=[fingerprint(flow),fingerprint(value)]
    initial_actor=fingerprint(learner.actor)
    initial_critic=fingerprint(learner.critic)
    replay=Replay(cfg['replay_capacity'])
    for s,a,r,ns,d in zip(w['states'],w['actions'],w['rewards'],w['next_states'],w['terminals']):
        replay.add(features(s,flow).numpy(),a.numpy(),r.item(),features(ns,flow).numpy(),d.item())
    warm_steps=len(w['states']); n=cfg['total_steps']-warm_steps
    if n<0: raise ValueError('Budget smaller than warmup')
    gate=Gate(flow,value,learner,cfg,method)
    evaluations=[]; episodes=[]; selected=0; decision=0.; density=0.; evaluation_steps=0
    try:
        initial=evaluate(eval_env,flow,value,learner,cfg,'A')
        initial.update(training_env_steps=0,measured_after_warmup_steps=warm_steps,label='initial_frozen_bc')
        evaluations.append(initial); evaluation_steps+=initial['evaluation_steps']
        s=env.reset(cfg['seed']+2000); ep=0; ep_steps=0; ep_success=False
        with (Path(out)/'decisions.jsonl').open('w') as log_file:
            for step in range(n):
                a,log=gate.act(s); ns,r,done,info=env.step(a)
                replay.add(features(tensor(s),flow).numpy(),a,r,features(tensor(ns),flow).numpy(),done)
                selected+=log['selected']=='flow'; decision+=log['decision_seconds']; density+=log['density_seconds']
                if method!='A':
                    for _ in range(cfg['utd']): losses=learner.update(replay.batch(cfg['batch_size']))
                else: losses={}
                ep_steps+=1; ep_success|=info['success']; s=ns
                total=warm_steps+step+1
                if step%cfg['log_stride']==0 or done:
                    log_file.write(json.dumps(dict(total_env_steps=total,reward=r,**log,**losses),allow_nan=False)+'\n')
                if done:
                    episodes.append(dict(total_env_steps=total,length=ep_steps,success=ep_success))
                    print(json.dumps(dict(method=method,stage='online',**episodes[-1])),flush=True)
                    ep+=1; ep_steps=0; ep_success=False; s=env.reset(cfg['seed']+2000+ep)
                if total%cfg['eval_interval']==0 or step==n-1:
                    ev=evaluate(eval_env,flow,value,learner,cfg,method)
                    ev.update(training_env_steps=total,label='online')
                    evaluations.append(ev); evaluation_steps+=ev['evaluation_steps']
        assert frozen==[fingerprint(flow),fingerprint(value)]
        assert len(replay.rows)==min(cfg['total_steps'],cfg['replay_capacity'])
        assert learner.updates==(0 if method=='A' else n*cfg['utd'])
        if method=='A':
            assert initial_actor==fingerprint(learner.actor)
            assert initial_critic==fingerprint(learner.critic)
        elif n:
            assert initial_actor!=fingerprint(learner.actor)
    finally: env.close(); eval_env.close()
    torch.save(dict(actor=learner.actor.state_dict(),critic=learner.critic.state_dict(),
                    method=method,flow_sha256=sha256(flow_path),warmup_sha256=sha256(warmup_path),
                    config=cfg,updates=learner.updates),Path(out)/'online.pt')
    write_json(Path(out)/'summary.json',dict(method=method,seed=cfg['seed'],training_env_steps=warm_steps+n,
        initial_rollout_steps=warm_steps,online_steps=n,evaluation_env_steps=evaluation_steps,
        all_simulator_steps=warm_steps+n+evaluation_steps,evaluations=evaluations,episodes=episodes,
        partial_online_episode_steps=ep_steps,updates=learner.updates,replay_size=len(replay.rows),
        flow_fraction=selected/max(n,1),decision_seconds=decision,density_seconds=density,
        elapsed_seconds=time.perf_counter()-start,frozen_unchanged=True,
        flow_fingerprint=frozen[0],value_fingerprint=frozen[1],
        initial_actor_fingerprint=initial_actor,initial_critic_fingerprint=initial_critic,
        final_actor_fingerprint=fingerprint(learner.actor),
        final_critic_fingerprint=fingerprint(learner.critic),
        all_executed_transitions_stored=True,
        flow_sha256=sha256(flow_path),warmup_sha256=sha256(warmup_path),
        max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,config=cfg))


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['audit-data','train-bc','qest','run','eval'])
    p.add_argument('--config',required=True); p.add_argument('--out',required=True)
    p.add_argument('--flow'); p.add_argument('--warmup'); p.add_argument('--online')
    p.add_argument('--method',choices=['A','B','C'],default='C'); args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text()); validate_config(cfg); seed(cfg['seed'])
    if args.command in ('qest','run','eval') and not args.flow: p.error('--flow is required')
    if args.command in ('run','eval') and not args.warmup: p.error('--warmup is required')
    if args.command=='audit-data': write_json(args.out,audit(cfg))
    elif args.command=='train-bc': train_bc(cfg,args.out)
    elif args.command=='qest': qest(cfg,args.flow,args.out)
    elif args.command=='run': run(cfg,args.flow,args.warmup,args.method,args.out)
    else:
        flow,ck,w,value,learner=load_warmup(cfg,args.flow,args.warmup)
        if args.online:
            online=torch.load(args.online,map_location='cpu',weights_only=False)
            if online['flow_sha256']!=sha256(args.flow) or online['warmup_sha256']!=sha256(args.warmup) or online['config']!=cfg:
                raise ValueError('Evaluation artifact mismatch')
            learner.actor.load_state_dict(online['actor']); learner.critic.load_state_dict(online['critic'])
            learner.target.load_state_dict(online['critic'])
            if args.method!=online['method']: raise ValueError('Evaluation method mismatch')
        env=Task(cfg,ck['manifest']['env'])
        try: write_json(args.out,evaluate(env,flow,value,learner,cfg,args.method))
        finally: env.close()


if __name__=='__main__': main()

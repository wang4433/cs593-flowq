"""Thin adapter around unmodified original robosuite tasks, not a custom task."""
import copy
import importlib.metadata
import numpy as np
from .data import TASKS, KEYS


class Task:
    def __init__(self,cfg,metadata=None):
        import robosuite
        from robosuite.controllers import load_controller_config
        for package in ('robosuite','mujoco'):
            actual=importlib.metadata.version(package)
            if actual!=cfg['versions'][package]:
                raise RuntimeError(f'{package}: expected {cfg["versions"][package]}, installed {actual}')
        self.cfg=cfg
        if metadata is None:
            # ONLY used by labeled simulator integration checks, never BC experiments.
            kwargs=dict(robots='Panda',controller_configs=load_controller_config(default_controller='OSC_POSE'))
        else:
            if metadata['env_name']!=TASKS[cfg['task']] or metadata.get('env_version')!='1.4.1':
                raise ValueError('Incompatible environment metadata')
            kwargs=copy.deepcopy(metadata['env_kwargs'])
        kwargs.update(has_renderer=False,has_offscreen_renderer=False,use_camera_obs=False,
                      reward_shaping=False,reward_scale=1.0,ignore_done=False,horizon=cfg['horizon'])
        self.env=robosuite.make(TASKS[cfg['task']],**kwargs)
        if self.env.action_dim!=7: raise ValueError('Expected 7D Panda action')
        self.t=0

    def observation(self,obs):
        s=np.concatenate([obs['object-state' if k=='object' else k] for k in KEYS]+[
            np.array([1-self.t/self.cfg['horizon']])]).astype(np.float64)
        if not np.isfinite(s).all(): raise FloatingPointError('Invalid simulator observation')
        return s

    def reset(self,seed):
        np.random.seed(seed)
        self.t=0
        return self.observation(self.env.reset())

    def step(self,a):
        a=np.asarray(a,dtype=np.float64)
        if a.shape!=(7,) or not np.isfinite(a).all() or np.any(np.abs(a[:6])>=1) or a[-1] not in (-1,0,1):
            raise ValueError('Invalid executed hybrid action; no inference clipping permitted')
        obs,raw,done,info=self.env.step(a)
        self.t+=1
        success=bool(self.env._check_success())
        timeout=self.t>=self.cfg['horizon']
        # Explicit finite-horizon objective: both success and horizon end the return.
        terminal=bool(success or timeout or done)
        reward=self.cfg['reward_scale']*float(raw)+self.cfg['reward_bias']
        if not np.isfinite(reward): raise FloatingPointError('Invalid reward')
        return self.observation(obs),reward,terminal,dict(success=success,timeout=timeout,raw_reward=float(raw))

    def close(self): self.env.close()

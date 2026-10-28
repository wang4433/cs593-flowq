"""Reward-return twin critic with a SAC-style entropy-regularized actor.

Entropy is deliberately excluded from critic backups to keep gate scores in
discounted task-reward units. q_alpha and sac_alpha are independent constants.
"""
import copy
import numpy as np
import torch
from torch import nn
from .flow import DTYPE, tensor, mlp, finite, log_jacobian


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        if act_dim != 7:
            raise ValueError('Only the original Panda 6D arm + gripper action is supported')
        self.net = mlp(obs_dim, 15, hidden)

    def distributions(self, s):
        out = self.net(s)
        return (torch.distributions.Normal(out[..., :6], out[..., 6:12].clamp(-5,1).exp()),
                torch.distributions.Categorical(logits=out[..., 12:]))

    def enumerate_actions(self, s):
        d, g = self.distributions(s)
        z = d.rsample()
        arm_lp = d.log_prob(z).sum(-1)-log_jacobian(z)
        actions = torch.cat([z.tanh()[:,None,:].expand(-1,3,-1),
                             torch.arange(-1,2,dtype=s.dtype).expand(len(s),3)[...,None]],-1)
        return actions, g.probs, arm_lp[:,None]+g.logits

    def sample(self, s, deterministic=False):
        d, g = self.distributions(s)
        z = d.mean if deterministic else d.rsample()
        grip = g.probs.argmax(-1) if deterministic else g.sample()
        return torch.cat([z.tanh(), (grip-1).to(z.dtype)[...,None]],-1), d.log_prob(z).sum(-1)-log_jacobian(z)+g.log_prob(grip)

    def log_prob(self, s, a):
        d, g = self.distributions(s)
        z = torch.atanh(a[..., :6])
        return d.log_prob(z).sum(-1)-log_jacobian(z)+g.log_prob(a[..., -1].long()+1)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.qs = nn.ModuleList([mlp(obs_dim+act_dim, 1, hidden) for _ in range(2)])

    def forward(self,s,a):
        return torch.stack([q(torch.cat([s,a],-1)).squeeze(-1) for q in self.qs])


class Replay:
    def __init__(self, capacity=1000000):
        self.capacity, self.rows, self.pos = capacity, [], 0

    def add(self,s,a,r,ns,terminal):
        for name,x in [('state',s),('action',a),('reward',r),('next_state',ns)]:
            finite(x,name)
        if np.max(np.abs(a[:6])) >= 1 or a[-1] not in (-1,0,1):
            raise ValueError('Invalid hybrid command')
        row = (np.array(s,copy=True),np.array(a,copy=True),float(r),np.array(ns,copy=True),float(terminal))
        if len(self.rows)<self.capacity: self.rows.append(row)
        else: self.rows[self.pos]=row
        self.pos=(self.pos+1)%self.capacity

    def batch(self,n):
        idx=np.random.randint(len(self.rows),size=n)
        return tuple(tensor(np.stack(x)) for x in zip(*(self.rows[i] for i in idx)))


class Learner:
    def __init__(self, obs_dim, act_dim, cfg):
        self.cfg=cfg
        self.actor=Actor(obs_dim,act_dim,cfg['hidden'])
        self.critic=Critic(obs_dim,act_dim,cfg['hidden'])
        self.target=copy.deepcopy(self.critic).requires_grad_(False)
        self.ao=torch.optim.Adam(self.actor.parameters(),lr=cfg['lr'])
        self.qo=torch.optim.Adam(self.critic.parameters(),lr=cfg['lr'])
        self.updates=0

    def update(self,batch):
        s,a,r,ns,done=batch
        with torch.no_grad():
            na,_=self.actor.sample(ns)
            y=r+self.cfg['gamma']*(1-done)*self.target(ns,na).min(0).values
        qloss=((self.critic(s,a)-y)**2).mean()
        self.qo.zero_grad(); qloss.backward(); self.qo.step()
        self.critic.requires_grad_(False)
        aa,prob,lp=self.actor.enumerate_actions(s)
        qs=self.critic(s[:,None,:].expand(-1,3,-1).reshape(-1,s.shape[-1]),aa.reshape(-1,7)).min(0).values.reshape(-1,3)
        aloss=(prob*(self.cfg['sac_alpha']*lp-qs)).sum(-1).mean()
        self.ao.zero_grad(); aloss.backward(); self.ao.step()
        self.critic.requires_grad_(True)
        with torch.no_grad():
            for t,p in zip(self.target.parameters(),self.critic.parameters()):
                t.lerp_(p,self.cfg['tau'])
        finite(qloss,'critic loss'); finite(aloss,'actor loss')
        self.updates+=1
        return {'critic_loss':qloss.item(),'actor_loss':aloss.item()}

    def initialize(self,s,a,q,steps,batch_size):
        for _ in range(steps):
            i=torch.randint(len(s),(batch_size,))
            loss=((self.critic(s[i],a[i])-q[i])**2).mean()
            self.qo.zero_grad(); loss.backward(); self.qo.step()
            loss=-self.actor.log_prob(s[i],a[i]).mean()
            self.ao.zero_grad(); loss.backward(); self.ao.step()
        self.target.load_state_dict(self.critic.state_dict())

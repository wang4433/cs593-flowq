"""Analytic conditional Gaussian oracle, separate from simulator experiments."""
import json
import math
import resource
import time
import unittest
from pathlib import Path
import numpy as np
import torch
from flowq.flow import Flow, DTYPE, freeze, fingerprint, log_jacobian, tensor
from flowq.runner import returns
from flowq.rl import Learner, Replay


class GaussianFlow(Flow):
    def __init__(self):
        super().__init__(2,2,16,hybrid=False)
        self.register_buffer('matrix',tensor([[.4,.35],[-.2,-.25]]))

    def velocity(self,t,x,s): return x@self.matrix.T+s*.3

    def oracle(self,s):
        exp=torch.matrix_exp(self.matrix)
        mean=(s*.3)@torch.linalg.solve(self.matrix,exp-torch.eye(2,dtype=DTYPE)).T
        cov=exp@exp.T
        return torch.distributions.MultivariateNormal(mean,covariance_matrix=cov)


class Numerics(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(9)

    def test_bounded_flow_avoids_ood_action_saturation(self):
        flow=freeze(Flow(3,hidden=16,max_velocity=6.0))
        s=tensor([[1000.,-1000.,1000.]])
        a=flow.sample(s,steps=16,generator=torch.Generator().manual_seed(7))
        self.assertTrue((a[:,:6].abs()<1).all())
        self.assertTrue(torch.isfinite(flow.log_prob(s,a,steps=16)).all())
        x=torch.zeros(1,6,dtype=DTYPE)
        self.assertTrue((flow.velocity(.5,x,s).abs()<=6).all())

    def test_gaussian_density_convergence_entropy_and_trace(self):
        flow=freeze(GaussianFlow()); before=fingerprint(flow)
        s=torch.randn(96,2,dtype=DTYPE); dist=flow.oracle(s)
        y=dist.sample(); a=y.tanh(); truth=dist.log_prob(y)-log_jacobian(y)
        errors=[]; latencies=[]
        for steps in [1,2,4,8,16,32]:
            t=time.perf_counter(); lp=flow.log_prob(s,a,steps=steps)
            latencies.append(time.perf_counter()-t); errors.append((lp-truth).abs().max().item())
        self.assertLess(errors[-1],1e-7)
        self.assertTrue(all(b<a for a,b in zip(errors,errors[1:])))
        # Unsquashed density recovers the Gaussian oracle too.
        self.assertLess((lp+log_jacobian(y)-dist.log_prob(y)).abs().max().item(),1e-7)
        state=tensor([[.4,-.6]])
        oracle=flow.oracle(state)
        nodes,weights=np.polynomial.hermite.hermgauss(100)
        yy=oracle.mean+tensor(nodes)[:,None]*torch.diagonal(oracle.covariance_matrix[0]).sqrt()*math.sqrt(2)
        entropy_truth=(oracle.entropy()+(tensor(weights)*log_jacobian(yy)).sum()/math.sqrt(math.pi)).item()
        h,se=flow.entropy(state,samples=8192,steps=16)
        self.assertLess(abs(h.item()-entropy_truth),5*se.item()+1e-5)
        trace_results=[]
        for k in [1,4,16]:
            estimates=torch.stack([flow.log_prob(s[:1],a[:1],steps=8,trace='hutchinson',trace_samples=k)
                                   for _ in range(128)]).flatten()
            bias=(estimates.mean()-truth[0]).item(); variance=estimates.var().item()
            self.assertLess(abs(bias),5*math.sqrt(variance/128)+1e-5)
            trace_results.append(dict(samples=k,bias=bias,variance=variance))
        self.assertLess(trace_results[-1]['variance'],trace_results[0]['variance']/4)
        self.assertEqual(before,fingerprint(flow))
        self.assertTrue(all(p.grad is None for p in flow.parameters()))
        report=dict(steps=[1,2,4,8,16,32],max_log_density_errors=errors,
                    batch96_density_seconds=latencies,entropy_oracle=entropy_truth,
                    entropy_mc=h.item(),entropy_mc_se=se.item(),trace=trace_results,
                    max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                    frozen_unchanged=True)
        Path('artifacts/checks').mkdir(parents=True,exist_ok=True)
        Path('artifacts/checks/numerics.json').write_text(json.dumps(report,indent=2)+'\n')

    def test_hybrid_probability_and_invalid_commands(self):
        flow=freeze(Flow(3,hidden=16)); s=torch.zeros(4,3,dtype=DTYPE)
        for p in flow.net.parameters(): p.data.zero_()
        for p in flow.gripper.parameters(): p.data.zero_()
        flow.gripper[-1].bias.data.copy_(tensor([.2,.3,.5]).log())
        a=torch.zeros(4,7,dtype=DTYPE); a[:,-1]=tensor([-1,0,1,1])
        truth=-3*math.log(2*math.pi)+tensor([.2,.3,.5,.5]).log()
        self.assertTrue(torch.allclose(flow.log_prob(s,a,steps=2),truth,atol=1e-10))
        a[0,-1]=.1
        with self.assertRaises(ValueError): flow.log_prob(s,a)
        a[0,-1]=1; a[0,0]=1
        with self.assertRaises(ValueError): flow.log_prob(s,a)

    def test_bc_gradient_and_offpolicy_updates(self):
        # Artificial tensors here test gradients only, never form a BC checkpoint.
        flow=Flow(3,hidden=16); s=torch.randn(32,3,dtype=DTYPE)
        a=torch.rand(32,7,dtype=DTYPE)*2-1
        loss=flow.loss(s,a); loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in flow.parameters()))
        cfg=dict(hidden=16,lr=.001,gamma=.99,sac_alpha=.01,tau=.005)
        learner=Learner(4,7,cfg); before=fingerprint(learner.actor)
        replay=Replay(20)
        for i in range(30):
            aa=np.zeros(7); aa[-1]=i%3-1
            replay.add(np.zeros(4),aa,-1,np.ones(4),i%5==0)
        self.assertEqual(len(replay.rows),20)
        learner.update(replay.batch(16))
        self.assertNotEqual(before,fingerprint(learner.actor))
        self.assertEqual(returns([1,2,3],.5),[2.75,3.5,3])


if __name__=='__main__': unittest.main()

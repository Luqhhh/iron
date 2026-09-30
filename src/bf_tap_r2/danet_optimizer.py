"""Quasi-hyperbolic Adam with coupled L2 decay and normalized EMA weights.

Algorithm source: facebookresearch/qhoptim e81dea3f2765780cf4fbb90b87b22ba7604b8625.
Independent implementation; see licenses/QHoptim-MIT.txt. No dependency install.
"""
import math
import torch


class QHAdam(torch.optim.Optimizer):
    def __init__(self,params,lr=.008,betas=(.9,.999),nus=(.8,1.),weight_decay=1e-5,eps=1e-8):
        if (not all(math.isfinite(v) for v in (lr,*betas,*nus,weight_decay,eps))
                or lr<0 or weight_decay<0 or eps<=0 or len(betas)!=2 or len(nus)!=2
                or any(not 0<=v<1 for v in betas) or any(not 0<=v<=1 for v in nus)):
            raise ValueError('Invalid QHAdam hyperparameters')
        super().__init__(params,dict(lr=lr,betas=tuple(betas),nus=tuple(nus),weight_decay=weight_decay,eps=eps))

    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():loss=closure()
        for group in self.param_groups:
            b1,b2=group['betas'];n1,n2=group['nus']
            for parameter in group['params']:
                if parameter.grad is None:continue
                if parameter.grad.is_sparse:raise ValueError('Dense QHAdam gradients required')
                gradient=parameter.grad.detach().clone()
                if not torch.isfinite(gradient).all():raise ValueError('Nonfinite QHAdam gradient')
                gradient.add_(parameter,alpha=group['weight_decay'])
                state=self.state[parameter]
                if not state:
                    state.update(beta1_weight=0.,beta2_weight=0.,exp_avg=torch.zeros_like(parameter),exp_avg_sq=torch.zeros_like(parameter))
                state['beta1_weight']=1+b1*state['beta1_weight'];state['beta2_weight']=1+b2*state['beta2_weight']
                adjusted1=1-1/state['beta1_weight'];adjusted2=1-1/state['beta2_weight']
                state['exp_avg'].mul_(adjusted1).add_(gradient,alpha=1-adjusted1)
                state['exp_avg_sq'].mul_(adjusted2).add_(gradient.square(),alpha=1-adjusted2)
                numerator=n1*state['exp_avg']+(1-n1)*gradient
                denominator=(n2*state['exp_avg_sq']+(1-n2)*gradient.square()).sqrt()+group['eps']
                parameter.addcdiv_(numerator,denominator,value=-group['lr'])
                if not torch.isfinite(parameter).all():raise ValueError('Nonfinite QHAdam parameter')
        return loss

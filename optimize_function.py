#from .optimizer import Optimizer, required
import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np
#import pdb
import math

from torch import nn


class SGD_layer(Optimizer):
    def __init__(self, params, lr=0.1, momentum=0, weight_decay=0, nesterov=False, bn_weight_decay='False'):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov, bn_weight_decay=bn_weight_decay)
        super(SGD_layer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_layer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            param = group['params']
            weight_decay = group['weight_decay']
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            bn_weight_decay = group['bn_weight_decay']

            for p in param:
                size = p.size()
                if p.grad is None:
                    continue
                grad = p.grad.data
                if len(p.size()) < 2:
                    p.data.add_(-lr * weight_decay, p.data)
                    p.data.add_(-lr, grad.data)
                    continue
                norm = p.norm(2, 1).unsqueeze(1) ** 2
                #grad_norm = grad.norm(2, 1).unsqueeze(1)
                #clip_by_norm(grad, 0.1)
                normalized_grad = norm * grad# / (grad_norm * norm + torch.ones_like(grad))
                if weight_decay > 0 and bn_weight_decay != 'True':
                    normalized_grad.add_(weight_decay, p.data)

                if weight_decay > 0 and bn_weight_decay == 'True':
                    normalized_grad.data.view(size[0], -1).add_(weight_decay, torch.matmul(
                        torch.matmul(p.data.view(size[0], -1), p.data.view(size[0], -1).t()),
                        p.data.view(size[0], -1)) - p.data.view(size[0], -1))
                if momentum != 0:
                    param_state = self.state[p]

                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(normalized_grad.data.view(size)).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum * torch.sqrt(norm)).add_(normalized_grad.data.view(size))

                    if nesterov:
                        normalized_grad = normalized_grad.add(momentum, buf)
                    else:
                        normalized_grad = buf

                p.data.add_(-lr, normalized_grad)

        return loss


class SGD_node(Optimizer):
    def __init__(self, params, lr=0.1, momentum=0, weight_decay=0, nesterov=False, bn_weight_decay='False'):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov, bn_weight_decay=bn_weight_decay)
        super(SGD_node, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_node, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            param = group['params']
            weight_decay = group['weight_decay']
            #group['lr'] *= 1.001
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            bn_weight_decay = group['bn_weight_decay']

            for p in param:
                size = p.size()
                if p.grad is None:
                    continue
                grad = p.grad.data
                if len(p.size()) <= 1:
                    p.data.add_(-lr * weight_decay, p.data)
                    p.data.add_(-lr, grad.data)
                    continue
                norm = torch.abs(p.data)
                # print(norm)
                # print('******************')
                grad_norm = torch.abs(grad)
                normalized_grad = norm * grad / (grad_norm * norm + torch.ones_like(grad))
                if weight_decay > 0 and bn_weight_decay != 'True':
                    p.data.add_(-lr * weight_decay, p.data)

                if weight_decay > 0 and bn_weight_decay == 'True':
                    p.data.view(size[0], -1).add_(-2 * lr * weight_decay, torch.matmul(
                        torch.matmul(p.data.view(size[0], -1), p.data.view(size[0], -1).t()),
                        p.data.view(size[0], -1)) - p.data.view(size[0], -1))

                if momentum != 0:
                    param_state = self.state[p]

                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(normalized_grad.data.view(size))

                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(normalized_grad.data.view(size))

                    if nesterov:
                        normalized_grad = normalized_grad.add(momentum, buf)

                    else:
                        normalized_grad = buf

                p.data.add_(-lr, normalized_grad)

        return loss


class SGD_spectral(Optimizer):
    def __init__(self, params, lr=0.1, weight_decay=0, momentum=0, lam=0, nesterov=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, lam=lam, nesterov=nesterov)
        super(SGD_spectral, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_spectral, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            param = group['params']
            weight_decay = group['weight_decay']
            lr = group['lr']
            lam = group['lam']
            momentum = group['momentum']
            nesterov = group['nesterov']
            #gpu = group['gpu']

            for p in param:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if len(p.size()) <= 1:
                    p.data.add_(-lr * weight_decay, p.data)
                    p.data.add_(-lr, grad.data)
                    continue
                size = p.size()
                if torch.cuda.is_available():
                    p_prime = torch.tensor(p.data.view([size[0], -1]), requires_grad=True).cuda()
                    _, eigenvalue, _ = torch.svd((torch.matmul(p_prime.t(), p_prime)))
                    eigenvalue = eigenvalue.cuda()
                else:
                    p_prime = torch.tensor(p.data.view([size[0], -1]), requires_grad=True)
                    _, eigenvalue, _ = torch.svd((torch.matmul(p_prime.t(), p_prime)))
                    eigenvalue = eigenvalue
                eigenvalue.max().backward()
                eig_grad = p_prime.grad.data
                grad_with_regularization = grad.data.view([size[0], -1]) + lam * eig_grad

                if weight_decay > 0:
                    p.data.add_(-lr * weight_decay, p.data)

                if momentum != 0:
                    param_state = self.state[p]

                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(grad_with_regularization.data.view(size))

                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(grad_with_regularization.data.view(size))

                    if nesterov:
                        grad_with_regularization = grad_with_regularization.add(momentum, buf)

                    else:
                        grad_with_regularization = buf

                    #p.data.add_(-lr, grad_with_regularization)
                p.data.add_(-lr, grad_with_regularization.data.view(size))
                del p_prime

        return loss


class SGDG(Optimizer):

    def __init__(self, params, lr=required, momentum=0, weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            lr = group['lr']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                size = p.size()
                if len(size) > 2:
                    unity = (p / (p.norm(2, 1).unsqueeze(1) + 1e-8)).data.view(size[0], -1)
                    g = p.grad.data.view(size[0], -1)

                    if weight_decay != 0:
                      # L=|Y'Y-I|^2/2=|YY'-I|^2/2+c
                      # dL/dY=2(YY'Y-Y)
                      g.add_(2 * weight_decay, torch.matmul(torch.matmul(unity.data, unity.t()), unity) - unity)

                    h = g - torch.sum(unity * g, dim=1, keepdim=True) * unity

                    h_hat = clip_by_norm(h, 0.1)

                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = torch.zeros(h_hat.size())
                        if p.is_cuda:
                          param_state['momentum_buffer'] = param_state['momentum_buffer'].cuda()

                    mom = param_state['momentum_buffer']
                    mom_new = momentum * mom - lr * h_hat

                    p.data.copy_(gexp(unity, mom_new).view(p.size()))
                    mom.copy_(gpt(unity, mom_new))

                else:
                    # This routine is from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
                    weight_decay = group['weight_decay']
                    nesterov = group['nesterov']

                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = d_p.clone()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                    p.data.add_(-lr, d_p)

        return loss


class SGD_average(Optimizer):
    def __init__(self, params, model_avg, lr=0.1, momentum=0, weight_decay=0, nesterov=False, step_number=0):
        defaults = dict(model_avg=model_avg, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov, step_number=step_number)
        super(SGD_average, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_average, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            lr = group['lr']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']
            step_number = group['step_number']
            param = group['params']
            param_avg = group['model_avg'].parameters()

            for p, p_avg in zip(param, param_avg):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(-lr, d_p)
                p_avg.data.copy_((p_avg.data * step_number + p.data) / (step_number + 1))

        return loss


class SGD_ensemble(Optimizer):
    def __init__(self, params, model_avg=None, lr=0.1, momentum=0, weight_decay=0, nesterov=False, step_number=0, epoch_end=False):
        defaults = dict(model_avg=model_avg, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov, step_number=step_number, epoch_end=epoch_end)
        super(SGD_ensemble, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_ensemble, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            lr = group['lr']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']
            step_number = group['step_number']
            param = group['params']
            param_avg = group['model_avg'].parameters()
            epoch_end = group['epoch_end']

            for p, p_avg in zip(param, param_avg):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(-lr, d_p)
                p_avg.data.copy_(p_avg.data + p.data)
                if epoch_end is True:
                    p.data.copy_(p_avg.data / step_number)
                    p_avg.data.copy_(torch.zeros_like(p_avg))

        return loss


def clip_by_norm(vec, norm):
    vec_norm = vec.norm(2, 1).unsqueeze(1)
    norm_vector = norm * torch.ones_like(vec_norm)
    vec.data.copy_(vec.data / vec_norm * torch.max(norm_vector, vec_norm))
    return vec


def norm(v):
  assert len(v.size()) == 2

  return v.norm(p=2, dim=1, keepdim=True)


def unit(v, eps=1e-8):
  vnorm = norm(v)

  return v/vnorm.add(eps), vnorm


def xTy(x, y):
  assert len(x.size())==2 and len(y.size())==2,'xTy'

  return torch.sum(x*y, dim=1, keepdim=True)


def gproj(y, g, normalize=False):
  if normalize:
    y = y / (y.norm(2, 1).unsqueeze(1) + 1e-8)
  yTg = xTy(y,g)

  return  g-(yTg*y)


def gexp(y, h, normalize=False):
  if normalize:
    y,_ = unit(y)
    h = gproj(y,h)

  u, hnorm = unit(h)
  return y*hnorm.cos() + u*hnorm.sin()


def gpt(y, h, normalize=False):
  if normalize:
    h = gproj(y, h)

  [u, unorm] = unit(h)
  return (u*unorm.cos() - y*unorm.sin())*unorm
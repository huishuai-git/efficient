from advertorch.attacks.iterative_projected_gradient import *
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import time


class W_PGDAttack(PGDAttack):
    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            ord=np.inf, targeted=False, predict_list=None, rep=False):
        """
        Create an instance of the PGDAttack.

        """
        super(W_PGDAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.ord = ord
        self.targeted = targeted
        self.predict_list = predict_list
        self.rep = rep
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

        assert is_float_or_torch_tensor(self.eps_iter)
        assert is_float_or_torch_tensor(self.eps)

    def perturb_new(self, x, y, delta_initial=None, delta_average=None, lip=0.01, exp='False'):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :param z: alpha for alpha-divergence
        :return: tensor containing perturbed inputs.
        """
        x, y = self._verify_and_process_inputs(x, y)

        # delta = Variable(torch.zeros_like(x), requires_grad=True)
        #
        # if self.rand_init:
        #     if torch.cuda.is_available():
        #         delta = torch.FloatTensor(*x.shape).uniform_(-self.eps, self.eps).cuda()
        #     else:
        #         delta = torch.FloatTensor(*x.shape).uniform_(-self.eps, self.eps)

        rval = perturb_iterative(
            x, y, delta_initial=delta_initial, delta_average=delta_average, lip=lip, exp=exp, predict=self.predict,
            nb_iter=self.nb_iter,eps=self.eps, eps_iter=self.eps_iter,
            loss_fn=self.loss_fn, minimize=self.targeted,
            ord=self.ord, clip_min=self.clip_min,
            clip_max=self.clip_max)

        return rval.data

class W_LinfPGDAttack(W_PGDAttack):
    """
    PGD Attack with order=Linf

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1., ord=np.inf,
            targeted=False, predict_list=None, rep=False):
        super(W_LinfPGDAttack, self).__init__(
            predict, loss_fn, eps, nb_iter, eps_iter, rand_init,
            clip_min, clip_max, ord, targeted, predict_list, rep)


def perturb_iterative(xvar, yvar, delta_initial, delta_average, lip, exp, predict, nb_iter, eps, eps_iter, loss_fn,
                      minimize=False, ord=np.inf, clip_min=0.0, clip_max=1.0):
    """
    Iteratively maximize the loss over the input. It is a shared method for
    iterative attacks including IterativeGradientSign, LinfPGD, etc.

    :param xvar: input data.
    :param yvar: input labels.
    :param predict: forward pass function.
    :param nb_iter: number of iterations.
    :param eps: maximum distortion.
    :param eps_iter: attack step size.
    :param loss_fn: loss function.
    :param delta_init: (optional) tensor contains the random initialization.
    :param minimize: (optional bool) whether to minimize or maximize the loss.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.

    :return: tensor containing the perturbed input.
    """
    # if delta_init is not None:
    #     delta = delta_init
    # else:
    #     delta = torch.zeros_like(xvar)

    if delta_initial is not None and delta_initial.size() == xvar.size():
        delta = delta_initial
    else:
        if torch.cuda.is_available():
            delta = torch.FloatTensor(*xvar.shape).uniform_(eps, eps).cuda()
        else:
            delta = torch.FloatTensor(*xvar.shape).uniform_(eps, eps)

    delta.requires_grad_(True)
    for ii in range(nb_iter):
        if exp == 'True':
            p = predict.forward_in_exp(xvar + delta)
        else:
            p = predict.forward(xvar + delta)

        loss = loss_fn.forward(p, yvar)
        if minimize:
            loss = -loss
        loss.backward()

        if delta_average is not None and delta_average.size() == delta.size():
            delta.grad.data = delta.grad.data + lip * (delta.data - delta_average)

        if ord == np.inf:
            grad_sign = delta.grad.detach().sign()
            delta.data.add_(eps_iter / (1 + eps_iter * lip) * grad_sign)
            delta.data = torch.min(torch.max(delta, -eps * torch.ones_like(delta)), eps * torch.ones_like(delta))
            x_per = xvar + delta
            x_per.data = torch.clamp(x_per.data, clip_min, clip_max)

        elif ord == 2:
            grad = delta.grad.detach()
            grad = normalize_by_pnorm(grad)
            delta.data.add_(eps_iter / (1 + eps_iter * lip) * xvar.size()[0] * grad)
            if eps is not None:
                delta.data = clamp_by_pnorm(delta.data, ord, eps)
            x_per = xvar + delta
            x_per.data = torch.clamp(x_per.data, clip_min, clip_max)
        else:
            error = "Only ord = inf and ord = 2 have been implemented"
            raise NotImplementedError(error)

        # x_per.grad.data.zero_()
        delta.grad.data.zero_()
        x_per.detach()

    return xvar + delta

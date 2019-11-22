import torch

import torch.nn as nn
import torch.nn.functional as F


class Renyi_divergence(nn.CrossEntropyLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(Renyi_divergence, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, inputs, target, alpha=0.5):
        probability = F.softmax(inputs, dim=1)
        indice = torch.tensor(range(inputs.size()[0])).long()
        pro = probability[indice, target].pow(1 - alpha).sum()

        return 1 / (alpha - 1) * torch.log(pro / inputs.size()[0])

    def forward_regularized(self, p, q, alpha=0.5):
        p_measure = F.softmax(p, dim=1)
        q_measure = F.softmax(q, dim=1)
        div = (p_measure.pow(alpha) * q_measure.pow(1 - alpha)).sum() / p.size()[0]

        return 1 / (alpha - 1) * torch.log(div)


class Hellinger(nn.CrossEntropyLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(Hellinger, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, inputs, target, alpha=0.5):
        probability = F.softmax(inputs, dim=1)
        indice = torch.tensor(range(inputs.size()[0])).long()
        pro = probability[indice, target].pow(1 - alpha).sum()

        return 1 - pro / inputs.size()[0]

    def forward_regularized(self, p, q, alpha=0.5):
        p_measure = F.softmax(p, dim=1)
        q_measure = F.softmax(q, dim=1)
        div = (p_measure.pow(alpha) * q_measure.pow(1 - alpha)).sum() / p.size()[0]

        return 1 - div


class Wasserstein(nn.CrossEntropyLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', init=None):
        super(Wasserstein, self).__init__(weight, size_average, reduce, reduction, init)
        self.ignore_index = ignore_index
        self.optimizer = init[0]
        self.f_function = init[1]
        self.W_dis = init[2]

    def forward(self, inputs, p, target):
        result = 0
        W_dis = self.W_dis

        p = F.softmax(p, dim=1)
        for i in range(100):
            output = self.f_function.forward(inputs)
            f_div = F.softmax(output, dim=1)
            loss = -W_dis.forward(p, target, f_div)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            self.optimizer.zero_grad()
            result = -loss
        #     print([result, i])
        # print('***************************')
        # for p in self.f_function.parameters():
        #     p.data.copy_(torch.randn_like(p))

        return result

    def forward_regularized(self, inputs, p, q):
        result = 0
        W_dis = self.W_dis

        output = self.f_function.forward(inputs)
        f_div = F.softmax(output, dim=1)

        p_measure = F.softmax(p, dim=1)
        q_measure = F.softmax(q, dim=1)

        for i in range(10):
            loss = -W_dis.forward(p_measure, q_measure, f_div)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            self.optimizer.zero_grad()
            result = -loss

        return result


class W_distance(nn.CrossEntropyLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(W_distance, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, p, target, f_div):
        indice = torch.tensor(range(p.size()[0])).long()
        E_p = (p * f_div).sum() / p.size()[0]
        E_q = f_div[indice, target].sum() / p.size()[0]

        return E_p - E_q

    def forward_regularized(self, p, q, f_div):
        E_p = (p * f_div).sum() / p.size()[0]
        E_q = (q * f_div).sum() / p.size()[0]

        return torch.abs(E_p - E_q)


class Jacobian(nn.CrossEntropyLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', criterion=nn.CrossEntropyLoss(), model=None):
        super(Jacobian, self).__init__(weight, reduce, reduction, criterion, model)
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.criterion = criterion
        self.model = model

    def forward(self, inputs, target, noise='uniform', sigma=1.0, repeat=1, attack=1.0):
        outputs = 0
        jacobian = 0
        inputs.requires_grad_(True)
        # for p in self.model.parameters():
        #     p.requires_grad_(False)
        for i in range(repeat):
            if noise == 'normal':
                noise_tmp = sigma * torch.randn_like(inputs).cuda()
            else:
                noise_tmp = torch.FloatTensor(*inputs.shape).uniform_(-sigma, sigma).cuda()

            outputs_tmp = self.model.forward(inputs + noise_tmp)
            outputs += outputs_tmp
            loss = self.criterion(outputs_tmp, target)
            loss.backward(retain_graph=True)
            p = 2 if attack == 2.0 else 1
            jacobian += inputs.grad.detach().data.view(inputs.size()[0], -1).norm(p, 1) ** p
            inputs.grad.data.zero_()

        # for p in self.model.parameters():
        #     p.requires_grad_(True)

        inputs.detach()
        inputs.requires_grad_(False)

        return outputs, jacobian.sum()

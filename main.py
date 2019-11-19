from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import time
import logging
import optimize_function
import json
import torchvision
import pickle
import torchvision.transforms as transforms
import copy
import numpy as np
from numpy import random
# import apex.amp as amp

from divergence import *
from adv_per import *
import os
import argparse

import models
#from utils import progress_bar

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', '-a', metavar='MODEL', default='resnet',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')
parser.add_argument('--depth', default=20, type=int,
                    help='depth for resnet')
parser.add_argument('--index', type=str, default=0,
                    help='experiment index')
parser.add_argument('--hyperindex', type=str, default=0)
parser.add_argument('--save_path', type=str, default='./results')
parser.add_argument('--bn_weight_decay', type=str, default='False')
##########################################################################
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default='[60, 120, 160]', type=str,
                    help='learning rate decay step')
parser.add_argument('--batch_size', default=128, type=int,
                    help='batch size')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr_decay_ratio', default=0.1, type=float,
                    help='learning rate decay ratio')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: None)')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--training_method', default='adv', type=str, metavar='training method',
                    help='The method of training')
parser.add_argument('--resume', default='True', type=str, help='resume from checkpoint')
parser.add_argument('--gpu', default=0, type=int, help='number of gpu')
parser.add_argument('--attack', default=np.inf, type=float, help='norm of attack')
parser.add_argument('--dynamic', default='True', type=str, help='dynamic lipschtiz constant')
parser.add_argument('--lip_1', default=0.01, type=float, help='the estimation of 1/lip constant for theta')
parser.add_argument('--lip_2', default=0.01, type=float, help='the estimation of 1/lip constant for x')
# parser.add_argument('--lip_exist', default='True', type=str, help='lip or not')
parser.add_argument('--exp', default='False', type=str, help='in expectation or not')
parser.add_argument('--noise', default='uniform', type=str, help='noise type')
parser.add_argument('--sigma', default=8/255, type=float, help='the scale of noise')
parser.add_argument('--cubic_x', default='False', type=str, help='cubic regularizer for x')
parser.add_argument('--cubic_theta', default='False', type=str, help='cubic regularizer for theta')
parser.add_argument('--full_avg', default='False', type=str, help='full average theta or not')
parser.add_argument('--eps_iter', default=10/255, type=float, help='lr for adv')


def main():
    global args
    torch.manual_seed(0)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.save = args.optimizer + '_' + args.model + '_' + args.dataset + '_' + '_' + args.training_method + '_' \
                + str(args.index) + '_' + str(args.attack) + '_' + str(args.exp) + '_' + args.noise + '_' \
                + str(args.sigma)
    save_path = os.path.join(args.save_path, str(args.hyperindex))
    save_path = os.path.join(save_path, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logging.info("creating model %s", args.model)
    model = models.__dict__[args.model]
    if args.dataset == 'cat_and_dog':
        model_config = {'input_size': 32, 'dataset': args.dataset, 'depth': args.depth, 'num_classes': 2}
    else:
        model_config = {'input_size': 32, 'dataset': args.dataset, 'depth': args.depth}

    model = model(**model_config)

    if device == 'cuda':
        model = model.to(device)
        #model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    print("created model with configuration: %s", model_config)
    print("run arguments: %s", args)
    with open(save_path+'/log.txt', 'a') as f:
        f.writelines(str(args) + '\n')

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    num_parameters = sum([l.nelement() for l in model.parameters()])
    print("number of parameters: {}".format(num_parameters))
    # Data
    print('==> Preparing data..')
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.24703225141799082, 0.24348516474564, 0.26158783926049628)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.24703225141799082, 0.24348516474564, 0.26158783926049628)),
        ])

    elif args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5070588235294118, 0.48666666666666664, 0.4407843137254902),
                                 (0.26745098039215687, 0.2564705882352941, 0.27607843137254906)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070588235294118, 0.48666666666666664, 0.4407843137254902),
                                 (0.26745098039215687, 0.2564705882352941, 0.27607843137254906)),
        ])
    elif args.dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.5),
            #                      (0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0:wq.5),
            #                      (0.5)),
        ])
    else:
        raise ValueError('No such dataset')

    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    elif args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError('There is no such dataset')
    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    #create optimzier

    criterion = nn.CrossEntropyLoss()
    adv_criterion_train = nn.CrossEntropyLoss()
    adv_criterion_test = nn.CrossEntropyLoss()

    if args.optimizer == 'SGD_layer':
        optimizer = optimize_function.SGD_layer(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, bn_weight_decay=args.bn_weight_decay)
    elif args.optimizer == 'SGD_spectral':
        optimizer = optimize_function.SGD_spectral(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, lam=args.lam)
    elif args.optimizer == 'SGD_node':
        optimizer = optimize_function.SGD_node(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, bn_weight_decay=args.bn_weight_decay)
    elif args.optimizer == 'SGDG':
        optimizer = optimize_function.SGDG(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        # model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=1.0, master_weights=False)

    # theta_tmp, grad_tmp, delta_inital, delta_average = [], [], [], []
    delta_initial, delta_average, theta_average = [], [], []
    # for inputs, targets in trainloader:
    #     inputs, targets = inputs.to(device), targets.to(device)
    #     loss = criterion(model.forward(inputs), targets)
    #     loss.backward()
    #

    for p in model.parameters():
        theta_average.append(torch.zeros_like(p).to(device).data.copy_(p.data))
        # grad_tmp.append(torch.zeros_like(p).to(device).data.copy_(p.grad.data))
    #
    #     break

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        delta_initial.append(torch.zeros_like(inputs))
        delta_average.append(torch.zeros_like(inputs))

    if args.resume == 'True':
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        if os.path.exists(save_path + '/ckpt.t7'):#, 'Error: no results directory found!'
            checkpoint = torch.load(save_path + '/ckpt.t7')
            model.load_state_dict(checkpoint['model'])
            best_acc = checkpoint['best_acc']
            start_epoch = checkpoint['epoch'] + 1
            delta_initial = checkpoint['delta_initial']
            delta_average = checkpoint['delta_average']
            theta_average = checkpoint['theta_average']

    result_vector = []
    for epoch in range(start_epoch, args.epochs):
        power = sum(epoch >= int(i) for i in [30, 60, 90])
        lr = args.lr * pow(args.lr_decay_ratio, power)
        strat_time = time.time()
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('create an optimizer with learning rate as:', lr)
        training_loss, training_acc = train(dataloader=trainloader, training=True, criterion=criterion, model=model,
                                            device=device, optimizer=optimizer, epoch=epoch,
                                            adv_criterion_train=adv_criterion_train,
                                            adv_criterion_test=adv_criterion_test, delta_average=delta_average,
                                            delta_initial=delta_initial, theta_average=theta_average)

        if epoch > 59:
            test_loss, test_acc = train(dataloader=testloader, training=False, criterion=criterion, model=model,
                                        device=device, optimizer=optimizer,
                                        adv_criterion_train=adv_criterion_train, adv_criterion_test=adv_criterion_test)
        else:
            test_loss, test_acc = 0.0, 0.0
        print('Training time for each epoch is %g, optimizer is %s, model is %s' % (
            time.time() - strat_time, args.optimizer, args.model + str(args.depth)))
        if test_acc > best_acc:
            best_acc = test_acc

        print('\n Epoch: {0}\t'
                     'Training Loss {training_loss:.4f} \t'
                     'Training Acc {training_acc:.3f} \t'
                     'Test Loss {test_loss:.4f} \t'
                     'Test Acc {test_acc:.3f} \n'
                     .format(epoch + 1, training_loss=training_loss, training_acc=training_acc,
                             test_loss=test_loss, test_acc=test_acc))
        with open(save_path + '/log.txt', 'a') as f:
            f.write(str('\n Epoch: {0}\t'
                     'Training Loss {training_loss:.4f} \t'
                     'Training Acc {training_acc:.3f} \t'
                     'Test Loss {test_loss:.4f} \t'
                     'Test Acc {Test_acc:.3f} \n'
                     .format(epoch + 1, training_loss=training_loss, training_acc=training_acc,
                             test_loss=test_loss, Test_acc=test_acc)) + '\n')
        state = {
            'model': model.state_dict(),
            'best_acc': best_acc,
            'epoch': epoch,
            'delta_initial': delta_initial,
            'delta_average': delta_average,
            'theta_average': theta_average
        }

        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        torch.save(state, save_path + '/ckpt.t7')
        if os.path.exists(save_path + '/result_vector'):
            with open(save_path + '/result_vector', 'rb') as fp:
                result_vector = pickle.load(fp)
        result_vector.append([epoch, training_loss, training_acc, test_loss, test_acc])
        with open(save_path + '/result_vector', 'wb') as fp:
            pickle.dump(result_vector, fp)


# Training
def train(dataloader, training, criterion, model, device, optimizer, epoch=0, adv_criterion_train=None,
          adv_criterion_test=None, delta_average=None, delta_initial=None, theta_average=None):

    training_loss = 0
    test_loss = 0
    correct = 0
    total = 0
    batch_num = 0
    exp = [args.exp, args.sigma, args.noise]
    if training:
        model.train()
        if args.training_method == 'adv':
            if args.cubic_theta == 'True':
                if args.full_avg == 'True':
                    iteration = epoch * len(dataloader)
                else:
                    iteration = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            lr = optimizer.param_groups[0]['lr']
            inputs, targets = inputs.to(device), targets.to(device)

            if args.training_method == 'adv':
                for p in model.parameters():
                    p.requires_grad_(False)
                if args.dataset == 'mnist':
                    if args.attack == 2.0:
                        ord, eps, eps_iter = 2, 1.0, 0.3
                    else:
                        ord, eps, eps_iter = np.inf, 0.3, 0.1
                    adversary = W_LinfPGDAttack(
                        model, loss_fn=adv_criterion_train, eps=eps,
                        nb_iter=40, eps_iter=eps_iter, rand_init=True, clip_min=0.0, clip_max=1.0, ord=ord,
                        targeted=False, rep=False)

                elif 'cifar' in args.dataset:
                    if args.attack == 2.0:
                        ord, eps, eps_iter = 2, 1.0, 1.57
                    else:
                        ord, eps, eps_iter = np.inf, 8/255, args.eps_iter
                    adversary = W_LinfPGDAttack(
                        model, loss_fn=adv_criterion_train, eps=eps,
                        nb_iter=1, eps_iter=eps_iter, rand_init=True, clip_min=0.0, clip_max=1.0, ord=ord,
                        targeted=False, rep=False)

                else:
                    raise ValueError('There is no such dataset.')

                if args.cubic_x == 'True':
                    adv_untargeted = adversary.perturb_new(inputs, targets, delta_initial[batch_idx],
                                                           delta_average[batch_idx], args.lip_2, args.exp)
                    if adv_untargeted.size() == delta_initial[batch_idx].size():
                        delta_average[batch_idx].data = (delta_average[batch_idx].data * epoch + adv_untargeted.data
                                                         - inputs.data) / (epoch + 1)
                        delta_initial[batch_idx].data = adv_untargeted.data - inputs.data

                else:
                    adv_untargeted = adversary.perturb_new(inputs, targets, None, None, 0)

                for p in model.parameters():
                    p.requires_grad_(True)

                if args.exp == 'True':
                    # outputs = model.forward_in_exp(adv_untargeted, rep=20, sigma=exp[1], noise=exp[2])
                    outputs = model.forward(adv_untargeted)
                else:
                    outputs = model.forward(adv_untargeted)

                if args.cubic_theta == 'True':
                    # if args.dynamic == 'True':
                    #     grad_norm_tmp = 0
                    #     theta_norm_tmp = 0
                    #     loss = criterion(outputs, targets)
                    #     loss.backward()
                    #
                    #     for p, q, r in zip(grad_tmp, theta_tmp, model.parameters()):
                    #         grad_norm_tmp += torch.pow(torch.dist(p, r.grad), 2).sum()
                    #         theta_norm_tmp += torch.pow(torch.dist(q, r), 2).sum()
                    #     lip = 2 * grad_norm_tmp / (theta_norm_tmp + 1e-8)
                    #     lip = torch.clamp(lip, 0, 0.1)

                       # print(lip)

                        # for p, q, r in zip(model.parameters(), theta_tmp, grad_tmp):
                        #     r.data.copy_(p.grad.data)
                        #     # p.grad.data.add_(-p.data / (2 * lr) + q.data / (2 * lr))
                        #     p.grad.data.add_(0.01 * q.data)
                        #     q.data = (q.data * iteration + p.data) / (iteration + 1)
                        #
                        # iteration += 1

                    loss = criterion(outputs, targets)
                    loss.backward()
                    for p, q in zip(model.parameters(), theta_average):
                        if p.grad is not None:
                            print('check')
                            p.grad.data = p.grad.data + args.lip_1 * (p.data - q.data)
                            q.data = (q.data * iteration + p.data) / (iteration + 1)

                    iteration += 1

                    optimizer.param_groups[0]['lr'] = lr / (1 + lr * args.lip_1)
                    optimizer.step()
                    optimizer.param_groups[0]['lr'] = lr
                    optimizer.zero_grad()

                else:
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            else:
                outputs = model.forward(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            training_loss += loss.item()
            _, predicted_ad = outputs.max(1)
            total += targets.size(0)
            correct += predicted_ad.eq(targets).sum().item()
            batch_num = batch_idx + 1

        training_loss /= batch_num
        training_acc = 100 * correct / total

        return training_loss, training_acc

    else:
        model.eval()
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            for p in model.parameters():
                p.requires_grad_(False)
            if args.dataset == 'mnist':
                if args.attack == 2.0:
                    ord, eps, eps_iter = 2, 1.0, 0.3
                else:
                    ord, eps, eps_iter = np.inf, 0.3, 0.1
                adversary = W_LinfPGDAttack(
                    model, loss_fn=adv_criterion_test, eps=eps,
                    nb_iter=40, eps_iter=eps_iter, rand_init=True, clip_min=0.0, clip_max=1.0, ord=ord,
                    targeted=False, rep=False)

            elif 'cifar' in args.dataset:
                if args.attack == 2.0:
                    ord, eps, eps_iter = 2, 1.0, 0.25
                else:
                    ord, eps, eps_iter = np.inf, 8/255, 2/255
                adversary = W_LinfPGDAttack(
                    model, loss_fn=adv_criterion_test, eps=eps,
                    nb_iter=20, eps_iter=eps_iter, rand_init=True, clip_min=0.0, clip_max=1.0, ord=ord,
                    targeted=False, rep=False)

            else:
                raise ValueError('There is no such dataset.')

            adv_untargeted = adversary.perturb_new(inputs, targets, None, None, 0, exp)
            # adv_untargeted = inputs
            for p in model.parameters():
                p.requires_grad_(True)
            outputs_ad = model.forward_in_exp(adv_untargeted, rep=20, sigma=exp[1], noise=exp[2])
            loss = criterion.forward(outputs_ad, targets)
            optimizer.zero_grad()
            test_loss += loss.item()
            _, predicted = outputs_ad.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            batch_num = batch_idx + 1

        test_loss /= batch_num
        test_acc = 100 * correct / total

        return test_loss, test_acc


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    set_seed(888)
    main()

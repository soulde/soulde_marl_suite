import numpy as np
import os
import torch
import random
import math


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def soft_update_model_weight(net, target_net, tau):
    # 复制网络参数
    target_net_state_dict = target_net.state_dict()
    net_state_dict = net.state_dict()
    for key in net_state_dict:
        target_net_state_dict[key] = net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau)
    target_net.load_state_dict(target_net_state_dict)


def epsilon_greedy(count, eps_start=0.9, eps_end=0.05, eps_decay=1000):
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * count / eps_decay)
    return sample > eps_threshold

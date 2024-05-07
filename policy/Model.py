from torch.nn import Module, Sequential, Linear, ReLU, Sigmoid, Tanh, Softplus, Parameter
import torch
import numpy as np
from utils import layer_init
from torch import Tensor


class ActorPPOContinuous(Module):
    def __init__(self, feature_in, feature_out, action_scale, action_bias):
        super(ActorPPOContinuous, self).__init__()
        self.backbone = Sequential(layer_init(Linear(feature_in, 128)),
                                   Tanh(),
                                   layer_init(Linear(128, 64)),
                                   Tanh(),
                                   layer_init(Linear(64, feature_out)),
                                   Tanh()
                                   )
        self.state_avg = Parameter(torch.zeros((feature_in,)), requires_grad=False)
        self.state_std = Parameter(torch.ones((feature_out,)), requires_grad=False)

        self.action_std_log = Parameter(torch.zeros((1, feature_out)), requires_grad=True)  # trainable parameter

        self.action_scale = Parameter(torch.tensor(action_scale), requires_grad=False)

        self.action_bias = Parameter(torch.tensor(action_bias), requires_grad=False)

    def state_norm(self, state: Tensor) -> Tensor:
        return (state - self.state_avg) / self.state_std

    def forward(self, x, action=None):
        mean = self.backbone(x)
        std = self.action_std_log.exp()
        dist = torch.distributions.Normal(mean, std)

        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = dist.entropy()
        action = action * self.action_scale + self.action_bias
        return action, log_prob, entropy


class Actor(Module):
    def __init__(self, feature_in, feature_out, action_scale, action_bias):
        super(Actor, self).__init__()
        self.backbone = Sequential(layer_init(Linear(feature_in, 128)),
                                   ReLU(),
                                   layer_init(Linear(128, 128)),
                                   ReLU(),
                                   )
        self.mean_head = Sequential(layer_init(Linear(128, feature_out)),
                                    Tanh())

        self.register_buffer(
            "action_scale", torch.tensor(action_scale, dtype=torch.float32, requires_grad=False)
        )
        self.register_buffer(
            "action_bias", torch.tensor(action_bias, dtype=torch.float32, requires_grad=False)
        )

    def forward(self, x):
        feature = self.backbone(x)
        mean = self.mean_head(feature)
        return mean * self.action_scale + self.action_bias


class ActorSAC(Module):
    def __init__(self, feature_in, feature_out, action_scale, action_bias):
        super(ActorSAC, self).__init__()
        self.backbone = Sequential(layer_init(Linear(feature_in, 128)),
                                   ReLU(),
                                   layer_init(Linear(128, 128)),
                                   ReLU(),
                                   )
        self.mean_head = Sequential(layer_init(Linear(128, feature_out)),
                                    Tanh())

        self.std_head = Sequential(layer_init(Linear(128, feature_out)),
                                   Softplus())
        self.register_buffer(
            "action_scale", torch.tensor(action_scale, dtype=torch.float32, requires_grad=False)
        )
        self.register_buffer(
            "action_bias", torch.tensor(action_bias, dtype=torch.float32, requires_grad=False)
        )

    def forward(self, x):
        feature = self.backbone(x)
        mean = self.mean_head(feature)
        std = self.std_head(feature)
        dist = torch.distributions.Normal(mean, std)
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample)
        action_norm = torch.tanh(normal_sample)
        log_prob = log_prob - torch.log(1 - torch.tanh(action_norm).pow(2) + 1e-7)
        action = action_norm * self.action_scale + self.action_bias
        return action, log_prob


class CriticTwin(Module):
    def __init__(self, state_in, action_in, q_in):
        super(CriticTwin, self).__init__()

        self.model = Sequential(layer_init(Linear((state_in + action_in) * q_in, 128)),
                                ReLU(),
                                layer_init(Linear(128, 128)),
                                ReLU(),
                                layer_init(Linear(128, 2 * q_in)))

    def forward(self, x, a):
        feature = torch.cat([x, a], dim=-1).view(x.shape[0], -1)
        return self.model(feature).reshape(x.shape[0], -1, 2)


class Critic(Module):
    def __init__(self, state_in, action_in, q_in):
        super(Critic, self).__init__()
        self.model = Sequential(layer_init(Linear((state_in + action_in) * q_in, 128)),
                                ReLU(),
                                layer_init(Linear(128, 128)),
                                ReLU(),
                                layer_init(Linear(128, q_in)))

    def forward(self, x, a):
        feature = torch.cat([x, a], dim=-1).view(x.shape[0], -1)
        return self.model(feature).reshape(x.shape[0], -1, 1)


class CriticPPO(Module):
    def __init__(self, state_in, action_in, q_in):
        super(CriticPPO, self).__init__()
        self.model = Sequential(layer_init(Linear(state_in * q_in, 128)),
                                ReLU(),
                                layer_init(Linear(128, 128)),
                                ReLU(),
                                layer_init(Linear(128, q_in)))

    def forward(self, x, a):
        return self.model(x.reshape(x.shape[0], -1)).reshape(x.shape[0], -1, 1)


class CriticPPOFork(Module):
    def __init__(self, state_in, action_in, q_in):
        super(CriticPPOFork, self).__init__()
        self.a_head = Sequential(layer_init(Linear(q_in * action_in, 64)), )
        self.s_head = Sequential(layer_init(Linear(state_in * action_in, 64)), )
        self.model = Sequential(
            ReLU(),
            layer_init(Linear(128, 64)),
            ReLU(),
            layer_init(Linear(64, q_in)))

    def forward(self, x, a):
        a_feature = self.a_head(a)
        s_feature = self.s_head(x)
        feature = torch.cat([a_feature, s_feature], dim=-1)
        # feature = torch.cat([x.reshape(x.shape[0],-1),a.reshape(a.shape[0],-1)], dim=-1)
        return self.model(feature).reshape(x.shape[0], -1, 1)

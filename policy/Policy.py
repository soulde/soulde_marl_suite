from abc import abstractmethod
import torch
from torch.nn import SmoothL1Loss
import os
from torch.optim import AdamW


class Policy(object):
    def __init__(self, actor_type, critic_type, n_agents, actor_state_size, actor_action_size, param, action_scale=1,
                 action_bias=0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.critic = critic_type(actor_state_size, actor_action_size, n_agents).to(self.device)
        self.critic_target = critic_type(actor_state_size, actor_action_size, n_agents).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actors = [actor_type(actor_state_size, actor_action_size, action_scale, action_bias).to(self.device) for _
                       in range(n_agents)]
        self.actors_target = [actor_type(actor_state_size, actor_action_size, action_scale, action_bias).to(self.device)
                              for _ in range(n_agents)]
        self.gamma = param.gamma
        self.tau = param.tau
        self.policy_update_frequency = param.policy_update_frequency
        self.target_network_frequency = param.target_network_frequency
        self.max_grad_norm = param.max_grad_norm
        self.train_count = 0
        self.critic_loss = SmoothL1Loss()

        self.actor_optimizer = AdamW(sum((list(actor.parameters()) for actor in self.actors), []),
                                     lr=param.actor_lr)
        self.critic_optimizer = AdamW(self.critic.parameters(), lr=param.critic_lr)

    def save_model(self, directory: str, name: str):
        for i, (actor, actor_target) in enumerate(zip(self.actors, self.actors_target)):
            torch.save(actor.state_dict(), os.path.join(directory, 'actor_{}_{}.pth'.format(i, name)))
            torch.save(actor_target.state_dict(), os.path.join(directory, 'actor_target_{}_{}.pth'.format(i, name)))

        torch.save(self.critic.state_dict(), os.path.join(directory, 'critic_{}.pth'.format(name)))
        torch.save(self.critic_target.state_dict(), os.path.join(directory, 'critic_target_{}.pth'.format(name)))

    def load_model(self, directory: str, name: int):
        for i, (actor, actor_target) in enumerate(zip(self.actors, self.actors_target)):
            actor.load_state_dict(torch.load(os.path.join(directory, 'actor_{}_{}.pth'.format(i, name))))
            actor_target.load_state_dict(
                torch.load(os.path.join(directory, 'actor_target_{}_{}.pth'.format(i, name))))

            self.critic.load_state_dict(torch.load(os.path.join(directory, 'critic_{}.pth'.format(name))))
            self.critic_target.load_state_dict(torch.load(os.path.join(directory, 'critic_target_{}.pth'.format(name))))

    def get_action(self, states, use_target=False):
        actions = tuple(
            actor(states[:, i]) for i, actor in enumerate(self.actors_target if use_target else self.actors))
        actions = torch.stack(actions, dim=1)
        return actions

    def get_value(self, states, actions, use_target=False):
        qf = self.critic_target(states, actions) if use_target else self.critic(states, actions)
        return qf

    @abstractmethod
    def train(self, batch_data):
        raise NotImplementedError

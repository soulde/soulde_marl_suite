from .Policy import *
from utils import soft_update_model_weight
import numpy as np


class MASAC(ACPolicy):
    def __init__(self, actor_type, critic_type, n_agents, actor_state_size, actor_action_size, param, action_scale=1,
                 action_bias=0):
        super(MASAC, self).__init__(actor_type, critic_type, n_agents, actor_state_size, actor_action_size, param,
                                    action_scale, action_bias)

        self.target_entropy = -torch.prod(torch.tensor(actor_action_size).to(self.device)).item()

        self.log_alpha = torch.tensor(np.log(0.01), requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()
        self.a_optimizer = AdamW([self.log_alpha], lr=param.actor_lr)

    def save_model(self, directory: str, name: str):
        super().save_model(directory=directory, name=name)
        torch.save(self.log_alpha, os.path.join(directory, 'log_alpha.pth'.format(name)))

    def load_model(self, directory: str, name: int):
        super().load_model(directory=directory, name=name)
        self.log_alpha = torch.load(os.path.join(directory, 'log_alpha.pth'.format(name)))
        self.alpha = self.log_alpha.exp().item()

    def get_action_and_value(self, states):
        actions, log_probs = zip(*(actor(states[:, i]) for i, actor in enumerate(self.actors)))
        actions = torch.stack(actions, dim=1)
        qf = self.critic_target(states, actions)

        min_qf_next_target = torch.min(qf, dim=-1, keepdim=True)[0]
        return actions, torch.stack(log_probs, dim=1), min_qf_next_target

    def train(self, batch_data):
        self.train_count += 1

        next_observations, observations, actions, rewards, done_factors = batch_data

        with torch.no_grad():
            next_state_actions, next_state_log_pi, min_qf = self.get_action_and_value(next_observations)

            min_qf_next_target = min_qf - self.alpha * torch.sum(next_state_log_pi, dim=-1, keepdim=True)
            next_q_values = rewards + done_factors * min_qf_next_target

        q_values = self.critic(observations, actions)
        next_q_values = torch.concatenate([next_q_values, next_q_values], dim=-1)

        qf_loss = self.critic_loss(q_values, next_q_values)

        # optimize the model
        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        alpha_total_loss = 0
        actor_total_loss = 0
        if self.train_count % self.policy_update_frequency == 0:  # TD 3 Delayed update support
            for _ in range(self.policy_update_frequency):
                pi, log_pi, _ = self.get_action_and_value(observations)

                qf = self.critic(observations, pi)
                min_qf_pi = torch.min(qf, dim=-1)[0]

                actor_loss = ((self.alpha * torch.sum(log_pi, dim=-1, keepdim=False)) - min_qf_pi).mean()
                actor_total_loss += actor_loss.item()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                with torch.no_grad():
                    _, log_pi, _ = self.get_action_and_value(observations)
                alpha_loss = (-self.log_alpha.exp() * (
                        torch.sum(log_pi, dim=-1, keepdim=False) + self.target_entropy)).mean()
                alpha_total_loss += alpha_loss.item()
                self.a_optimizer.zero_grad()
                alpha_loss.backward()
                self.a_optimizer.step()
                self.alpha = self.log_alpha.exp().item()

        if self.train_count % self.target_network_frequency == 0:
            soft_update_model_weight(self.critic, self.critic_target, self.tau)
        return alpha_total_loss / self.policy_update_frequency, actor_total_loss / self.policy_update_frequency, qf_loss.item()

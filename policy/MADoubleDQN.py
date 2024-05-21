import torch

from .Policy import *
from utils import soft_update_model_weight


class MADoubleDQN(ValueBase):
    def __init__(self, model_type, n_agents, actor_state_size, actor_action_size, param):
        super(MADoubleDQN, self).__init__(model_type, n_agents, actor_state_size, actor_action_size, param)

        self.noise_param = param.noise_param

    def train(self, batch_data):
        self.train_count += 1

        next_observations, observations, actions, rewards, done_factors = batch_data

        with torch.no_grad():
            qf = self.get_value(next_observations, True)

            # max_qf = qf.gather(-1, actions.long())
            max_qf = torch.max(qf, dim=-1, keepdim=True)[0]
            next_q_values = rewards + done_factors * max_qf

        q_values = self.get_value(observations, False)
        # max_q_values = torch.max(q_values, dim=-1, keepdim=True)[0]
        max_q_values = q_values.gather(-1, actions.long())
        qf_loss = self.critic_loss(max_q_values, next_q_values)

        # optimize the model

        self.optimizer.zero_grad()
        qf_loss.backward()

        torch.nn.utils.clip_grad_norm_(sum((list(actor.parameters()) for actor in self.models), []), self.max_grad_norm)
        self.optimizer.step()


        for m, m_target in zip(self.models, self.models_target):
            soft_update_model_weight(m, m_target, self.tau)

        return qf_loss.item()

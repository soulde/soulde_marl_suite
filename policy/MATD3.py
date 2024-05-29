from .Policy import *
from utils import soft_update_model_weight


class MATD3(ACPolicy):
    def __init__(self, actor_type, critic_type, n_agents, actor_state_size, actor_action_size, param, action_scale=1,
                 action_bias=0):
        super(MATD3, self).__init__(actor_type, critic_type, n_agents, actor_state_size, actor_action_size, param,
                                    action_scale, action_bias)

        self.noise_param = param.noise_param
        self.action_scale = action_scale
        self.action_bias = action_bias

    def train(self, batch_data):

        self.train_count += 1

        next_observations, observations, actions, rewards, done_factors = batch_data

        with torch.no_grad():
            next_state_actions = self.get_action(next_observations, True)
            actions_with_noise = torch.clamp(
                torch.clamp(torch.randn_like(next_state_actions) * self.noise_param, min=-0.5,
                            max=0.5) + next_state_actions,
                min=torch.tensor(-self.action_scale + self.action_bias).to(next_state_actions),
                max=torch.tensor(self.action_scale + self.action_bias).to(next_state_actions))
            qf = self.get_value(next_observations, actions_with_noise, True)
            min_qf = torch.min(qf, dim=-1, keepdim=True)[0]
            next_q_values = rewards + done_factors * min_qf

        q_values = self.get_value(observations, actions, False)
        next_q_values = torch.concatenate([next_q_values, next_q_values], dim=-1)
        qf_loss = self.critic_loss(q_values, next_q_values)

        # optimize the model

        self.critic_optimizer.zero_grad()
        qf_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        if self.train_count % self.target_network_frequency == 0:
            soft_update_model_weight(self.critic, self.critic_target, self.tau)

        actor_total_loss = 0
        if self.train_count % self.policy_update_frequency == 0:  # TD 3 Delayed update support
            pi = self.get_action(observations, False)
            self.critic.eval()

            qf_pi = self.get_value(observations, pi, False)
            min_qf_pi = torch.min(qf_pi, dim=-1, keepdim=True)[0]
            actor_loss = -min_qf_pi.mean()
            actor_total_loss += actor_loss.item()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            for a in self.actors:
                torch.nn.utils.clip_grad_norm_(a.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            for a, a_target in zip(self.actors, self.actors_target):
                soft_update_model_weight(a, a_target, self.tau)
            self.critic.train()
        return actor_total_loss, qf_loss.item()

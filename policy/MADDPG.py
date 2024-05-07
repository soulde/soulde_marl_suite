from .Policy import *
from utils import soft_update_model_weight


class MADDPG(Policy):
    def __init__(self, actor_type, critic_type, n_agents, actor_state_size, actor_action_size, param, action_scale=1,
                 action_bias=0):
        super(MADDPG, self).__init__(actor_type, critic_type, n_agents, actor_state_size, actor_action_size, param,
                                     action_scale, action_bias)

    def train(self, batch_data):
        self.train_count += 1

        next_observations, observations, actions, rewards, done_factors = batch_data
        # print(next_state_batch.shape, state_batch.shape, action_batch.shape, reward_batch.shape)
        with torch.no_grad():
            next_state_actions = self.get_action(next_observations, True)
            next_q = self.get_value(next_observations, next_state_actions, True)
            next_q_values = rewards + done_factors * next_q

        q = self.critic(observations, actions)

        critic_loss = self.critic_loss(q, next_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        soft_update_model_weight(self.critic, self.critic_target, self.tau)

        pi = self.get_action(observations, False)
        self.critic.eval()
        no_grad_q = self.get_value(observations,pi, False)
        actor_loss = -torch.mean(no_grad_q)

        # print(actor_loss)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        for a in self.actors:
            torch.nn.utils.clip_grad_norm_(a.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        for a, a_target in zip(self.actors, self.actors_target):
            soft_update_model_weight(a, a_target, self.tau)
        self.critic.train()
        return actor_loss.cpu().detach().numpy(), critic_loss.cpu().detach().numpy()

from .Policy import *
import numpy as np


class MAPPO(ACPolicy):
    def __init__(self, actor_type, critic_type, n_agents, actor_state_size, actor_action_size, param, action_scale=1,
                 action_bias=0):
        super(MAPPO, self).__init__(actor_type, critic_type, n_agents, actor_state_size, actor_action_size, param,
                                    action_scale, action_bias)
        self.clip_param = param.clip_param
        self.ent_param = param.ent_param
    def save_model(self, directory: str, name: str):
        for i, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), os.path.join(directory, 'actor_{}_{}.pth'.format(i, name)))

        torch.save(self.critic.state_dict(), os.path.join(directory, 'critic_{}.pth'.format(name)))

    def load_model(self, directory: str, name: int):
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(torch.load(os.path.join(directory, 'actor_{}_{}.pth'.format(i, name))))

        self.critic.load_state_dict(torch.load(os.path.join(directory, 'critic_{}.pth'.format(name))))
    def get_action_and_value(self, states, actions=None):
        if actions is None:
            actions, log_probs, entropy = zip(*(actor(states[:, i], actions) for i, actor in enumerate(self.actors)))
        else:
            actions, log_probs, entropy = zip(
                *(actor(states[:, i], actions[:, i]) for i, actor in enumerate(self.actors)))
        actions = torch.stack(actions, dim=1)

        qf = self.critic(states, actions)
        return actions, torch.stack(log_probs, dim=1), torch.stack(entropy, dim=1), qf

    def train(self, batch_data, learning_rate=None):
        if learning_rate is not None:
            self.actor_optimizer.param_groups[0]["lr"] = learning_rate
            self.critic_optimizer.param_groups[0]["lr"] = learning_rate

        batch_states, batch_actions, batch_rewards, batch_values, batch_log_probs, batch_advantages = batch_data

        _, new_log_prob, entropy, new_value = self.get_action_and_value(batch_states, batch_actions)

        batch_returns = batch_advantages + batch_values

        logratio = new_log_prob - batch_log_probs

        ratio = logratio.exp()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfracs = [((ratio - 1.0).abs() > self.clip_param).float().mean().item()]

        batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std(dim=0) + 1e-8)

        # policy loss
        pg_loss1 = -batch_advantages * ratio
        # print(torch.concatenate([ratio,torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)], dim=1))
        pg_loss2 = -batch_advantages * torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)

        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        v_loss_unclipped = self.critic_loss(new_value, batch_returns)
        # v_clipped = batch_values + torch.clamp(
        #     new_value - batch_values,
        #     -self.clip_param,
        #     self.clip_param,
        # )
        # v_loss_clipped = self.critic_loss(v_clipped, batch_returns)
        # v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        # v_loss = v_loss_max.mean()
        v_loss = v_loss_unclipped.mean()
        entropy_loss = entropy.mean()

        loss = pg_loss - self.ent_param * entropy_loss

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        loss.backward()
        v_loss.backward()

        for actor in self.actors:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        y_pred, y_true = batch_values.cpu().numpy(), batch_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        return v_loss.item(), pg_loss.item(), entropy_loss.item(), \
            old_approx_kl.item(), approx_kl.item(), np.mean(clipfracs), explained_var

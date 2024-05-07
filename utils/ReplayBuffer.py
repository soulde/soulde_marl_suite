import torch
import gymnasium as gym
from itertools import count
import numpy as np


class ReplayBuffer(object):
    def __init__(self, capacity: int, n_env: int, n_agent: int, shapes: tuple,
                 additional_input_formate: tuple = ('values', 'log_probs'), mode='off_policy',
                 use_advantage=False,
                 gae_lambda=0.95, device='cuda'):
        reserved_formate = ('states', 'actions', 'rewards', 'non_terminate_factors')
        input_formate = reserved_formate + additional_input_formate
        sample_formate = reserved_formate[:-1] + additional_input_formate + ('advantages',) if use_advantage else input_formate
        assert len(shapes) == len(input_formate)
        self.capacity_ = capacity
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.use_advantage_ = use_advantage
        self.sample_formate_ = sample_formate
        self.input_formate_ = input_formate
        self.shapes_ = shapes
        self.memory = dict()
        self.count_ = 0
        self.mode_ = mode
        self.n_agent_ = n_agent
        self.n_env_ = n_env
        self.gae_lambda_ = gae_lambda
        self.loop_flag = False
        self._temp_buffer = dict()
        self.reset_memory()

    def __getattr__(self, name):
        try:
            return self.memory[name]
        except KeyError as e:
            raise AttributeError('No such attribute named {}'.format(name))

    '''push experience to memory
    @brief add parallel env and multi-agent experience to memory in one time
    @params state observation from env (dim [n_env, state_size])(all agents' states store as one state)
    @params action agent action (dim [n_env, action_size])(all agents' states store as one state)
    @params reward reward from env (dim [n_env, n_agent])
    @params done(dim [n_env, n_agent])
    '''

    def reset_memory(self):
        for k, shape in zip(self.input_formate_, self.shapes_):
            self.memory[k] = torch.zeros((self.n_env_, self.capacity_ + 1, self.n_agent_, shape), device='cpu').float()

        if self.use_advantage_:
            self.memory['advantages'] = torch.zeros((self.n_env_, self.capacity_ + 1, self.n_agent_, 1),
                                                    device='cpu').float()

    def push(self, *args):
        assert len(args) == len(self.input_formate_)

        if self.count_ == 0:
            if self.mode_ == 'on_policy':
                self.reset_memory()

        for k, s, d in zip(self.input_formate_, self.shapes_, args):
            assert (self.n_env_, self.n_agent_, s) == tuple(d.shape), 'require {} but have {}'.format(
                (self.n_env_, self.n_agent_, s), d.shape)
            if type(d) is torch.Tensor:
                self.memory[k][:, self.count_] = d
            else:
                self.memory[k][self.count_] = torch.tensor(d, device='cpu')
        self.count_ += 1
        if self.count_ > self.capacity_:
            if self.mode_ == 'on_policy' and self.use_advantage_:
                self._calc_advantages()
            self.count_ = 0
            self.loop_flag = True
            return True
        return False

    def _calc_advantages(self):
        last_gae_lambda = 0
        for t in reversed(range(self.capacity_)):
            delta1 = self.values[:, t + 1] * self.non_terminate_factors[:, t + 1]
            delta = self.rewards[:, t] - self.values[:, t] + delta1
            last_gae_lambda = delta + self.gae_lambda_ * self.non_terminate_factors[:, t + 1] * last_gae_lambda
            self.advantages[:, t] = last_gae_lambda

    def _off_policy_sample(self, batch_size: int):
        idx = np.random.choice(self.__len__(), batch_size, replace=False)
        next_state_idx = idx + 1
        env_id = np.random.choice(self.n_env_, batch_size, replace=True)
        next_state = self.memory['states'][env_id, next_state_idx].to(self.device)
        ret = sum(((self.memory[k][env_id, idx].to(self.device),) for k in self.sample_formate_), (next_state,))
        return ret

    def _on_policy_sample(self, batch_size: int):

        # print('-' * 10)
        idx = np.arange(num_data := self.capacity_ * self.n_env_)
        parts = num_data // batch_size
        np.random.shuffle(idx)
        idx = idx[:parts * batch_size]
        data_split_index = np.array_split(idx, parts)
        data = tuple(
            torch.reshape(self.memory[k], [-1, self.n_agent_, self.memory[k].shape[-1]]).to(self.device) for k in
            self.sample_formate_)
        # print(data)
        # print('data', data[0])
        ret = tuple(tuple(map(lambda d: d[i], data)) for i in data_split_index)

        # temp_list = sum(self.states, start=[])
        # state_ret = tuple(map(lambda idx_: tuple(temp_list[i] for i in idx_), data_split_index))
        #
        # temp_ret = tuple((s, *r) for s, r in zip(state_ret, ret))

        return ret

    def sample(self, batch_size):
        assert batch_size <= self.capacity_
        if self.mode_ == 'on_policy':
            return self._on_policy_sample(batch_size)
        elif self.mode_ == 'off_policy':
            return self._off_policy_sample(batch_size)

    def __len__(self):

        if self.mode_ == 'on_policy':
            return self.count_

        elif self.mode_ == 'off_policy':
            if self.loop_flag:
                return self.capacity_
            else:
                return self.count_


if __name__ == '__main__':
    buffer = ReplayBuffer(1000, 4, 3, (4, 2, 1, 1, 1, 1), ('values', 'log_probs'),
                          mode='on_policy', use_advantage=True)
    states = torch.rand((4, 3, 4), device='cpu')
    actions = torch.rand((4, 3, 2), device='cuda')
    rewards = torch.rand((4, 3, 1), device='cpu')
    terminations = torch.rand((4, 3, 1), device='cpu')
    values = torch.rand((4, 3, 1), device='cpu')
    log_probs = torch.rand((4, 3, 1), device='cpu')
    for i in range(10000):
        # print('ready to push')
        ret = buffer.push(states, actions, rewards, terminations, values, log_probs)
        if ret:
            batch_data = buffer.sample(16)
            for mini_batch in batch_data:
                s, a, r, t, v, l, ad = mini_batch
                print(s.shape)

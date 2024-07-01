from abc import ABC, abstractmethod

from . import SandBox
from .Agent import USVAgent

from utils import Config

import numpy as np


class Mission(ABC):
    def __init__(self, config: Config, sandbox: SandBox):
        self.config = config
        self.sandbox = sandbox
        self.num_agents = config['num_agents']
        self.agents_profile = config['agent_profile']

    def action_transform(self, actions):
        return actions

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def is_termination(self) -> (bool, bool):
        raise NotImplementedError

    @abstractmethod
    def step(self):
        raise NotImplementedError

    @abstractmethod
    def calculate_reward(self):
        raise NotImplementedError

    @abstractmethod
    def get_state(self):
        raise NotImplementedError

    @abstractmethod
    def skip_frame(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def sample(self, zero=False):
        raise NotImplementedError

    @property
    @abstractmethod
    def action_dim(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def state_dim(self):
        raise NotImplementedError


class USVMission(Mission):
    @abstractmethod
    def calculate_reward(self):
        raise NotImplementedError

    def __init__(self, config: Config, sandbox: SandBox):
        super(USVMission, self).__init__(config, sandbox)
        self.max_step = config['max_step']
        self.hit_wall_threshold = config['hit_wall_threshold']
        self.collision_info = None
        self.hit_wall_info = None

        for i in range(self.num_agents):
            self.sandbox.register_agent('agent_{}'.format(i), USVAgent, self.agents_profile)

    def reset(self):
        pass

    def is_termination(self) -> (bool, bool):
        max_step_termination = self.sandbox.n_step > self.max_step

        collision_termination = len(self.collision_info) > 0

        return collision_termination, max_step_termination

    def step(self):
        self.collision_info = self.sandbox.collision_server.check_collision_all()
        lower_range = np.ones_like(self.sandbox.size_) * self.hit_wall_threshold
        upper_range = self.sandbox.size_ - lower_range

    def get_state(self):
        return np.stack([agent.state for agent in self.sandbox.agents])

    def skip_frame(self) -> bool:
        return False

    def sample(self, zero=False):
        actions = []

        for n, a, p in self.sandbox.agents_type_profile_list:
            high, low = p['input_range']
            single_action = np.random.uniform(low, high, size=(high.shape[-1]))
            if zero:
                single_action = np.zeros_like(single_action)
            actions.append(single_action)
        return actions

    @property
    def action_dim(self):
        dims = []
        for n, a, p in self.sandbox.agents_type_profile_list:
            high, low = p['input_range']
            dims.append(len(high))
        return tuple(dims)

    @property
    def state_dim(self):
        dims = []
        for n, a, p in self.sandbox.agents_type_profile_list:
            high, low = p['state_range']
            dims.append(len(high))
        return tuple(dims)

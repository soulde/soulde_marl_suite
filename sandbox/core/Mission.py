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

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def is_termination(self):
        raise NotImplementedError

    @abstractmethod
    def step(self):
        raise NotImplementedError

    @abstractmethod
    def calculate_reward(self):
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

    def creat_agent_from_profile(self, name, agent_type, agent_profile, init_state='random'):
        state_range = np.array(agent_profile['state_range'])
        input_range = np.array(agent_profile['input_range'])
        radius = agent_profile['collision/radius']
        collision_type = agent_profile['collision/type']
        agent = agent_type(name, state_range, input_range, init_state, {'args': (radius,), 'type': collision_type})
        return agent

    def reset(self):
        self.sandbox.agents.clear()

        for i in range(self.num_agents):
            self.sandbox.register_agent(
                self.creat_agent_from_profile('agent_{}'.format(1), USVAgent, self.agents_profile))


    def is_termination(self):
        max_step_termination = self.sandbox.n_step > self.max_step

        collision_termination = len(self.collision_info) > 0

        return any([collision_termination] + self.hit_wall_info), max_step_termination

    def step(self):
        self.collision_info = self.sandbox.collision_server.check_collision_all()
        lower_range = np.ones_like(self.sandbox.size_) * self.hit_wall_threshold
        upper_range = self.sandbox.size_ - lower_range
        self.hit_wall_info = [(np.any(agent.pos < lower_range) or np.any(agent.pos > upper_range)) for agent in
                              self.sandbox.agents]

from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable

import numpy as np
from numpy import clip

# from .Transmission import Transmitter
from .CollisionServer import CollisionServer


class Agent(ABC):
    def __init__(self, name, state_range, input_range, initial_state=None, collision_info=None):
        if collision_info is None:
            collision_info = {'type': 'circle', 'args': (1,)}
        self.name_ = name
        if initial_state is None:
            self.state_ = np.zeros(self.dim_state)
        elif type(initial_state) is str and initial_state == 'random':
            self.state_ = np.random.uniform(high=state_range[0], low=state_range[1], size=self.dim_state)
        else:

            if not initial_state.shape == self.dim_state:
                initial_state = np.append(initial_state, np.zeros(self.dim_state - initial_state.shape[0]))
            self.state_ = initial_state

        self.state_range_ = state_range
        self.input_range_ = input_range
        self.collision_info_ = collision_info

    @abstractmethod
    def __call__(self, u, t):
        raise NotImplementedError

    @property
    @abstractmethod
    def pos(self):
        raise NotImplementedError

    @property
    def state(self):
        return self.state_

    def __str__(self):
        return self.name_

    @classmethod
    @property
    @abstractmethod
    def dim_state(cls):

        raise NotImplementedError

    @classmethod
    @property
    @abstractmethod
    def dim_input(cls):
        raise NotImplementedError

    @property
    def name(self):
        return self.name_


class USVAgent(Agent):
    def __init__(self, name, state_range, input_range, initial_state=None, collision_info=None):
        super().__init__(name, state_range, input_range, initial_state, collision_info=collision_info)

    @property
    def pos(self):
        return self.state[:2]

    @property
    def velocity(self):
        return self.state[2]

    def __call__(self, u, t):
        # state: x, y, v, phi
        # u: a,omega

        u = clip(u, self.input_range_[1], self.input_range_[0])

        self.state_[2:] += u * t
        self.state_[2:] = clip(self.state_[2:], self.state_range_[1, 2:], self.state_range_[0, 2:])
        self.state_[3] = (self.state_[3] + np.pi) % (2 * np.pi) - np.pi
        self.state_[0] += self.state_[2] * np.cos(self.state_[3])
        self.state_[1] += self.state_[2] * np.sin(self.state_[3])
        self.state_[:2] = clip(self.state_[:2], self.state_range_[1, :2], self.state_range_[0, :2])

    @classmethod
    @property
    def dim_state(cls):
        return np.array([4, ])

    @classmethod
    @property
    def dim_input(cls):
        return np.array([2, ])


# class USVWithTransmission(USVAgent):
#     def __init__(self, name, state_range, input_range, radius, transmitter_buffer_length, max_communicate_distance,
#                  type_=None, initial_state=None):
#         super().__init__(name, state_range, input_range, radius, type_, initial_state=initial_state)
#         # self.transmitter = Transmitter(buffer_length=transmitter_buffer_length,
#         #                                max_communicate_distance=max_communicate_distance)
#         self.transmitter.installed_in = self
#
#     def publish(self, data, timestamp):
#         self.transmitter.send(data, timestamp)
#
#     def receive(self):
#         return self.transmitter.receive()

AGENT_TYPE_LUT = {
    'usv': USVAgent,
}

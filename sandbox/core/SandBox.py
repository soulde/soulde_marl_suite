from .Agent import Agent
from .CollisionServer import CollisionServer
from utils import Config
from .Mission import Mission
from .StateWrapper import StateWrapper
from .MapGenerator import MapGenerator
from .Renderer import Renderer
import numpy as np
import cv2


class SandBox:
    def __init__(self, config: Config, map_generator, mission, states_wrapper, renderer, info_logger=None):

        assert issubclass(map_generator, MapGenerator)
        assert issubclass(mission, Mission)
        assert issubclass(states_wrapper, StateWrapper)
        assert issubclass(renderer, Renderer)
        # environment config
        self.config = config
        self.name = self.config['name']
        self.tick = self.config['tick']
        self.mode = self.config['mode']

        # sandbox mechanism component
        self.agents: list[Agent] = list()

        self.time_ = 0

        self.map_generator_ = map_generator(self.config['map'])
        self.states_wrapper_ = states_wrapper(self.config['state'], self)
        self.mission_ = mission(self.config['mission'], self)
        self.renderer_ = renderer(self.config['renderer'], self)

        self.logger = info_logger(self) if info_logger is not None else None
        self.info_buffer = {}

        self.collision_server = CollisionServer()
        self.frame = None
        self.size_ = None

    def register_agent(self, agent: Agent):
        self.agents.append(agent)
        self.collision_server.register(agent.name, agent.pos, agent.collision_info_['type'],
                                       *agent.collision_info_['args'])

    def _agent_dynamic_step(self, actions):
        for agent_, actions in zip(self.agents, actions):

            agent_(actions, self.tick)
        self.time_ += self.tick

    def _collision_server_update(self):
        for agent in self.agents:
            self.collision_server.update_pos(agent.name, agent.pos)

    def register_var(self, name, value):
        self.info_buffer[name] = value

    def lookup_var(self, name):
        return self.info_buffer[name]

    @property
    def time(self):
        return self.time_

    @property
    def n_step(self):
        return self.time_ / self.tick

    def step(self, actions):
        self._agent_dynamic_step(actions)
        self._collision_server_update()
        self.mission_.step()
        states = self.states_wrapper_()
        reward = self.mission_.calculate_reward()
        done, truncation = self.mission_.is_termination()
        if self.logger is None:
            info = {'_final_info': self.n_step} if done or truncation else {}
        else:
            info = self.logger()
        return states, reward, done, truncation, info

    def sample(self, zero=False):
        high, low = self.agents[0].input_range_
        if zero:
            return np.zeros((len(self.agents), high.shape[0]))
        else:
            return np.random.uniform(low, high, size=(len(self.agents), high.shape[0]))

    def reset(self):
        self.time_ = 0
        self.collision_server.reset()
        self.frame, collision_info = self.map_generator_.generate()
        self.size_ = self.frame.shape[:-1]
        for x, y, r in collision_info:
            self.collision_server.register('coli_{}_{}_{}'.format(x, y, r), np.array([x, y]), 'circle', r)

        self.mission_.reset()
        for agent in self.agents:
            self.collision_server.register(agent.name, agent.pos, agent.collision_info_['type'],
                                           *agent.collision_info_['args'])
        states = self.states_wrapper_()
        if self.logger is None:
            info = {}
        else:
            info = self.logger()
        return states, info

    def render(self, mode='human'):
        self.renderer_.render(mode=mode)

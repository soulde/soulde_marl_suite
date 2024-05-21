from .Agent import Agent
from .CollisionServer import CollisionServer
from utils import Config
from .Mission import Mission
from .MapGenerator import MapGenerator
from .Renderer import Renderer
import numpy as np
import cv2


class SandBox:
    def __init__(self, config: Config, map_generator, mission, renderer, info_logger=None):

        assert issubclass(map_generator, MapGenerator)
        assert issubclass(mission, Mission)

        assert issubclass(renderer, Renderer)
        # environment config
        self.config = config
        self.name = self.config['name']
        self.tick = self.config['tick']
        self.mode = self.config['mode']

        # sandbox mechanism component
        self.agents: list[Agent] = list()
        self.agents_type_profile_list = []
        self.time_ = 0

        self.map_generator_ = map_generator(self.config['map'])
        self.mission_ = mission(self.config['mission'], self)
        self.renderer_ = renderer(self.config['renderer'], self)

        self.logger = info_logger(self) if info_logger is not None else None
        self.info_buffer = {}

        self.collision_server = CollisionServer()
        self.frame = None
        self.size_ = None

    @staticmethod
    def creat_agent_from_profile(name, agent_type, agent_profile, init_state='random'):
        state_range = np.array(agent_profile['state_range'])
        input_range = np.array(agent_profile['input_range'])
        radius = agent_profile['collision/radius']
        collision_type = agent_profile['collision/type']
        agent = agent_type(name, state_range, input_range, init_state, {'args': (radius,), 'type': collision_type})
        return agent

    def register_agent(self, name, agent_type, profile):
        self.agents_type_profile_list.append((name, agent_type, profile))

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
        if actions is not None:
            self._agent_dynamic_step(actions)
            self._collision_server_update()
        self.mission_.step()
        states = self.mission_.get_state()
        reward = self.mission_.calculate_reward()
        done, truncation = self.mission_.is_termination()
        if self.logger is None:
            info = {'_final_info': self.n_step} if done or truncation else {}
        else:
            info = self.logger()
        return states, reward, done, truncation, info

    def sample(self, zero=False):
        actions = []

        for i in self.agents_type_profile_list:
            n, a, p = i
            high, low = p['input_range']
            single_action = np.random.uniform(low, high, size=(high.shape[0]))
            if zero:
                single_action = np.zeros_like(single_action)
            actions.append(single_action)
        return actions

    def reset(self):
        self.time_ = 0
        self.collision_server.reset()
        self.frame, collision_info = self.map_generator_.generate()
        self.size_ = self.frame.shape[:-1]
        for x, y, r in collision_info:
            self.collision_server.register('coli_{}_{}_{}'.format(x, y, r), np.array([x, y]), 'circle', r)
        for n, a, p in self.agents_type_profile_list:
            self.agents.append(self.creat_agent_from_profile(n, a, p))
        self.mission_.reset()

        states = self.mission_.get_state()
        if self.logger is None:
            info = {}
        else:
            info = self.logger()
        return states, info

    def render(self, mode='human'):
        self.renderer_.render(mode=mode)

from .Agent import Agent
from .CollisionServer import CollisionServer
from utils import Config
from .Mission import Mission
from .MapGenerator import MapGenerator
from .Renderer import Renderer
import numpy as np
import cv2


class SandBox:
    def __init__(self, config: Config, map_generator, mission, renderer, info_logger=None, render_mode='human'):

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
        self.render_mode = render_mode

    @staticmethod
    def creat_agent_from_profile(name, agent_type, agent_profile):
        state_range = np.array(agent_profile['state_range'])
        input_range = np.array(agent_profile['input_range'])
        radius = agent_profile['collision/radius']
        collision_type = agent_profile['collision/type']
        try:
            init_area = agent_profile['init_area']
            init_state = np.random.uniform(init_area[0], init_area[1])
        except KeyError:
            init_state = 'random'
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
        actions = self.mission_.action_transform(actions)
        while True:
            if actions is not None:
                self._agent_dynamic_step(actions)
                self._collision_server_update()
                self.renderer_.render(mode=self.render_mode)
            self.mission_.step()
            if not self.mission_.skip_frame():
                print('break')
                break
        states = self.mission_.get_state()
        reward = self.mission_.calculate_reward()
        done, truncation = self.mission_.is_termination()
        if self.logger is None:
            info = {'_final_info': self.n_step} if done or truncation else {}
        else:
            info = self.logger()
        return states, reward, done, truncation, info

    def reset(self):
        self.time_ = 0
        self.collision_server.reset()
        self.frame = self.map_generator_.generate()
        self.size_ = self.frame.shape
        self.collision_server.set_background(self.frame)
        self.agents.clear()
        for n, a, p in self.agents_type_profile_list:
            agent = self.creat_agent_from_profile(n, a, p)
            self.agents.append(agent)
            self.collision_server.register(agent.name, agent.pos, agent.collision_info_['type'],
                                           *agent.collision_info_['args'])
        self.mission_.reset()

        states = self.mission_.get_state()
        if self.logger is None:
            info = {}
        else:
            info = self.logger()
        return states, info

    def render(self, mode='human'):
        self.renderer_.render(mode=mode)

    def sample(self, zero=False):
        return self.mission_.sample(zero=zero)

    @property
    def action_dim(self):
        return self.mission_.action_dim

    @property
    def state_dim(self):
        return self.mission_.state_dim

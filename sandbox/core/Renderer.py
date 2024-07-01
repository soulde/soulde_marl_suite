from abc import ABC, abstractmethod
from .Agent import USVAgent
from . import SandBox
from utils import Config

import numpy as np
import cv2


def corners(p):
    floor = np.floor(p)
    return [tuple(p.astype(int)) for p in
            [floor, floor + np.array([1, 0]), floor + np.array([0, 1]), floor + np.array([1, 1])]]


class Renderer(ABC):
    def __init__(self, config: Config, sandbox: SandBox):
        self.sandbox = sandbox
        self.config = config
        self.plot_scale = config['plot_scale']

    @abstractmethod
    def render(self, mode):
        raise NotImplementedError


class BasicRenderer(Renderer):
    def __init__(self, config: Config, sandbox: SandBox):
        super(BasicRenderer, self).__init__(config, sandbox)

    def render(self, mode):
        if mode is None:
            return
        frame_copy = self.sandbox.frame.copy()
        frame_copy = np.stack([frame_copy, frame_copy, frame_copy], axis=-1)
        # for i in self.sandbox.collision_server.bodies.values():
        #     for p in set(sum([corners(p) for p in (i.mask + i.pos)], [])):
        #         cv2.circle(frame_copy, p, 1, (124,0,0), -1)
        for agent in self.sandbox.agents:
            v, dire = agent.state[2:]
            # print(v, dire)
            pos: np.ndarray = agent.pos.copy()
            next_pos = pos + 6 * v * np.array((np.cos(dire), np.sin(dire)))
            pos = pos.astype(int)
            next_pos = next_pos.astype(int)
            cv2.circle(frame_copy, pos, agent.collision_info_['args'][0], (0, 255, 0), -1)
            cv2.arrowedLine(frame_copy, pos, next_pos, (0, 255, 0), 2)
        if mode == 'rgb_array':
            return frame_copy
        elif mode == 'human':
            cv2.imshow(self.sandbox.name, frame_copy)
            cv2.waitKey(1)

from abc import ABC, abstractmethod
from .Agent import USVAgent
from . import SandBox
from utils import Config

import numpy as np
import cv2


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
        frame_copy = self.sandbox.frame.copy()
        frame_copy = cv2.resize(frame_copy,
                                (frame_copy.shape[0] * self.plot_scale, frame_copy.shape[1] * self.plot_scale))
        for agent in self.sandbox.agents:
            v, dire = agent.state[2:]
            # print(v, dire)
            pos: np.ndarray = agent.pos.copy() * self.plot_scale
            next_pos = pos + 5 * self.plot_scale * v * np.array((np.cos(dire), np.sin(dire)))
            pos = pos.astype(int)
            next_pos = next_pos.astype(int)
            cv2.circle(frame_copy, pos, self.plot_scale, (0, 255, 0), -1)
            cv2.arrowedLine(frame_copy, pos, next_pos, (0, 255, 0), 2)
        if mode == 'rgb_array':
            return frame_copy
        elif mode == 'human':
            cv2.imshow(self.sandbox.name, frame_copy)
            cv2.waitKey(1)

import numpy as np

from . import SandBox
from abc import ABC, abstractmethod
from utils import Config


class StateWrapper(ABC):
    def __init__(self, config: Config, sandbox: SandBox):
        self.sandbox_ = sandbox

    @abstractmethod
    def __call__(self):
        raise NotImplementedError


class RawStateWrapper(StateWrapper):
    def __init__(self, config: Config, sandbox: SandBox):
        super().__init__(config, sandbox)

    def __call__(self):
        return np.stack([agent.state for agent in self.sandbox_.agents])

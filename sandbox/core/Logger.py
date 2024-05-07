from .Config import Config
from abc import ABC, abstractmethod
import numpy as np


class Logger(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self):
        raise NotImplementedError

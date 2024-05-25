from utils import Config
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image


class MapGenerator(ABC):
    def __init__(self, config: Config):
        self.size_ = config['size']

    @abstractmethod
    def generate(self) -> (np.ndarray, list):
        raise NotImplementedError


class EmptyMapGenerator(MapGenerator):
    def __init__(self, config: Config):
        super(EmptyMapGenerator, self).__init__(config)

    def generate(self):
        return np.ones((self.size_[0], self.size_[1], 3)) * 255, []


class ObstacleMapGenerator(MapGenerator):
    def __init__(self, config: Config):
        super(ObstacleMapGenerator, self).__init__(config)
        self.num_obstacles_ = config['num_obstacles']
        self.max_obstacle_radius = config['max_obstacle_radius']

    def generate(self):
        frame = np.ones((self.size_[0], self.size_[1], 3)) * 255
        obstacle_info = []
        for i in range(self.num_obstacles_):
            obstacle_info.append(np.random.uniform(low=np.array([0, 0, 0.001]), high=np.array(
                [self.size_[0], self.size_[1], self.max_obstacle_radius])))
        return frame, obstacle_info


class ImageMapGenerator(MapGenerator):
    def generate(self) -> (np.ndarray, list):
        return self.map, []

    def __init__(self, config: Config):
        super(ImageMapGenerator, self).__init__(config)
        self.url = config['url']
        self.map = Image.open(self.url)
        self.size = self.map.size

import cv2

from utils import Config
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image, ImageDraw


class MapGenerator(ABC):
    def __init__(self, config: Config):
        self.size_ = config['size']

    @abstractmethod
    def generate(self) -> np.ndarray:
        raise NotImplementedError


class EmptyMapGenerator(MapGenerator):
    def __init__(self, config: Config):
        super(EmptyMapGenerator, self).__init__(config)

    def generate(self):
        return np.ones((self.size_[0], self.size_[1])) * 255


class ObstacleMapGenerator(MapGenerator):
    def __init__(self, config: Config):
        super(ObstacleMapGenerator, self).__init__(config)
        self.num_obstacles_ = config['num_obstacles']
        self.max_obstacle_radius = config['max_obstacle_radius']

    def generate(self):
        frame = np.ones(self.size_, dtype=np.uint8) * 255
        cv2.rectangle(frame, (0, 0), self.size_, 0, 1)
        for i in range(self.num_obstacles_):
            center = np.random.uniform(low=np.array([0, 0]), high=self.size_)
            r = np.random.uniform(low=0.01, high=self.max_obstacle_radius)
            cv2.circle(frame, center.astype(int), int(r), 0, -1)

        return frame


class ImageMapGenerator(MapGenerator):
    def generate(self) -> np.ndarray:
        return self.map

    def __init__(self, config: Config):
        super(ImageMapGenerator, self).__init__(config)
        self.url = config['url']
        self.map = cv2.imread(self.url, -1)
        self.map = cv2.resize(self.map, self.size_)

import numpy as np

from core import SandBox, ObstacleMapGenerator, USVMission, BasicRenderer
from functools import partial
from utils import Config


class PathPlanningMission(USVMission):
    def __init__(self, config: Config, sandbox: SandBox):
        super(PathPlanningMission, self).__init__(config, sandbox)
        self.reach_threshold = config['reach_threshold']
        self.smooth_factor = config['reward/smooth_factor']
        self.reward_scale = config['reward/reward_scale']
        self.offset_factor = config['reward/offset_factor']
        self.collision_factor = config['reward/collision_factor']
        self.success_reward = config['reward/success_reward']
        self.target = np.zeros(2)

    def reset(self):
        super(PathPlanningMission, self).reset()
        self.target = np.random.uniform(low=np.array([0, 0]),
                                        high=np.array([self.sandbox.size_[0], self.sandbox.size_[1]]))

    def is_termination(self):
        termination, truncation = super(PathPlanningMission, self).is_termination()
        reach_termination = False
        for a in self.sandbox.agents:
            if np.linalg.norm(a.pos - self.target) < self.reach_threshold:
                reach_termination = True
                break
        return any([termination, reach_termination]), truncation

    def calculate_reward(self):
        reach_reward = np.array(
            [self._distance_reward(np.linalg.norm(a.pos - self.target)) for a in self.sandbox.agents])
        collision_mask = np.array(
            [not any(a.name in item for item in self.collision_info) for a in self.sandbox.agents])

        return self.reward_scale * sum(reach_reward) * collision_mask + ~collision_mask * self.collision_factor

    def _distance_reward(self, distance):
        if distance > self.reach_threshold:
            return self.smooth_factor / (self.smooth_factor + distance) - self.offset_factor
        else:
            return self.success_reward


PathPlanningRenderer = BasicRenderer

PathPlanningSandbox = partial(SandBox, map_generator=ObstacleMapGenerator, mission=PathPlanningMission,
                              renderer=PathPlanningRenderer)

import cv2

from core import SandBox, Config, ObstacleMapGenerator, RawStateWrapper, USVMission, BasicRenderer, Mission

from core.Agent import Agent, Dynamic
import numpy as np
import matplotlib.pyplot as plt


class Manual(Dynamic):
    @property
    def dim_input(self):
        return 2

    @property
    def dim_state(self):
        return 2

    def __call__(self, u, t):
        self.state_ = u

    def __init__(self, state_range, input_range):
        super().__init__(state_range, input_range)


class ManualAgent(Agent):

    @property
    def pos(self):
        return self.state

    def __init__(self, name, state_range, input_range):
        super().__init__(name, Manual(state_range, input_range))


class HeatMapMission(Mission):
    def __init__(self, config, sandbox):
        super(HeatMapMission, self).__init__(config, sandbox)

        self.reach_threshold = config['reach_threshold']
        self.smooth_factor = config['reward/smooth_factor']
        self.reward_scale = config['reward/reward_scale']
        self.offset_factor = config['reward/offset_factor']
        self.collision_factor = config['reward/collision_factor']
        self.success_reward = config['reward/success_reward']
        self.target = np.zeros(2)
        self.collision_info = []

    def reset(self):
        frame_size = self.sandbox.size_
        self.sandbox.register_agent(ManualAgent('heatmap', np.stack([frame_size, np.zeros_like(frame_size)]),
                                                np.stack([frame_size, np.zeros_like(frame_size)])))
        self.target = np.random.uniform(low=np.array([0, 0]),
                                        high=np.array([self.sandbox.size_[0], self.sandbox.size_[1]]))

    def is_termination(self):
        return False, False

    def step(self):
        self.collision_info = self.sandbox.collision_server.check_collision_all()

    def calculate_reward(self):
        reach_reward = np.array(
            [self._distance_reward(np.linalg.norm(a.pos - self.target)) for a in self.sandbox.agents])
        if len(self.collision_info) == 0:
            collision_mask = np.ones_like(reach_reward).astype(bool)
        else:
            collision_mask = np.array(
                [not any([a.name in item for item in self.collision_info]) for a in self.sandbox.agents])
        # print(reach_reward, self.collision_info)

        return self.reward_scale * sum(reach_reward) * collision_mask + self.collision_factor * ~collision_mask

    def _distance_reward(self, distance):
        if distance > self.reach_threshold:
            # print('distance: ', distance)
            return self.smooth_factor / (self.smooth_factor + distance) + self.offset_factor
        else:
            return self.success_reward


def main():
    config = Config('reward_heat_map.json')
    sandbox = SandBox(config, ObstacleMapGenerator, HeatMapMission, RawStateWrapper, BasicRenderer)
    sandbox.reset()
    size = sandbox.size_
    heatmap = np.zeros([size[0] * 10, size[1] * 10], dtype=float)
    for i in range(size[0] * 10):
        for j in range(size[1] * 10):
            o, r, ter, tru, info = sandbox.step(np.array([[i, j]]) / 10)
            # print(r.item())
            heatmap[i, j] = r.item()
    max_v = np.max(heatmap)

    min_v = np.min(heatmap)
    # print(min_v, max_v)
    heatmap = (1 - np.exp(-(heatmap - min_v))) * 255
    heatmap = heatmap.astype(np.uint8)
    # heatmap = cv2.resize(heatmap, (6 * size[1], 6 * size[0]))
    cv2.imshow('heatmap', heatmap)
    print('end')
    cv2.waitKey(0)
    # x_ = list(range(size[0] * 10))
    # y_ = list(range(size[1] * 10))
    # plt.pcolormesh(x_, y_, heatmap, cmap='viridis_r', shading='gouraud')
    # plt.show()


if __name__ == '__main__':
    print('start')
    main()

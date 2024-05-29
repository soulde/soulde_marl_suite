import numpy as np

from sandbox.core import SandBox, USVMission, BasicRenderer, USVAgent, EmptyMapGenerator
from functools import partial
from utils import Config
import cv2


class DefendMission(USVMission):
    def __init__(self, config: Config, sandbox: SandBox):
        super(DefendMission, self).__init__(config, sandbox)
        self.enemy_waves = config['enemy_info/waves']
        self.enemy_profile = config['enemy_info/enemy_profile']
        self.init_enemy = config['enemy_info/init']
        self.agent_start_area = config['agent_start_area']
        self.enemy_start_area = config['enemy_info/enemy_start_area']
        self.thread_threshold = config['thread_threshold']
        self.capture_num_agents_need = config['capture_num_agents_need']
        self.collision_factor = config['reward/collision_factor']
        self.success_reward = config['reward/success_reward']
        self.distance_factor = config['reward/distance_factor']
        self.reach_threshold = config['reach_threshold']
        self.failure_reward = config['reward/failure_reward']
        self.enemy: list[USVAgent] = []
        self.captured_enemy = []
        self.agents_add_score = []
        self.enemy_counter = 0
        self.failure_flag = False
        self.last_distance = np.ones(self.num_agents) * abs(
            np.array(self.enemy_start_area).min() - np.array(self.agent_start_area).max())
        self.distance = np.zeros(self.num_agents)
        self.err_distance = np.zeros(self.num_agents)

        for i in range(self.num_agents):
            self.sandbox.register_agent('agent_{}'.format(i), USVAgent, self.agents_profile)

    def reset(self):
        self.enemy_counter = 0
        for i in range(self.init_enemy['num_agents']):
            self._add_single_enemy()

    def _add_single_enemy(self):
        pos = np.random.uniform(self.enemy_start_area[:2], self.enemy_start_area[2:])
        enemy: USVAgent = self.sandbox.creat_agent_from_profile('enemy_{}'.format(self.enemy_counter), USVAgent,
                                                                self.enemy_profile, pos)
        self.sandbox.collision_server.register(enemy.name, enemy.pos, enemy.collision_info_['type'],
                                               *enemy.collision_info_['args'])
        self.enemy.append(enemy)
        self.enemy_counter += 1

    def step(self):
        # add enemy if necessary
        for wave in self.enemy_waves:
            wave['time'] -= 1
        ready_enemy = filter(lambda x: x['time'] < 0, self.enemy_waves)
        for i in ready_enemy:
            for _ in range(i['num_agents']):
                self._add_single_enemy()
        self.enemy_waves = list(filter(lambda x: x['time'] >= 0, self.enemy_waves))
        # enemy move (agent move in sandbox main step)
        for i in self.enemy:
            action = self.enemy_policy(i)
            i(action, self.sandbox.tick)
            self.sandbox.collision_server.update_pos(i.name, i.pos)

        # defend state detect
        self.captured_enemy = []
        reach_enemy = []
        self.agents_add_score = []
        self.failure_flag = False
        for e in self.enemy:
            agent_nearby = []
            for i, a in enumerate(self.sandbox.agents):
                if np.linalg.norm(e.pos - a.pos) < self.thread_threshold:
                    agent_nearby.append(i)
            if len(agent_nearby) >= self.capture_num_agents_need:
                self.captured_enemy.append(e)
                self.agents_add_score += agent_nearby
            if abs(e.pos[0] - self.sandbox.size_[0]) < self.reach_threshold:
                self.failure_flag |= True
                reach_enemy.append(e)

        tmp_len = len(self.enemy)
        self.enemy = list(filter(lambda x: x not in self.captured_enemy and x not in reach_enemy, self.enemy))
        if len(self.enemy) == 0:
            self.last_distance = np.zeros(self.num_agents)
            self.distance = np.zeros(self.num_agents)
            self.err_distance = np.zeros(self.num_agents)
            return
        if tmp_len > len(self.enemy):
            self.last_distance = np.array(
                [np.array([np.linalg.norm(a.pos - e.pos) for e in self.enemy]).min() for a in self.sandbox.agents])

        for i in self.captured_enemy:
            self.sandbox.collision_server.unregister(i.name)
        super(DefendMission, self).step()
        self.distance = np.array(
            [np.array([np.linalg.norm(a.pos - e.pos) for e in self.enemy]).min() for a in self.sandbox.agents])
        self.err_distance = self.last_distance - self.distance
        self.last_distance = self.distance

    def enemy_policy(self, enemy: USVAgent):
        action = np.zeros(enemy.dim_input)
        vec = np.zeros(2)
        for a in self.sandbox.agents:
            vec -= 20 / (20 + (a.pos - enemy.pos))
        vec += np.array([1, 0])
        vec /= np.linalg.norm(vec)
        theta_target = np.arctan2(vec[1], vec[0])
        theta_err = theta_target - enemy.state[3]
        action = np.array([0.5 - theta_err / np.pi, 0.1 * theta_err])
        return action

    def calculate_reward(self):
        capture_reward = np.zeros(self.num_agents)
        for i in self.agents_add_score:
            capture_reward[i] += self.success_reward
        agents_collision = [any([a.name in i for i in self.collision_info]) for a in self.sandbox.agents]
        total_reward = self.collision_factor * (np.array(self.hit_wall_info) | np.array(
            agents_collision))
        if self.failure_flag:
            return self.failure_reward * np.ones(self.num_agents) + total_reward
        else:
            return total_reward + capture_reward + self.distance_factor * self.err_distance

    def is_termination(self):

        no_enemy_termination = True if len(self.enemy_waves) == 0 and len(self.enemy) == 0 else False

        max_step_termination = self.sandbox.n_step > self.max_step
        return any([no_enemy_termination]), max_step_termination

    def get_state(self):
        states = []
        for a in self.sandbox.agents:
            try:
                ret = min(self.sandbox.mission_.enemy, key=lambda x: np.linalg.norm(a.pos - x.pos)).state
            except ValueError:
                ret = np.zeros_like(a.state)

            states.append(np.concatenate([*tuple(a.state for a in self.sandbox.agents), ret]))

        return np.stack(states)


class DefendingRenderer(BasicRenderer):
    def __init__(self, config: Config, sandbox: SandBox):
        super(DefendingRenderer, self).__init__(config, sandbox)

    def render(self, mode):
        frame_copy = super(DefendingRenderer, self).render('rgb_array')
        for agent in self.sandbox.mission_.enemy:
            v, dire = agent.state[2:]
            # print(v, dire)
            pos: np.ndarray = agent.pos.copy() * self.plot_scale
            next_pos = pos + 5 * self.plot_scale * v * np.array((np.cos(dire), np.sin(dire)))
            pos = pos.astype(int)
            next_pos = next_pos.astype(int)
            cv2.circle(frame_copy, pos, self.plot_scale, (255, 0, 0), -1)
            cv2.arrowedLine(frame_copy, pos, next_pos, (255, 0, 0), 2)
        if mode == 'human':
            cv2.imshow(self.sandbox.name, frame_copy)
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return frame_copy


DefendingSandbox = SandBox = partial(SandBox, map_generator=EmptyMapGenerator, mission=DefendMission,
                                     renderer=DefendingRenderer)

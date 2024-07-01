import numpy as np

from sandbox.core import SandBox, EmptyMapGenerator, Renderer, Mission, TaskAllocationAgent
from functools import partial
from utils import Config
import cv2


def creat_task_from_profile(profile):
    state_range = np.array(profile['pos_range'])
    radius_range = np.array(profile['radius_range'])
    pos = np.random.uniform(high=state_range[0], low=state_range[1])
    r = np.random.uniform(high=radius_range[0], low=radius_range[1])

    return Task(pos, r)


class Task:
    def __init__(self, pos, radius):
        self.state = pos
        self.process = 0
        self.r = radius

    @property
    def pos(self):
        return self.state

    def __str__(self):
        return f'Task(pos={self.pos}, r={self.r})'


class TaskAllocationMission(Mission):
    @property
    def action_dim(self):
        return (len(self.tasks),) * self.num_agents

    @property
    def state_dim(self):
        return (len(self.tasks) * 3 + TaskAllocationAgent.dim_state[0],) * self.num_agents

    def sample(self, zero=False):
        return np.random.randint(0, self.action_dim, self.num_agents, dtype=int)

    def __init__(self, config: Config, sandbox) -> None:
        super().__init__(config, sandbox)
        self.work_mode = config['work_mode']
        self.task_queue_length = config['task_queue_length']
        self.num_tasks_total = config['num_tasks_total']
        self.success_reward = config['reward/success_reward']
        self.reach_threshold = config['reach_threshold']
        self.num_tasks_done = 0
        self.task_chosen = [None, ] * self.num_agents
        self.tasks = []
        self.task_profile = config['task_profile']
        self.not_next_frame = True

        for i in range(self.num_agents):
            self.sandbox.register_agent('agent_{}'.format(i), TaskAllocationAgent, self.agents_profile)

    def action_transform(self, actions):
        for i, a in enumerate(actions):
            self.task_chosen[i] = self.tasks[a]
        return [self.tasks[a].pos for a in actions]

    def reset(self) -> None:
        self.tasks = []
        self.num_tasks_done = 0
        self.task_chosen = [None, ] * self.num_agents
        self._add_task()

    def is_termination(self) -> (bool, bool):
        return self.num_tasks_done == self.num_tasks_total, False

    def _add_task(self):
        while (len(self.tasks) < self.task_queue_length) and (
                self.num_tasks_done + len(self.tasks) < self.num_tasks_total):
            self.tasks.append(creat_task_from_profile(self.task_profile))

    def get_state(self):
        agents_states = np.stack(
            [np.concatenate(
                [a.state, np.array([self.task_chosen[i].process if self.task_chosen[i] is not None else 0])])
                for i, a in enumerate(self.sandbox.agents)], axis=0)
        if len(self.tasks) == 0:
            tasks_states = None
        else:
            tasks_states = np.stack([np.concatenate([t.pos, np.array([t.process])]) for t in self.tasks], axis=0)
        return agents_states, tasks_states

    def calculate_reward(self):
        pass

    def skip_frame(self) -> bool:
        return self.not_next_frame

    def step(self):
        statistic = {str(i): 0 for i in self.task_chosen}
        for i, agent in enumerate(self.sandbox.agents):
            t_chosen: Task = self.task_chosen[i]
            if np.linalg.norm(t_chosen.pos - agent.pos) < t_chosen.r + self.reach_threshold:
                statistic[str(t_chosen)] += 1
        for t in self.tasks:
            if str(t) in statistic.keys():
                t.process += self.sandbox.tick * statistic[str(t)]
        old_task_len = len(self.tasks)
        self.tasks = [t for t in self.tasks if t.process < 1]
        done_tasks = old_task_len - len(self.tasks)
        if done_tasks > 0:
            self.not_next_frame = False
            self.num_tasks_done += done_tasks
            print(self.num_tasks_done)
            self._add_task()
        else:
            self.not_next_frame = True


class TaskAllocationRenderer(Renderer):

    def __init__(self, config, sandbox):
        super(TaskAllocationRenderer, self).__init__(config, sandbox)

    def render(self, mode):
        frame_copy = self.sandbox.frame.copy()
        frame_copy = np.stack([frame_copy, frame_copy, frame_copy], axis=-1)
        for agent in self.sandbox.agents:
            pos: np.ndarray = agent.pos.copy()
            pos = pos.astype(int)
            cv2.circle(frame_copy, pos, int(agent.collision_info_['args'][0]), (0, 255, 0), -1)
        for t in self.sandbox.mission_.tasks:
            pos: np.ndarray = t.pos.copy()
            pos = pos.astype(int)
            cv2.circle(frame_copy, pos, int(t.r), (255, 0, 0), 1)
        if mode == 'rgb_array':
            return frame_copy
        elif mode == 'human':
            cv2.imshow(self.sandbox.name, frame_copy)
            cv2.waitKey(1)


TaskAllocationSandBox = partial(SandBox, map_generator=EmptyMapGenerator, mission=TaskAllocationMission,
                                renderer=TaskAllocationRenderer)

import torch
from torch.multiprocessing import Process, Pipe, Semaphore
import numpy as np
from utils import ReplayBuffer


class ParallelContainer:
    def __init__(self, n_env, env_creator):
        self.n_env = n_env
        self.env_creator = env_creator
        self.processes = []
        self.env_pipes: list[Pipe] = []
        self.container_pipe: Pipe = Pipe(duplex=False)
        self.stop_byte = torch.tensor(False).share_memory_()

    def reset(self):
        for i in range(self.n_env):
            p = Pipe(duplex=False)
            self.env_pipes.append(p[1])
            self.processes.append(
                EnvProcess(id=i, env_creator=self.env_creator, pipe_env=p[0], pipe_container=self.container_pipe,
                           stop_flag=self.stop_byte))

        [process.start() for process in self.processes]

        return self._sync_recv()

    def join(self):
        self.stop_byte = True
        for proc, pipe in zip(self.processes, self.env_pipes):
            pipe.send(proc.env.sample(zero=True))
        [process.join() for process in self.processes]

    def step(self, actions):

        for i, (a, p) in enumerate(zip(actions, self.env_pipes)):
            # print('sending action to {}'.format(i))
            p.send(a)
        return self._sync_recv()

    def _sync_recv(self):
        # print('waiting for state')
        sorted_ret = sorted([self.container_pipe[0].recv() for _ in range(self.n_env)], key=lambda x: x[0])
        # print('received {} states'.format(len(sorted_ret)))
        sorted_ret = [ret[1] for ret in sorted_ret]
        ret = tuple(i for i in zip(*sorted_ret))
        ob = ret[:-1]
        ob = [np.stack(i) for i in ob]
        return *ob, ret[-1]


class EnvProcess(Process):
    def __init__(self, id, env_creator, pipe_env, pipe_container, stop_flag):
        super(EnvProcess, self).__init__()
        self.id = id
        self.env = env_creator()
        self.pipe_env = pipe_env
        self.pipe_container = pipe_container[1]
        self.stop_flag = stop_flag

    def run(self):
        ret = self.env.reset()
        self.pipe_container.send((self.id, ret))

        while True:

            if self.stop_flag:
                break
            # print('{} wait to receive action'.format(self.id))
            action = self.pipe_env.recv()
            # print('{} received action'.format(self.id))

            observation, reward, termination, truncation, info = self.env.step(action)

            if termination or truncation:
                ob, info_ = self.env.reset()
                observation = ob
            # print('{} send state'.format(self.id))
            self.pipe_container.send((self.id, (observation, reward, termination, truncation, info)))
        print('process {} terminated'.format(self.id))

import torch
from torch.multiprocessing import Process, Pipe
from utils import ReplayBuffer


class ParallelContainer:
    def __init__(self, n_env, env_creator):
        self.n_env = n_env
        self.env_creator = env_creator
        self.processes = []

    def reset(self):
        for i in range(self.n_env):
            self.processes.append(EnvProcess(env_creator=self.env_creator, pipe=Pipe(duplex=False)))
            self.processes[i].start()
            self.processes[i].join()


class EnvProcess(Process):
    def __init__(self, env_creator, pipe):
        super(EnvProcess, self).__init__()
        self.env = env_creator()
        self.pipe = pipe

    def run(self):
        ret = self.env.reset()
        self.pipe.send(ret)
        while True:
            action = self.pipe.recv()
            ret = self.env.step(action)
            self.pipe.send(ret)

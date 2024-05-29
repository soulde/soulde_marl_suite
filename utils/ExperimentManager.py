import os
import json
from datetime import datetime
from typing import Union
from torch.utils.tensorboard import SummaryWriter
from .Config import Config


class ExperimentManager:
    def __init__(self, directory):
        self.directory = directory
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.args = None
        self.experiments_name = None
        self.tensorboard_writer = None
        self.counter = dict()

    def __del__(self):
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()

    def start_experiment(self, config_url, tag='train'):
        self.args = Config(config_url)
        self.experiments_name = tag + '_' + self.args['/sandbox/name'] + '_' + f'{datetime.now():%Y%m%d_%H%M%S}'
        experiment_url: Union[str, bytes] = os.path.join(self.directory, self.experiments_name)
        os.makedirs(experiment_url, exist_ok=True)
        with open(os.path.join(experiment_url, 'config.json'), 'w') as f:
            json.dump(self.args.as_dict(), f, indent=3)
        os.makedirs(os.path.join(experiment_url, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(experiment_url, 'weights'), exist_ok=True)
        self.args['weights_dir'] = os.path.join(experiment_url, 'weights')
        self.tensorboard_writer = SummaryWriter(log_dir=os.path.join(experiment_url, 'logs'))

    def add_figure(self, tag, fig):
        if tag not in self.counter.keys():
            self.counter[tag] = 0
        self.tensorboard_writer.add_figure(tag, fig, global_step=self.counter[tag])
        self.counter[tag] += 1

    def add_image(self, tag: str, image):
        if tag not in self.counter.keys():
            self.counter[tag] = 0
        self.tensorboard_writer.add_image(tag, image, global_step=self.counter[tag], dataformats='HWC')
        self.counter[tag] += 1

    def add_scalar(self, tag: str, value):
        if tag not in self.counter.keys():
            self.counter[tag] = 0
        self.tensorboard_writer.add_scalar(tag, value, global_step=self.counter[tag])
        self.counter[tag] += 1

    def add_scalars(self, tag: str, value):
        if tag not in self.counter.keys():
            self.counter[tag] = 0
        self.tensorboard_writer.add_scalars(tag, value, global_step=self.counter[tag])
        self.counter[tag] += 1


if __name__ == '__main__':
    experiment_manager = ExperimentManager('../runs')
    experiment_manager.start_experiment('../configs/path_planning.json')

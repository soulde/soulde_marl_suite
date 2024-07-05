import numpy as np
from mujoco import MjData, mj_step, mj_resetData

from .MJCF import MJCF
import mujoco

import time


class Simulation:
    def __init__(self, sim_model: MJCF):
        t1 = time.perf_counter()
        self.xml_model = sim_model

        # MuJoCo data structures
        self.model = sim_model.get_model("mujoco")  # MuJoCo model
        self.data = MjData(self.model)  # MuJoCo data

        t2 = time.perf_counter()
        print("create mujoco model time:", (t2 - t1) * 1000, 'ms')
        t3 = time.perf_counter()
        self.renderer = None
        t4 = time.perf_counter()
        print("create renderer time:", (t4 - t3) * 1000, 'ms')
        self.control_dict = {}
        self.state_dict = {}
        self.observe_dict = {}
        self.body_dict = {}
        # print(self.model.body_quat)
        # print("qpos", self.data.qpos)
        # print("qvel", self.data.qvel)
        # print("control", self.data.ctrl)
        # print("", self.data.userdata)
        self.collect_parts_ids()

        self._initialize()
        for _ in range(400):
            self.step()

    def get_data(self):
        return self.data

    def get_model(self):
        return self.model

    def collect_parts_ids(self):
        for i in self.xml_model.info['ns']:
            self.state_dict[i] = [j[len(i):] for j in self.xml_model.joints if j[:len(i)] == i]
            self.control_dict[i] = [j[len(i):] for j in self.xml_model.actuators if j[:len(i)] == i]
            self.observe_dict[i] = [j[len(i):] for j in self.xml_model.sensors if j[:len(i)] == i]
            self.body_dict[i] = [j[len(i):] for j in self.xml_model.bodies if j[:len(i)] == i]

    def observe(self, robot):
        return np.concatenate([self.data.sensor(robot + i).data for i in self.observe_dict[robot]])

    def monitor(self, robot):
        return np.concatenate([self.data.joint(robot + i).qpos for i in self.state_dict[robot]]), \
            {i: self.data.body(robot + i).xpos for index, i in enumerate(self.body_dict[robot])}

    def control(self, robot, u: np.array):
        ids = [self.data.actuator(robot + i).id for i in self.control_dict[robot]]
        for i, u_i in zip(ids, u):
            if u_i == np.nan:
                continue
            self.data.ctrl[i] = u_i

        # print([self.data.actuator(robot + i) for i in self.control_dict[robot]])
        # assert len(ids) == u.shape[0]

    # def fix_time_step(self, t):

    #     def step_decorator(f):
    #         @ wraps(f)
    #         def wrapper():
    #
    #             simstart = self.data.time
    #             while self.data.time - simstart < t:
    #                 mj_step(self.model, self.data)
    #             return f()
    #         return wrapper
    #     return step_decorator

    @property
    def is_alive(self):
        return self.renderer.is_alive

    # @ fix_time_step(0.01)
    def step(self):
        sim_start = self.data.time
        while self.data.time - sim_start < 0.01:
            mj_step(self.model, self.data)

    def reset(self):
        mj_resetData(self.model, self.data)
        self._initialize()
        for _ in range(400):
            self.step()

    def _initialize(self):
        print(self.xml_model.info['init_qpos'])
        print(self.xml_model.info['init_ctrl'])
        for i in self.xml_model.info['ns']:
            print(i)
            self.robot_reset(i)

    def robot_reset(self, name):
        try:
            if name in self.xml_model.info['init_qpos'].keys():
                for i, d in zip(self.state_dict[name], self.xml_model.info['init_qpos'][name]):
                    print(name)
                    print(self.data.joint(name + i).qpos)
                    # continue

                    self.data.joint(name + i).qpos = d
        except Exception as e:
            print('warning',e)
        try:
            if name in self.xml_model.info['init_ctrl'].keys():
                self.control(name, self.xml_model.info['init_ctrl'][name])
        except Exception as e:
            print('warning', e)

    def robot_initial_state(self, name):
        return self.xml_model.info['init_qpos'][name]

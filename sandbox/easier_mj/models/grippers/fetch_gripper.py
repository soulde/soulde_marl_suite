"""
Gripper with two fingers.
"""
import numpy as np

from ...sim_core.mjcf_utils import xml_path_completion
from .gripper import Gripper


class FetchGripperBase(Gripper):
    """
    Gripper with two fingers.
    """

    def __init__(self, unique_name):
        super().__init__(xml_path_completion("grippers/fetch_gripper.xml"), unique_name)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.05, 0.05])

    @property
    def joints(self):
        return ["r_gripper_finger", "l_gripper_finger"]

    @property
    def sensors(self):
        return ["force_ee", "torque_ee"]

    @property
    def dof(self):
        return 2

    @property
    def visualization_sites(self):
        return []

    @property
    def contact_geoms(self):
        return [
            "gripper_link_collision",
            "r_gripper_finger_link_collision",
            "l_gripper_finger_link_collision",
        ]

    @property
    def left_finger_geoms(self):
        return ["l_gripper_finger_link_collision"]

    @property
    def right_finger_geoms(self):
        return ["r_gripper_finger_link_collision"]


class FetchGripper(FetchGripperBase):
    """
    Modifies two finger base to only take one action.
    """

    def format_action(self, action):
        """
        1 => open, -1 => closed
        """
        assert len(action) == 1
        return np.array([action[0], action[0]])

    @property
    def dof(self):
        return 1

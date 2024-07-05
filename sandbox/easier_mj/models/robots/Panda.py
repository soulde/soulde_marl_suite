import numpy as np

from .robot import Robot
from ..mjcf_utils import array_to_string, xml_path_completion


class Panda(Robot):
    """Panda is a sensitive single-arm robot designed by Franka."""

    def __init__(
            self,
            use_torque=False,
            xml_path="robots/panda/robot.xml"
    ):
        if use_torque:
            xml_path = "robots/panda/robot_torque.xml"
        super().__init__(xml_path_completion(xml_path))
        self._model_name = "panda"
        self._init_qpos = np.array([0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, -np.pi / 4])
        self.bottom_offset = np.array([0, 0, -0.913])
        self._set_joint_frictionloss()

    def set_pose(self, pos=np.array([0, 0, 0]), quat=np.array([1, 0, 0, 0])):
        node = self.worldbody.find("./body[@name='link0']")
        node.set("pos", array_to_string(pos - self.bottom_offset))
        node = self.worldbody.find("./body[@name='link0']")
        node.set("quat", array_to_string(quat))

    def _set_joint_damping(self, damping=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01))):
        """Set joint damping """
        body = self._base_body
        for i in range(len(self._link_body)):
            body = body.find("./body[@name='{}']".format(self._link_body[i]))
            joint = body.find("./joint[@name='{}']".format(self._joints[i]))
            joint.set("damping", array_to_string(np.array([damping[i]])))

    def _set_joint_frictionloss(self, friction=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01))):
        """Set joint friction loss (static friction)"""
        body = self._base_body
        for i in range(len(self._link_body)):
            body = body.find("./body[@name='{}']".format(self._link_body[i]))
            joint = body.find("./joint[@name='{}']".format(self._joints[i]))
            joint.set("frictionloss", array_to_string(np.array([friction[i]])))

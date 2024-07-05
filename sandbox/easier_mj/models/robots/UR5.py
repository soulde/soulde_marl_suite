import numpy as np

from ...sim_core.mjcf_utils import array_to_string, xml_path_completion
from .robot import Robot


class UR5(Robot):
    """Panda is a sensitive single-arm robot designed by Franka."""

    def __init__(
            self,
            unique_name='ur5',
            use_torque=False,
            xml_path="robots/UR5/UR5gripper_2_finger.xml"
    ):
        if use_torque:
            xml_path = "robots/panda/robot_torque.xml"
        self._init_qpos = np.array(
            [np.pi / 2.0, -2.7 * np.pi / 4.0, 3 * np.pi / 5.0, -3*np.pi / 8., -np.pi / 2.0, 3 * np.pi / 4, -np.pi / 4])
        self._init_ctrl = np.array(
            [np.pi / 2.0, -2.7 * np.pi / 4.0, 3 * np.pi / 5.0, -3*np.pi / 8., -np.pi / 2.0, 3 * np.pi / 4, -np.pi / 4])

        self.bottom_offset = np.array([0, 0, -0.913])

        self._model_name = "ur5"
        super().__init__(xml_path_completion(xml_path), unique_name)
        self.set_joint_damping()
        # Careful of init_qpos -- certain init poses cause ik controller to go unstable (e.g: pi/4 instead of -pi/4
        # for the final joint angle)

    def set_base_xpos(self, pos):
        """Places the robot on position @pos."""
        node = self.worldbody.find("./body[@name='{}box_link']".format(self.prefix))
        node.set("pos", array_to_string(pos - self.bottom_offset))

    def set_base_xquat(self, quat):
        """Places the robot on position @quat."""
        node = self.worldbody.find("./body[@name='{}box_link']".format(self.prefix))
        node.set("quat", array_to_string(quat))

    def set_joint_damping(self, damping=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01))):
        """Set joint damping """
        body = self._base_body.find("./body[@name='{}']".format(self._link_body[0]))
        for i in range(1, len(self._link_body)):
            body = body.find("./body[@name='{}']".format(self._link_body[i]))
            try:
                joint = body.find("./joint[@name='{}']".format(self._joints[i - 1]))
                joint.set("damping", array_to_string(np.array([damping[i - 1]])))
            except AttributeError:
                pass

    def set_joint_frictionloss(self, friction=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01))):
        """Set joint friction loss (static friction)"""
        body = self._base_body.find("./body[@name='{}']".format(self._link_body[0]))
        for i in range(1, len(self._link_body)):
            body = body.find("./body[@name='{}']".format(self._link_body[i]))
            try:
                joint = body.find("./joint[@name='{}']".format(self._joints[i - 1]))
                joint.set("frictionloss", array_to_string(np.array([friction[i - 1]])))
            except UnboundLocalError:
                pass

    @property
    def dof(self):
        return 6

    @property
    def joints(self):
        raise NotImplementedError
        # return ["joint{}".format(x) for x in range(1, 8)]

    @property
    def init_qpos(self):
        return self._init_qpos

    @property
    def init_ctrl(self):
        return self._init_ctrl

    @property
    def contact_geoms(self):
        raise NotImplementedError
        # return ["link{}_collision".format(x) for x in range(1, 8)]

    @property
    def _base_body(self):
        node = self.worldbody.find("./body[@name='{}box_link']".format(self.prefix))
        return node

    @property
    def _link_body(self):

        return [self.prefix + i for i in
                ["base", "shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link",
                 "wrist_3_link"]]

    @property
    def _joints(self):
        return [self.prefix + i for i in
                ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
                 "wrist_3_joint"]]

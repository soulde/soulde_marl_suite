import numpy as np

from ...sim_core.mjcf_utils import array_to_string, xml_path_completion
from .robot import Robot


class Jaco(Robot):
    """Jaco is a single-arm robot designed by Kinova."""

    def __init__(
            self,
            unique_name,
            use_torque=False,
            xml_path="robots/jaco/robot.xml",
    ):
        if use_torque:
            xml_path = "robots/jaco/robot_torque.xml"
        super().__init__(xml_path_completion(xml_path), unique_name)

        self.bottom_offset = np.array([0, 0, -0.913])

    def set_base_xpos(self, pos):
        """
        Places the robot on position @pos.
        """
        node = self.worldbody.find("./body[@name='{}jaco_link_base']".format(self.prefix))
        node.set("pos", array_to_string(pos - self.bottom_offset))

    def set_base_xquat(self, quat):
        """
        Places the robot on position @quat.
        """
        node = self.worldbody.find("./body[@name='{}jaco_link_base']".format(self.prefix))
        node.set("quat", array_to_string(quat))

    @property
    def dof(self):
        return 6

    @property
    def joints(self):
        return ["jaco_joint_{}".format(x) for x in range(1, 7)]

    @property
    def init_qpos(self):
        return np.array([0, np.pi * 3 / 4, -np.pi * 1 / 3, 0, 0.5, 0])

    @property
    def contact_geoms(self):
        return ["jaco_link_geom_{}".format(x) for x in range(7)]

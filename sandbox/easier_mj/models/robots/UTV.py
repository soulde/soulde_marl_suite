import numpy as np

from ...sim_core.mjcf_utils import array_to_string, xml_path_completion
from .robot import Robot


class UTV(Robot):
    """Panda is a sensitive single-arm robot designed by Franka."""

    def __init__(
            self,
            unique_name='utv',
            xml_path="robots/UTV/robot.xml"
    ):
        self.bottom_offset = np.array([0, 0, -1.05])

        self._model_name = "utv"
        # Careful of init_qpos -- certain init poses cause ik controller to go unstable (e.g: pi/4 instead of -pi/4
        # for the final joint angle)

        self._init_ctrl = np.array([0, 0, 0, 0])
        super().__init__(xml_path_completion(xml_path), unique_name)

    def set_base_xpos(self, pos):
        """Places the robot on position @pos."""
        node = self.worldbody.find("./body[@name='{}base']".format(self.prefix))
        node.set("pos", array_to_string(pos - self.bottom_offset))

    def set_base_xquat(self, quat):
        """Places the robot on position @quat."""
        node = self.worldbody.find("./body[@name='{}base']".format(self.prefix))
        node.set("quat", array_to_string(quat))

    @property
    def dof(self):
        return 3

    @property
    def joints(self):
        raise NotImplementedError
        # return ["joint{}".format(x) for x in range(1, 8)]

    @property
    def init_qpos(self):
        return self._init_qpos

    @property
    def init_ctrl(self):
        return self._init_qpos

    @property
    def contact_geoms(self):
        raise NotImplementedError
        # return ["link{}_collision".format(x) for x in range(1, 8)]

    @property
    def _base_body(self):
        node = self.worldbody.find("./body[@name='{}base']".format(self.prefix))
        return node

    @property
    def _link_body(self):
        return [self.prefix + i for i in ["base_link", 'torso_lift_link', 'estop_link', 'laser_link']]

    @property
    def _joints(self):
        return [self.prefix + i for i in
                []]

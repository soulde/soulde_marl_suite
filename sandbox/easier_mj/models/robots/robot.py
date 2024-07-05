from easier_mj.sim_core.MJCF import MJCF


# only 1dof rotate joint support


class Robot(MJCF):
    """Base class for all robot models."""

    def __init__(self, url, unique_name):
        """Initializes from file @fname."""
        super().__init__(unique_name=unique_name, url=url)
        # key: gripper name and value: gripper model
        self.gripper = None
        try:
            self.info['init_qpos'] = {unique_name: self.init_qpos}

        except:
            pass
        try:
            self.info['init_ctrl'] = {unique_name: self.init_ctrl}
        except:
            pass
        # self._joints = dict()
        # self.find_arm_path()

    def find_arm_path(self, arm_root):
        terminal = list(self.worldbody.find(arm_root).iter('body'))[-1]
        print(terminal)

    def add_gripper(self, arm_name, gripper):
        """
        Mounts gripper to arm.

        Throws error if robot already has a gripper or gripper type is incorrect.

        Args:
            arm_name (str): name of arm mount
            gripper (MujocoGripper instance): gripper MJCF model
        """
        if self.gripper is not None:
            raise ValueError("gripper has been attached")

        self.attach(arm_name, gripper, 'all')

        self.gripper = gripper

    def is_robot_part(self, geom_name):
        """
        Checks if @geom_name is part of robot.
        """
        is_robot_geom = False

        # check geoms of robot.
        if geom_name in self.contact_geoms:
            is_robot_geom = True

        # check geoms of grippers.
        for gripper in self.gripper.values():
            if geom_name in gripper.contact_geoms:
                is_robot_geom = True

        return is_robot_geom

    @property
    def dof(self):
        """Returns the number of DOF of the robot, not including gripper."""
        raise NotImplementedError

    @property
    def joints(self):
        """Returns a list of joint names of the robot."""
        raise NotImplementedError
        return list(self._joints.keys())

    @property
    def init_qpos(self):
        """Returns default qpos."""
        raise NotImplementedError

    @property
    def init_ctrl(self):
        """Returns default ctrl."""
        raise NotImplementedError

import time

from .world import BasicWorld

from ..grippers import PandaGripper, Robotiq85Gripper
from ..robots import Panda, UTV, UR5
from ..objects import BoxObject, Assembly, Floor, Table


class CollisionWorld(BasicWorld):
    def __init__(self):
        super().__init__()
        floor_full_size = (1.5, 1.0)
        floor_friction = (2.0, 0.005, 0.0001)
        floor = Floor("floor")
        box = BoxObject([1.5, 1.5, 1.5])
        box.set_pos([1, 1.5, 3])
        table = Table('table')
        self.add_object_to_world(box)
        self.add_object_to_world(floor)
        self.add_object_to_world(table)


class GraspTestWorld(BasicWorld):
    def __init__(self):
        super().__init__()
        floor = Floor("floor")

        ur5 = UR5('ur5')
        utv = UTV('utv')
        box = BoxObject([0.03, 0.03, 0.03])
        gripper = Robotiq85Gripper('robotiq85_gripper')
        ur5.add_gripper('hand', gripper)
        utv.attach('arm_install_block', ur5, 'base')
        utv.set_base_xpos([0.4, -0.1, 0.1])
        utv.set_base_xquat([1, 0, 0, 0])
        self.add_object_to_world(floor)
        self.add_object_to_world(box)
        self.add_object_to_world(utv)



class TestSimWorld(BasicWorld):
    def __init__(self):
        super().__init__()

        t1 = time.perf_counter()
        floor_full_size = (1.5, 1.0)
        floor_friction = (2.0, 0.005, 0.0001)
        floor = Floor("floor")

        ur5_1 = UR5('ur5e_1')
        utv_1 = UTV('utv_1')

        ur5_2 = UR5('ur5e_2')
        utv_2 = UTV('utv_2')

        box = BoxObject([0.02, 0.02, 0.5])
        table = Table('table')
        furniture = Assembly('table_bjorkudden_0207')
        # furniture.set_base_xpos([0, 0, 0.3])
        box = BoxObject([0.03, 0.03, 0.03])
        gripper1, gripper2 = Robotiq85Gripper('robotiq85_gripper1'), Robotiq85Gripper('robotiq85_gripper2')
        ur5_1.add_gripper('hand', gripper1)
        ur5_2.add_gripper('hand', gripper2)
        # # robot1 = UR5('ur5')
        # robot1.set_base_xpos([0.75, 0.0, -0.5])
        # robot1.set_base_xquat([0.707, 0, 0, 0.707])
        print(ur5_1)
        utv_1.attach('arm_install_block', ur5_1, 'base')
        utv_2.attach('arm_install_block', ur5_2, 'base')
        # del robot1
        utv_1.set_base_xpos([0.4, -0.1, 0.1])
        utv_1.set_base_xquat([1, 0, 0, 0])
        utv_2.set_base_xpos([-0.4, 0.1, 0.1])
        utv_2.set_base_xquat([0, 0, 0, 1])
        table.set_base_xpos([0, 0, -3])

        # robot2 = Fetch('fetch')
        # gripper2 = FetchGripper('fetch_gripper2')
        # # robot2.set_base_xpos([-0.7, 0, 0])
        # # robot2.set_base_xquat([1, 0, 0, 0])
        # robot2.add_gripper('right_hand', gripper2)
        # self.add_object_to_world(furniture)
        # self.add_object_to_world(box)
        self.add_object_to_world(floor)
        # self.add_object_to_world(robot1)
        self.add_object_to_world(utv_1)
        self.add_object_to_world(utv_2)
        self.add_object_to_world(table)

        t2 = time.perf_counter()
        print("create model time:", (t2 - t1) * 1000, 'ms')

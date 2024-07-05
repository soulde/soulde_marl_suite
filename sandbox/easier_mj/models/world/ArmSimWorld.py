import time

from .world import BasicWorld

from ..grippers import PandaGripper
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


class ArmSimWorld(BasicWorld):
    def __init__(self):
        super().__init__()

        t1 = time.perf_counter()
        floor_full_size = (1.5, 1.0)
        floor_friction = (2.0, 0.005, 0.0001)
        floor = Floor("floor")

        ur5_1 = UR5('ur5_1')
        utv_1 = UTV('utv_1')

        ur5_2 = UR5('ur5_2')
        utv_2 = UTV('utv_2')

        box = BoxObject([0.02, 0.02, 0.5])
        table = Table('table')
        furniture = Assembly('table_bjorkudden_0207')
        # furniture.set_base_xpos([0, 0, 0.3])
        box = BoxObject([0.03, 0.03, 0.03])

        # # robot1 = UR5('ur5')
        # robot1.set_base_xpos([0.75, 0.0, -0.5])
        # robot1.set_base_xquat([0.707, 0, 0, 0.707])

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

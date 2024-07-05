import numpy as np

from ...sim_core.MJCF import MJCF
import xml.etree.ElementTree as ET
from ...sim_core.mjcf_utils import array_to_string


class MujocoWorldBase(MJCF):
    """Base class to inherit all mujoco worlds from."""

    def __init__(self):
        super().__init__(unique_name='base')
        self.assemblies = []

    def _add_basic_body(self, name, pos, type, size):
        body = ET.Element('body', {'name': name, 'pos': pos})
        body.append(
            ET.Element('geom', {'pos': '0 0 0', 'size': size, 'type': type, 'group': "0", 'name': name + '_collision'}))
        body.append(ET.Element('geom', {'pos': '0 0 0', 'size': size, 'type': type, 'conaffinity': '0', 'contype': '0',
                                        'group': "1", 'name': name + '_visual'}))
        self.worldbody.insert(body)

    def set_background(self):
        self.asset.append(
            ET.Element('texture', {'builtin': "gradient", 'height': '256', 'rgb2': '1 1 1', 'rgb1': '0.733 0.149 0.286',
                                   'type': 'skybox', 'width': "256"}))

    def add_camera(self):
        pass

    def add_light(self, pos):
        self.worldbody.append(ET.Element('light', {'diffuse': ".8 .8 .8", 'dir': ".5 -.3 -.8", 'directional': "true",
                                                   'pos': array_to_string(pos), 'specular': "0.3 0.3 0.3",
                                                   'castshadow': "true"}))

    def add_object_to_world(self, object_):
        self.attach('root', object_, 'all')

    def add_assembly_to_world(self, object_):
        self.attach('root', object_, 'all')
        self.assemblies.append(object_)

    def set_radian(self):
        self.compiler.set('angle', 'radian')


class BasicWorld(MujocoWorldBase):
    def __init__(self):
        super().__init__()
        self.set_background()
        self.add_light(np.array([0, 0, 2.0]))
        self.set_radian()


if __name__ == '__main__':
    basic_world = BasicWorld()
    basic_world.set_background()

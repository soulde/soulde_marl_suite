import math
import xml.etree.ElementTree as ET

import numpy as np

from .object import Object
from easier_mj.sim_core.mjcf_utils import array_to_string

'''
<body name="Box" pos="0 0.0085 0.056">
                    <inertial pos="0 0 0" quat="0 0 0 1" mass="0.01" diaginertia="0.01 0.01 0.01" />
                    <geom size="size" pos="0 -0.005 -0.015" quat="0 0 0 1" type="box" solref="0.01 0.5" friction = "2 0.05 0.0001" conaffinity="1" contype="1" name="box_tip_collision"/>
                </body>
'''


class BoxObject(Object):
    def __init__(self, size, unique_name='box', rgba=None, friction=10, density=10000):
        super().__init__(unique_name=unique_name, url=None)
        if rgba is None:
            self.rgba_ = [1, 0, 0, 1]
        self.size_ = size
        self.friction_ = friction
        self.parts['worldbody'].append(self.get_collision(np.array([0.05, -0.2, 0.8]), 'box_collision', False))

    def set_pos(self, pos):
        self.parts['worldbody'].find('body').attrib['pos'] = array_to_string(pos)

    def get_bottom_offset(self):
        return np.array([0, 0, -self.size_[-1] / 2])

    def get_top_offset(self):
        return np.array([0, 0, self.size_[-1] / 2])

    def get_horizontal_radius(self):
        return math.sqrt(self.size_[0] ** 2 + self.size_[1] ** 2)

    def get_collision(self, pos, name=None, site=False):
        collision = ET.Element('body', attrib={'pos': array_to_string(pos)})
        if name is not None:
            collision.attrib["name"] = name
        collision.append(ET.Element('geom', attrib={
            'size': array_to_string(self.size_),
            'pos': '0 0 0',
            'quat': array_to_string([1, 0, 0, 0]),
            'type': "box",
            'friction': array_to_string([1, 0.005, 0.0001]),
            'conaffinity': "1",
            'contype': "1",
            'condim': "4"
        }))
        collision.append(ET.Element('freejoint'))
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            template["rgba"] = "1 0 0 0"
            if name is not None:
                template["name"] = name
            collision.append(ET.Element("site", attrib=template))
        return collision

    def get_visual(self, name=None, site=False):
        visual = ET.Element('body')
        if name is not None:
            visual.attrib["name"] = name

        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            template["rgba"] = "1 0 0 0"
            if name is not None:
                template["name"] = name
            visual.append(ET.Element("site", attrib=template))
        return visual

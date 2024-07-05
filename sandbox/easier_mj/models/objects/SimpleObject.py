from .object import Object
from ...sim_core.mjcf_utils import array_to_string, xml_path_completion


class SimpleObject(Object):
    def __init__(self, type_, unique_name=None):
        super().__init__(url=xml_path_completion('objects/{}.xml'.format(type_)),
                         unique_name=str(type_) if unique_name is None else unique_name)

    def set_base_xpos(self, pos):
        """Places the robot on position @pos."""
        try:
            node = self.worldbody.find("./body[@name='{}base']".format(self.prefix))
            node.set("pos", array_to_string(pos))
        except:
            print(Warning('this object can not set base position'))

    def set_base_xquat(self, quat):
        """Places the robot on position @quat."""
        try:
            node = self.worldbody.find("./body[@name='{}base']".format(self.prefix))
            node.set("quat", array_to_string(quat))
        except:
            print(Warning('this object can not set base quaternion'))




Floor = lambda unique_name: SimpleObject('floor', unique_name=unique_name)
Table = lambda unique_name: SimpleObject('table', unique_name=unique_name)
Sink = lambda unique_name: SimpleObject('sink', unique_name=unique_name)
Bullets = lambda unique_name: SimpleObject('bullets', unique_name=unique_name)

from .SimpleObject import SimpleObject
from ...sim_core.mjcf_utils import array_to_string, xml_path_completion, string_to_array
import re
import networkx as nx


class Assembly(SimpleObject):
    def __init__(self, type, unique_name=None):
        super().__init__(type, unique_name)
        items = self.parts['custom'].findall(".//numeric[@name]")
        init_pos = dict()
        for b in self.bodies:
            for i in items:
                if i.get("name").startswith(b):
                    data = string_to_array(i.get("data"))
                    init_pos[b] = (data[:3], data[3:])
                    break

        sites = self.parts['worldbody'].findall(".//site")
        self.connect_graph = nx.DiGraph()
        for site in sites:
            ret = re.match('{}[a-z]+-[a-z]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,conn_site[0-9]+'.format(self.prefix),
                           site.get("name"))
            if ret is None:
                continue
            info = ret.string[self.prefix_len:]
            info = info.split(',')
            info = (info[0].split('-'), info[1:-1], info[-1], site.get('name'))
            print(info)

    def set_part_init_pos(self, part, pos):
        pass

    def set_part_init_quat(self, quat):
        pass

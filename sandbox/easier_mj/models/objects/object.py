from easier_mj.sim_core.MJCF import MJCF


class Object(MJCF):
    def __init__(self, url, unique_name):
        super().__init__(url=url, unique_name=unique_name)

import io
import os
import xml.dom.minidom
import xml.etree.ElementTree as ET

unique_tags = ['compiler', 'visual', ]
no_atrrib_tags = ['worldbody', 'deformable', 'asset', 'contact', 'default', 'equality', 'tendon', 'actuator', 'sensor',
                  'keyframe', 'custom', 'extension']


class MJCF(object):
    def __init__(self, unique_name='', url=None):
        """
        Loads a mujoco xml from file.

        Args:
            url (str): path to the MJCF xml file.
        """

        self.info = dict()
        self.parts = dict()

        self._creat_empty()
        if url is not None:
            self.file = url
            self.folder = os.path.dirname(url)
            self.tree = ET.parse(url)
            self.root = self.tree.getroot()
            self.parts = {i: self.create_default_element(i) for i in
                          no_atrrib_tags + unique_tags}

        self.prefix = str(unique_name) if unique_name != '' else self.root.attrib['model']
        self.prefix_len = len(self.prefix)

        self.resolve_asset_dependency()

        self._add_prefix4all()
        self.collect_info()

    def __str__(self):
        return ET.tostring(self.root, encoding='unicode')

    def __getattr__(self, attr):
        if attr in no_atrrib_tags + unique_tags:
            return self.parts[attr]
        else:
            raise AttributeError('No such attribute ' + attr)

    def _creat_empty(self):
        self.tree = ET.ElementTree(ET.Element('mujoco', {'model': ''}))
        self.root = self.tree.getroot()

        for i in no_atrrib_tags + unique_tags:
            self.parts[i] = ET.Element(i)
            self.root.append(self.parts[i])

    # <geom pos="0 0 0" size="0.4 0.4 0.4" type="box" conaffinity="0" contype="0" group="1" name="table_visual"
    #                   material="light-wood"/>

    def collect_info(self):
        if self.prefix == '':
            return None
        self.info['ns'] = [self.prefix]
        # print('=========================  ' + self.prefix + '  =============================')
        # print('actuator:')
        self.info['actuator'] = [node.attrib['name'] for node in self.parts['actuator'].findall('.//*')]
        # print('sensor:')
        self.info['sensor'] = [node.attrib['name'] for node in self.parts['sensor'].findall('.//*')]

        # print('body:')
        self.info['joint'] = [node.attrib['name'] for node in self.parts['worldbody'].iter('joint')]
        self.info['body'] = [node.attrib['name'] for node in self.parts['worldbody'].iter('body')]
        # print(self.info)

    @property
    def joints(self):
        return list(self.info['joint'])

    @property
    def actuators(self):
        return list(self.info['actuator'])

    @property
    def sensors(self):
        return list(self.info['sensor'])

    @property
    def bodies(self):
        return list(self.info['body'])

    def _add_prefix4all(self):
        pre = self._model_name if '_model_name' in self.__dir__() else self.prefix
        local_default = ET.Element('default', {'class': pre})
        for item in self.default:
            try:
                if item.attrib['class'] == pre:
                    local_default = item
                    break
            except:
                pass
            local_default.append(item)

        self.default.clear()
        self.default.append(local_default)
        for node in self.root.iter():
            if node.tag in ['mesh', 'texture', 'material', 'default']:
                continue
            # if node.tag in ['exclude']:
            #     for i in node.attrib:
            #         # print(i)
            #         node.set(i, self.prefix + node.attrib.get(i))
            # print(node.attrib.get(i))
            for attr in ['name', 'joint', 'site', 'target', 'joint1', 'joint2', 'body1',
                         'body2', 'objname']:
                name = node.attrib.get(attr)

                if name is not None:
                    node.set(attr, self.prefix + name)

    def resolve_asset_dependency(self):
        """
        Converts every file dependency into absolute path so when we merge we don't.xml break things.
        """

        for node in self.parts['asset'].findall("./*[@file]"):
            file = node.get("file")
            abs_path = os.path.abspath(self.folder)
            abs_path = os.path.join(abs_path, file)
            node.set("file", abs_path)

    def create_default_element(self, name):
        """
        Creates a <@name/> tag under root if there is none.
        """

        found = self.root.find(name)
        if found is not None:
            return found
        ele = ET.Element(name)
        self.root.append(ele)
        return ele

    def attach(self, body_parent: str, object_from, body_child):
        node_parent = self.parts['worldbody'].find(".//body[@name='{}']".format(self.prefix + body_parent)) \
            if body_parent != 'root' else self.parts['worldbody']
        node_child = object_from.parts['worldbody'].find(".//body[@name='{}']".format(object_from.prefix + body_child)) \
            if body_child != 'all' else object_from.parts['worldbody']

        if body_child == 'all':
            for body in node_child:
                node_parent.append(body)
        else:
            node_child.attrib['pos'] = '0 0 0'
            node_parent.append(node_child)

        for item in ["actuator", "asset", "equality", "sensor", "contact", "default", "custom", 'tendon']:
            try:
                for i in object_from.parts[item]:
                    code = 'name' if item != 'default' else 'class'
                    name_ = i.get(code)
                    type_ = i.tag

                    # Avoids duplication
                    pattern = "./{}[@{}='{}']".format(type_, code, name_)
                    if self.parts[item].find(pattern) is None:
                        self.parts[item].append(i)
                    elif item != 'asset':
                        print('Warning: find duplication in {}, model parts may missed!!!!'.format(i))
            except Exception as e:
                pass
        try:
            for i in object_from.info.keys():

                if i in self.info.keys():
                    if isinstance(self.info[i], dict):
                        self.info[i].update(object_from.info[i])
                    else:
                        self.info[i] += object_from.info[i]

                else:
                    self.info[i] = object_from.info[i]

        except:
            pass

    # def merge(self, other, merge_body=True):
    #     """
    #     Default merge method.
    #
    #     Args:
    #         other: another MujocoXML instance
    #             raises XML error if @other is not a MujocoXML instance.
    #             merges <worldbody/>, <actuator/> and <asset/> of @other into @self
    #         merge_body: True if merging child bodies of @other. Defaults to True.
    #     """
    #     # if not isinstance(other, MujocoXML):
    #     #     raise XMLError("{} is not a MujocoXML instance.".format(type(other)))
    #     if merge_body:
    #         for body in other.worldbody:
    #             self.worldbody.append(body)
    #     self.merge_asset(other)
    #     for one_actuator in other.actuator:
    #         self.actuator.append(one_actuator)
    #     for one_equality in other.equality:
    #         self.equality.append(one_equality)
    #     for one_sensor in other.sensor:
    #         self.sensor.append(one_sensor)
    #     for one_contact in other.contact:
    #         self.contact.append(one_contact)
    #     for one_default in other.default:
    #         self.default.append(one_default)
    #     # self.config.append(other.config)
    #     for i in other.info.keys():
    #         if i in self.info.keys():
    #             self.info[i] += other.info[i]
    #         else:
    #             self.info[i] = other.info[i]

    def get_model(self, mode="mujoco"):
        """
        Returns a MjModel instance from the current xml tree.
        """

        available_modes = ["mujoco"]

        if mode == "mujoco":
            from mujoco import MjModel
            xml_str = ET.tostring(self.root, encoding="unicode")
            print(xml_str)
            model: MjModel = MjModel.from_xml_string(xml_str)

            return model
        raise ValueError(
            "Unkown model mode: {}. Available options are: {}".format(
                mode, ",".join(available_modes)
            )
        )

    def get_xml(self):
        """
        Returns a string of the MJCF XML file.
        """
        with io.StringIO() as string:
            string.write(ET.tostring(self.root, encoding="unicode"))
            return string.getvalue()

    def save_model(self, fname, pretty=False):
        """
        Saves the xml to file.

        Args:
            fname: output file location
            pretty: attempts!! to pretty print the output
        """
        with open(fname, "w") as f:
            xml_str = ET.tostring(self.root, encoding="unicode")
            if pretty:
                # TODO: get a better pretty print library
                parsed_xml = xml.dom.minidom.parseString(xml_str)
                xml_str = parsed_xml.toprettyxml(newl="")
            f.write(xml_str)

    # def merge_asset(self, other):
    #     """
    #     Useful for merging other files in a custom logic.
    #     """
    #     for asset in other.asset:
    #         asset_name = asset.get("name")
    #         asset_type = asset.tag
    #         # Avoids duplication
    #         pattern = "./{}[@name='{}']".format(asset_type, asset_name)
    #         if self.asset.find(pattern) is None:
    #             self.asset.append(asset)

    # def
    def get_children_names(self):
        if self.debug:
            print('Reading object xml')
        names = []
        for child in self.root.iter("body"):
            if self.debug:
                print('\t', child.tag, child.get('name'))
            names.append(child.get('name'))
        return names

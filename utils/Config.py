import json
import os


class Config:
    def __init__(self, url: str or dict):
        if isinstance(url, dict):
            self._data: dict = url
        else:
            with open(url, 'r') as f:
                self._data = json.load(f)

    def __setitem__(self, key, value):
        self._data[key] = value

    def __getitem__(self, key):
        keys_list = key.split('/')
        ret = self._data
        for k in keys_list:
            if k == '':
                continue
            if k in ret.keys():
                ret = ret[k]
            else:
                raise KeyError('Key {} not found'.format(k))
        if isinstance(ret, dict):
            return Config(ret)
        else:
            return ret

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def as_dict(self):
        return self._data


if __name__ == '__main__':
    config = Config('../configs/defending.json')
    print(config.as_dict())

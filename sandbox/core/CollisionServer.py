import numpy as np
from abc import ABC, abstractmethod
from itertools import combinations


def corners(p):
    return np.round(p).astype(int)


class Shape(ABC):
    shape = (
        'circle',
        'occupancy'
    )

    def __init__(self, *args):
        self.args = args
        self.mask = None
        self.pos = np.zeros(2)

    @abstractmethod
    def is_point_inside(self, pos):
        raise NotImplementedError

    def set_position(self, pos):
        assert self.pos.shape == pos.shape
        self.pos = pos


class Circle(Shape):
    type = 'circle'

    def __init__(self, *args):
        super(Circle, self).__init__(*args)
        assert len(args) == 1
        self.radius = args[0]
        self.pos = np.zeros(2)
        self.mask = self.radius * np.stack([np.array([np.cos(theta), np.sin(theta)]) for theta in
                                            np.linspace(0, 2 * np.pi, 180)], axis=0)


    def is_point_inside(self, pos):
        return np.linalg.norm(self.pos - pos) < self.radius


class Occupancy(Shape):
    type = 'occupancy'

    def __init__(self, *args):
        super(Occupancy, self).__init__(*args)
        self.mask = np.stack(np.where(args[0] == 0), axis=1)

    def is_point_inside(self, pos):
        return pos.astype(int) in self.mask


class CollisionServer:
    shape = {'circle': Circle,
             'occupancy': Occupancy,
             }

    def __init__(self):
        self.bodies: dict[str:Shape] = {}
        self.backgound = None

    #     self.map = None

    # def set_occupancy(self, oc_map):
    #     self.map = oc_map

    def reset(self):
        self.bodies = {}

    def register(self, name, init_pos=np.zeros(2), collision_type='circle', *args):
        try:
            shape = CollisionServer.shape[collision_type](*args)
        except KeyError:
            raise KeyError("No Collision Type {}".format(collision_type))

        shape.set_position(init_pos)
        self.bodies[name] = shape

    def unregister(self, name):
        self.bodies.pop(name)

    def set_background(self, background):
        self.backgound = background

    def update_pos(self, name, pos):
        self.bodies[name].set_position(pos)

    def check_collision_all(self):
        collision_info = []
        for name, body in self.bodies.items():
            if self.background_check(body):
                collision_info.append((name, 'env'))

        for name1, name2 in combinations(self.bodies.keys(), 2):
            if self.collide(self.bodies[name1], self.bodies[name2]):
                collision_info.append((name1, name2))
        return collision_info

    def check_pair_collision(self, name1, name2):
        ret = self.collide(self.bodies[name1], self.bodies[name2])
        if ret:
            return True
        return False

    def collide_check(self, name):
        for k, v in self.bodies.items():
            if k == name:
                continue
            ret = self.collide(self.bodies[name], v)
            if ret:
                return True
        return False

    def point_collision(self, pos):
        for shape in self.bodies.values():
            if pos in set(tuple(np.round(p).astype(int)) for p in (shape.mask + shape.pos)):
                return True
        if self.backgound[pos[1], pos[0]] == 0:
            return True
        return False

    @property
    def collision_view(self):
        return {k: body.getTransform().getTranslation() for k, body in self.bodies.items()}

    def collide(self, shape1, shape2):
        p1 = set(tuple(np.round(p).astype(int)) for p in (shape1.mask + shape1.pos))
        p2 = set(tuple(np.round(p).astype(int)) for p in (shape2.mask + shape2.pos))

        return len(p1 & p2) > 0

    def background_check(self, shape):
        if self.backgound is None:
            print("Background is not set")
            return False

        for i in set(tuple(np.round(p).astype(int)) for p in (shape.mask + shape.pos)):
            try:
                if self.backgound[i[1], i[0]] == 0:
                    return True
            except IndexError as e:
                return True

        return False


if __name__ == '__main__':
    server = CollisionServer()
    server.register('obs1', np.array([0, 0]), 'sphere', 1)

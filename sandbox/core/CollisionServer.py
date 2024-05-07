import numpy as np
from abc import ABC, abstractmethod


class Shape(ABC):
    shape = (
        'circle',
    )

    def __init__(self, *args):
        self.args = args

    @abstractmethod
    def is_point_inside(self, pos):
        raise NotImplementedError


class Circle(Shape):
    type = 'circle'

    def __init__(self, *args):
        super(Circle, self).__init__(*args)
        assert len(args) == 1
        self.radius = args[0]
        self.pos = np.zeros(2)

    def set_position(self, pos):
        assert self.pos.shape == pos.shape
        self.pos = pos

    def is_point_inside(self, pos):
        return np.linalg.norm(self.pos - pos) < self.radius


class CollisionServer:
    shape = {'circle': Circle}

    def __init__(self):
        self.bodies: dict[str:Shape] = {}

    def reset(self):
        self.bodies = {}

    def register(self, name, init_pos, collision_type='circle', *args):
        try:
            shape = CollisionServer.shape[collision_type](*args)
        except KeyError:
            raise KeyError("No Collision Type {}".format(collision_type))

        shape.set_position(init_pos)
        self.bodies[name] = shape

    def unregister(self, name):
        self.bodies.pop(name)

    def update_pos(self, name, pos):
        self.bodies[name].set_position(pos)

    def check_collision_all(self):
        collision_info = []
        for name1 in self.bodies.keys():
            for name2 in self.bodies.keys():
                if name1 == name2:
                    continue
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
        for body in self.bodies.values():
            if body.is_point_inside(pos):
                return True
        return False

    @property
    def collision_view(self):
        return {k: body.getTransform().getTranslation() for k, body in self.bodies.items()}

    def collide(self, shape1, shape2):
        if shape1.type == shape2.type == 'circle':
            return np.linalg.norm(shape1.pos - shape2.pos) < shape1.radius + shape2.radius
        else:
            raise NotImplementedError


if __name__ == '__main__':
    server = CollisionServer()
    server.register('obs1', np.array([0, 0]), 'sphere', 1)

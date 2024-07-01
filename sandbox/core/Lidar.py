import numpy as np


class Lidar:
    def __init__(self, num_lines, sep, range):
        self.check_points = np.array([i * np.array([np.cos(line), np.sin(line)]) for i in
                                      np.linspace(range[0], range[1], (range[1] - range[0]) / sep) for line in
                                      np.linspace(0, 2 * np.pi, num_lines)])
        self.range = range
        self.sep = sep

    def detect(self, pos, check_obstacle_fn):
        ck_pts = self.check_points + pos

        def check_line(line):
            for i, p in enumerate(line):
                if check_obstacle_fn(p):
                    return self.range[0] + i * self.sep
        return np.array(map(check_line, ck_pts))

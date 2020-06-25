"""
We frame the finding a point that is closest to a set of lines as a least square problem.
More details can be found here:
https://math.stackexchange.com/questions/36398/point-closest-to-a-set-four-of-lines-in-3d
"""

import numpy as np


class Cluster:
    def __init__(self, center):
        self.assignments = []
        self.center = center
        self.A = np.zeros((3, 3))
        self.b = np.zeros((3, 1))

    def add(self, x):
        self.assignments.append(x)
        P = np.eye(3) - np.dot(x.direction.T, x.direction)
        self.b += P.dot(x.x0.T)
        self.A += P
        x.assigned_cluster = self

    def subtract(self, x):
        self.remove_assignment_by_id(x.id)
        P = np.eye(3) - np.dot(x.direction.T, x.direction)
        self.b -= P.dot(x.x0.T)
        self.A -= P
        x.assigned_cluster = None

    def compute_center(self):
        x, _, _, _ = np.linalg.lstsq(self.A, self.b, rcond=-1)
        self.center = x.T  # [3,1] -> [1,3] 
        return np.squeeze(x)

    def remove_assignment_by_id(self, id_):
        before_remove = len(self.assignments)
        self.assignments = [f for f in self.assignments if f.id != id_]
        after_remove = len(self.assignments)
        assert before_remove - after_remove == 1, "remove more than one element"

    def __repr__(self):
        string = "cluster center: {} {} {}, # assignments: {} \n".format(self.center[0, 0], self.center[0, 1], self.center[0, 2], len(self.assignments))
        return string
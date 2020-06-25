class Line:
    def __init__(self, x0, direction, norm, id_):
        self.x0 = x0
        self.direction = direction
        self.norm = norm
        self.id = id_
        self.assigned_cluster = None

    def get_mid_pt(self):
        mid_pt = self.x0 + self.norm / 2. * self.direction
        return mid_pt

    def __repr__(self):
        string = "x0: {:.3f}, {:.3f}, {:.3f}, direction: {:.3f}, {:.3f}, {:.3f}, norm: {:.3f} \n".format(
            self.x0[0, 0], self.x0[0, 1], self.x0[0, 2], self.direction[0, 0], self.direction[0, 1], self.direction[0, 2], self.norm
        )
        return string
import numpy as np
from utils.vec2d import Vec2d


class Box2d:
    def __init__(self, center:Vec2d, heading, length, width):
        self.center_ = center
        self.length_ = length
        self.width_ = width
        self.half_length_ = length / 2.0
        self.half_width_ = width / 2.0
        self.heading_ = heading
        self.cos_heading_ = np.cos(heading)
        self.sin_heading_ = np.sin(heading)
        self.corners_ = []
        self.max_x_ = -np.inf
        self.min_x_ = np.inf
        self.max_y_ = -np.inf
        self.min_y_ = np.inf
        self.InitCorners()

    def InitCorners(self):
        dx1 = self.cos_heading_ * self.half_length_
        dy1 = self.sin_heading_ * self.half_length_
        dx2 = self.sin_heading_ * self.half_width_
        dy2 = -self.cos_heading_ * self.half_width_

        self.corners_.clear()
        self.corners_.append(Vec2d(self.center_.x_ + dx1 + dx2, self.center_.y_ + dy1 + dy2))
        self.corners_.append(Vec2d(self.center_.x_ + dx1 - dx2, self.center_.y_ + dy1 - dy2))
        self.corners_.append(Vec2d(self.center_.x_ - dx1 - dx2, self.center_.y_ - dy1 - dy2))
        self.corners_.append(Vec2d(self.center_.x_ - dx1 + dx2, self.center_.y_ - dy1 + dy2))

        for corner in self.corners_:
            self.max_x_ = max(corner.x_, self.max_x_)
            self.min_x_ = min(corner.x_, self.min_x_)
            self.max_y_ = max(corner.y_, self.max_y_)
            self.min_y_ = min(corner.y_, self.min_y_)

    @staticmethod
    def CreateAABox(one_corner, opposite_corner):
        x1 = min(one_corner.x, opposite_corner.x)
        x2 = max(one_corner.x, opposite_corner.x)
        y1 = min(one_corner.y, opposite_corner.y)
        y2 = max(one_corner.y, opposite_corner.y)
        return Box2d(Vec2d((x1 + x2) / 2.0, (y1 + y2) / 2.0), 0.0, x2 - x1, y2 - y1)

    def GetAllCorners(self):
        return self.corners_
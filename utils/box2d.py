import numpy as np
from utils.vec2d import Vec2d

class LineSegment:
    """
    简单的线段类，用 Vec2d 表示起点终点
    """
    def __init__(self, p0: Vec2d, p1: Vec2d):
        self.p0 = p0
        self.p1 = p1

    def direction(self) -> Vec2d:
        return self.p1 - self.p0

    def length(self) -> float:
        return self.direction().length()

    def __repr__(self):
        return f"LineSegment(p0={self.p0}, p1={self.p1})"

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
    
    def _world_to_local(self, point: Vec2d) -> Vec2d:
        """
        把世界坐标系下的点转换到 box 自身坐标系（box 中心为原点，heading 方向为 x 轴）
        """
        dx = point.x_ - self.center_.x_
        dy = point.y_ - self.center_.y_
        # 旋转 -heading
        local_x = dx * self.cos_heading_ + dy * self.sin_heading_
        local_y = -dx * self.sin_heading_ + dy * self.cos_heading_
        return Vec2d(local_x, local_y)

    def contains_point(self, point: Vec2d, eps: float = 1e-8) -> bool:
        """
        判断一个点是否在 box 内部（含边界）
        """
        lp = self._world_to_local(point)
        return (
            -self.half_length_ - eps <= lp.x_ <= self.half_length_ + eps
            and -self.half_width_ - eps <= lp.y_ <= self.half_width_ + eps
        )

    def overlap_with_segment(self, seg: LineSegment, eps: float = 1e-8) -> bool:
        """
        判断 box 是否与线段产生 overlap（相交或线段在 box 内部）
        算法：把线段两端点变换到 box 的局部坐标系，对 axis-aligned AABB 做线段裁剪
        """
        p0_local = self._world_to_local(seg.p0)
        p1_local = self._world_to_local(seg.p1)

        # 1. 端点在 box 内，直接 overlap
        if self.contains_point(seg.p0, eps) or self.contains_point(seg.p1, eps):
            return True

        # 2. 用 slab 方法判断线段与 AABB 是否相交
        min_x = -self.half_length_
        max_x = self.half_length_
        min_y = -self.half_width_
        max_y = self.half_width_

        dx = p1_local.x_ - p0_local.x_
        dy = p1_local.y_ - p0_local.y_

        t_min = 0.0
        t_max = 1.0

        # 处理 x 方向
        if abs(dx) < eps:
            # 线段与 x 轴平行，如果起点 x 不在区间内，必不相交
            if p0_local.x_ < min_x or p0_local.x_ > max_x:
                return False
        else:
            tx1 = (min_x - p0_local.x_) / dx
            tx2 = (max_x - p0_local.x_) / dx
            if tx1 > tx2:
                tx1, tx2 = tx2, tx1
            t_min = max(t_min, tx1)
            t_max = min(t_max, tx2)
            if t_min > t_max:
                return False

        # 处理 y 方向
        if abs(dy) < eps:
            if p0_local.y_ < min_y or p0_local.y_ > max_y:
                return False
        else:
            ty1 = (min_y - p0_local.y_) / dy
            ty2 = (max_y - p0_local.y_) / dy
            if ty1 > ty2:
                ty1, ty2 = ty2, ty1
            t_min = max(t_min, ty1)
            t_max = min(t_max, ty2)
            if t_min > t_max:
                return False

        # t_min <= t_max 说明线段在 [0,1] 上存在与 box 的交点
        return t_max >= t_min and t_max >= 0.0 and t_min <= 1.0
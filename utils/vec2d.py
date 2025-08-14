import numpy as np

class Vec2d:
    def __init__(self, x, y):
        self.x_ = x
        self.y_ = y

    @staticmethod
    def create_unit_vec2d(angle):
        """
        根据角度生成单位向量
        :param angle: 角度（单位：弧度）
        :return: 单位向量 (Vec2d)
        """
        return Vec2d(np.cos(angle), np.sin(angle))

    def length(self):
        """
        计算向量的长度
        :return: 向量的长度
        """
        return np.hypot(self.x_, self.y_)

    def length_square(self):
        """
        计算向量的长度平方
        :return: 向量的长度平方
        """
        return self.x_ * self.x_ + self.y_ * self.y_

    def angle(self):
        """
        计算向量的方向角度
        :return: 角度（单位：弧度）
        """
        return np.arctan2(self.y_, self.x_)

    def normalize(self):
        """
        归一化向量
        """
        length = self.length()
        if length > 1e-5:
            self.x_ /= length
            self.y_ /= length

    def distance_to(self, other):
        """
        计算到另一个向量的距离
        :param other: 另一个向量 (Vec2d)
        :return: 距离
        """
        return np.hypot(self.x_ - other.x_, self.y_ - other.y_)

    def distance_square_to(self, other):
        """
        计算到另一个向量的距离平方
        :param other: 另一个向量 (Vec2d)
        :return: 距离平方
        """
        dx = self.x_ - other.x_
        dy = self.y_ - other.y_
        return dx * dx + dy * dy

    def cross_prod(self, other):
        """
        计算向量的叉积
        :param other: 另一个向量 (Vec2d)
        :return: 叉积结果
        """
        return self.x_ * other.y_ - self.y_ * other.x_

    def inner_prod(self, other):
        """
        计算向量的点积
        :param other: 另一个向量 (Vec2d)
        :return: 点积结果
        """
        return self.x_ * other.x_ + self.y_ * other.y_

    def rotate(self, angle):
        """
        旋转向量
        :param angle: 旋转角度（单位：弧度）
        :return: 旋转后的向量 (Vec2d)
        """
        return Vec2d(self.x_ * np.cos(angle) - self.y_ * np.sin(angle),
                     self.x_ * np.sin(angle) + self.y_ * np.cos(angle))

    def self_rotate(self, angle):
        """
        自身旋转
        :param angle: 旋转角度（单位：弧度）
        """
        tmp_x = self.x_
        self.x_ = self.x_ * np.cos(angle) - self.y_ * np.sin(angle)
        self.y_ = tmp_x * np.sin(angle) + self.y_ * np.cos(angle)

    def __add__(self, other):
        return Vec2d(self.x_ + other.x_, self.y_ + other.y_)

    def __sub__(self, other):
        return Vec2d(self.x_ - other.x_, self.y_ - other.y_)

    def __mul__(self, ratio):
        return Vec2d(self.x_ * ratio, self.y_ * ratio)

    def __truediv__(self, ratio):
        assert abs(ratio) > 1e-5, "Division by zero"
        return Vec2d(self.x_ / ratio, self.y_ / ratio)

    def __iadd__(self, other):
        self.x_ += other.x_
        self.y_ += other.y_
        return self

    def __isub__(self, other):
        self.x_ -= other.x_
        self.y_ -= other.y_
        return self

    def __imul__(self, ratio):
        self.x_ *= ratio
        self.y_ *= ratio
        return self

    def __itruediv__(self, ratio):
        assert abs(ratio) > 1e-5, "Division by zero"
        self.x_ /= ratio
        self.y_ /= ratio
        return self

    def __eq__(self, other):
        return abs(self.x_ - other.x_) < 1e-5 and abs(self.y_ - other.y_) < 1e-5

    @staticmethod
    def lerp(v1, v2, ratio):
        """
        线性插值
        :param v1: 起始向量 (Vec2d)
        :param v2: 目标向量 (Vec2d)
        :param ratio: 插值比例
        :return: 插值结果 (Vec2d)
        """
        return v1 + (v2 - v1) * ratio

    def __repr__(self):
        return f"{{x: {self.x_}, y: {self.y_}}}"

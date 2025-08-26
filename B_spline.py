import numpy as np
from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline
import matplotlib.pyplot as plt


# ---------- 1. 弦长参数化 ----------
def chord_length_parameterize(pts: np.ndarray) -> np.ndarray:
    diff = np.diff(pts, axis=0)
    dist = np.sqrt((diff ** 2).sum(axis=1))
    params = np.concatenate(([0], np.cumsum(dist)))
    params /= params[-1] + 1e-12        # 归一到 [0,1]
    return params


# ---------- 2. 生成“合法”节点 ----------
def build_valid_knots(params: np.ndarray, degree: int = 3):
    """
    返回一组能被 LSQUnivariateSpline 接受的节点。
    规则：
      - 内节点严格递增
      - 内节点个数 <= len(params) - degree - 1
      - 每条数据区间至少有一个节点（Schoenberg-Whitney）
    """
    n = len(params)
    k = degree
    if n <= k + 1:
        raise ValueError("点数不足，无法拟合")

    # 最多允许的内节点个数
    max_inner = n - k - 1
    # 按“每两个数据点之间最多一个节点”的规则生成
    idx = np.linspace(1, n - 2, max(max_inner, 1)).astype(int)
    inner = params[idx]                 # 保证落在参数区间内
    # 去重并排序（保险）
    inner = np.unique(inner)
    # 多重端点
    knots = np.r_[[params[0]] * k, inner, [params[-1]] * k]
    return knots


# ---------- 3. 鲁棒拟合 ----------
def fit_bspline(pts: np.ndarray, degree: int = 3):
    params = chord_length_parameterize(pts)
    if len(pts) <= degree + 1:
        # 点数太少，退化成插值（自然边界）
        spl_x = UnivariateSpline(params, pts[:, 0], k=degree, s=0)
        spl_y = UnivariateSpline(params, pts[:, 1], k=degree, s=0)
    else:
        knots = build_valid_knots(params, degree)
        # 有时 inner 为空，也退化成插值
        if len(knots) == 2 * degree:
            spl_x = UnivariateSpline(params, pts[:, 0], k=degree, s=0)
            spl_y = UnivariateSpline(params, pts[:, 1], k=degree, s=0)
        else:
            spl_x = LSQUnivariateSpline(params, pts[:, 0], t=knots[degree:-degree], k=degree)
            spl_y = LSQUnivariateSpline(params, pts[:, 1], t=knots[degree:-degree], k=degree)
    return spl_x, spl_y, params[0], params[-1]


# ---------- 4. 采样 ----------
def sample_spline(spl_x, spl_y, t0, t1, step=0.05):
    t_eval = np.arange(t0, t1 + 1e-12, step)
    xy = np.column_stack((spl_x(t_eval), spl_y(t_eval)))
    return t_eval, xy


# =========================================================
if __name__ == "__main__":
    # 用更少的点做极限测试
    pts = np.array([[0,0], [1,1], [2,2], [3,3], [4,4], [5,5], [6,6]])
    spl_x, spl_y, t0, t1 = fit_bspline(pts, degree=3)
    t_eval, smooth_pts = sample_spline(spl_x, spl_y, t0, t1, step=0.02)

    plt.plot(pts[:, 0], pts[:, 1], 'ro-', label='raw')
    plt.scatter(smooth_pts[:, 0], smooth_pts[:, 1], c='b', s=5,label='B-spline')
    plt.legend(); plt.axis('equal'); plt.show()
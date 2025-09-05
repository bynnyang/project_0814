import numpy as np
from scipy.special import comb


def bezier_curve(points, num_points, item_number):
    n = len(points) - 1
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, item_number))
    for i in range(num_points):
        for j in range(n + 1):
            curve[i] += comb(n, j) * (1 - t[i])**(n - j) * t[i]**j * points[j]
    return curve

def fitting_curve(raw_data, num_points, item_number):
    control_points = np.array(raw_data)
    smoothed_curve = bezier_curve(control_points, num_points, item_number)
    return smoothed_curve.tolist()

def get_safe_yaw(yaw):
    yaw = np.where(yaw <= -np.pi, yaw + 2*np.pi, yaw)
    yaw = np.where(yaw > np.pi, yaw - 2*np.pi, yaw)
    return yaw
def calculate_tangent(points, mode):
    num_points = len(points)

    if num_points < 2:
        return [0.]
    
    max_index = np.argmax(points[:, 0])

    max_points = points.shape[0]

    if max_index != 0 and max_index != (max_points -1):
        tangent = np.zeros((max_index+1, 2))
        for i in range(max_index + 1):
            if i == 0:
                tangent[i] = (points[i + 1] - points[i])
            elif i == max_index:
                tangent[i] = (points[i] - points[i - 1])
            elif (i == 1) or (i == max_index - 1):
                tangent[i] = (points[i] - points[i - 1])
            else:
                tangent[i] = ((points[i - 2] - 8 * points[i - 1] + 8 * points[i + 1] - points[i + 2]) / 12)
        traj_heading_head = np.arctan2(tangent[:, 1], tangent[:, 0])
        cnt = 0
        tangent_tail = np.zeros((num_points-max_index, 2))
        for i in range(max_index, num_points):
            if i == max_index:
                tangent_tail[cnt] = (points[i + 1] - points[i])
            elif i == num_points - 1:
                tangent_tail[cnt] = (points[i] - points[i - 1])
            elif (i == max_index+1) or (i == num_points - 2):
                tangent_tail[cnt] = (points[i] - points[i - 1])
            else:
                tangent_tail[cnt] = ((points[i - 2] - 8 * points[i - 1] + 8 * points[i + 1] - points[i + 2]) / 12)
            cnt = cnt + 1  
        traj_heading_tail = np.arctan2(tangent_tail[:, 1], tangent_tail[:, 0])
        traj_heading_tail = traj_heading_tail + np.pi
        traj_heading_tail = get_safe_yaw(traj_heading_tail)



        traj_heading_list = np.rad2deg(traj_heading_head).tolist() + np.rad2deg(traj_heading_tail).tolist()[1:]

    else:
        tangent = np.zeros((num_points, 2))
        heading_pos = 3.14 if max_index == 0 else 0.0
        for i in range(num_points):
            if mode == "three_point":
                if i == 0:
                    tangent[i] = -(points[i + 1] - points[i])
                elif i == num_points - 1:
                    tangent[i] = -(points[i] - points[i - 1])
                else:
                    tangent[i] = -(points[i + 1] - points[i - 1])
            elif mode == "five_point":
                if i == 0:
                    tangent[i] = (points[i + 1] - points[i])
                elif i == num_points - 1:
                    tangent[i] = (points[i] - points[i - 1])
                elif (i == 1) or (i == num_points - 2):
                    tangent[i] = (points[i] - points[i - 1])
                else:
                    tangent[i] = ((points[i - 2] - 8 * points[i - 1] + 8 * points[i + 1] - points[i + 2]) / 12)
            elif mode == "three_point_back":
                if i == 0:
                    tangent[i] = -(points[i + 1] - points[i])
                else:
                    tangent[i] = -(points[i] - points[i - 1])
            elif mode == "three_point_front":
                if i == num_points - 1:
                    tangent[i] = -(points[i] - points[i - 1])
                else:
                    tangent[i] = -(points[i + 1] - points[i])
            else:
                assert print("Error mode!")

        traj_heading = np.arctan2(tangent[:, 1], tangent[:, 0])
        traj_heading = traj_heading + heading_pos
        traj_heading = get_safe_yaw(traj_heading)

        traj_heading_list = np.rad2deg(traj_heading).tolist()

    if np.any(np.isnan(traj_heading_list)):
        print("NaN")
    return traj_heading_list


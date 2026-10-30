import math
import numpy as np
def compute_angle(x, y, ego_position):
    dx = x - ego_position[0]
    dy = ego_position[1] - y 
    if dx != 0 or dy != 0:
        current_angle = math.degrees(math.atan2(dy, dx))
    else:
        return 0.0, 0.0, 1.0
    rad = np.deg2rad(current_angle)
    angle_sin = float(np.sin(rad))
    angle_cos = float(np.cos(rad))
    return float(current_angle), angle_sin, angle_cos

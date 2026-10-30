import math
from configs.settings import DISTANCE_CONFIG
def estimate_distance(bbox, class_name, frame_width=2560, horizontal_fov=None):
    height_pixel = bbox[3] - bbox[1]  # y2 - y1
    if height_pixel <= 0:
        return -1.0
    if horizontal_fov is None:
        horizontal_fov = DISTANCE_CONFIG.get("horizontal_fov", 130)
    focal_length_px = (frame_width / 2) / math.tan(math.radians(horizontal_fov / 2))
    real_heights = DISTANCE_CONFIG.get("real_heights", {})
    real_height = real_heights.get(str(class_name).lower(), 1.6)
    distance_m = (real_height * focal_length_px) / height_pixel
    return round(float(distance_m), 2)

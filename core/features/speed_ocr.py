import re
import cv2
from rapidocr_onnxruntime import RapidOCR
ocr_engine = RapidOCR()

_last_speed = 0 
_MU = 0.8       # Hệ số ma sát mặt đường
_G = 9.8        # Gia tốc trọng trường (m/s^2)

def extract_speed(text):
    if not text:
        return None
    text = re.sub(r'\b[OQ](?=km)', '0', text, flags=re.IGNORECASE)  
    text = re.sub(r'\bB(?=km)', '8', text, flags=re.IGNORECASE)  
    match = re.search(r'(\d{1,3})\s*(?:km/h|km/|km)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match_num = re.search(r'\b(\d{1,3})\b', text)
    if match_num and ("km" in text.lower() or "/h" in text.lower()):
        return int(match_num.group(1))
    return None

def get_speed(frame):
    h, w = frame.shape[:2]
    if h == 1920 and w == 2560:
        roi = frame[1800:1918, 1480:1800]
    else:
        ymin = int(h * (1742 / 1920))
        ymax = int(h * (1918 / 1920))
        xmin = int(w * (1390 / 2560))
        xmax = int(w * (1800 / 2560))
        roi = frame[ymin:ymax, xmin:xmax]
    res, _ = ocr_engine(roi, use_det=False, use_cls=False)
    if res and len(res) > 0 and isinstance(res[0], (list, tuple)):
        text = res[0][0]
        speed = extract_speed(text)
        if speed is not None:
            return speed
    res, _ = ocr_engine(roi, use_det=True, use_cls=False)
    if res:
        joined_text = " ".join([item[1] for item in res])
        speed = extract_speed(joined_text)
        if speed is not None:
            return speed
        for item in res:
            speed = extract_speed(item[1])
            if speed is not None:
                return speed
    return None

def safe_distance_estimation(frame):
    global _last_speed
    speed_kmh = get_speed(frame)
    if speed_kmh is None:
        speed_kmh = _last_speed
    else:
        speed_kmh = int(speed_kmh)
        _last_speed = speed_kmh
    speed_mps = float(speed_kmh * 1000.0 / 3600.0)
    braking_distance = float(speed_mps**2 / (2 * _MU * _G))
    total_safe_distance = round(speed_mps + braking_distance, 1)
    return speed_kmh, total_safe_distance

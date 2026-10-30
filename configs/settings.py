import os
import sys
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_DETECTION_PATH = MODELS_DIR / "detection" / "best.pt"
MODEL_MLP_PATH = MODELS_DIR / "decision" / "mlp_model.pth"
SCALER_PATH = MODELS_DIR / "decision" / "preprocessor.pkl"
STATUS_MAP = {
    0: ("crossing", (0, 255, 255)),     # Màu vàng: Cắt ngang
    1: ("normal", (0, 255, 0)),         # Màu xanh lá: Bình thường
    2: ("wrong-way", (0, 0, 255))       # Màu đỏ: Đi ngược chiều
}
ROI_CONFIG = {
    "right": {
        "x_min": 0.50,
        "x_max": 0.95,
        "y_min": 0.10,
        "y_max": 0.70
    },
    "left": {
        "x_min": 0.05,
        "x_max": 0.4999,
        "y_min": 0.10,
        "y_max": 0.70
    }
}
DISTANCE_CONFIG = {
    "horizontal_fov": 130,  # Độ
    "real_heights": {
        "person": 1.6,
        "bicycle": 1.0,
        "motorcycle": 1.2,
        "car": 1.5,
        "bus": 3.2,
        "truck": 3.5,
    }
}
AUDIO_CONFIG = {
    "cooldown_seconds": 5.0,
    "enabled": True
}

# Cấu hình thuật toán suy luận rủi ro sớm TSDA-EW
TSDA_EW_CONFIG = {
    "tau_ttc": 3.0,               # Ngưỡng Time-To-Collision nguy cơ (giây)
    "tau_h": 0.5,                 # Ngưỡng Hazard Score kích hoạt cảnh báo
    "K": 5,                       # Kích thước cửa sổ trượt khẳng định thời gian (frames)
    "eta": 0.6,                   # Tỷ lệ khẳng định rủi ro (>= 3/5 frames)
    "weights": (0.4, 0.4, 0.2),   # Trọng số: (w_d khoảng cách, w_t TTC, w_m vận tốc)
    "v_min": 0.5,                 # Vận tốc tiếp cận tối thiểu có ý nghĩa (m/s)
    "max_history_len": 10         # Cửa sổ lịch sử khoảng cách để tính v_rel
}

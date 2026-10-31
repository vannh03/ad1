import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ENV_FILE_PATH = PROJECT_ROOT / ".env"


def load_dotenv(dotenv_path=None):
    """Đọc file .env bằng Python thuần (không cần thư viện ngoài trên Jetson Nano)."""
    path = Path(dotenv_path) if dotenv_path else ENV_FILE_PATH
    if not path.exists():
        return False
    try:
        with open(str(path), "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                    val = val[1:-1]
                elif "#" in val:
                    val = val.split("#", 1)[0].strip()
                if key:
                    os.environ.setdefault(key, val)
        return True
    except Exception:
        return False


ENV_LOADED = load_dotenv(ENV_FILE_PATH)


def _get_env_bool(key, default=False):
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _get_env_int(key, default):
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


def _get_env_float(key, default):
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return float(val.strip())
    except ValueError:
        return default


DEFAULT_VIDEO_SOURCE = os.environ.get("ADAS_VIDEO_SOURCE", "NO20250604-063730-010007F.MP4")
DEFAULT_WEIGHTS = os.environ.get("ADAS_WEIGHTS", "auto")
DEFAULT_IMGSZ = _get_env_int("ADAS_IMGSZ", 320)
DEFAULT_DEVICE = os.environ.get("ADAS_DEVICE", "auto")
DEFAULT_MAX_FRAMES = _get_env_int("ADAS_MAX_FRAMES", 100)
DEFAULT_WARMUP_FRAMES = _get_env_int("ADAS_WARMUP_FRAMES", 5)
DEFAULT_CONF_THRES = _get_env_float("ADAS_CONF_THRES", 0.25)
DEFAULT_IOU_THRES = _get_env_float("ADAS_IOU_THRES", 0.45)

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
    "cooldown_seconds": _get_env_float("ADAS_AUDIO_COOLDOWN", 5.0),
    "enabled": _get_env_bool("ADAS_AUDIO_ENABLED", False)
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

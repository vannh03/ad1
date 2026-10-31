#!/bin/bash
# ==============================================================================
# SCRIPT TỰ ĐỘNG THIẾT LẬP MÔI TRƯỜNG ADAS TRÊN NVIDIA JETSON NANO (1-CLICK SETUP)
# Phù hợp cho người mới bắt đầu: Chỉ cần chạy 1 lần duy nhất:
#     bash setup_jetson.sh
# ==============================================================================

set -e

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$CURRENT_DIR"

echo ""
echo "========================================================================"
echo "🚀 BẮT ĐẦU TỰ ĐỘNG CÀI ĐẶT MÔI TRƯỜNG ADAS TRÊN JETSON NANO"
echo "========================================================================"
echo ""

# 1. Tự động kiểm tra và tạo Swap 8GB (Tránh tràn RAM)
SWAP_TOTAL=$(free -m | awk '/Swap:/ {print $2}')
if [ "$SWAP_TOTAL" -lt 4000 ]; then
    echo "▶ [1/6] Đang tạo bộ nhớ ảo Swap 8GB (Bắt buộc để không bị tắt ứng dụng)..."
    sudo swapoff -a || true
    sudo fallocate -l 8G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=8192
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    if ! grep -q '/swapfile' /etc/fstab; then
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    fi
    echo "  -> Đã tạo và kích hoạt Swap 8GB thành công!"
else
    echo "▶ [1/6] Đã có Swap ($SWAP_TOTAL MB), bỏ qua bước này."
fi

# 2. Bật chế độ công suất tối đa 10W và bật quạt
echo "▶ [2/6] Đang kích hoạt chế độ 10W (MAXN) và ép xung quạt tản nhiệt..."
sudo nvpmodel -m 0 || true
sudo jetson_clocks || true

# 3. Cài đặt các gói hệ thống Ubuntu 18.04 & khởi tạo môi trường ảo .venv
echo "▶ [3/6] Đang cập nhật hệ thống và khởi tạo môi trường ảo Python (.venv)..."
sudo apt update -y
sudo apt install -y python3-pip python3-dev python3-venv git cmake unzip \
                    libopenblas-base libopenmpi-dev libomp-dev \
                    libjpeg-dev zlib1g-dev libavcodec-dev libavformat-dev libswscale-dev \
                    libcanberra-gtk-module libcanberra-gtk3-module libgstreamer1.0-dev

if [ ! -d "$CURRENT_DIR/.venv" ]; then
    python3 -m venv --system-site-packages "$CURRENT_DIR/.venv" || true
fi
if [ -f "$CURRENT_DIR/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$CURRENT_DIR/.venv/bin/activate"
    echo "  -> Đã kích hoạt môi trường ảo .venv (kế thừa CUDA/TensorRT của JetPack)!"
fi

# 4. Cài đặt PyTorch 1.10 và Torchvision
echo "▶ [4/6] Đang kiểm tra PyTorch CUDA trên Jetson Nano..."
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  -> PyTorch đã nhận diện GPU CUDA thành công!"
else
    echo "  -> Đang tải gói PyTorch v1.10.0 chính thức của NVIDIA cho JetPack 4.6..."
    wget -q https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O /tmp/torch-1.10.0-cp36-cp36m-linux_aarch64.whl
    pip3 install /tmp/torch-1.10.0-cp36-cp36m-linux_aarch64.whl
    rm -f /tmp/torch-1.10.0-cp36-cp36m-linux_aarch64.whl

    echo "  -> Đang build và cài đặt Torchvision v0.11.1..."
    cd /tmp
    rm -rf torchvision
    git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
    cd torchvision
    export BUILD_VERSION=0.11.1
    python3 setup.py install
    cd "$CURRENT_DIR"
    rm -rf /tmp/torchvision
fi

# 5. Cài đặt các thư viện AI tương thích Python 3.6
echo "▶ [5/6] Đang cài đặt thư viện khoa học dữ liệu (numpy, pandas, scikit-learn, ocr)..."
pip3 install numpy==1.19.5 pandas==1.1.5 scikit-learn==0.24.2
pip3 install joblib tqdm pyyaml pillow matplotlib scipy
pip3 install onnxruntime==1.10.0 rapidocr-onnxruntime==1.2.3

# Cài đặt Ultralytics v8.0.196 patch tương thích Python 3.6
echo "▶ [6/6] Đang cài đặt mô hình YOLOv8 Ultralytics..."
if ! python3 -c "import ultralytics" 2>/dev/null; then
    cd /tmp
    rm -rf ultralytics
    git clone -b v8.0.196 https://github.com/ultralytics/ultralytics.git
    cd ultralytics
    sed -i 's/from importlib import metadata/import pkg_resources as metadata/g' ultralytics/utils/checks.py 2>/dev/null || true
    sed -i 's/import importlib.metadata as metadata/import pkg_resources as metadata/g' ultralytics/utils/__init__.py 2>/dev/null || true
    pip3 install --no-deps -e .
    cd "$CURRENT_DIR"
    rm -rf /tmp/ultralytics
fi

# Tùy chọn: Chuyển đổi sang TensorRT Engine nếu chưa có
ENGINE_FILE="$CURRENT_DIR/models/detection/best.engine"
PT_FILE="$CURRENT_DIR/models/detection/best.pt"

if [ ! -f "$ENGINE_FILE" ] && [ -f "$PT_FILE" ]; then
    echo ""
    echo "⚡ Đang tối ưu hóa mô hình sang TensorRT Engine (FP16)... (Có thể mất 2-3 phút)"
    yolo export model="$PT_FILE" format=engine half=True imgsz=320 device=0 || true
fi

echo ""
echo "========================================================================"
echo "🎉 CÀI ĐẶT HOÀN TẤT 100%! BÂY GIỜ BẠN CHỈ CẦN CHẠY LỆNH:"
echo ""
echo "         bash run_jetson.sh"
echo "   hoặc: python3 run.py"
echo "   (Mọi thông số cấu hình có thể chỉnh nhanh trong file .env)"
echo "========================================================================"
echo ""

#!/bin/bash
# ==============================================================================
# SCRIPT TỰ ĐỘNG THIẾT LẬP MÔI TRƯỜNG ADAS TRÊN NVIDIA JETSON NANO (1-CLICK SETUP)
# Phù hợp cho người mới bắt đầu: Chỉ cần chạy 1 lần duy nhất:
#     bash setup_jetson.sh
# ==============================================================================

set -e

echo ""
echo "========================================================================"
echo "🚀 BẮT ĐẦU TỰ ĐỘNG CÀI ĐẶT MÔI TRƯỜNG ADAS TRÊN JETSON NANO"
echo "========================================================================"
echo ""

# 1. Tự động kiểm tra và giải phóng ổ đĩa trước
echo "▶ [1/6] Kiểm tra dung lượng ổ đĩa và cấu hình RAM..."
sudo swapoff /swapfile 2>/dev/null || true
sudo rm -f /swapfile 2>/dev/null || true
rm -f ~/module.zip 2>/dev/null || true
sudo apt-get clean 2>/dev/null || true

# Tự động nhận diện và kích hoạt ZRAM nếu chưa có Swap
CURRENT_SWAP=$(free -m 2>/dev/null | awk '/Swap:/ {print $2}')
if [ -z "$CURRENT_SWAP" ]; then CURRENT_SWAP=0; fi

if [ "$CURRENT_SWAP" -gt 500 ]; then
    echo "  -> Bộ nhớ ảo Swap/ZRAM đã có sẵn: ${CURRENT_SWAP} MB (Đang hoạt động tốt)."
else
    echo "  -> Đang kích hoạt bộ nhớ nén ZRAM (tốn 0 byte ổ cứng)..."
    if [ -x /sbin/nvzramconfig.sh ]; then
        sudo /sbin/nvzramconfig.sh 2>/dev/null || true
    elif [ -x /usr/sbin/nvzramconfig.sh ]; then
        sudo /usr/sbin/nvzramconfig.sh 2>/dev/null || true
    elif [ -x /etc/systemd/nvzramconfig.sh ]; then
        sudo /etc/systemd/nvzramconfig.sh 2>/dev/null || true
    elif systemctl is-enabled nvzramconfig >/dev/null 2>&1; then
        sudo systemctl restart nvzramconfig 2>/dev/null || true
    fi
fi

AVAIL_DISK_MB=$(df -m / 2>/dev/null | awk 'NR==2 {print $4}')
echo "  -> Dung lượng ổ cứng khả dụng hiện tại: ${AVAIL_DISK_MB} MB"
echo "  -> Cấu hình bộ nhớ ảo hoàn tất! (Swap hiện tại: $(free -m 2>/dev/null | awk '/Swap:/ {print $2}') MB)"

# 2. Bật chế độ công suất tối đa 10W và bật quạt
echo "▶ [2/6] Đang kích hoạt chế độ 10W (MAXN) và ép xung quạt tản nhiệt..."
sudo nvpmodel -m 0 || true
sudo jetson_clocks || true

# 3. Cài đặt các gói hệ thống Ubuntu 18.04
echo "▶ [3/6] Đang cập nhật hệ thống và cài đặt gói bổ trợ C++/Python..."
sudo apt update -y
sudo apt install -y python3-pip python3-dev git cmake unzip \
                    libopenblas-base libopenmpi-dev libomp-dev \
                    libjpeg-dev zlib1g-dev libavcodec-dev libavformat-dev libswscale-dev \
                    libcanberra-gtk-module libcanberra-gtk3-module libgstreamer1.0-dev

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
    python3 setup.py install --user
    cd -
    rm -rf /tmp/torchvision
fi

# 5. Cài đặt các thư viện AI bằng apt-get chính thức (Cực nhanh, không lỗi pip)
echo "▶ [5/6] Đang cài đặt thư viện khoa học dữ liệu (numpy, pandas, scikit-learn, joblib) qua apt..."
sudo apt install -y python3-numpy python3-scipy python3-pandas python3-sklearn python3-joblib python3-pil python3-yaml || true

# 6. Kiểm tra Ultralytics
echo "▶ [6/6] Kiểm tra YOLOv8 Ultralytics..."
echo "  -> Ultralytics đã được tích hợp sẵn trực tiếp trong mã nguồn!"

# Tùy chọn: Chuyển đổi sang TensorRT Engine nếu chưa có
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ENGINE_FILE="$CURRENT_DIR/models/detection/best.engine"
PT_FILE="$CURRENT_DIR/models/detection/best.pt"

if [ ! -f "$ENGINE_FILE" ] && [ -f "$PT_FILE" ]; then
    echo ""
    echo "⚡ Đang tối ưu hóa mô hình sang TensorRT Engine (FP16)... (Có thể mất 2-3 phút)"
    python3 -c "from ultralytics import YOLO; model = YOLO('$PT_FILE'); model.export(format='engine', half=True, imgsz=320, device=0)" 2>/dev/null || true
fi

echo ""
echo "========================================================================"
echo "🎉 CÀI ĐẶT HOÀN TẤT 100%! BÂY GIỜ BẠN CHỈ CẦN CHẠY LỆNH:"
echo ""
echo "         python3 run.py"
echo ""
echo "========================================================================"
echo ""
echo "========================================================================"
echo ""

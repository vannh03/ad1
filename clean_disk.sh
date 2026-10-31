#!/bin/bash
# ==============================================================================
# SCRIPT DỌN DẸP BỘ NHỚ Ổ ĐĨA TRÊN NVIDIA JETSON NANO (GIẢI PHÓNG 3GB - 10GB)
# ==============================================================================

echo "========================================================================"
echo "🧹 ĐANG BẮT ĐẦU DỌN DẸP Ổ ĐĨA JETSON NANO..."
echo "========================================================================"

# 1. Tắt và xóa file swap 8GB nếu đang chiếm đĩa
if [ -f /swapfile ]; then
    echo "▶ [1/6] Đang gỡ bỏ /swapfile cũ để giải phóng dung lượng..."
    sudo swapoff /swapfile 2>/dev/null || true
    sudo rm -f /swapfile
fi

# 2. Xóa file zip tải về sau khi đã giải nén
echo "▶ [2/6] Xóa file nén module.zip tạm thời..."
rm -f ~/module.zip
rm -f /home/*/module.zip 2>/dev/null || true
rm -f /tmp/*.zip /tmp/*.whl 2>/dev/null || true

# 3. Dọn dẹp bộ nhớ đệm APT và Pip
echo "▶ [3/6] Dọn dẹp cache của apt và pip..."
sudo apt-get clean -y
sudo apt-get autoremove -y --purge
rm -rf ~/.cache/pip
rm -rf ~/.cache/thumbnails

# 4. Gỡ bỏ các ứng dụng văn phòng thừa không dùng trên Jetson (LibreOffice, Thunderbird)
echo "▶ [4/6] Gỡ bỏ phần mềm văn phòng LibreOffice (tiết kiệm ~1GB)..."
sudo apt-get purge -y libreoffice* thunderbird aisleriot gnome-mines gnome-sudoku 2>/dev/null || true
sudo apt-get autoremove -y --purge
sudo apt-get clean -y

# 5. Xóa các file mẫu CUDA / cuDNN demo không cần thiết
echo "▶ [5/6] Dọn dẹp các mẫu demo của CUDA/VisionWorks..."
sudo rm -rf /usr/local/cuda/samples 2>/dev/null || true
sudo rm -rf /usr/src/cudnn_samples* 2>/dev/null || true
sudo rm -rf ~/VisionWorks-Repo 2>/dev/null || true

# 6. Kiểm tra lại dung lượng trống
echo ""
echo "========================================================================"
echo "✅ DỌN DẸP HOÀN TẤT! DUNG LƯỢNG Ổ ĐĨA HIỆN TẠI:"
echo "========================================================================"
df -h /
echo ""

# HƯỚNG DẪN 3 BƯỚC ĐO FPS & LATENCY TRÊN JETSON NANO
> **Dành riêng cho người mới bắt đầu:** Mọi cấu hình trong file `.env`, chọn model, chọn GPU, tạo môi trường ảo `.venv` và tính toán độ trễ đã được tự động hóa 100%.

---

## 🎯 BƯỚC 1: Tải Code về Jetson Nano từ GitHub
Trên màn hình Jetson Nano (kết nối qua RealVNC hoặc SSH), mở **Terminal** và gõ:
```bash
cd /home/jetson
git clone https://github.com/vannh03/camera.git
cd camera
```

---

## ⚙️ BƯỚC 2: Cài đặt Tự động 1-Click (Chỉ chạy đúng 1 lần đầu)
Trong thư mục `camera`, chạy lệnh:
```bash
bash setup_jetson.sh
```
*Script này sẽ tự động làm hết mọi việc: tạo bộ nhớ Swap 8GB, bật nguồn 10W, tạo môi trường ảo `.venv` kế thừa CUDA/TensorRT của JetPack, cài đặt PyTorch CUDA, YOLOv8 và tối ưu hóa TensorRT.*

---

## 🚀 BƯỚC 3: Chạy và Lấy kết quả FPS & Latency
Khi cần đo hiệu năng, bạn chỉ cần gõ đúng 1 dòng:
```bash
bash run_jetson.sh
```
*(hoặc: `python3 run.py`)*

> 💡 **Mẹo chỉnh cấu hình nhanh qua file `.env`:**
> Nếu muốn đổi video kiểm thử, đổi kích thước ảnh (`ADAS_IMGSZ=320`), hoặc đổi số frame kiểm tra (`ADAS_MAX_FRAMES=100`), bạn chỉ cần mở file `.env` ngay trong thư mục gốc để sửa mà không cần đụng vào code Python!

### 📊 Mẫu kết quả hiển thị trên màn hình:
```text
=================================================================
🚗 HỆ THỐNG CẢNH BÁO VA CHẠM THÔNG MINH (ADAS) - JETSON NANO
=================================================================
 • Cấu hình (.env)    : Đã nạp (.env)
 • Thiết bị tính toán : NVIDIA Tegra X1 (device='0')
 • Mô hình YOLO       : models/detection/best.pt
 • Video kiểm thử     : NO20250604-063730-010007F.MP4
 • Kích thước ảnh     : 320x320 pixels
 • Số frame kiểm tra  : 100 frames (5 frame đầu khởi động bỏ qua)
=================================================================
[CHUAN_XAC] Frame 100/100 | FPS:  24.8 | Latency:  40.3ms (YOLO: 37.1ms, Rule:  3.2ms)

=================================================================
📊 BẢNG TỔNG HỢP KẾT QUẢ ĐO ĐẠC TRÊN JETSON NANO
=================================================================
 • Tốc độ trung bình (Realtime FPS) : 24.85 FPS
 • Độ trễ xử lý trung bình (Latency): 40.24 ms
   ├─ Thời gian YOLO Detect & Track  : 37.41 ms
   └─ Thời gian Suy luận Luật + MLP : 2.83 ms
 • Độ trễ AI nhỏ nhất (Min Latency) : 36.10 ms
 • Độ trễ AI lớn nhất (Max Latency) : 51.20 ms
 • Độ trễ phân vị 95% (P95 Latency) : 44.30 ms
 • Độ trễ toàn trình (End-to-End)   : 43.15 ms
=================================================================
🎯 ĐÁNH GIÁ TIÊU CHUẨN THỜI GIAN THỰC:
  ✅ ĐẠT CHUẨN REALTIME TRÊN JETSON NANO (FPS >= 20, Latency < 50ms)
=================================================================
```
Bạn chỉ việc sao chép bảng số liệu này vào báo cáo đồ án hoặc slide thuyết trình!

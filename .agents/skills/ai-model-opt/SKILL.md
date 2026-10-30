---
name: ai-model-opt
description: >-
  Kỹ năng tối ưu hóa mô hình AI, xử lý Computer Vision và kỹ thuật trích xuất đặc trưng
  cho agent AI. Bao gồm YOLOv8, ByteTrack, ước lượng khoảng cách quang học và mô hình phân loại MLP.
---

# AI Model & Computer Vision Optimization Skill

Kỹ năng này hướng dẫn AI Engineer tối ưu hóa các thành phần thị giác máy tính và học máy trong dự án ADAS.

## 1. Tối ưu phát hiện đối tượng (YOLOv8 & ONNX)
- Lựa chọn kích thước ảnh đầu vào (`imgsz`): 320x320 cho realtime trên máy cấu hình vừa, 640x640 cho độ chính xác cao.
- Sử dụng mô hình xuất ONNX (`session = ort.InferenceSession('best.onnx')`) với `providers=['CUDAExecutionProvider', 'CPUExecutionProvider']` để tăng tốc độ gấp 2-3 lần so với PyTorch mặc định trên môi trường deploy.

## 2. Kỹ thuật ước lượng khoảng cách (Pinhole Camera Model)
- Công thức: `Distance = (Real_Height * Focal_Length_px) / Bounding_Box_Height_px`
- Bảng tham chiếu chiều cao thực tế:
  - `person`: 1.6m - 1.7m
  - `bicycle` / `motorcycle`: 1.0m - 1.2m
  - `car`: 1.5m
  - `bus` / `truck`: 3.0m - 3.5m
- Chú ý: Bounding box bị cắt bởi mép khung hình (boundary clipping) sẽ cho kết quả khoảng cách không chính xác -> cần lọc các bounding box chạm viền ảnh.

## 3. Trích xuất đặc trưng quỹ đạo & Mô hình quyết định
- Theo dõi lịch sử di chuyển theo chuỗi (Buffer `N=10` frames).
- Tính biến thiên toạ độ `dx = x_t - x_0`, `dy = y_t - y_0` và biến thiên góc `dtheta = theta_t - theta_0`.
- Kết hợp vùng quan tâm ROI (trái / phải) để phân loại rủi ro:
  - Vùng bên phải (`right`): Xe di chuyển theo chiều `dy > 0` (tiến gần xe ta ngược chiều) -> Cảnh báo **wrong-way**.
  - Vùng bên trái hoặc giao lộ: Biến thiên tọa độ `abs(dx) > threshold` và hướng chếch vào làn xe ta -> Cảnh báo **crossing**.

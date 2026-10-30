---
name: project-management
description: >-
  Kỹ năng quản lý dự án cho agent PM. Hướng dẫn lập kế hoạch sprint, phân rã task,
  theo dõi tiến độ hoàn thành các module trong hệ thống ADAS AIP491.
---

# Project Management Skill for ADAS Project

Kỹ năng này hướng dẫn PM cách theo dõi tiến độ, điều phối task và duy trì tài liệu của dự án ADAS.

## 1. Phân rã công việc (Work Breakdown Structure)
- **Phase 1: Chuẩn hóa & Dọn dẹp Codebase**:
  - Dọn dẹp file tàn dư không dùng (Dead code từ YOLOv5 cũ, file typo).
  - Sửa lỗi hardcoded path và chuẩn hóa thư mục.
- **Phase 2: Nâng cấp Lõi AI & Features**:
  - Tối ưu OCR nhận diện tốc độ.
  - Chuẩn hóa mô hình nhận diện hành vi (MLP / Rules).
  - Khử false positive trong phát hiện wrong-way và crossing.
- **Phase 3: Tối ưu Pipeline Real-time**:
  - Đảm bảo đa tiến trình đạt FPS ổn định (>= 25 FPS).
  - Thêm cảnh báo bằng âm thanh & giao diện hiển thị cảnh báo (Warning UI/Audio).
- **Phase 4: Báo cáo & Bàn giao đồ án AIP491**:
  - Benchmark số liệu (mAP, FPS, độ trễ cảnh báo).
  - Hoàn thiện tài liệu kiến trúc hệ thống và hướng dẫn cài đặt.

## 2. Tiêu chuẩn bàn giao (Definition of Done)
1. Code chạy không gặp lỗi FileNotFoundError trên bất kỳ máy nào (không dùng đường dẫn cứng).
2. Khi dừng chương trình (`Ctrl+C`), toàn bộ tiến trình con phải tắt sạch, không sinh zombie process.
3. Có kết quả log hoặc video output rõ ràng tại thư mục chỉ định.

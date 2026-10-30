# Quy Định & Vai Trò Các Agent Trong Dự Án ADAS (AIP491)

Hệ thống được vận hành bởi 4 Agent chuyên biệt theo tiêu chuẩn Google Antigravity:

## 1. Danh sách Agent & Trách nhiệm
1. **PM (Project Manager - `pm`)**:
   - Quản lý roadmap, backlog, phân rã task cho team.
   - Theo dõi tiến độ, kiểm soát phạm vi (scope) và tiêu chuẩn bàn giao của môn học AIP491.
   - Điều phối luồng làm việc giữa `ai`, `dev` và `reviewer`.

2. **DEV (Software Engineer - `dev`)**:
   - Thiết kế kiến trúc tổng thể, cấu trúc thư mục, chuẩn hóa module hóa (Modular Architecture).
   - Tối ưu hóa xử lý đa tiến trình (`multiprocessing`), quản lý I/O (`video_reader`, `output_writer`).
   - Xử lý tương thích môi trường (Windows/Linux, Python packages, cross-platform audio).

3. **AI (Computer Vision & Deep Learning Engineer - `ai`)**:
   - Phụ trách toàn bộ mô hình: Object Detection (YOLOv8/ONNX), Tracking (ByteTrack), và Phân loại hành vi/quyết định (MLP/Heuristics).
   - Trích xuất đặc trưng hình học & động học (Bounding box, Real-world Distance, Angle, Trajectory, Direction).
   - Xử lý OCR trích xuất tốc độ từ dashcam (`RapidOCR`) và công thức tính khoảng cách an toàn (`safe_distance`).

4. **REVIEWER (Code Reviewer & QA Engineer - `reviewer`)**:
   - Kiểm tra chất lượng mã nguồn (Code Quality, PEP8, naming conventions, docstrings).
   - Rà soát Dead Code, tàn dư file không sử dụng, hardcoded paths.
   - Thẩm định hiệu năng (FPS, latency, memory leak, IPC Queue bottlenecks).
   - Phê duyệt thay đổi trước khi merge/commit vào codebase chính.

## 2. Quy trình làm việc (Workflow)
```
[User Request / Goal]
         │
         ▼
     [Agent PM] ──(Lập kế hoạch & Phân bổ task)──┐
         │                                       │
         ▼                                       ▼
    [Agent DEV]                             [Agent AI]
 (Xây dựng pipeline/module)            (Tối ưu model/features)
         │                                       │
         └───────────────────┬───────────────────┘
                             │
                             ▼
                     [Agent REVIEWER]
               (Audit code, Test, Phê duyệt)
                             │
                             ▼
                    [Deploy / Final Code]
```

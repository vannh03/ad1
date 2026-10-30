---
name: code-review-qa
description: >-
  Kỹ năng rà soát chất lượng code, kiểm toán lỗi tiềm ẩn, và phát hiện dead code
  cho agent REVIEWER trong dự án ADAS.
---

# Code Review & QA Skill for ADAS Project

Kỹ năng này cung cấp checklist và phương pháp luận để reviewer đánh giá chất lượng mã nguồn.

## 1. Checklist Rà soát Dead Code & File thừa
1. **Kiểm tra import**: Mọi hàm hoặc class import từ module khác phải được sử dụng. Nếu import mà không dùng, gắn cờ warning.
2. **Kiểm tra file tàn dư (Orphaned files)**:
   - Các file script độc lập không có trong pipeline `main.py` (ví dụ `onxx_model.py` tồn tại song song với `detection.py` nhưng không dùng).
   - Các file checkpoint model cũ không được load trong `decision.py` (như `hgb_baseline.pkl`, `label_encoder.pkl`).
   - Các file tiện ích từ framework gốc (ví dụ các file huấn luyện YOLOv5: `loss.py`, `plots.py`, `autoanchor.py`, `autobatch.py`, `benchmarks.py`, thư mục `wandb`).
3. **Thư mục rỗng**: Kiểm tra các thư mục như `Warning/`, `interface/` nếu không có file hoặc chưa implement thì cần dọn dẹp hoặc tạo file `.gitkeep` kèm tài liệu kế hoạch.

## 2. Checklist Lỗi Tiềm Ẩn (Defect Inspection)
- **Hardcoded Path**: Tuyệt đối không chấp nhận các đường dẫn bắt đầu bằng ký tự ổ đĩa (ví dụ `D:\...`, `C:\...`).
- **Memory Leaks**: Trong `decision.py`, biến `position_history` lưu theo `obj_id`. Nếu các object biến mất mà không xóa khỏi dict, dict sẽ phình to liên tục theo thời gian.
- **IPC Blocking / Deadlock**: Khi một process con bị Exception và crash đột ngột, nó không gửi `None` sang process tiếp theo, dẫn đến các process còn lại bị block vô hạn.
- **Cross-platform**: Lệnh `espeak &` sẽ fail trên Windows.

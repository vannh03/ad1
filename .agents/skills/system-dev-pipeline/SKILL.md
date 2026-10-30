---
name: system-dev-pipeline
description: >-
  Kỹ năng xây dựng kiến trúc hệ thống và luồng đa tiến trình (multiprocessing pipeline)
  cho agent DEV. Bao gồm quản lý Queue IPC, xử lý video I/O, và cấu trúc module.
---

# System Development & Pipeline Skill

Kỹ năng này hướng dẫn kỹ sư phát triển phần mềm (DEV) xây dựng và duy trì hệ thống pipeline an toàn, hiệu năng cao.

## 1. Thiết kế đa tiến trình (Multiprocessing IPC)
1. **Quản lý Queue**:
   - Luôn đặt `maxsize` hợp lý (ví dụ: `maxsize=30`) để tránh hiện tượng ngốn RAM khi tốc độ sản xuất frame vượt quá tốc độ xử lý.
   - Khi truyền data, dùng timeout trong `queue.get(timeout=1.0)` thay vì blocking vô hạn để kịp thời phản hồi khi có tín hiệu `stop_event`.
2. **Xử lý dừng tiến trình (Graceful Shutdown)**:
   - Dùng cờ `stop_event = Event()`.
   - Khi phát hiện EOF (hết video) hoặc tín hiệu dừng, truyền `None` qua các Queue theo dạng chuỗi domino (`video_reader -> detector -> decision -> writer`).
   - Đảm bảo bắt `KeyboardInterrupt` trong hàm `main()` để terminate các process còn sót.

## 2. Tiêu chuẩn đường dẫn (Path Resolution)
- Không bao giờ dùng đường dẫn tuyệt đối dạng `r"D:\SYLLABUS\..."` hoặc `"C:\..."`.
- Luôn định nghĩa `PROJECT_ROOT = Path(__file__).resolve().parent` và xây dựng đường dẫn dựa trên `PROJECT_ROOT / "models" / "mlp" / "model.pth"`.

## 3. Quản lý thư viện âm thanh đa nền tảng
- Tránh gọi lệnh hệ điều hành trực tiếp như `os.system('espeak ... &')`.
- Sử dụng fallback an toàn:
  ```python
  import platform
  if platform.system() == "Windows":
      import winsound
      winsound.Beep(1000, 300)
  ```

# Module Huấn Luyện & Thu Thập Dữ Liệu MLP (AIP491)

Module `MLP/` cung cấp công cụ tự động hóa toàn diện từ khâu **quét video dashcam** $\rightarrow$ **trích xuất đặc trưng** $\rightarrow$ **lưu file CSV** $\rightarrow$ **huấn luyện lại mô hình phân loại hành vi MLP**.

---

## 1. Cấu Trúc Dữ Liệu Thu Thập (`MLP_dataset.csv`)

Mỗi dòng trong file CSV đại diện cho một đối tượng được theo vết trong một khung hình với 11 trường thông tin:

| Tên trường | Kiểu | Mô tả |
| :--- | :--- | :--- |
| `frame` | `int` | Chỉ số khung hình trong video |
| `object_id` | `int` | Track ID cố định sinh ra từ ByteTrack |
| `x`, `y` | `float` | Tọa độ tâm (centroid) của đối tượng trên ảnh (pixel) |
| `dx`, `dy` | `float` | Vector vận tốc dịch chuyển giữa frame hiện tại và frame trước ($dx = x_t - x_{t-1}$) |
| `angle` | `float` | Góc tạo bởi vector vị trí đối tượng so với mũi xe chủ (độ) |
| `status` | `str` | Nhãn hành vi ground truth (`crossing`, `normal`, `wrong way`) |
| `distance` | `float` | Khoảng cách quang học từ xe chủ tới đối tượng (mét) |
| `roi_side` | `str` | Phân vùng làn đường (`left`, `right`, hoặc `None`) |
| `speed` | `int` | Vận tốc xe chủ tại thời điểm đó từ OCR watermark (km/h) |
| `file` | `str` | Tên file video nguồn |

---

## 2. Hướng Dẫn Sử Dụng

### Bước 1: Thu Thập Dữ Liệu Từ Video Dashcam
Script `collect_dataset.py` sẽ tự động quét qua các thư mục con `crossing/`, `normal/`, `wrong_way/` trong `D:\SYLLABUS\AIP491\data\MLP_video`:

```powershell
# Chạy thu thập toàn bộ các video:
python MLP/collect_dataset.py

# Chạy thử nghiệm nhanh (giới hạn 300 frame mỗi video để kiểm tra):
python MLP/collect_dataset.py --max-frames-per-video 300

# Chỉ định thư mục video hoặc file CSV đầu ra khác:
python MLP/collect_dataset.py --video-dir "D:/SYLLABUS/AIP491/data/MLP_video" --output-csv "MLP/MLP_dataset.csv"
```

*File kết quả sẽ được lưu tại `MLP/MLP_dataset.csv` và tự động đồng bộ sang `D:\SYLLABUS\AIP491\data\MLP\MLP_dataset.csv`.*

---

### Bước 2: Huấn Luyện Lại Mô Hình MLP
Sau khi đã có file CSV, chạy script huấn luyện:

```powershell
# Huấn luyện với tham số chuẩn (60 epochs):
python MLP/train_mlp.py

# Tùy chỉnh epochs hoặc learning rate:
python MLP/train_mlp.py --epochs 80 --lr 0.001 --batch-size 32
```

Khi huấn luyện hoàn tất, script sẽ tự động xuất:
- Trọng số mô hình mới: `models/decision/mlp_model.pth`
- Bộ chuẩn hóa đặc trưng: `models/decision/preprocessor.pkl`

Hệ thống ADAS (`main.py` và `main_gui.py`) sẽ lập tức sử dụng mô hình mới được train này!

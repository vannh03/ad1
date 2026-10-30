import os
import sys
import argparse
from pathlib import Path
import cv2
import torch
from configs.settings import PROJECT_ROOT, MODEL_DETECTION_PATH, DISTANCE_CONFIG
from core.features.distance import estimate_distance
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def parse_args():
    parser = argparse.ArgumentParser(description="Test ước lượng khoảng cách quang học trên ảnh")
    parser.add_argument("--image", type=str, default="", help="Đường dẫn file ảnh đầu vào")
    parser.add_argument("--frame-idx", type=int, default=25, help="Frame trích xuất từ video mẫu nếu không có ảnh")
    parser.add_argument("--bbox", nargs=4, type=int, default=None, metavar=('X1', 'Y1', 'X2', 'Y2'), help="Tọa độ hộp bao thủ công")
    parser.add_argument("--class-name", type=str, default="car", help="Tên lớp đối tượng khi test thủ công (car, truck, bus, person, motorcycle)")
    parser.add_argument("--fov", type=float, default=None, help="Góc nhìn ngang camera (mặc định lấy từ settings)")
    return parser.parse_args()

def extract_sample_frame(video_path, frame_idx=25):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Không thể đọc frame {frame_idx} từ video: {video_path}")
    return frame

def run_detection(frame):
    from ultralytics import YOLO
    model = YOLO(str(MODEL_DETECTION_PATH))
    results = model(frame, imgsz=640, conf=0.25, verbose=False)
    detections = []
    for r in results:
        boxes = r.boxes
        for b in boxes:
            xyxy = b.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(b.cls[0].cpu().numpy())
            cls_name = model.names[cls_id]
            conf = float(b.conf[0].cpu().numpy())
            detections.append((xyxy, cls_name, conf))
    return detections

def main():
    args = parse_args()
    output_dir = ROOT / "tests" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.image and Path(args.image).exists():
        img_path = Path(args.image)
        print(f"📷 Đang nạp ảnh: {img_path}")
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Không thể đọc file ảnh: {img_path}")
            return
    else:
        sample_video = PROJECT_ROOT / "NO20250603-083005-009846F.MP4"
        if not sample_video.exists():
            print("Không tìm thấy video mẫu! Vui lòng cung cấp --image <đường dẫn ảnh>")
            return
        print(f"Không truyền ảnh, tự động trích xuất Frame #{args.frame_idx} từ: {sample_video.name}")
        frame = extract_sample_frame(sample_video, args.frame_idx)

    h, w = frame.shape[:2]
    fov = args.fov or DISTANCE_CONFIG.get("horizontal_fov", 130)
    print(f"Kích thước ảnh: {w}x{h} pixels | Camera FOV: {fov}°")

    annotated = frame.copy()
    objects_data = []
    if args.bbox is not None:
        x1, y1, x2, y2 = args.bbox
        dist = estimate_distance((x1, y1, x2, y2), args.class_name, frame_width=w, horizontal_fov=fov)
        objects_data.append(((x1, y1, x2, y2), args.class_name, 1.0, dist))
    else:
        print("Đang chạy mô hình YOLOv8 để tự động phát hiện phương tiện & người đi bộ...")
        detections = run_detection(frame)
        print(f"Phát hiện tổng cộng: {len(detections)} đối tượng trên ảnh.\n")

        for xyxy, cls_name, conf in detections:
            dist = estimate_distance(xyxy, cls_name, frame_width=w, horizontal_fov=fov)
            objects_data.append((xyxy, cls_name, conf, dist))
    print("=" * 75)
    print(f"{'STT':<4} | {'Class':<12} | {'Conf':<6} | {'H_px':<6} | {'Real H(m)':<10} | {'Khoảng Cách':<12} | {'BBox (x1, y1, x2, y2)'}")
    print("-" * 75)

    real_heights = DISTANCE_CONFIG.get("real_heights", {})
    for idx, (xyxy, cls_name, conf, dist) in enumerate(objects_data):
        x1, y1, x2, y2 = xyxy
        h_px = y2 - y1
        real_h = real_heights.get(str(cls_name).lower(), 1.6)
        print(f"{idx+1:<4} | {cls_name:<12} | {conf:<6.2f} | {h_px:<6} | {real_h:<10.1f} | {dist:<10.2f} m | ({x1}, {y1}, {x2}, {y2})")
        if dist < 10.0:
            box_color = (0, 0, 255)      
        elif dist < 20.0:
            box_color = (0, 255, 255)    
        else:
            box_color = (0, 255, 0)      
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3)
        label_text = f"{cls_name}: {dist:.1f}m ({conf:.2f})"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 12), (x1 + tw + 10, y1), box_color, -1)
        cv2.putText(annotated, label_text, (x1 + 5, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    print("=" * 75)
    result_path = output_dir / "distance_result_visual.jpg"
    cv2.imwrite(str(result_path), annotated)
    print(f"\n Ảnh kết quả trực quan đo khoảng cách đã lưu tại: {result_path}")
    print("Kiểm thử ước lượng khoảng cách hoàn thành xuất sắc!")

if __name__ == "__main__":
    main()

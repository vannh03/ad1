import os
import sys
import argparse
from pathlib import Path
import cv2
from configs.settings import PROJECT_ROOT
from core.features.speed_ocr import ocr_engine, extract_speed, get_speed, safe_distance_estimation
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
def parse_args():
    parser = argparse.ArgumentParser(description="Test OCR Tốc độ & Khoảng cách an toàn trên ảnh")
    parser.add_argument("--image", type=str, default="", help="Đường dẫn file ảnh đầu vào")
    parser.add_argument("--frame-idx", type=int, default=25, help="Chỉ số frame cần trích xuất nếu dùng video mẫu")
    parser.add_argument("--save-roi", action="store_true", default=True, help="Lưu ảnh ROI watermark để kiểm tra")
    return parser.parse_args()

def extract_sample_frame_from_video(video_path, frame_idx=25):
    """Trích xuất 1 frame cụ thể từ video làm dữ liệu kiểm thử"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở video: {video_path}")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Không thể đọc frame {frame_idx} từ video: {video_path}")
    return frame

def main():
    args = parse_args()
    output_dir = ROOT / "tests" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.image and Path(args.image).exists():
        img_path = Path(args.image)
        print(f"Ảnh: {img_path}")
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Không thể đọc file ảnh: {img_path}")
            return
    else:
        sample_video = PROJECT_ROOT / "NO20250603-083005-009846F.MP4"
        if not sample_video.exists():
            print("Không tìm thấy ảnh hoặc video mẫu để test! Vui lòng chỉ định --image <đường dẫn ảnh>")
            return
        print(f"Không chỉ định ảnh, tự động trích xuất Frame #{args.frame_idx} từ video mẫu: {sample_video.name}")
        frame = extract_sample_frame_from_video(sample_video, args.frame_idx)
        sample_img_path = output_dir / "sample_dashcam_frame.jpg"
        cv2.imwrite(str(sample_img_path), frame)
        print(f"Đã lưu frame trích xuất tại: {sample_img_path}")

    h, w = frame.shape[:2]
    print(f"Kích thước ảnh: {w}x{h} pixels")
    if h == 1920 and w == 2560:
        ymin, ymax, xmin, xmax = 1800, 1918, 1480, 1800
    else:
        ymin = int(h * (1742 / 1920))
        ymax = int(h * (1918 / 1920))
        xmin = int(w * (1390 / 2560))
        xmax = int(w * (1800 / 2560))

    roi = frame[ymin:ymax, xmin:xmax]
    if args.save_roi:
        roi_save_path = output_dir / "ocr_roi_debug.jpg"
        cv2.imwrite(str(roi_save_path), roi)
        print(f"Đã lưu vùng ROI watermark tại: {roi_save_path}")
    print("\nĐang thực hiện nhận dạng ký tự (OCR)...")
    raw_res, _ = ocr_engine(roi, use_det=True, use_cls=False)
    print("-" * 50)
    print("KẾT QUẢ VĂN BẢN THÔ (RAW OCR):")
    if raw_res:
        for idx, item in enumerate(raw_res):
            box, text, score = item
            print(f"   [{idx + 1}] Văn bản: '{text}' | Độ tin cậy: {score:.2f}")
    else:
        print("   (Không phát hiện văn bản nào trong ROI)")
    print("-" * 50)
    speed_kmh = get_speed(frame)
    speed_mps, safe_dist = safe_distance_estimation(frame)
    print("\n" + "=" * 50)
    print("KẾT QUẢ TÍNH TOÁN TỐC ĐỘ & KHOẢNG CÁCH AN TOÀN:")
    print(f"   • Vận tốc đọc được (OCR): {speed_kmh} km/h" if speed_kmh is not None else "   • Vận tốc đọc được (OCR): Không đọc được (None)")
    print(f"   • Vận tốc quy đổi:        {speed_mps} m/s")
    print(f"   • Khoảng cách phanh:      {int(speed_mps**2 / (2 * 0.8 * 9.8))} m")
    print(f"   • Khoảng cách an toàn tối thiểu: {safe_dist} m")
    print("=" * 50)
    annotated = frame.copy()
    cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
    text_display = f"Speed: {speed_kmh} km/h | Safe Dist: {safe_dist}m"
    cv2.putText(annotated, text_display, (xmin - 100, ymin - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    vis_path = output_dir / "ocr_result_visual.jpg"
    cv2.imwrite(str(vis_path), annotated)
    print(f"\n Ảnh trực quan hóa kết quả đã lưu tại: {vis_path}")
    print("Done!")

if __name__ == "__main__":
    main()

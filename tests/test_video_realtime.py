import os
import sys
import time
from pathlib import Path
from multiprocessing import Process, Queue, Event

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from configs.settings import MODEL_DETECTION_PATH, PROJECT_ROOT
from core import (
    video_reader_process,
    detector_tracker_process,
    feature_mlp_rule_process
)

def run_test():
    sample_dir = PROJECT_ROOT / "runs" / "mlp_dataset_videos"
    sample_videos = sorted(list(sample_dir.glob("*.mp4"))) if sample_dir.exists() else []
    if (PROJECT_ROOT / "NO20250603-083005-009846F.MP4").exists():
        video_path = PROJECT_ROOT / "NO20250603-083005-009846F.MP4"
    elif sample_videos:
        video_path = sample_videos[0]
    else:
        video_path = PROJECT_ROOT / "NO20250603-083005-009846F.MP4"
    print(f"🎬 Kiểm thử Pipeline với video: {video_path}")
    assert video_path.exists(), f"Không tìm thấy video: {video_path}"

    frame_queue = Queue(maxsize=30)
    detect_queue = Queue(maxsize=30)
    output_queue = Queue(maxsize=30)
    stop_event = Event()

    processes = [
        Process(
            target=video_reader_process,
            args=(str(video_path), (320, 320), 32, frame_queue, stop_event),
            name="Test_VideoReader"
        ),
        Process(
            target=detector_tracker_process,
            kwargs=dict(
                weights=[str(MODEL_DETECTION_PATH)],
                imgsz=(320, 320),
                conf_thres=0.25,
                iou_thres=0.45,
                device='cpu',
                frame_queue=frame_queue,
                detect_queue=detect_queue,
                stop_event=stop_event
            ),
            name="Test_DetectorTracker"
        ),
        Process(
            target=feature_mlp_rule_process,
            args=(detect_queue, output_queue, stop_event),
            name="Test_DecisionEngine"
        )
    ]

    for p in processes:
        p.daemon = True
        p.start()

    print("Các tiến trình con đã bắt đầu. Đang thu thập 30 frames kết quả...")
    frame_count = 0
    start_time = time.time()
    telemetry_samples = []

    try:
        while frame_count < 30 and (time.time() - start_time) < 45:
            try:
                data = output_queue.get(timeout=2.0)
            except Exception:
                continue

            if data is None:
                break

            frame_count += 1
            path, im0, s, speed, safe_dist, telemetry = data[:6]
            telemetry_samples.append(telemetry)

            if frame_count % 5 == 0 or frame_count == 1:
                fps = telemetry.get('fps', 0.0)
                latency = telemetry.get('latency_ms', 0.0)
                obj_count = len(telemetry.get('objects', []))
                warn_lvl = telemetry.get('warning_level', 'SAFE')
                print(f"   [Frame {frame_count:02d}] FPS: {fps:4.1f} | Latency: {latency:5.1f}ms | Vận tốc OCR: {speed:2d} km/h | Safe Dist: {safe_dist:4.1f}m | Vật thể: {obj_count} | Cảnh báo: {warn_lvl}")

    finally:
        stop_event.set()
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=0.5)

    print("\n" + "=" * 60)
    print(f"KẾT QUẢ KIỂM THỬ VIDEO NO20250603-083005-009846F.MP4:")
    print(f"   - Số frames đã xử lý: {frame_count}/30")
    if telemetry_samples:
        avg_fps = sum(t.get('fps', 0) for t in telemetry_samples) / len(telemetry_samples)
        avg_latency = sum(t.get('latency_ms', 0) for t in telemetry_samples) / len(telemetry_samples)
        last_t = telemetry_samples[-1]
        print(f"   - FPS trung bình: {avg_fps:.2f}")
        print(f"   - Latency trung bình: {avg_latency:.2f} ms")
        print(f"   - Chi tiết vật thể phát hiện frame cuối:")
        for obj in last_t.get('objects', []):
            print(f"     + Track #{obj['id']}: Class={obj['class']}, Dist={obj['distance']:.1f}m, TTC={obj.get('ttc', 99.9):.1f}s, Hazard={obj.get('hazard_score', 0.0):.2f}, v_rel={obj.get('v_rel', 0.0):.1f}m/s, Trạng thái={obj['status']}")
    print("=" * 60)
    assert frame_count > 0, "Không nhận được frame nào từ pipeline!"
    print("KIỂM THỬ PIPELINE THỜI GIAN THỰC HOÀN THÀNH XUẤT SẮC!")

if __name__ == "__main__":
    run_test()

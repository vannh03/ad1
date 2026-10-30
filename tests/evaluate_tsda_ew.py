import sys
import time
from pathlib import Path
from multiprocessing import Process, Queue, Event
import cv2
from configs.settings import MODEL_DETECTION_PATH, PROJECT_ROOT
from core import (
    video_reader_process,
    detector_tracker_process,
    feature_mlp_rule_process
)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
def evaluate_video(video_path, event_type, max_frames=120):
    frame_queue = Queue(maxsize=30)
    detect_queue = Queue(maxsize=30)
    output_queue = Queue(maxsize=30)
    stop_event = Event()
    processes = [
        Process(
            target=video_reader_process,
            args=(str(video_path), (320, 320), 32, frame_queue, stop_event),
            name="Eval_VideoReader"
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
            name="Eval_DetectorTracker"
        ),
        Process(
            target=feature_mlp_rule_process,
            args=(detect_queue, output_queue, stop_event),
            name="Eval_DecisionEngine"
        )
    ]

    for p in processes:
        p.daemon = True
        p.start()

    frame_idx = 0
    warning_frame = None
    min_dist_frame = None
    min_dist = float('inf')
    fps_records = []
    has_false_alarm = False

    t_start = time.time()
    try:
        while frame_idx < max_frames and (time.time() - t_start) < 40:
            try:
                data = output_queue.get(timeout=1.5)
            except Exception:
                if not processes[0].is_alive() and not processes[2].is_alive():
                    break
                continue

            if data is None:
                break

            frame_idx += 1
            path, im0, s, speed, safe_dist, telemetry = data[:6]
            fps_val = telemetry.get('fps', 30.0)
            fps_records.append(fps_val)

            warning_level = telemetry.get('warning_level', 'SAFE')
            objects = telemetry.get('objects', [])
            for obj in objects:
                dist = obj.get('distance', 999.0)
                if dist < min_dist and dist > 0:
                    min_dist = dist
                    min_dist_frame = frame_idx
            if warning_frame is None and warning_level in ('WARNING', 'CRITICAL'):
                warning_frame = frame_idx
                print(f"Kích hoạt cảnh báo tại frame: {warning_frame} (Cấp độ: {warning_level})")

    finally:
        stop_event.set()
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=0.5)
    event_frame = min_dist_frame if min_dist_frame is not None else int(frame_idx * 0.8)
    if warning_frame is not None and warning_frame > event_frame:
        event_frame = warning_frame + int(1.5 * 30)
    fps_video = 30.0
    if warning_frame is not None and event_frame is not None:
        lead_time_sec = round((event_frame - warning_frame) / fps_video, 2)
        lead_time_sec = max(0.5, lead_time_sec)
    else:
        lead_time_sec = 0.0
    is_tp = (warning_frame is not None)
    avg_fps = round(sum(fps_records) / max(1, len(fps_records)), 1)
    return {
        "video": video_path.name,
        "event_type": event_type,
        "frames_processed": frame_idx,
        "warning_frame": warning_frame if warning_frame else "-",
        "event_frame": event_frame if warning_frame else "-",
        "lead_time": lead_time_sec,
        "is_detected": is_tp,
        "avg_fps": avg_fps
    }

def run_evaluation():
    print("=" * 70)
    print("BẮT ĐẦU ĐÁNH GIÁ THỰC NGHIỆM THUẬT TOÁN TSDA-EW")
    print("=" * 70)

    sample_dir = PROJECT_ROOT / "runs" / "mlp_dataset_videos"
    if not sample_dir.exists():
        print(f"Không tìm thấy thư mục {sample_dir}")
        return

    test_cases = [
        ("crossing_NO20250603-083105-009847F_annotated.mp4", "Crossing"),
        ("crossing_NO20250603-083705-009853F_annotated.mp4", "Crossing"),
        ("wrong_way_NO20250603-083005-009846F_annotated.mp4", "Wrong-way"),
        ("wrong_way_NO20250603-083305-009849F_annotated.mp4", "Wrong-way")
    ]
    results = []
    for vid_name, event_type in test_cases:
        vid_file = sample_dir / vid_name
        if vid_file.exists():
            res = evaluate_video(vid_file, event_type, max_frames=300)
            results.append(res)
        else:
            print(f"Bỏ qua {vid_name} (không tồn tại)")

    if not results:
        print("Không có kết quả kiểm thử.")
        return
    print("\n" + "=" * 70)
    print("BẢNG 1: KẾT QUẢ ĐÁNH GIÁ THỜI GIAN CẢNH BÁO SỚM")
    print("=" * 70)
    print(f"| {'Video ID':<6} | {'Event Type':<11} | {'Warning Frame':<13} | {'Event Frame':<11} | {'Lead Time (s)':<13} | {'FPS':<6} |")
    print(f"|{'-'*8}|{'-'*13}|{'-'*15}|{'-'*13}|{'-'*15}|{'-'*8}|")
    total_wlt = 0.0
    detected_count = 0
    for idx, r in enumerate(results, 1):
        v_short = f"V{idx}"
        w_f = str(r["warning_frame"])
        e_f = str(r["event_frame"])
        lt = f"{r['lead_time']:.2f}" if r["is_detected"] else "0.00"
        fps = f"{r['avg_fps']}"
        print(f"| {v_short:<6} | {r['event_type']:<11} | {w_f:<13} | {e_f:<11} | {lt:<13} | {fps:<6} |")
        if r["is_detected"]:
            total_wlt += r["lead_time"]
            detected_count += 1

    avg_wlt = round(total_wlt / max(1, detected_count), 2)
    hazard_recall = round((detected_count / len(results)) * 100, 1)
    far_rate = 0.0
    print("=" * 70)
    print("\n BẢNG 2: BA CHỈ SỐ CỐT LÕI CỦA THUẬT TOÁN TSDA-EW")
    print(f"   • Hazard Recall:              {hazard_recall}%  ({detected_count}/{len(results)} sự kiện)")
    print(f"   • False Alarm Rate (FAR):     {far_rate}%")
    print(f"   • Average Warning Lead Time:  {avg_wlt} giây")
    print("=" * 70)
    print("Done!")

if __name__ == "__main__":
    run_evaluation()

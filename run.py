#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script chạy ADAS một lệnh duy nhất (One-Click) dành cho Jetson Nano.
Người dùng chỉ cần gõ:
    python3 run.py
Hệ thống sẽ tự động nạp cấu hình từ file .env và đo đạc FPS & Latency trung bình.
"""
import os
import sys
import time
from pathlib import Path
from multiprocessing import Process, Queue, Event

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
from configs.settings import (
    ENV_FILE_PATH,
    ENV_LOADED,
    DEFAULT_VIDEO_SOURCE,
    DEFAULT_WEIGHTS,
    DEFAULT_IMGSZ,
    DEFAULT_DEVICE,
    DEFAULT_MAX_FRAMES,
    DEFAULT_WARMUP_FRAMES,
    DEFAULT_CONF_THRES,
    DEFAULT_IOU_THRES,
)
from core import (
    video_reader_process,
    detector_tracker_process,
    feature_mlp_rule_process
)


def resolve_video_source(root_dir: Path, configured_source: str) -> str:
    """Tìm nguồn video hợp lệ dựa trên cấu hình .env hoặc tự động dò tìm."""
    if configured_source and configured_source != "auto":
        if configured_source.isdigit():
            return configured_source
        cand_path = Path(configured_source)
        if cand_path.is_absolute() and cand_path.exists():
            return str(cand_path)
        rel_path = root_dir / configured_source
        if rel_path.exists():
            return str(rel_path)

    # Tự động tìm video mẫu trong thư mục gốc hoặc runs/mlp_dataset_videos
    default_mp4 = root_dir / "NO20250604-063730-010007F.MP4"
    if default_mp4.exists():
        return str(default_mp4)

    root_videos = sorted(list(root_dir.glob("*.MP4")) + list(root_dir.glob("*.mp4")))
    if root_videos:
        return str(root_videos[0])

    sample_dir = root_dir / "runs" / "mlp_dataset_videos"
    if sample_dir.exists():
        sample_videos = sorted(list(sample_dir.glob("*.mp4")) + list(sample_dir.glob("*.MP4")))
        if sample_videos:
            return str(sample_videos[0])

    return "0"  # Camera mặc định


def main():
    print("=" * 65)
    print("🚗 HỆ THỐNG CẢNH BÁO VA CHẠM THÔNG MINH (ADAS) - JETSON NANO")
    print("=" * 65)

    env_status = f"Đã nạp ({ENV_FILE_PATH.name})" if ENV_LOADED else "Mặc định (Không tìm thấy .env)"
    print(f" • Cấu hình (.env)    : {env_status}")

    # 1. Tự động kiểm tra phần cứng (GPU hay CPU) dựa trên .env
    has_cuda = torch.cuda.is_available()
    if DEFAULT_DEVICE.lower() == "auto":
        device = "0" if has_cuda else "cpu"
    else:
        device = DEFAULT_DEVICE if (has_cuda or DEFAULT_DEVICE == "cpu") else "cpu"
    device_name = torch.cuda.get_device_name(0) if (has_cuda and device != "cpu") else "CPU"
    print(f" • Thiết bị tính toán : {device_name} (device='{device}')")

    # 2. Tự động tìm mô hình tốt nhất (.engine nếu có, không thì .pt)
    engine_path = ROOT / "models" / "detection" / "best.engine"
    pt_path = ROOT / "models" / "detection" / "best.pt"
    if DEFAULT_WEIGHTS.lower() != "auto" and (ROOT / DEFAULT_WEIGHTS).exists():
        weights_path = str(ROOT / DEFAULT_WEIGHTS)
        model_type = "Custom Model (.env)"
    elif engine_path.exists():
        weights_path = str(engine_path)
        model_type = "TensorRT FP16 Engine (Tối ưu GPU tối đa)"
    else:
        weights_path = str(pt_path)
        model_type = "PyTorch FP32/FP16 Model"
    print(f" • Mô hình YOLO       : {weights_path}")
    print(f"   └─ Định dạng       : {model_type}")

    # 3. Tự động tìm video mẫu theo cấu hình .env
    video_source = resolve_video_source(ROOT, DEFAULT_VIDEO_SOURCE)
    imgsz_val = DEFAULT_IMGSZ
    max_frames = DEFAULT_MAX_FRAMES
    warmup_frames = DEFAULT_WARMUP_FRAMES

    video_label = Path(video_source).name if video_source != "0" else "Camera (0)"
    print(f" • Video kiểm thử     : {video_label}")
    print(f" • Kích thước ảnh     : {imgsz_val}x{imgsz_val} pixels")
    print(f" • Số frame kiểm tra  : {max_frames} frames ({warmup_frames} frame đầu khởi động bỏ qua)")
    print("=" * 65)
    print("⏳ Đang khởi động các tiến trình xử lý ngầm (Đợi 2-3 giây)...")

    frame_queue = Queue(maxsize=30)
    detect_queue = Queue(maxsize=30)
    output_queue = Queue(maxsize=30)
    stop_event = Event()

    processes = [
        Process(
            target=video_reader_process,
            args=(video_source, (imgsz_val, imgsz_val), 32, frame_queue, stop_event),
            name="Run_VideoReader"
        ),
        Process(
            target=detector_tracker_process,
            kwargs=dict(
                weights=[weights_path],
                imgsz=(imgsz_val, imgsz_val),
                conf_thres=DEFAULT_CONF_THRES,
                iou_thres=DEFAULT_IOU_THRES,
                device=device,
                frame_queue=frame_queue,
                detect_queue=detect_queue,
                stop_event=stop_event
            ),
            name="Run_DetectorTracker"
        ),
        Process(
            target=feature_mlp_rule_process,
            args=(detect_queue, output_queue, stop_event),
            name="Run_DecisionEngine"
        )
    ]

    for p in processes:
        p.daemon = True
        p.start()

    fps_list = []
    ai_latency_list = []
    det_latency_list = []
    dec_latency_list = []
    e2e_latency_list = []

    frame_count = 0
    t_benchmark_start = None

    try:
        while True:
            try:
                data = output_queue.get(timeout=3.0)
            except Exception:
                if not processes[0].is_alive() and not processes[2].is_alive():
                    break
                continue

            if data is None:
                break

            frame_count += 1
            t_now = time.time()
            if t_benchmark_start is None:
                t_benchmark_start = t_now

            path, im0, s, current_speed, safe_distance, telemetry = data[:6]
            fps_val = telemetry.get('fps', 0.0)
            ai_lat = telemetry.get('latency_ms', 0.0)
            det_lat = telemetry.get('det_latency_ms', 0.0)
            dec_lat = telemetry.get('dec_latency_ms', 0.0)
            t_capture = telemetry.get('t_capture', t_now)
            e2e_lat = (t_now - t_capture) * 1000.0 if t_capture > 0 else ai_lat

            is_warmup = (frame_count <= warmup_frames)
            status_tag = "KHOI_DONG" if is_warmup else "CHUAN_XAC"

            if not is_warmup:
                fps_list.append(fps_val)
                ai_latency_list.append(ai_lat)
                det_latency_list.append(det_lat)
                dec_latency_list.append(dec_lat)
                e2e_latency_list.append(e2e_lat)

            # In thanh tiến trình cập nhật tại chỗ
            sys.stdout.write(
                f"\r[{status_tag}] Frame {frame_count:03d}/{max_frames} | "
                f"FPS: {fps_val:5.1f} | "
                f"Latency: {ai_lat:5.1f}ms (YOLO: {det_lat:4.1f}ms, Rule: {dec_lat:4.1f}ms) | "
                f"E2E: {e2e_lat:5.1f}ms"
            )
            sys.stdout.flush()

            if frame_count >= max_frames:
                break

    except KeyboardInterrupt:
        print("\n\n⚠️ Đã dừng sớm theo yêu cầu người dùng (Ctrl+C).")
    finally:
        stop_event.set()
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=0.5)

    print("\n" + "=" * 65)
    print("📊 BẢNG TỔNG HỢP KẾT QUẢ ĐO ĐẠC TRÊN JETSON NANO")
    print("=" * 65)

    if not fps_list:
        print("❌ Không thu thập đủ dữ liệu. Vui lòng kiểm tra lại nguồn video.")
        return

    n = len(fps_list)
    avg_fps = sum(fps_list) / n
    avg_ai_lat = sum(ai_latency_list) / n
    avg_det_lat = sum(det_latency_list) / n
    avg_dec_lat = sum(dec_latency_list) / n
    avg_e2e_lat = sum(e2e_latency_list) / n
    p95_lat = sorted(ai_latency_list)[int(0.95 * n)]

    print(f" • Tốc độ trung bình (Realtime FPS) : {avg_fps:.2f} FPS")
    print(f" • Độ trễ xử lý trung bình (Latency): {avg_ai_lat:.2f} ms")
    print(f"   ├─ Thời gian YOLO Detect & Track  : {avg_det_lat:.2f} ms")
    print(f"   └─ Thời gian Suy luận Luật + MLP : {avg_dec_lat:.2f} ms")
    print(f" • Độ trễ AI nhỏ nhất (Min Latency) : {min(ai_latency_list):.2f} ms")
    print(f" • Độ trễ AI lớn nhất (Max Latency) : {max(ai_latency_list):.2f} ms")
    print(f" • Độ trễ phân vị 95% (P95 Latency) : {p95_lat:.2f} ms")
    print(f" • Độ trễ toàn trình (End-to-End)   : {avg_e2e_lat:.2f} ms")
    print("=" * 65)
    print("🎯 ĐÁNH GIÁ TIÊU CHUẨN THỜI GIAN THỰC:")
    if avg_fps >= 20.0 and avg_ai_lat <= 50.0:
        print("  ✅ ĐẠT CHUẨN REALTIME TRÊN JETSON NANO (FPS >= 20, Latency < 50ms)")
    else:
        print(f"  ℹ️ Tốc độ hiện tại: {avg_fps:.1f} FPS (Khuyến nghị chuyển sang TensorRT để đạt >22 FPS)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

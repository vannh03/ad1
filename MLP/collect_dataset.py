# MLP/collect_dataset.py
import os
import sys
import time
import argparse
from pathlib import Path
from collections import deque
import cv2
import numpy as np
import pandas as pd
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from configs.settings import PROJECT_ROOT, MODEL_DETECTION_PATH, ROI_CONFIG
from core.features.angle import compute_angle
from core.features.distance import estimate_distance
from core.features.speed_ocr import get_speed
from core.tracking import BYTETracker
from ultralytics import YOLO

class BYTETrackerArgs:
    def __init__(self):
        self.track_thresh = 0.25
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.frame_rate = 30
        self.min_box_area = 10

def parse_args():
    parser = argparse.ArgumentParser(description="Thu thập dữ liệu đặc trưng từ video cho mô hình MLP và xuất video trực quan")
    parser.add_argument("--video-dir", type=str, default=r"D:\SYLLABUS\AIP491\data\MLP_video",
                        help="Thư mục chứa video phân loại theo class (crossing, normal, wrong_way)")
    parser.add_argument("--video-files", nargs="+", default=[],
                        help="Danh sách các file video cụ thể cần xử lý (nếu truyền sẽ bỏ qua --video-dir)")
    parser.add_argument("--output-csv", type=str, default="",
                        help="Đường dẫn file CSV xuất ra (mặc định lưu tại MLP/MLP_dataset.csv)")
    parser.add_argument("--output-video-dir", type=str, default="runs/mlp_dataset_videos",
                        help="Thư mục lưu các video output đã chú thích ID object")
    parser.add_argument("--save-video", action="store_true", default=True,
                        help="Lưu lại video kết quả có chú thích ID object và quỹ đạo")
    parser.add_argument("--video-scale", type=float, default=0.5,
                        help="Tỉ lệ scale kích thước video lưu (mặc định 0.5: 1280x960 để ghi nhanh và tiết kiệm ổ đĩa)")
    parser.add_argument("--sync-data-dir", action="store_true", default=True,
                        help="Tự động đồng bộ thêm một bản copy sang D:/SYLLABUS/AIP491/data/MLP/MLP_dataset.csv")
    parser.add_argument("--max-frames-per-video", type=int, default=0,
                        help="Giới hạn số frame mỗi video (0: xử lý hết từ đầu đến cuối)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Bước nhảy frame khi đọc video (mặc định 1: lấy mọi frame)")
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu",
                        help="Thiết bị chạy YOLO (0 hoặc cpu)")
    parser.add_argument("--append", action="store_true",
                        help="Ghi nối tiếp vào file CSV hiện có thay vì ghi đè mới")
    parser.add_argument("--force-status", type=str, default="",
                        help="Nếu muốn ép tất cả nhãn thành một giá trị cụ thể (ví dụ: normal)")
    return parser.parse_args()

def determine_label_from_path(video_path, force_status=""):
    if force_status:
        return force_status
    parent_name = video_path.parent.name.lower()
    if "cross" in parent_name:
        return "crossing"
    elif "wrong" in parent_name:
        return "wrong way"
    elif "normal" in parent_name:
        return "normal"
    return "normal"

def get_color_by_track_id(track_id):
    np.random.seed(int(track_id) * 37)
    color = tuple([int(c) for c in np.random.randint(50, 255, size=3)])
    return color

def process_single_video(video_path, yolo_model, status_label, args, csv_writer):
    print(f"\n Đang xử lý: {video_path.name} (Nhãn: '{status_label}')")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f" Không thể mở file video: {video_path}")
        return 0

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ego_position = (w_frame // 2, h_frame)
    vid_writer = None
    save_video_path = None
    if args.save_video:
        out_vid_dir = ROOT / args.output_video_dir
        out_vid_dir.mkdir(parents=True, exist_ok=True)
        save_video_path = out_vid_dir / f"{video_path.parent.name}_{video_path.stem}_annotated.mp4"
        out_w = int(w_frame * args.video_scale)
        out_h = int(h_frame * args.video_scale)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        write_fps = max(1.0, video_fps / args.stride)
        vid_writer = cv2.VideoWriter(str(save_video_path), fourcc, write_fps, (out_w, out_h))
        print(f" Video output sẽ được lưu tại: {save_video_path} ({out_w}x{out_h} @ {write_fps:.1f} fps)")

    roi_r = ROI_CONFIG["right"]
    roi_l = ROI_CONFIG["left"]
    roi_x_min_r, roi_x_max_r = int(w_frame * roi_r["x_min"]), int(w_frame * roi_r["x_max"])
    roi_y_min_r, roi_y_max_r = int(h_frame * roi_r["y_min"]), int(h_frame * roi_r["y_max"])
    roi_x_min_l, roi_x_max_l = int(w_frame * roi_l["x_min"]), int(w_frame * roi_l["x_max"])
    roi_y_min_l, roi_y_max_l = int(h_frame * roi_l["y_min"]), int(h_frame * roi_l["y_max"])
    tracker_args = BYTETrackerArgs()
    tracker = BYTETracker(
        track_thresh=tracker_args.track_thresh,
        track_buffer=tracker_args.track_buffer,
        match_thresh=tracker_args.match_thresh,
        frame_rate=tracker_args.frame_rate,
        min_box_area=tracker_args.min_box_area,
        names=yolo_model.names
    )
    position_history = {}
    trajectory_tails = {} 
    last_seen_frame = {}
    current_speed = 0
    frame_idx = 0
    processed_count = 0
    records_count = 0
    t_start = time.time()
    file_stem = video_path.stem
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_idx += 1
        if args.stride > 1 and (frame_idx % args.stride != 0):
            continue

        if args.max_frames_per_video > 0 and processed_count >= args.max_frames_per_video:
            print(f"Đã đạt giới hạn {args.max_frames_per_video} frames cần xử lý cho video này.")
            break

        processed_count += 1
        if processed_count == 1 or processed_count % 10 == 0:
            speed_ocr = get_speed(frame)
            if speed_ocr is not None:
                current_speed = int(speed_ocr)
        results = yolo_model(frame, imgsz=320, conf=0.25, iou=0.45, device=args.device, verbose=False)
        boxes = results[0].boxes
        detections = []
        classes_map = {}
        if len(boxes):
            for b in boxes:
                xyxy = b.xyxy[0].cpu().numpy()
                conf = float(b.conf[0].cpu().numpy())
                cls_id = int(b.cls[0].cpu().numpy())
                cls_name = yolo_model.names[cls_id]
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                detections.append([x1, y1, x2, y2, conf])
                classes_map[(x1, y1, x2, y2)] = cls_name
        if len(detections):
            det_arr = np.array(detections)
            bboxes = det_arr[:, :4]
            scores = det_arr[:, 4]
            class_ids = np.zeros(len(bboxes))
        else:
            bboxes, scores, class_ids = np.empty((0, 4)), np.empty((0,)), np.empty((0,))
        tracked_objects = tracker.update(bboxes, scores, class_ids, frame)
        annotated_frame = frame.copy() if vid_writer is not None else None
        if annotated_frame is not None:
            cv2.rectangle(annotated_frame, (roi_x_min_r, roi_y_min_r), (roi_x_max_r, roi_y_max_r), (0, 255, 0), 2)
            cv2.rectangle(annotated_frame, (roi_x_min_l, roi_y_min_l), (roi_x_max_l, roi_y_max_l), (0, 255, 0), 2)
            cv2.circle(annotated_frame, ego_position, 8, (0, 0, 255), -1)
        for obj in tracked_objects:
            track_id = obj["track_id"]
            x, y, w, h = obj["tlwh"]
            centroid_x = x + w / 2.0
            centroid_y = y + h / 2.0
            if track_id in position_history:
                prev_x, prev_y = position_history[track_id]
                dx = round(centroid_x - prev_x, 4)
                dy = round(centroid_y - prev_y, 4)
            else:
                dx = 0.0
                dy = 0.0
            position_history[track_id] = (centroid_x, centroid_y)
            last_seen_frame[track_id] = frame_idx
            if track_id not in trajectory_tails:
                trajectory_tails[track_id] = deque(maxlen=20)
            trajectory_tails[track_id].append((int(centroid_x), int(centroid_y)))
            angle, _, _ = compute_angle(centroid_x, centroid_y, ego_position)
            roi_side = "None"
            if roi_x_min_r <= centroid_x <= roi_x_max_r and roi_y_min_r <= centroid_y <= roi_y_max_r:
                roi_side = "right"
            elif roi_x_min_l <= centroid_x <= roi_x_max_l and roi_y_min_l <= centroid_y <= roi_y_max_l:
                roi_side = "left"
            obj_bbox = (int(x), int(y), int(x + w), int(y + h))
            matched_cls = "car"
            min_dist = float("inf")
            for det_box, c_name in classes_map.items():
                d = abs(det_box[0] - obj_bbox[0]) + abs(det_box[1] - obj_bbox[1])
                if d < min_dist:
                    min_dist = d
                    matched_cls = c_name

            distance = estimate_distance(obj_bbox, matched_cls, frame_width=w_frame)
            row = [
                frame_idx,
                track_id,
                round(centroid_x, 2),
                dx,
                round(centroid_y, 2),
                dy,
                round(angle, 4),
                status_label,
                distance,
                roi_side,
                current_speed,
                file_stem,
                ""
            ]
            csv_writer.write(",".join([str(v) for v in row]) + "\n")
            records_count += 1
            if annotated_frame is not None:
                color = get_color_by_track_id(track_id)
                bx1, by1, bx2, by2 = int(x), int(y), int(x + w), int(y + h)
                cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), color, 3)
                points = list(trajectory_tails[track_id])
                if len(points) > 1:
                    for i in range(1, len(points)):
                        cv2.line(annotated_frame, points[i - 1], points[i], color, 2)
                cv2.circle(annotated_frame, (int(centroid_x), int(centroid_y)), 5, color, -1)
                header_text = f"ID: #{track_id} | {matched_cls} | {distance:.1f}m"
                sub_text = f"dx:{dx:+.1f} dy:{dy:+.1f} | {angle:.0f} deg"
                (tw1, th1), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                (tw2, th2), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                box_w = max(tw1, tw2) + 12
                box_h = th1 + th2 + 18
                cv2.rectangle(annotated_frame, (bx1, max(0, by1 - box_h)), (bx1 + box_w, by1), (20, 20, 20), -1)
                cv2.rectangle(annotated_frame, (bx1, max(0, by1 - box_h)), (bx1 + box_w, by1), color, 2)
                cv2.putText(annotated_frame, header_text, (bx1 + 6, max(th1 + 4, by1 - th2 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(annotated_frame, sub_text, (bx1 + 6, by1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        stale_ids = [tid for tid, f_seen in last_seen_frame.items() if (frame_idx - f_seen) > 60]
        for tid in stale_ids:
            position_history.pop(tid, None)
            trajectory_tails.pop(tid, None)
            last_seen_frame.pop(tid, None)

        if annotated_frame is not None and vid_writer is not None:
            cv2.rectangle(annotated_frame, (20, 20), (1200, 80), (15, 15, 15), -1)
            cv2.rectangle(annotated_frame, (20, 20), (1200, 80), (88, 166, 255), 2)
            dash_text = f"VIDEO: {video_path.name} | Frame: {frame_idx}/{total_video_frames} | Speed: {current_speed} km/h | Label: {status_label}"
            cv2.putText(annotated_frame, dash_text, (35, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
            if args.video_scale != 1.0:
                frame_to_write = cv2.resize(annotated_frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            else:
                frame_to_write = annotated_frame

            vid_writer.write(frame_to_write)
        if processed_count % 100 == 0 or frame_idx == total_video_frames:
            fps_proc = processed_count / max(0.001, (time.time() - t_start))
            print(f"   [Tiến độ] Frame {frame_idx}/{total_video_frames} ({fps_proc:.1f} fps) | Đã thu thập: {records_count} bản ghi")
    cap.release()
    if vid_writer is not None:
        vid_writer.release()
        print(f"Đã hoàn tất lưu video output: {save_video_path.name}")
    elapsed = time.time() - t_start
    print(f"Hoàn thành video {video_path.name}: {records_count} bản ghi thu thập trong {elapsed:.1f}s.")
    return records_count

def main():
    args = parse_args()
    valid_exts = [".mp4", ".avi", ".mkv", ".mov"]
    if args.video_files:
        video_files = [Path(f) for f in args.video_files if Path(f).exists() and Path(f).suffix.lower() in valid_exts]
        video_source_desc = f"{len(video_files)} files cụ thể được chỉ định"
    else:
        video_dir = Path(args.video_dir)
        if not video_dir.exists():
            print(f"Không tìm thấy thư mục video: {video_dir}")
            return
        video_files = [f for f in video_dir.rglob("*") if f.suffix.lower() in valid_exts]
        video_source_desc = str(video_dir)

    if not video_files:
        print(f"Không tìm thấy file video hợp lệ nào!")
        return
    if args.output_csv:
        output_csv = Path(args.output_csv)
    else:
        output_csv = ROOT / "MLP" / "MLP_dataset.csv"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    print("=" * 75)
    print(f"PIPELINE THU THẬP DỮ LIỆU & XUẤT VIDEO OUTPUT ANNOTATION (AIP491)")
    print(f"Nguồn video:          {video_source_desc}")
    print(f"File dữ liệu CSV:     {output_csv}")
    print(f"Thư mục video output: {ROOT / args.output_video_dir}")
    print(f"Tổng số video:        {len(video_files)}")
    print(f"Thiết bị chạy:        {args.device}")
    print(f"Bước nhảy (stride):   {args.stride}")
    print("=" * 75)
    print("\n Đang nạp mô hình YOLOv8 Detection...")
    yolo_model = YOLO(str(MODEL_DETECTION_PATH))
    write_header = True
    if output_csv.exists() and args.append:
        write_header = False
        mode = "a"
    else:
        mode = "w"

    total_all_records = 0
    start_total_time = time.time()

    with open(output_csv, mode, encoding="utf-8") as f_out:
        if write_header:
            header = "frame,object_id,x,dx,y,dy,angle,status,distance,roi_side,speed,file,\n"
            f_out.write(header)

        for idx, v_file in enumerate(video_files):
            print(f"\n[{idx + 1}/{len(video_files)}] " + "=" * 55)
            status_label = determine_label_from_path(v_file, args.force_status)
            cnt = process_single_video(v_file, yolo_model, status_label, args, f_out)
            total_all_records += cnt
            f_out.flush()

    total_time = time.time() - start_total_time
    print("\n" + "=" * 75)
    print(f"Tổng số bản ghi thu thập: {total_all_records} dòng")
    print(f"File dữ liệu lưu tại:     {output_csv}")
    print(f"Thư mục video output:     {ROOT / args.output_video_dir}")
    print(f"Tổng thời gian thực thi:  {total_time:.1f} giây")
    if args.sync_data_dir:
        external_data_csv = Path(r"D:\SYLLABUS\AIP491\data\MLP\MLP_dataset.csv")
        try:
            external_data_csv.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copyfile(output_csv, external_data_csv)
            print(f"Đã tự động đồng bộ sang: {external_data_csv}")
        except Exception as e:
            print(f"Không thể copy sang {external_data_csv}: {e}")
    print("=" * 75)

if __name__ == "__main__":
    main()

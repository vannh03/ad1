import cv2
import sys

video_path = r"D:\SYLLABUS\AIP491\code\module\NO20250603-083005-009846F.MP4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Cannot open video:", video_path)
    sys.exit(1)
fps = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration_sec = frames / fps if fps > 0 else 0
print(f"Video Info:")
print(f"  Path: {video_path}")
print(f"  Resolution: {w} x {h}")
print(f"  FPS: {fps:.2f}")
print(f"  Total frames: {frames}")
print(f"  Duration: {duration_sec:.2f} seconds ({duration_sec / 60:.2f} minutes)")
cap.release()

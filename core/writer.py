# core/writer.py
from pathlib import Path
import time
import cv2
from core.utils import increment_path

def output_writer_process(output_queue, stop_event, project, name, view_img, nosave, exist_ok):
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    if not nosave:
        save_dir.mkdir(parents=True, exist_ok=True)
    vid_writer = None
    vid_path = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    output_fps = 30
    while not stop_event.is_set():
        try:
            data = output_queue.get(timeout=0.5)
        except Exception:
            continue
        if data is None:
            break
        if len(data) >= 6:
            path, im0, s, current_speed, safe_distance, telemetry = data[:6]
        else:
            path, im0, s, current_speed, safe_distance = data[:5]
            telemetry = {}
        p = Path(path)
        save_path = str(save_dir / p.name)
        if view_img:
            cv2.imshow("ADAS Collision Warning System", im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
        if not nosave:
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                cv2.imwrite(save_path, im0)
            else:
                if vid_path != save_path:
                    vid_path = save_path
                    if vid_writer is not None:
                        vid_writer.release()
                    save_path_mp4 = str(Path(save_path).with_suffix('.mp4'))
                    h, w = im0.shape[:2]
                    vid_writer = cv2.VideoWriter(save_path_mp4, fourcc, output_fps, (w, h))
                if vid_writer is not None:
                    vid_writer.write(im0)
    if vid_writer is not None:
        vid_writer.release()
    cv2.destroyAllWindows()

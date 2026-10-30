# core/utils.py
"""
Các hàm tiện ích dùng chung cho hệ thống ADAS (Được tinh gọn hoàn toàn độc lập):
- scale_coords: Chuyển đổi toạ độ BBox về kích thước ảnh gốc
- clip_coords: Cắt toạ độ không vượt quá biên ảnh
- increment_path: Tự tăng tên thư mục chạy (runs/detect/exp*)
- select_device: Tự động phát hiện và chọn thiết bị CPU/CUDA
- letterbox: Resize ảnh giữ tỷ lệ kèm padding
- LoadImages: Lớp nạp ảnh hoặc file video
- LoadStreams: Lớp nạp luồng camera hoặc RTSP đa luồng
"""
import os
import glob
import re
import math
import time
from pathlib import Path
from threading import Thread
import cv2
import numpy as np
import torch

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']

def clip_coords(boxes, shape):
    """Cắt toạ độ bounding box không vượt ra ngoài khung hình (H, W)"""
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'
            if not os.path.exists(p):
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)

    return path

def select_device(device=''):
    device = str(device).strip().lower().replace('cuda:', '').replace('none', '')
    cpu = (device == 'cpu')
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device

    cuda = not cpu and torch.cuda.is_available()
    device_obj = torch.device('cuda:0' if cuda else 'cpu')
    return device_obj

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]  # [H, W]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

class LoadImages:
    def __init__(self, path, img_size=640, stride=32, auto=True):
        p = str(Path(path).resolve())
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))
        elif os.path.isfile(p):
            files = [p]
        else:
            raise FileNotFoundError(f'ERROR: Không tìm thấy đường dẫn {p}')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)
        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        if any(videos):
            self.new_video(videos[0])
        else:
            self.cap = None
        assert self.nf > 0, f'Không tìm thấy file hợp lệ trong {p}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:
                    raise StopIteration
                path = self.files[self.count]
                self.new_video(path)
                ret_val, img0 = self.cap.read()

            self.frame += 1
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '
        else:
            self.count += 1
            img0 = cv2.imread(path)
            assert img0 is not None, f'Lỗi đọc ảnh {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap, s

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf

class LoadStreams:
    """Nạp luồng video liên tục từ Webcam (0), Camera IP, hoặc RTSP Stream"""
    def __init__(self, sources='0', img_size=640, stride=32, auto=True):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.auto = auto

        sources = [sources]
        n = len(sources)
        self.imgs = [None] * n
        self.fps = [0] * n
        self.frames = [0] * n
        self.threads = [None] * n
        self.sources = sources

        for i, s in enumerate(sources):
            src = int(s) if str(s).isnumeric() else s
            cap = cv2.VideoCapture(src)
            assert cap.isOpened(), f'Không thể mở nguồn camera/stream: {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30

            _, self.imgs[i] = cap.read()
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            self.threads[i].start()

    def update(self, i, cap, stream):
        n = 0
        while cap.isOpened():
            n += 1
            cap.grab()
            if n % 4 == 0:
                success, im = cap.retrieve()
                self.imgs[i] = im if success else self.imgs[i] * 0
            time.sleep(0.01)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            raise StopIteration

        img = [letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0] for x in img0]
        img = np.stack(img, 0)
        img = img[..., ::-1].transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None, ''

    def __len__(self):
        return len(self.sources)

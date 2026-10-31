#!/usr/bin/env python3
# ==============================================================================
# TỰ ĐỘNG VÁ TOÀN BỘ CÚ PHÁP ULTRALYTICS CHO PYTHON 3.6 TRÊN JETSON NANO
# ==============================================================================
import os
import re

print("🔧 Đang tự động vá cú pháp Python 3.6 cho Ultralytics...")

# 1. ultralytics/utils/__init__.py
p1 = "ultralytics/utils/__init__.py"
if os.path.exists(p1):
    with open(p1, "r", encoding="utf-8") as f:
        t = f.read()
    t = t.replace('if env_dir := os.getenv("YOLO_CONFIG_DIR"):', 'env_dir = os.getenv("YOLO_CONFIG_DIR")\n    if env_dir:')
    with open(p1, "w", encoding="utf-8") as f:
        f.write(t)

# 2. ultralytics/nn/tasks.py
p2 = "ultralytics/nn/tasks.py"
if os.path.exists(p2):
    with open(p2, "r", encoding="utf-8") as f:
        t = f.read()
    t = t.replace('if task := BaseBackend.read_metadata(model).get("task"):', 'task = BaseBackend.read_metadata(model).get("task")\n        if task:')
    with open(p2, "w", encoding="utf-8") as f:
        f.write(t)

# 3. ultralytics/engine/predictor.py
p3 = "ultralytics/engine/predictor.py"
if os.path.exists(p3):
    with open(p3, "r", encoding="utf-8") as f:
        t = f.read()
    t = t.replace('if seen := self.seen:', 'seen = self.seen\n            if seen:')
    t = re.sub(
        r'if not getattr\(self\.model, "base_model", True\) and \(\s*unsupported := \[k for k in \("augment", "embed", "visualize"\) if getattr\(self\.args, k\)\]\s*\):',
        'unsupported = [k for k in ("augment", "embed", "visualize") if getattr(self.args, k)]\n        if not getattr(self.model, "base_model", True) and unsupported:',
        t
    )
    with open(p3, "w", encoding="utf-8") as f:
        f.write(t)

# 4. ultralytics/engine/model.py
p4 = "ultralytics/engine/model.py"
if os.path.exists(p4):
    with open(p4, "r", encoding="utf-8") as f:
        t = f.read()
    t = t.replace(
        'or self.predictor.args.quantize != QUANTIZE_ALIASES.get(str(q := args.get("quantize")).lower(), q)',
        'or self.predictor.args.quantize != QUANTIZE_ALIASES.get(str(args.get("quantize", "")).lower(), args.get("quantize", ""))'
    )
    with open(p4, "w", encoding="utf-8") as f:
        f.write(t)

# 5. ultralytics/trackers/track.py
p5 = "ultralytics/trackers/track.py"
if os.path.exists(p5):
    with open(p5, "r", encoding="utf-8") as f:
        t = f.read()
    t = t.replace(
        'if (task := predictor.args.task) in TASKS and task not in trackable:',
        'task = predictor.args.task\n    if task in TASKS and task not in trackable:'
    )
    t = t.replace(
        'det = (src := result.obb if is_obb else result.boxes).cpu().numpy()',
        'src = result.obb if is_obb else result.boxes\n        det = src.cpu().numpy()'
    )
    with open(p5, "w", encoding="utf-8") as f:
        f.write(t)

# 6. ultralytics/utils/checks.py
p6 = "ultralytics/utils/checks.py"
if os.path.exists(p6):
    with open(p6, "r", encoding="utf-8") as f:
        t = f.read()
    t = t.replace(
        'if match := re.match(r"([a-zA-Z0-9-_]+)\\s*([<>!=~]+.*)?", line):',
        'match = re.match(r"([a-zA-Z0-9-_]+)\\s*([<>!=~]+.*)?", line)\n            if match:'
    )
    t = t.replace(
        'if s := clean_url(f).rpartition(".")[-1].lower().strip():',
        's = clean_url(f).rpartition(".")[-1].lower().strip()\n            if s:'
    )
    with open(p6, "w", encoding="utf-8") as f:
        f.write(t)

# 7. Quét toàn bộ và thay thế các dạng `if var := expr:` còn lại
count = 0
for root, _, files in os.walk("ultralytics"):
    for file in files:
        if file.endswith(".py"):
            fpath = os.path.join(root, file)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            new_content = re.sub(
                r'([ \t]+)if ([a-zA-Z_0-9]+) := ([^:\n]+):',
                r'\1\2 = \3\n\1if \2:',
                content
            )
            if new_content != content:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(new_content)
                count += 1

print("✅ Đã hoàn tất vá lỗi cú pháp Python 3.6!")
print("🚀 Bạn có thể chạy ngay: python3 run.py")

# core/__init__.py
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .reader import video_reader_process
from .detector import detector_tracker_process
from .decision import feature_mlp_rule_process
from .writer import output_writer_process

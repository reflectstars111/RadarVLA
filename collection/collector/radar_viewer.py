"""Compatibility import; the CARLA K-Radar viewer lives in C-Shenron."""
from pathlib import Path
import sys

AAA_TOOLS = Path(__file__).resolve().parents[3] / "Carla/AAA_tools"
if str(AAA_TOOLS) not in sys.path:
    sys.path.insert(0, str(AAA_TOOLS))

from kradar_carla_viewer import (  # noqa: E402,F401
    build_explorer, load_frame, render_frame, save_examples,
)

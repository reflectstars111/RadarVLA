"""Compatibility import; CARLA scatterer mapping lives in C-Shenron."""
from pathlib import Path
import sys

TEAM_CODE = Path(__file__).resolve().parents[3] / "Carla/C-Shenron/team_code"
if str(TEAM_CODE) not in sys.path:
    sys.path.insert(0, str(TEAM_CODE))

from e2e_agent_sem_lidar2shenron_package.carla_radar_source import (  # noqa: E402,F401
    DYNAMIC_TAGS, carla_rotation_matrix, prepare_scatterers,
)

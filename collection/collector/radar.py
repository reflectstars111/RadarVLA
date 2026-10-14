"""Compatibility entrypoint; CARLA radar generation lives in C-Shenron."""
from pathlib import Path
import sys

TEAM_CODE = Path(__file__).resolve().parents[3] / "Carla/C-Shenron/team_code"
if str(TEAM_CODE) not in sys.path:
    sys.path.insert(0, str(TEAM_CODE))

from e2e_agent_sem_lidar2shenron_package.carla_kradar_episode import (  # noqa: E402,F401
    main, model_axes, preflight, process, select_scatterers, validate_products,
)


if __name__ == "__main__":
    main()

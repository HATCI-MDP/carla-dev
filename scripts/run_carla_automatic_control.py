#!/usr/bin/env python
"""
Launch CARLA's automatic_control.py: drive with an autopilot agent, camera view + HUD.

Usage:
    python scripts/run_carla_automatic_control.py [--host 127.0.0.1] [--port 2000]

Override CARLA install path with CARLA_ROOT env var, e.g.:
    set CARLA_ROOT=C:\CARLA_0.9.16
"""

import os
import subprocess
import sys

CARLA_ROOT = os.environ.get("CARLA_ROOT", r"C:\CARLA_0.9.16")
AUTOMATIC_CONTROL = os.path.join(CARLA_ROOT, "PythonAPI", "examples", "automatic_control.py")


def main():
    if not os.path.isfile(AUTOMATIC_CONTROL):
        print("CARLA automatic_control.py not found: %s" % AUTOMATIC_CONTROL)
        print("Set CARLA_ROOT to your CARLA install directory.")
        return 1
    rc = subprocess.call([sys.executable, AUTOMATIC_CONTROL] + sys.argv[1:])
    return rc


if __name__ == "__main__":
    sys.exit(main())

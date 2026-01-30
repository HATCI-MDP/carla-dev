#!/usr/bin/env python
"""
Launch CARLA's generate_traffic.py: spawn NPC vehicles and pedestrians in the world.

Usage:
    python scripts/run_carla_generate_traffic.py [--host 127.0.0.1] [--port 2000]

Override CARLA install path with CARLA_ROOT env var, e.g.:
    set CARLA_ROOT=C:\CARLA_0.9.16
"""

import os
import subprocess
import sys

CARLA_ROOT = os.environ.get("CARLA_ROOT", r"C:\CARLA_0.9.16")
GENERATE_TRAFFIC = os.path.join(CARLA_ROOT, "PythonAPI", "examples", "generate_traffic.py")


def main():
    if not os.path.isfile(GENERATE_TRAFFIC):
        print("CARLA generate_traffic.py not found: %s" % GENERATE_TRAFFIC)
        print("Set CARLA_ROOT to your CARLA install directory.")
        return 1
    rc = subprocess.call([sys.executable, GENERATE_TRAFFIC] + sys.argv[1:])
    return rc


if __name__ == "__main__":
    sys.exit(main())

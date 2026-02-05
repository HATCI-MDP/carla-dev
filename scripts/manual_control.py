#!/usr/bin/env python
"""
Launch CARLA's official manual control: full Pygame HUD, keybinds, camera angles,
reverse, hand-brake, autopilot toggle, weather, etc. Runs in async mode by default
(no --sync), so it stays responsive.

Usage:
    python scripts/manual_control.py [--host 127.0.0.1] [--port 2000]

Defaults to sedan (vehicle.audi.a2). Override with --filter, e.g.:
    python scripts/manual_control.py --filter vehicle.tesla.model3

Override CARLA install path with CARLA_ROOT env var, e.g.:
    set CARLA_ROOT=C:\CARLA_0.9.16
"""

import os
import subprocess
import sys

CARLA_ROOT = os.environ.get("CARLA_ROOT", r"C:\CARLA_0.9.16")
MANUAL_CONTROL = os.path.join(CARLA_ROOT, "PythonAPI", "examples", "manual_control.py")


def main():
    if not os.path.isfile(MANUAL_CONTROL):
        print("CARLA manual_control.py not found: %s" % MANUAL_CONTROL)
        print("Set CARLA_ROOT to your CARLA install directory.")
        return 1
    args = list(sys.argv[1:])
    if not any(a == "--filter" for a in args):
        args.extend(["--filter", "vehicle.audi.a2"])  # sedan by default
    rc = subprocess.call([sys.executable, MANUAL_CONTROL] + args)
    return rc


if __name__ == "__main__":
    sys.exit(main())

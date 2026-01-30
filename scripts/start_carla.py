#!/usr/bin/env python
"""
Start the CARLA simulator (CarlaUE4.exe).
Blocks until CARLA is closed.

Usage:
    python scripts/start_carla.py

Override the install path with the CARLA_ROOT environment variable, e.g.:
    set CARLA_ROOT=D:\OtherCarla
    python scripts/start_carla.py
"""

import os
import subprocess
import sys

# Default path; override with CARLA_ROOT env var
CARLA_ROOT = os.environ.get("CARLA_ROOT", r"C:\CARLA_0.9.16")
CARLA_EXE = os.path.join(CARLA_ROOT, "CarlaUE4.exe")


def main():
    if not os.path.isfile(CARLA_EXE):
        print("CARLA executable not found: %s" % CARLA_EXE)
        print("Set CARLA_ROOT to your CARLA install directory, e.g.: set CARLA_ROOT=C:\\CARLA_0.9.16")
        return 1

    print("Starting CARLA: %s" % CARLA_EXE)
    print("Close the CARLA window to exit this script.")
    try:
        p = subprocess.Popen(
            [CARLA_EXE],
            cwd=CARLA_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        p.wait()
    except KeyboardInterrupt:
        p.terminate()
        p.wait()
    except FileNotFoundError:
        print("Could not run %s" % CARLA_EXE)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

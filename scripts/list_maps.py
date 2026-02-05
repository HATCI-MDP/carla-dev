#!/usr/bin/env python
"""
Print the list of maps available on the running CARLA server.
Run CARLA first (e.g. python scripts/start_carla.py), then run this in another terminal.

Usage:
    python scripts/list_maps.py [--host 127.0.0.1] [--port 2000]
"""

import argparse
import sys

try:
    import carla
except ImportError:
    print(
        "CARLA Python package not found. Install the wheel from your CARLA install:\n"
        '  pip install "C:\\CARLA_0.9.16\\PythonAPI\\carla\\dist\\carla-0.9.16-cp312-cp312-win_amd64.whl"'
    )
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="List maps available on the CARLA server.")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    args = parser.parse_args()

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)
        client.get_server_version()
    except RuntimeError as e:
        print("Failed to connect: %s" % e)
        print("Start CARLA first (e.g. python scripts/start_carla.py).")
        return 1

    maps = client.get_available_maps()
    # Paths are like /Game/Carla/Maps/Town01; we want the short name
    names = sorted(set(m.replace("/Game/Carla/Maps/", "").strip("/") for m in maps))
    print("Available maps on %s:%d:" % (args.host, args.port))
    for n in names:
        print("  ", n)
    if not names:
        print("  (none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

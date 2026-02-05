#!/usr/bin/env python
"""
Load a CARLA map and exit. Use this to switch maps before running other scripts.
CARLA must already be running.

Usage:
    python scripts/load_map.py --map Town07
    python scripts/load_map.py --map Town03_Opt --host 127.0.0.1 --port 2000
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
    parser = argparse.ArgumentParser(description="Load a map in CARLA (server must be running).")
    parser.add_argument("--map", "-m", required=True, metavar="NAME", help="Map to load (e.g. Town01, Town07, Town03_Opt)")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    args = parser.parse_args()

    print("Connecting to CARLA at %s:%d ..." % (args.host, args.port))
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        client.get_server_version()
    except RuntimeError as e:
        print("Failed to connect: %s" % e)
        print("Start CARLA first (e.g. python scripts/start_carla.py).")
        return 1

    print("Loading map: %s (may take 30–60 s on first load) ..." % args.map)
    client.set_timeout(90.0)
    try:
        client.load_world(args.map)
    except RuntimeError as e:
        if "not found" in str(e).lower():
            print("Map '%s' is not available." % args.map)
            print("Run: python scripts/list_maps.py  (with CARLA running) to see available maps.")
        raise
    world = client.get_world()
    print("Map loaded: %s" % world.get_map().name)
    return 0


if __name__ == "__main__":
    sys.exit(main())

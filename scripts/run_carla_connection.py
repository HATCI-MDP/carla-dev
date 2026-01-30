#!/usr/bin/env python
"""
Connect to CARLA and verify the simulator is running.
Run this after starting CARLA (CarlaUE4.exe) with a map loaded.

Usage:
    python scripts/run_carla_connection.py [--host 127.0.0.1] [--port 2000] [--demo]
"""

import argparse
import math
import random
import sys
import time

try:
    import carla
except ImportError:
    print(
        "CARLA Python package not found. Install the wheel from your CARLA install:\n"
        '  pip install "C:\\CARLA_0.9.16\\PythonAPI\\carla\\dist\\carla-0.9.16-cp312-cp312-win_amd64.whl"'
    )
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Connect to CARLA and optionally run a short demo.")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port (default: 2000)")
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Connection timeout in seconds (default: 10)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Spawn a vehicle with a camera for a few seconds, then cleanup",
    )
    args = parser.parse_args()

    print("Connecting to CARLA at %s:%d ..." % (args.host, args.port))
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        version = client.get_server_version()
        print("Connected. CARLA server version: %s" % version)
    except RuntimeError as e:
        print("Failed to connect: %s" % e)
        print("Make sure CARLA is running (CarlaUE4.exe) and a map is loaded.")
        return 1

    world = client.get_world()
    world_map = world.get_map()
    print("Current map: %s" % world_map.name)
    spawn_points = world_map.get_spawn_points()
    print("Spawn points available: %d" % len(spawn_points))

    if not args.demo:
        print("Connection OK. Use --demo to spawn a vehicle and camera.")
        return 0

    # Short demo: spawn vehicle + camera, move spectator to follow, run a few seconds, cleanup
    def spectator_follow(spectator, vehicle, distance=8.0, height=3.0):
        """Place spectator behind and above the vehicle so you see it in the CARLA window."""
        loc = vehicle.get_location()
        rot = vehicle.get_transform().rotation
        yaw_rad = math.radians(rot.yaw)
        x = loc.x - distance * math.cos(yaw_rad)
        y = loc.y - distance * math.sin(yaw_rad)
        z = loc.z + height
        spectator.set_transform(
            carla.Transform(
                carla.Location(x=x, y=y, z=z),
                carla.Rotation(yaw=rot.yaw, pitch=-20),
            )
        )

    actor_list = []
    try:
        blueprint_library = world.get_blueprint_library()
        vehicles = blueprint_library.filter("vehicle")
        if not vehicles:
            print("No vehicle blueprints found; skipping demo.")
            return 0
        vehicle_bp = random.choice(vehicles)
        if not spawn_points:
            print("No spawn points on this map; skipping demo.")
            return 0
        transform = random.choice(spawn_points)
        vehicle = world.try_spawn_actor(vehicle_bp, transform)
        if vehicle is None:
            print("Spawn failed (spot may be blocked). Try again or pick another map.")
            return 1
        actor_list.append(vehicle)
        print("Spawned vehicle: %s" % vehicle.type_id)
        vehicle.set_autopilot(True)

        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        print("Spawned camera. Moving spectator to follow vehicle...")

        spectator = world.get_spectator()
        camera.listen(lambda img: None)  # discard frames for this test

        print("Running for 8 seconds (watch the CARLA window)...")
        deadline = time.time() + 8.0
        while time.time() < deadline:
            spectator_follow(spectator, vehicle)
            time.sleep(0.05)  # ~20 Hz update so view tracks smoothly
    except Exception as e:
        print("Demo error: %s" % e)
        raise
    finally:
        print("Cleaning up actors...")
        for actor in actor_list:
            try:
                actor.destroy()
            except Exception:
                pass
        print("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

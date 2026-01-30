#!/usr/bin/env python
"""
Spawn a vehicle in CARLA and show its RGB camera feed in a separate OpenCV window.
Car drives on autopilot (view only). This is the view your perception stack will use.

Usage:
    python scripts/run_autopilot_camera.py [--host 127.0.0.1] [--port 2000]
    Press 'q' or ESC in the camera window to exit.
"""

import argparse
import random
import sys

try:
    import carla
except ImportError:
    print(
        "CARLA Python package not found. Install the wheel from your CARLA install:\n"
        '  pip install "C:\\CARLA_0.9.16\\PythonAPI\\carla\\dist\\carla-0.9.16-cp312-cp312-win_amd64.whl"'
    )
    sys.exit(1)

try:
    import cv2
    import numpy as np
except ImportError:
    print("OpenCV and NumPy required: pip install opencv-python numpy")
    sys.exit(1)


def carla_image_to_bgr(carla_image):
    """Convert a carla.Image (BGRA) to a NumPy BGR array for OpenCV."""
    array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
    array = array.reshape((carla_image.height, carla_image.width, 4))
    return array[:, :, :3].copy()  # drop alpha, copy so callback buffer can be reused


def main():
    parser = argparse.ArgumentParser(
        description="Show ego vehicle RGB camera in an OpenCV window (autopilot)."
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument(
        "--width", type=int, default=1280, help="Camera image width (default: 1280)"
    )
    parser.add_argument(
        "--height", type=int, default=720, help="Camera image height (default: 720)"
    )
    args = parser.parse_args()

    print("Connecting to CARLA at %s:%d ..." % (args.host, args.port))
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        client.get_server_version()
    except RuntimeError as e:
        print("Failed to connect: %s" % e)
        print("Make sure CARLA is running (CarlaUE4.exe) and a map is loaded.")
        return 1

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("No spawn points on this map.")
        return 1

    vehicles = blueprint_library.filter("vehicle")
    if not vehicles:
        print("No vehicle blueprints found.")
        return 1

    vehicle_bp = random.choice(vehicles)
    transform = random.choice(spawn_points)
    vehicle = world.try_spawn_actor(vehicle_bp, transform)
    if vehicle is None:
        print("Spawn failed (spot may be blocked). Try again or pick another map.")
        return 1

    # RGB camera in front of the hood so you see the road ahead (not interior)
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(args.width))
    camera_bp.set_attribute("image_size_y", str(args.height))
    camera_transform = carla.Transform(carla.Location(x=2.8, z=1.2))  # hood-level, road view
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    vehicle.set_autopilot(True)  # car drives itself; this script is view-only (no keyboard control)

    # Latest frame from camera (callback runs on CARLA thread)
    latest_frame = [None]

    def on_image(carla_image):
        latest_frame[0] = carla_image_to_bgr(carla_image)

    camera.listen(on_image)

    window_name = "Ego camera (q or ESC to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)
    print("Camera window open. Car drives on autopilot (view only). Press 'q' or ESC to exit.")

    try:
        while True:
            img = latest_frame[0]
            if img is not None:
                cv2.imshow(window_name, img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q or ESC
                break
    finally:
        cv2.destroyAllWindows()
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        print("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

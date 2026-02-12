#!/usr/bin/env python
"""
Manual drive + DeepLabV3+ (ORFD) freespace segmentation: you control the car
with WASD and see binary drivable/non-drivable segmentation live.
Green = predicted drivable area.

Requires: CARLA running, pip install segmentation-models-pytorch opencv-python torchvision pynput

Usage:
    python scripts/manual_control_deeplabv3_orfd.py [--host 127.0.0.1] [--port 2000] [--map Town02]
    python scripts/manual_control_deeplabv3_orfd.py --model training/models/deeplabv3_orfd/best_model.pth

Controls:
    W = throttle, S = brake, A = left, D = right
    SPACE = hand brake, Q = toggle reverse
    Press 'q' or ESC in the OpenCV window to exit.
"""

import argparse
import math
import random
import sys
import threading
import time

try:
    import carla
except ImportError:
    print("CARLA Python package not found. Install the wheel from your CARLA install.")
    sys.exit(1)

try:
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    import segmentation_models_pytorch as smp
except ImportError as e:
    print("Missing dependency: %s" % e)
    print("Install: pip install segmentation-models-pytorch opencv-python torchvision")
    sys.exit(1)

try:
    from pynput import keyboard
except ImportError:
    print("pynput required for key state (WASD). Install: pip install pynput")
    sys.exit(1)


def carla_image_to_bgr(carla_image):
    array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
    array = array.reshape((carla_image.height, carla_image.width, 4))
    return array[:, :, :3].copy()


def build_legend(height):
    legend_width = 180
    legend = np.ones((height, legend_width, 3), dtype=np.uint8) * 255
    cv2.putText(legend, "ORFD Freespace", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    # Drivable
    cv2.rectangle(legend, (8, 36), (28, 52), (0, 200, 0), -1)
    cv2.rectangle(legend, (8, 36), (28, 52), (80, 80, 80), 1)
    cv2.putText(legend, "Drivable", (34, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    # Non-drivable
    cv2.rectangle(legend, (8, 60), (28, 76), (60, 60, 60), -1)
    cv2.rectangle(legend, (8, 60), (28, 76), (80, 80, 80), 1)
    cv2.putText(legend, "Non-drivable", (34, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return legend


def main():
    parser = argparse.ArgumentParser(
        description="Manual drive (WASD) + DeepLabV3+ (ORFD freespace) segmentation."
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--map", metavar="NAME", default=None, help="Load this map")
    parser.add_argument(
        "--model",
        default="./training/models/deeplabv3_orfd/best_model.pth",
        help="Path to DeepLabV3+ ORFD checkpoint (.pth)",
    )
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--scale", type=float, default=1.5, help="Display scale factor")
    parser.add_argument("--infer-every", type=int, default=1, metavar="N", help="Run model every Nth frame")
    parser.add_argument("--no-thread", action="store_true", help="Run inference in main loop")
    parser.add_argument("--max-inference-fps", action="store_true", help="Run inference as fast as possible")
    args = parser.parse_args()

    # Global key state (pynput listener updates this)
    pressed = set()
    special_pressed = set()
    reverse_on = [False]

    def on_press(key):
        try:
            c = key.char.lower()
            if c == "q":
                reverse_on[0] = not reverse_on[0]
            else:
                pressed.add(c)
        except AttributeError:
            special_pressed.add(key)

    def on_release(key):
        try:
            pressed.discard(key.char.lower())
        except AttributeError:
            special_pressed.discard(key)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Connect to CARLA first
    print("Connecting to CARLA at %s:%d ..." % (args.host, args.port))
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(30.0)
        client.get_server_version()
    except RuntimeError as e:
        print("Failed to connect: %s" % e)
        print("Make sure CARLA is running (CarlaUE4.exe) and a map is loaded.")
        return 1

    if args.map:
        print("Loading map: %s ..." % args.map)
        client.set_timeout(120.0)
        try:
            client.load_world(args.map)
        except RuntimeError as e:
            if "not found" in str(e).lower():
                print("Map '%s' not available." % args.map)
            raise

    client.set_timeout(30.0)
    world = client.get_world()
    time.sleep(2.0)

    # Load DeepLabV3+ model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading DeepLabV3+ ORFD from %s (device: %s) ..." % (args.model, device))
    model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print("Model loaded (epoch %d, IoU %.4f)" % (ckpt.get("epoch", -1), ckpt.get("iou", 0.0)))

    # Preprocessing: standard ImageNet normalization
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Spawn vehicle
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("No spawn points on this map.")
        return 1
    vehicles = blueprint_library.filter("vehicle")
    if not vehicles:
        print("No vehicle blueprints found.")
        return 1
    vehicle_bp = blueprint_library.find("vehicle.dodge.charger_2020")
    if vehicle_bp is None:
        vehicle_bp = random.choice(vehicles)
    vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    if vehicle is None:
        print("Spawn failed. Try again or another map.")
        return 1

    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(args.width))
    camera_bp.set_attribute("image_size_y", str(args.height))
    camera_bp.set_attribute("fov", "90")
    camera = world.spawn_actor(
        camera_bp,
        carla.Transform(carla.Location(x=2.8, z=1.2)),
        attach_to=vehicle,
    )

    latest_frame = [None]

    def on_image(carla_image):
        latest_frame[0] = carla_image_to_bgr(carla_image)

    camera.listen(on_image)

    window_name = "Manual + DeepLabV3+ ORFD"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    display_w = int((args.width * 2 + 180) * args.scale)
    display_h = int(args.height * args.scale)
    cv2.resizeWindow(window_name, display_w, display_h)
    print("W=throttle, S=brake, A=left, D=right, SPACE=handbrake, Q=toggle reverse.")
    print("Press 'q' or ESC in the OpenCV window to exit.")
    if not args.no_thread:
        print("Inference runs in background thread.")

    last_seg_bgr = [None]
    last_fps_timestamp = [None]
    fps_smooth = [0.0]
    stop_inference = threading.Event()

    def inference_worker():
        infer_interval = 0.0 if args.max_inference_fps else (args.infer_every / 30.0)
        while not stop_inference.is_set():
            stop_inference.wait(timeout=infer_interval)
            if stop_inference.is_set():
                break
            bgr = latest_frame[0]
            if bgr is None:
                continue
            bgr = bgr.copy()
            h, w = bgr.shape[:2]
            try:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                input_tensor = preprocess(pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(input_tensor)
                mask = torch.sigmoid(output).squeeze().cpu().numpy()
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                binary = (mask > 0.5).astype(np.uint8)
                seg_display = np.zeros((h, w, 3), dtype=np.uint8)
                seg_display[binary == 1] = (0, 200, 0)   # BGR green = drivable
                seg_display[binary == 0] = (60, 60, 60)  # dark gray = non-drivable
                last_seg_bgr[0] = seg_display
                t_now = time.perf_counter()
                if last_fps_timestamp[0] is not None:
                    dt = t_now - last_fps_timestamp[0]
                    if dt > 0:
                        fps_smooth[0] = 0.85 * fps_smooth[0] + 0.15 * (1.0 / dt)
                last_fps_timestamp[0] = t_now
            except Exception:
                pass

    use_thread = not args.no_thread
    if use_thread:
        worker = threading.Thread(target=inference_worker, daemon=True)
        worker.start()

    frame_count = 0
    t_last_inference = None
    legend = build_legend(args.height)
    try:
        while True:
            # Apply manual control
            throttle = 0.6 if "w" in pressed else 0.0
            brake = 0.5 if "s" in pressed else 0.0
            steer = 0.0
            if "a" in pressed:
                steer = -0.6
            if "d" in pressed:
                steer = 0.6
            hand_brake = keyboard.Key.space in special_pressed
            ctrl = carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake,
                hand_brake=hand_brake,
                reverse=reverse_on[0],
            )
            vehicle.apply_control(ctrl)

            bgr = latest_frame[0]
            if bgr is None:
                key = cv2.waitKey(100) & 0xFF
                if key == ord("q") or key == 27:
                    break
                continue

            h, w = bgr.shape[:2]
            if use_thread:
                seg_bgr = last_seg_bgr[0]
                disp_fps = fps_smooth[0]
            else:
                run_inference = (frame_count % args.infer_every == 0)
                if run_inference:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)
                    input_tensor = preprocess(pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = model(input_tensor)
                    mask = torch.sigmoid(output).squeeze().cpu().numpy()
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                    binary = (mask > 0.5).astype(np.uint8)
                    seg_display = np.zeros((h, w, 3), dtype=np.uint8)
                    seg_display[binary == 1] = (0, 200, 0)
                    seg_display[binary == 0] = (60, 60, 60)
                    last_seg_bgr[0] = seg_display
                    seg_bgr = seg_display
                    t_now = time.perf_counter()
                    if t_last_inference is not None:
                        dt = t_now - t_last_inference
                        if dt > 0:
                            fps_smooth[0] = 0.85 * fps_smooth[0] + 0.15 * (1.0 / dt)
                    t_last_inference = t_now
                else:
                    seg_bgr = last_seg_bgr[0]
                disp_fps = fps_smooth[0]

            if seg_bgr is not None:
                combined = np.hstack([bgr, seg_bgr, legend])
                if args.scale != 1.0:
                    new_w = int(combined.shape[1] * args.scale)
                    new_h = int(combined.shape[0] * args.scale)
                    combined = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                v = vehicle.get_velocity()
                speed_kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
                rev_str = " [R]" if reverse_on[0] else ""
                cv2.putText(combined, "Inference FPS: %.1f" % disp_fps, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(combined, "Inference FPS: %.1f" % disp_fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.putText(combined, "Speed: %.0f km/h%s" % (speed_kmh, rev_str), (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(combined, "Speed: %.0f km/h%s" % (speed_kmh, rev_str), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.imshow(window_name, combined)
            else:
                cv2.imshow(window_name, bgr)

            frame_count += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        stop_inference.set()
        listener.stop()
        cv2.destroyAllWindows()
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        print("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

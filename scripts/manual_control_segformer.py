#!/usr/bin/env python
"""
Manual drive + SegFormer (full): you control the car with WASD and see segmentation live.
Full defaults: 640×480, inference every frame. Use on a PC with GPU for smooth FPS.

Requires: CARLA running, pip install -r requirements-segmentation.txt (includes pynput for key state).

Usage:
    python scripts/manual_control_segformer.py [--host 127.0.0.1] [--port 2000] [--map Town02]
    Controls: W=throttle, S=brake, A=left, D=right. Press 'q' or ESC in the window to exit.
"""

import argparse
import math
import random
import sys
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
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
except ImportError as e:
    print("Missing dependency: %s" % e)
    print("Install: pip install -r requirements-segmentation.txt  and  pip install opencv-python")
    sys.exit(1)

try:
    from pynput import keyboard
except ImportError:
    print("pynput required for key state (WASD). Install: pip install pynput")
    sys.exit(1)


def _ade20k_palette():
    palette = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        palette[i] = [(37 * i) % 256, (97 * i + 31) % 256, (157 * i + 67) % 256]
    return palette


def carla_image_to_bgr(carla_image):
    array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
    array = array.reshape((carla_image.height, carla_image.width, 4))
    return array[:, :, :3].copy()


# Driving/outdoor-relevant ADE20K classes only (no ceiling, bed, cabinet, windowpane, etc.)
LEGEND_CLASSES = frozenset([
    "road", "sidewalk", "building", "sky", "tree", "grass", "earth", "path",
    "car", "vehicle", "person", "pole", "fence", "wall", "plant", "bus", "truck",
    "bicycle", "motorcycle", "traffic light", "traffic sign", "bridge", "water",
    "rock", "stone", "sand", "ground", "terrain", "vegetation", "house",
    "mountain", "sea", "field", "runway", "river", "tower", "skyscraper",
    "floor", "pavement", "dirt", "mud", "snow", "lane", "dirt track",
])

def build_legend(id2label, palette, height, num_rows=20):
    filtered = {i: n for i, n in id2label.items() if n.lower().strip() in LEGEND_CLASSES}
    id2label = filtered if filtered else id2label
    legend_width = 220
    row_h = max(18, height // num_rows)
    actual_rows = min(num_rows, len(id2label)) if id2label else 0
    if actual_rows == 0:
        legend = np.ones((height, legend_width, 3), dtype=np.uint8) * 240
        cv2.putText(legend, "No labels", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return legend
    legend = np.ones((height, legend_width, 3), dtype=np.uint8) * 255
    cv2.putText(legend, "Class (id)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    items = sorted(id2label.items())[:actual_rows]
    for i, (idx, name) in enumerate(items):
        y = 36 + i * row_h
        if y + row_h > height:
            break
        color = palette[idx % 256]
        color_bgr = (int(color[2]), int(color[1]), int(color[0]))
        cv2.rectangle(legend, (8, y - 14), (8 + 20, y + 2), color_bgr, -1)
        cv2.rectangle(legend, (8, y - 14), (8 + 20, y + 2), (80, 80, 80), 1)
        label = (name[:18] + "..") if len(name) > 18 else name
        cv2.putText(legend, label, (34, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return legend


def main():
    parser = argparse.ArgumentParser(
        description="Manual drive (WASD) + SegFormer segmentation; full res, every frame (for GPU)."
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--map", metavar="NAME", default=None, help="Load this map (e.g. Town01, Town02).")
    parser.add_argument(
        "--model",
        default="nvidia/segformer-b0-finetuned-ade-512-512",
        help="Hugging Face model id",
    )
    parser.add_argument("--width", type=int, default=640, help="Camera width (default 640)")
    parser.add_argument("--height", type=int, default=480, help="Camera height (default 480)")
    parser.add_argument(
        "--infer-every",
        type=int,
        default=1,
        metavar="N",
        help="Run model every Nth frame (default 1 = every frame)",
    )
    parser.add_argument("--scale", type=float, default=1.5, help="Display scale factor (default 1.5)")
    args = parser.parse_args()

    # Global key state (pynput listener updates this)
    pressed = set()

    def on_press(key):
        try:
            pressed.add(key.char.lower())
        except AttributeError:
            pass

    def on_release(key):
        try:
            pressed.discard(key.char.lower())
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading SegFormer model %s (device: %s) ..." % (args.model, device))
    processor = SegformerImageProcessor.from_pretrained(args.model)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model)
    model.to(device)
    model.eval()
    palette = _ade20k_palette()
    id2label = getattr(model.config, "id2label", None) or {}
    id2label = {int(k): str(v) for k, v in id2label.items()}

    print("Connecting to CARLA at %s:%d ..." % (args.host, args.port))
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        client.get_server_version()
    except RuntimeError as e:
        print("Failed to connect: %s" % e)
        return 1

    if args.map:
        print("Loading map: %s ..." % args.map)
        client.set_timeout(90.0)
        try:
            client.load_world(args.map)
        except RuntimeError as e:
            if "not found" in str(e).lower():
                print("Map '%s' not available. Run: python scripts/list_maps.py" % args.map)
            raise
        finally:
            client.set_timeout(10.0)

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
    vehicle_bp = blueprint_library.find("vehicle.audi.tt") or random.choice(vehicles)
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
    # No autopilot — we drive with WASD

    latest_frame = [None]
    def on_image(carla_image):
        latest_frame[0] = carla_image_to_bgr(carla_image)
    camera.listen(on_image)

    window_name = "Manual + SegFormer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    display_w = int((args.width * 2 + 220) * args.scale)
    display_h = int(args.height * args.scale)
    cv2.resizeWindow(window_name, display_w, display_h)
    print("W=throttle, S=brake, A=left, D=right. Press 'q' or ESC in window to exit.")

    frame_count = 0
    last_seg_bgr = None
    t_last_inference = None
    fps_smooth = 0.0
    try:
        while True:
            # Apply manual control from key state
            throttle = 0.6 if "w" in pressed else 0.0
            brake = 0.5 if "s" in pressed else 0.0
            steer = 0.0
            if "a" in pressed:
                steer = -0.6
            if "d" in pressed:
                steer = 0.6
            ctrl = carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake,
                hand_brake=False,
                reverse=False,
            )
            vehicle.apply_control(ctrl)

            bgr = latest_frame[0]
            if bgr is None:
                key = cv2.waitKey(100) & 0xFF
                if key == ord("q") or key == 27:
                    break
                continue

            h, w = bgr.shape[:2]
            run_inference = (frame_count % args.infer_every == 0)

            if run_inference:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                inputs = processor(images=pil, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                logits = outputs.logits
                logits = torch.nn.functional.interpolate(
                    logits, size=(h, w), mode="bilinear", align_corners=False
                )
                seg = logits.argmax(dim=1).squeeze(0).cpu().numpy()
                seg_display = palette[seg % 256]
                last_seg_bgr = cv2.cvtColor(seg_display, cv2.COLOR_RGB2BGR)
                t_now = time.perf_counter()
                if t_last_inference is not None:
                    dt = t_now - t_last_inference
                    if dt > 0:
                        fps_smooth = 0.85 * fps_smooth + 0.15 * (1.0 / dt)
                t_last_inference = t_now

            if last_seg_bgr is not None:
                legend = build_legend(id2label, palette, h, num_rows=20)
                combined = np.hstack([bgr, last_seg_bgr, legend])
                if args.scale != 1.0:
                    new_w = int(combined.shape[1] * args.scale)
                    new_h = int(combined.shape[0] * args.scale)
                    combined = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                v = vehicle.get_velocity()
                speed_kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
                cv2.putText(combined, "Inference FPS: %.1f" % fps_smooth, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(combined, "Inference FPS: %.1f" % fps_smooth, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.putText(combined, "Speed: %.0f km/h" % speed_kmh, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(combined, "Speed: %.0f km/h" % speed_kmh, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.imshow(window_name, combined)

            frame_count += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        listener.stop()
        cv2.destroyAllWindows()
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

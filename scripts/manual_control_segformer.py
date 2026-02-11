#!/usr/bin/env python
"""
Manual drive + SegFormer segmentation: you control the car with WASD and see
segmentation live. Works with both pretrained ADE20K models and fine-tuned
off-road models (RELLIS-3D, RUGD, etc.).

For fine-tuned models (<=30 classes), uses hand-picked off-road colors and shows
all classes in the legend. For ADE20K (150 classes), uses a generic palette and
filters the legend to driving-relevant classes only.

Requires: CARLA running, pip install -r requirements-segmentation.txt

Usage:
    python scripts/manual_control_segformer.py [--host 127.0.0.1] [--port 2000] [--map Town02]
    python scripts/manual_control_segformer.py --model training/models/rellis3d_segformer_b0

Controls:
    W = throttle, S = brake, A = left, D = right
    SPACE = hand brake, Q = toggle reverse
    Press 'q' or ESC in the OpenCV window to exit.

Performance options (for lower-spec machines):
    --width 320 --height 240       Lower camera resolution
    --infer-every 5                Skip frames (reuse last segmentation)
    --no-thread                    Run inference in main loop instead of background thread
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
    print(
        "CARLA Python package not found. Install the wheel from your CARLA install."
    )
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


# Hand-picked visually distinct colors for off-road classes (BGR order for OpenCV).
OFFROAD_COLORS_BGR = {
    "dirt":       (43, 90, 139),
    "grass":      (0, 160, 0),
    "tree":       (34, 120, 34),
    "pole":       (50, 50, 255),
    "water":      (220, 100, 30),
    "sky":        (235, 206, 135),
    "vehicle":    (200, 0, 200),
    "object":     (70, 130, 180),
    "asphalt":    (80, 80, 80),
    "building":   (150, 150, 150),
    "log":        (0, 50, 100),
    "person":     (100, 0, 255),
    "fence":      (220, 150, 190),
    "bush":       (100, 180, 0),
    "concrete":   (170, 170, 170),
    "barrier":    (0, 200, 255),
    "puddle":     (180, 50, 50),
    "mud":        (30, 60, 90),
    "rubble":     (90, 120, 160),
    "sign":       (50, 220, 255),
    "rock":       (100, 128, 128),
    "sand":       (160, 200, 220),
    "gravel":     (130, 170, 180),
    "mulch":      (40, 100, 160),
    "rock-bed":   (80, 100, 110),
    "bridge":     (140, 140, 100),
    "trash":      (50, 200, 200),
    "bicycle":    (200, 100, 255),
}

# Driving/outdoor-relevant ADE20K classes (for filtering the 150-class legend)
LEGEND_CLASSES = frozenset([
    "road", "sidewalk", "building", "sky", "tree", "grass", "earth", "path",
    "car", "vehicle", "person", "pole", "fence", "wall", "plant", "bus", "truck",
    "bicycle", "motorcycle", "traffic light", "traffic sign", "bridge", "water",
    "rock", "stone", "sand", "ground", "terrain", "vegetation", "house",
    "mountain", "sea", "field", "runway", "river", "tower", "skyscraper",
    "floor", "pavement", "dirt", "mud", "snow", "lane", "dirt track",
])


def _generic_palette():
    """Generic deterministic palette for ADE20K and other large-class models (RGB order)."""
    palette = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        palette[i] = [(37 * i) % 256, (97 * i + 31) % 256, (157 * i + 67) % 256]
    return palette


def _offroad_palette_bgr(id2label, num_entries=256):
    """BGR palette using hand-picked colors for known off-road classes."""
    palette = np.zeros((num_entries, 3), dtype=np.uint8)
    used = set()
    for idx, name in id2label.items():
        key = name.lower().strip()
        if key in OFFROAD_COLORS_BGR:
            palette[idx] = OFFROAD_COLORS_BGR[key]
            used.add(idx)
    for idx in range(num_entries):
        if idx not in used:
            hue = int((idx * 137.508) % 180)
            hsv = np.uint8([[[hue, 200, 220]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
            palette[idx] = bgr
    return palette


def carla_image_to_bgr(carla_image):
    """Convert a carla.Image (BGRA) to a NumPy BGR array for OpenCV."""
    array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
    array = array.reshape((carla_image.height, carla_image.width, 4))
    return array[:, :, :3].copy()


def build_legend(id2label, palette, height, is_finetuned, palette_is_bgr):
    """Build a vertical legend panel: color swatch + class name."""
    legend_width = 220

    if is_finetuned:
        num_items = len(id2label) if id2label else 0
        if num_items == 0:
            legend = np.ones((height, legend_width, 3), dtype=np.uint8) * 240
            cv2.putText(legend, "No labels", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            return legend
        header_h = 26
        row_h = max(10, (height - header_h) // num_items)
        font_scale = 0.35 if row_h >= 14 else 0.28
        legend = np.ones((height, legend_width, 3), dtype=np.uint8) * 255
        cv2.putText(legend, "Class (id)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        items = sorted(id2label.items())
        for i, (idx, name) in enumerate(items):
            y = header_h + 10 + i * row_h
            if y + 4 > height:
                break
            color = palette[idx % 256]
            if palette_is_bgr:
                color_bgr = (int(color[0]), int(color[1]), int(color[2]))
            else:
                color_bgr = (int(color[2]), int(color[1]), int(color[0]))
            box_h = min(row_h - 2, 14)
            cv2.rectangle(legend, (8, y - box_h), (8 + 16, y), color_bgr, -1)
            cv2.rectangle(legend, (8, y - box_h), (8 + 16, y), (80, 80, 80), 1)
            label = "%s (%d)" % (name[:14] if len(name) > 14 else name, idx)
            cv2.putText(legend, label, (30, y - 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
    else:
        filtered = {i: n for i, n in id2label.items() if n.lower().strip() in LEGEND_CLASSES}
        display_labels = filtered if filtered else id2label
        num_rows = 20
        row_h = max(18, height // num_rows)
        actual_rows = min(num_rows, len(display_labels)) if display_labels else 0
        if actual_rows == 0:
            legend = np.ones((height, legend_width, 3), dtype=np.uint8) * 240
            cv2.putText(legend, "No labels", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            return legend
        legend = np.ones((height, legend_width, 3), dtype=np.uint8) * 255
        cv2.putText(legend, "Class (id)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        items = sorted(display_labels.items())[:actual_rows]
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
        description="Manual drive (WASD) + SegFormer segmentation. Works with ADE20K and fine-tuned models."
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--map", metavar="NAME", default=None, help="Load this map (e.g. Town01, Town02).")
    parser.add_argument(
        "--model",
        default="nvidia/segformer-b0-finetuned-ade-512-512",
        help="HuggingFace model id or local path to fine-tuned model (default: ADE20K SegFormer-B0)",
    )
    parser.add_argument("--width", type=int, default=640, help="Camera width (default 640)")
    parser.add_argument("--height", type=int, default=480, help="Camera height (default 480)")
    parser.add_argument(
        "--infer-every",
        type=int,
        default=1,
        metavar="N",
        help="Run model every Nth frame (default 1 = every frame; increase for lower-spec machines)",
    )
    parser.add_argument("--scale", type=float, default=1.5, help="Display scale factor (default 1.5)")
    parser.add_argument(
        "--no-thread",
        action="store_true",
        help="Run inference in main loop instead of background thread (camera may stutter)",
    )
    parser.add_argument(
        "--max-inference-fps",
        action="store_true",
        help="Run inference as fast as possible (no throttle between frames)",
    )
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

    # Connect to CARLA FIRST (before model loading)
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
                print("Map '%s' not available. Run: python scripts/list_maps.py" % args.map)
            raise

    client.set_timeout(30.0)
    world = client.get_world()
    time.sleep(2.0)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading SegFormer model %s (device: %s) ..." % (args.model, device))
    processor = SegformerImageProcessor.from_pretrained(args.model)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model)
    model.to(device)
    model.eval()

    id2label = getattr(model.config, "id2label", None) or {}
    id2label = {int(k): str(v) for k, v in id2label.items()}
    is_finetuned = len(id2label) <= 30

    if is_finetuned:
        palette = _offroad_palette_bgr(id2label)
        palette_is_bgr = True
        print("Fine-tuned model (%d classes): %s" % (len(id2label), ", ".join(id2label.values())))
    else:
        palette = _generic_palette()
        palette_is_bgr = False
        print("Pretrained model (%d classes), showing driving-relevant classes in legend." % len(id2label))

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

    # Attach camera
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

    window_name = "Manual + SegFormer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    display_w = int((args.width * 2 + 220) * args.scale)
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
                if palette_is_bgr:
                    last_seg_bgr[0] = seg_display
                else:
                    last_seg_bgr[0] = cv2.cvtColor(seg_display, cv2.COLOR_RGB2BGR)
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
                    if palette_is_bgr:
                        seg_bgr = seg_display
                    else:
                        seg_bgr = cv2.cvtColor(seg_display, cv2.COLOR_RGB2BGR)
                    last_seg_bgr[0] = seg_bgr
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
                legend = build_legend(id2label, palette, h, is_finetuned, palette_is_bgr)
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

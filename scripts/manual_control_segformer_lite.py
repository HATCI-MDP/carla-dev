#!/usr/bin/env python
"""
Manual drive + SegFormer (lite): you control the car with WASD and see segmentation live.
Lite defaults: 320×240, infer every 5th frame. Use this on lower-spec machines.

Requires: CARLA running, pip install -r requirements-segmentation.txt (includes pynput for key state).

Usage:
    python scripts/manual_control_segformer_lite.py [--host 127.0.0.1] [--port 2000] [--map Town02]
    Controls: W=throttle, S=brake, A=left, D=right. Press 'q' or ESC in the window to exit.
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
        description="Manual drive (WASD) + SegFormer segmentation; lite defaults for lower-spec machines."
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--map", metavar="NAME", default=None, help="Load this map (e.g. Town01, Town02).")
    parser.add_argument(
        "--model",
        default="nvidia/segformer-b0-finetuned-ade-512-512",
        help="Hugging Face model id",
    )
    parser.add_argument("--width", type=int, default=320, help="Camera width (default 320)")
    parser.add_argument("--height", type=int, default=240, help="Camera height (default 240)")
    parser.add_argument(
        "--infer-every",
        type=int,
        default=5,
        metavar="N",
        help="Run model every Nth frame (default 5)",
    )
    parser.add_argument("--scale", type=float, default=1.5, help="Display scale factor (default 1.5)")
    parser.add_argument(
        "--no-thread",
        action="store_true",
        help="Run inference in main loop (camera can stutter); default is background thread so camera updates smoothly",
    )
    parser.add_argument(
        "--max-inference-fps",
        action="store_true",
        help="Run inference as fast as possible (no throttle); use with background thread to get highest inference FPS your GPU/CPU can do",
    )
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

    # Connect to CARLA FIRST (before model loading) to avoid stale connection / UE4 crash
    print("Connecting to CARLA at %s:%d ..." % (args.host, args.port))
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(30.0)
        client.get_server_version()
    except RuntimeError as e:
        print("Failed to connect: %s" % e)
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

    # Give large/streaming maps time to finish loading assets
    client.set_timeout(30.0)
    world = client.get_world()
    time.sleep(2.0)

    # Now load the model (after CARLA connection is stable)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading SegFormer model %s (device: %s) ..." % (args.model, device))
    processor = SegformerImageProcessor.from_pretrained(args.model)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model)
    model.to(device)
    model.eval()
    palette = _ade20k_palette()
    id2label = getattr(model.config, "id2label", None) or {}
    id2label = {int(k): str(v) for k, v in id2label.items()}
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("No spawn points on this map.")
        return 1
    vehicles = blueprint_library.filter("vehicle")
    if not vehicles:
        print("No vehicle blueprints found.")
        return 1
    vehicle_bp = blueprint_library.find("vehicle.dodge.charger_2020") or random.choice(vehicles)
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

    window_name = "Manual + SegFormer (lite)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    display_w = int((args.width * 2 + 220) * args.scale)
    display_h = int(args.height * args.scale)
    cv2.resizeWindow(window_name, display_w, display_h)
    print("W=throttle, S=brake, A=left, D=right. Press 'q' or ESC in window to exit.")
    if not args.no_thread:
        print("Inference runs in background so camera (left) updates at full rate.")
    if args.max_inference_fps:
        print("Max inference FPS: running model as fast as possible (no throttle).")

    last_seg_bgr = [None]
    inference_timestamp = [None]
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
                last_seg_bgr[0] = cv2.cvtColor(seg_display, cv2.COLOR_RGB2BGR)
                t_now = time.perf_counter()
                inference_timestamp[0] = t_now
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
                legend = build_legend(id2label, palette, h, num_rows=20)
                combined = np.hstack([bgr, seg_bgr, legend])
                if args.scale != 1.0:
                    new_w = int(combined.shape[1] * args.scale)
                    new_h = int(combined.shape[0] * args.scale)
                    combined = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                v = vehicle.get_velocity()
                speed_kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
                cv2.putText(combined, "Inference FPS: %.1f" % disp_fps, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(combined, "Inference FPS: %.1f" % disp_fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.putText(combined, "Speed: %.0f km/h" % speed_kmh, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(combined, "Speed: %.0f km/h" % speed_kmh, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
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

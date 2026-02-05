#!/usr/bin/env python
"""
Run SegFormer (ADE20K) on the live CARLA ego camera. Shows camera and segmentation side-by-side.
Car drives on autopilot. Requires: CARLA running, pip install -r requirements-segmentation.txt.

Usage:
    python scripts/autopilot_segformer.py [--host 127.0.0.1] [--port 2000] [--map Town02] [--model MODEL_ID]
    Press 'q' or ESC to exit.

Model is downloaded from Hugging Face on first run (no manual download).
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


# ADE20K uses 150 labels; we need a consistent color per class for display
def _ade20k_palette():
    # Simple deterministic palette (same as run_segformer_image.py style)
    palette = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        palette[i] = [(37 * i) % 256, (97 * i + 31) % 256, (157 * i + 67) % 256]
    return palette


def carla_image_to_bgr(carla_image):
    """Convert a carla.Image (BGRA) to a NumPy BGR array for OpenCV."""
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
    """Build a vertical legend panel: color swatch + class name (driving-relevant classes only)."""
    # Filter to driving/outdoor-relevant classes only
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
        description="Run SegFormer (ADE20K) on CARLA ego camera; show camera + segmentation."
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument(
        "--map",
        metavar="NAME",
        default=None,
        help="Load this map (e.g. Town01, Town02). If omitted, use current map.",
    )
    parser.add_argument(
        "--model",
        default="nvidia/segformer-b0-finetuned-ade-512-512",
        help="Hugging Face model id (default: ADE20K SegFormer-B0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera width (default 640; smaller = faster inference)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera height (default 480)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.5,
        help="Scale factor for display (default 1.5 = larger window)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading SegFormer model %s (device: %s) ..." % (args.model, device))
    print("(First run may download the model from Hugging Face.)")
    processor = SegformerImageProcessor.from_pretrained(args.model)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model)
    model.to(device)
    model.eval()
    palette = _ade20k_palette()
    # Class id -> name from model config (for legend)
    id2label = getattr(model.config, "id2label", None) or {}
    id2label = {int(k): str(v) for k, v in id2label.items()}

    print("Connecting to CARLA at %s:%d ..." % (args.host, args.port))
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        client.get_server_version()
    except RuntimeError as e:
        print("Failed to connect: %s" % e)
        print("Make sure CARLA is running (CarlaUE4.exe) and a map is loaded.")
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
    vehicle_bp = blueprint_library.find("vehicle.audi.tt")
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
    vehicle.set_autopilot(True)

    latest_frame = [None]

    def on_image(carla_image):
        latest_frame[0] = carla_image_to_bgr(carla_image)

    camera.listen(on_image)

    window_name = "Autopilot + SegFormer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("Running. Press 'q' or ESC to exit.")

    t_last_inference = None
    fps_smooth = 0.0
    try:
        while True:
            bgr = latest_frame[0]
            if bgr is None:
                key = cv2.waitKey(100) & 0xFF
                if key == ord("q") or key == 27:
                    break
                continue

            # BGR -> RGB, then PIL for the processor
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            h, w = bgr.shape[:2]

            inputs = processor(images=pil, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits
            logits = torch.nn.functional.interpolate(
                logits, size=(h, w), mode="bilinear", align_corners=False
            )
            seg = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)

            # Colorize: seg has class ids 0..149; map to palette (use mod 256 for display)
            seg_display = palette[seg % 256]
            seg_bgr = cv2.cvtColor(seg_display, cv2.COLOR_RGB2BGR)
            t_now = time.perf_counter()
            if t_last_inference is not None:
                dt = t_now - t_last_inference
                if dt > 0:
                    fps_smooth = 0.85 * fps_smooth + 0.15 * (1.0 / dt)
            t_last_inference = t_now

            # Legend: color swatch + class name for first N classes
            legend = build_legend(id2label, palette, h, num_rows=20)
            # Side-by-side: camera | segmentation | legend
            combined = np.hstack([bgr, seg_bgr, legend])
            # Scale up for display so the window isn't tiny
            if args.scale != 1.0:
                new_w = int(combined.shape[1] * args.scale)
                new_h = int(combined.shape[0] * args.scale)
                combined = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # FPS and speed overlay
            v = vehicle.get_velocity()
            speed_kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
            cv2.putText(combined, "Inference FPS: %.1f" % fps_smooth, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(combined, "Inference FPS: %.1f" % fps_smooth, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(combined, "Speed: %.0f km/h" % speed_kmh, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(combined, "Speed: %.0f km/h" % speed_kmh, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.imshow(window_name, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
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

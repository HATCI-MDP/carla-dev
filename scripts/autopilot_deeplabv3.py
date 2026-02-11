#!/usr/bin/env python
"""
Run DeepLabV3+ (binary road segmentation) on the live CARLA ego camera.
Shows camera and road overlay side-by-side. Car drives on autopilot.
Requires: CARLA running, pip install torch torchvision opencv-python numpy Pillow.

Usage:
    python scripts/autopilot_deeplabv3.py [--host 127.0.0.1] [--port 2000] [--map Town02]
    python scripts/autopilot_deeplabv3.py --model training/models/deeplabv3_orfd/best_model.pth
    Press 'q' or ESC to exit.

The model outputs binary segmentation: road (1) vs not-road (0).
Green overlay = predicted road area.
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
    from torchvision import transforms
except ImportError as e:
    print("Missing dependency: %s" % e)
    print("Install: pip install torch torchvision opencv-python numpy Pillow")
    sys.exit(1)


def load_deeplabv3_model(model_path, num_classes, device):
    """Load a DeepLabV3+ .pth model. Tries full model first, then state_dict."""
    # Try loading as a full model (torch.save(model, path))
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
        if hasattr(model, "eval"):
            print("Loaded full model from %s" % model_path)
            model.to(device)
            model.eval()
            return model
    except Exception:
        pass

    # Try loading as state_dict into torchvision DeepLabV3
    try:
        from torchvision.models.segmentation import deeplabv3_resnet50
        model = deeplabv3_resnet50(num_classes=num_classes, weights_backbone=None)
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        print("Loaded state_dict into DeepLabV3-ResNet50 (num_classes=%d)" % num_classes)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print("ERROR: Could not load model from %s" % model_path)
        print("  Full model load failed, state_dict load failed: %s" % e)
        print("")
        print("Check with your teammate how the model was saved.")
        print("Common formats:")
        print("  torch.save(model, path)              -> full model")
        print("  torch.save(model.state_dict(), path)  -> state dict only")
        sys.exit(1)


def carla_image_to_bgr(carla_image):
    """Convert a carla.Image (BGRA) to a NumPy BGR array for OpenCV."""
    array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
    array = array.reshape((carla_image.height, carla_image.width, 4))
    return array[:, :, :3].copy()


def build_road_overlay(bgr, mask, alpha=0.4):
    """Overlay green on road pixels (mask == 1) over the original BGR image."""
    overlay = bgr.copy()
    road_color = np.array([0, 200, 0], dtype=np.uint8)  # green in BGR
    overlay[mask == 1] = (
        (1 - alpha) * overlay[mask == 1].astype(np.float32)
        + alpha * road_color.astype(np.float32)
    ).astype(np.uint8)
    return overlay


def main():
    parser = argparse.ArgumentParser(
        description="Run DeepLabV3+ (binary road segmentation) on CARLA ego camera."
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
        default="training/models/deeplabv3_orfd/best_model.pth",
        help="Path to DeepLabV3+ .pth model file (default: training/models/deeplabv3_orfd/best_model.pth)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of output classes (default: 2 for binary road/not-road)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=512,
        help="Resize input to this resolution for inference (default: 512)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera width (default 640)",
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
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for road class (default: 0.5)",
    )
    args = parser.parse_args()

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

    # Load DeepLabV3+ model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading DeepLabV3+ model from %s (device: %s) ..." % (args.model, device))
    model = load_deeplabv3_model(args.model, args.num_classes, device)

    # Standard ImageNet normalization (used by torchvision models)
    preprocess = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
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
    vehicle.set_autopilot(True)

    latest_frame = [None]

    def on_image(carla_image):
        latest_frame[0] = carla_image_to_bgr(carla_image)

    camera.listen(on_image)

    window_name = "Autopilot + DeepLabV3+ Road Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("Running. Press 'q' or ESC to exit.")
    print("Green overlay = predicted road area.")

    t_last_inference = None
    fps_smooth = 0.0
    road_pct = 0.0
    try:
        while True:
            bgr = latest_frame[0]
            if bgr is None:
                key = cv2.waitKey(100) & 0xFF
                if key == ord("q") or key == 27:
                    break
                continue

            h, w = bgr.shape[:2]

            # Preprocess: BGR -> RGB -> PIL -> tensor
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            input_tensor = preprocess(pil).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                output = model(input_tensor)

            # Handle both torchvision dict output and raw tensor output
            if isinstance(output, dict):
                logits = output.get("out", list(output.values())[0])
            elif isinstance(output, (tuple, list)):
                logits = output[0]
            else:
                logits = output

            # Upsample to original resolution
            logits = torch.nn.functional.interpolate(
                logits, size=(h, w), mode="bilinear", align_corners=False
            )

            # Binary segmentation: class 1 = road
            if logits.shape[1] == 1:
                # Single channel sigmoid output
                road_mask = (torch.sigmoid(logits[0, 0]) > args.threshold).cpu().numpy().astype(np.uint8)
            elif logits.shape[1] >= 2:
                # Multi-channel softmax output — road is class 1
                probs = torch.softmax(logits, dim=1)
                road_mask = (probs[0, 1] > args.threshold).cpu().numpy().astype(np.uint8)
            else:
                road_mask = np.zeros((h, w), dtype=np.uint8)

            road_pct = 100.0 * road_mask.sum() / (h * w)

            # Build display: camera | road overlay | binary mask
            overlay = build_road_overlay(bgr, road_mask)
            mask_display = np.zeros((h, w, 3), dtype=np.uint8)
            mask_display[road_mask == 1] = [0, 255, 0]
            mask_display[road_mask == 0] = [40, 40, 40]

            t_now = time.perf_counter()
            if t_last_inference is not None:
                dt = t_now - t_last_inference
                if dt > 0:
                    fps_smooth = 0.85 * fps_smooth + 0.15 * (1.0 / dt)
            t_last_inference = t_now

            combined = np.hstack([bgr, overlay, mask_display])

            # Scale for display
            if args.scale != 1.0:
                new_w = int(combined.shape[1] * args.scale)
                new_h = int(combined.shape[0] * args.scale)
                combined = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # HUD overlay
            v = vehicle.get_velocity()
            speed_kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
            hud_lines = [
                "Inference FPS: %.1f" % fps_smooth,
                "Speed: %.0f km/h" % speed_kmh,
                "Road: %.1f%%" % road_pct,
            ]
            for i, line in enumerate(hud_lines):
                y = 32 + i * 30
                cv2.putText(combined, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(combined, line, (10, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

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

# CARLA Setup & Off-Road Perception Guide

This document covers: running CARLA, project scripts, camera and map options, and how to prototype with pre-trained segmentation models for off-road perception.

---

## 1. CARLA setup (quick reference)

- **CARLA install path:** `C:\CARLA_0.9.16` (override with `CARLA_ROOT` env var).
- **Python:** 3.12 required for the CARLA 0.9.16 Windows wheel.
- **Start server:** `python scripts/start_carla.py` (or run `CarlaUE4.exe`). Wait for the Unreal window and a map to load. To run **without the spectator window** (saves resources; cameras still work): `python scripts/start_carla.py --headless`.
- **Install client:**
  `pip install "C:\CARLA_0.9.16\PythonAPI\carla\dist\carla-0.9.16-cp312-cp312-win_amd64.whl"`
  Then: `pip install -r requirements.txt`
- **Test connection:** With CARLA running, `python scripts/connection.py` then `python scripts/connection.py --demo`.

---

## 2. Project scripts

| Script | Purpose |
|--------|--------|
| `scripts/start_carla.py` | Start CARLA server (CarlaUE4.exe). Use `--headless` to run without the spectator window (cameras still work). |
| `scripts/connection.py` | Test connection; `--demo` spawns vehicle + camera (default vehicle: Audi TT). Use `--vehicle <id>` or `--vehicle random`. |
| `scripts/autopilot.py` | Ego RGB camera while car drives on autopilot. Defaults: 120° FOV, ~5 MP, sporty car (`vehicle.audi.tt`). Supports `--map`, `--vehicle`, `--fov`, `--width`, `--height`. |
| `scripts/autopilot_segformer.py` | Run SegFormer (ADE20K) on the live ego camera; shows camera + segmentation side-by-side. Needs `pip install -r requirements-segmentation.txt`. |
| `scripts/autopilot_segformer_lite.py` | Same as above but for lower-spec machines: smaller res (320×240), infer every 3rd frame, sensible window size. |
| `scripts/manual_control_segformer.py` | Manual drive (WASD) + SegFormer; full res (640×480), every frame. For GPU. |
| `scripts/manual_control_segformer_lite.py` | Manual drive (WASD) + SegFormer; lite (320×240, infer every 3rd frame). |
| `scripts/manual_control.py` | CARLA’s manual control (HUD, keybinds, camera angles, reverse). Defaults to sedan (`vehicle.audi.a2`). |
| `scripts/generate_traffic.py` | Spawn NPC vehicles and pedestrians. |
| `scripts/automatic_control.py` | Autopilot agent + camera + HUD. Uses `--loop` by default so it keeps driving to new targets. |
| `scripts/list_maps.py` | Print maps available on the running CARLA server (run CARLA first). |
| `scripts/load_map.py` | Load a map and exit (e.g. `python scripts/load_map.py --map Town07`). Use before running other scripts so they see the desired map. |

**Order of use:** Start CARLA with `start_carla.py` (or the exe), then in another terminal run any of the other scripts.

### Vehicle selection

Scripts that spawn an ego vehicle (`autopilot.py`, `connection.py --demo`) **no longer pick at random**: they default to a **sporty car** and let you override.

- **Default:** `vehicle.audi.tt` (Audi TT).
- **Override:** `--vehicle <blueprint_id>` (e.g. `--vehicle vehicle.ford.mustang`) or `--vehicle random` for a random vehicle from the catalogue.

**Sporty / performance car options (blueprint IDs):**

| Blueprint ID | Model |
|--------------|--------|
| `vehicle.audi.tt` | Audi TT (default in this project) |
| `vehicle.ford.mustang` | Ford Mustang |
| `vehicle.tesla.model3` | Tesla Model 3 |
| `vehicle.mercedes.coupe` | Mercedes Coupe |
| `vehicle.mercedes.coupe_2020` | Mercedes Coupe 2020 (Gen 2) |
| `vehicle.dodge.charger_2020` | Dodge Charger 2020 |
| `vehicle.mini.cooper_s` | Mini Cooper S |
| `vehicle.mini.cooper_s_2021` | Mini Cooper S 2021 |

**Other scripts:**

- **manual_control.py:** Uses CARLA’s example; default is `vehicle.audi.a2` (sedan). Pass `--filter vehicle.audi.tt` (or any blueprint ID) to choose the vehicle.
- **automatic_control.py:** Launches CARLA’s automatic_control example; vehicle selection is that example’s default (check CARLA docs or pass through any `--filter` your CARLA version supports).

Full vehicle catalogue: [CARLA Vehicles](https://carla.readthedocs.io/en/latest/catalogue_vehicles/).

---

## 3. Camera parameters (autopilot camera)

The ego camera in `autopilot.py` is set to match a real-world spec: **120° FOV**, **~5 MP RGB**.

- **FOV:** `--fov` (degrees). Default: 120.
- **Resolution:** `--width`, `--height`. Default: 2560×1920 (~5 MP). Override for different resolution or aspect ratio.
- **Other options (in CARLA):** On the camera blueprint you can also set `sensor_tick`, `gamma`, and camera transform (position/rotation).

Example:

```powershell
python scripts/autopilot.py --map Town03 --fov 120 --width 2560 --height 1920
```

---

## 4. Maps

### Loading a map from the client

- **Standalone (recommended):** Run once, then use any script on the new map:
  ```powershell
  python scripts/load_map.py --map Town07
  ```
  Then run `autopilot.py`, `manual_control.py`, etc. without `--map`; they will use the current map.

- From Python, after connecting: `client.load_world("Town02")` then `world = client.get_world()`.
- From the autopilot camera script: `python scripts/autopilot.py --map Town03` (still supported).
- Map load can take **30–90 seconds** on first load; `load_map.py` uses a 90 s timeout for `load_world()`.

### Which maps are available?

- **Base install (typical):** Town01, Town02, Town03, Town04, Town05. Some builds also include Town06.
- **Extra maps (optional):** Town06, Town07, Town10, etc. require the **additional maps package** from CARLA (download from CARLA releases, import into the `Import` folder, run ImportAssets). If you did not install that package, **Town07 (and sometimes Town06) will not exist** and you get: `RuntimeError: Map 'Town07' not found`.

**If you get “Map 'Town07' not found”:**
Your CARLA build doesn’t include that map. Use a map that is in the base install, e.g.:

```powershell
python scripts/autopilot.py --map Town03
```

To see exactly which maps your server has, run (with CARLA already running):

```powershell
python scripts/list_maps.py
```

### Most “off-road-y” maps (when available)

- **Town07** – Country road, most rural (requires extra maps package).
- **Town03** – Mix of town and open/hilly; good fallback.
- **Town06** – Long highway with rural sections (may require extra maps).

If Town07 is not installed, use **Town03** as the most varied / least urban option in the base set.

### How to get Town06 and Town07 (additional maps)

Town06 and Town07 are not in the base CARLA package; you have to download the **Additional Maps** asset and merge it into your CARLA install.

**1. Download and extract**

- Download **AdditionalMaps Nightly Build (Windows)** from the [CARLA GitHub](https://github.com/carla-simulator/carla) (or the release page for your version).
- Extract the zip so you have a folder, e.g.:
  `C:\CARLA_0.9.16\Import\AdditionalMaps_Latest\`
  with subfolders `CarlaUE4\` and `Engine\` inside it.

**2. Merge into CARLA (Windows pre-built has no ImportAssets script)**

Pre-built Windows CARLA does not ship with `ImportAssets.sh`/`.bat`. You “import” by **merging** the AdditionalMaps content into the main CARLA folder.

From an **elevated Command Prompt** or PowerShell (optional but avoids permission issues), run:

```powershell
# Replace with your CARLA path if different
$CARLA = "C:\CARLA_0.9.16"
$IMP = "$CARLA\Import\AdditionalMaps_Latest"

# Merge CarlaUE4 content (maps and assets) into main CarlaUE4
robocopy "$IMP\CarlaUE4\Content" "$CARLA\CarlaUE4\Content" /E /IS /IT

# Merge plugin content (e.g. TaggedMaterials for additional maps)
robocopy "$IMP\CarlaUE4\Plugins" "$CARLA\CarlaUE4\Plugins" /E /IS /IT
```

- **`/E`** = copy subdirectories including empty.
- **`/IS /IT`** = include same and modified files (so you don’t skip existing files that are the same).

If you prefer to do it manually: copy the **contents** of `Import\AdditionalMaps_Latest\CarlaUE4\Content\` into `CarlaUE4\Content\`, and the contents of `Import\AdditionalMaps_Latest\CarlaUE4\Plugins\` into `CarlaUE4\Plugins\`, so that new maps (e.g. Town06, Town07) and assets are added alongside the existing ones. Do not replace the whole `Content` or `Plugins` folder.

**3. Restart CARLA**

Close CARLA if it’s running, then start it again. Run:

```powershell
python scripts/list_maps.py
```

You should see **Town06** and **Town07** in the list. Then:

```powershell
python scripts/autopilot.py --map Town07
```

---

## 5. Pre-trained segmentation models for off-road (dirt roads)

### Suggested models

- **General outdoor/driving (easy start):**
  **SegFormer** (B2/B5) or **DeepLabV3+** pretrained on **Cityscapes** or **ADE20K**. Classes like road, vegetation, terrain, obstacles; not dirt-specific but usable for “road vs trees vs obstacles”.

- **Better for off-road / dirt:**
  Models trained or fine-tuned on **RUGD** or **RELLIS-3D** (off-road/trail datasets). Search for “RUGD semantic segmentation” or “off-road segmentation” (e.g. GitHub, papers); many use SegFormer or DeepLab and release weights. These give dirt-road vs vegetation vs obstacles more directly.

- **Other options:** **BiSeNet**, **PIDNet** (real-time driving). **Mapillary Vistas** or **Semantic KITTI** pretrained models for road/terrain/vegetation.

**Summary:** Start with **SegFormer or DeepLabV3+** (Cityscapes/ADE20K); for dirt-road behavior, prefer models trained on **RUGD/RELLIS** or similar off-road datasets.

**How to run ADE20K and OffSeg:** See **[SEGMENTATION_MODELS.md](SEGMENTATION_MODELS.md)** for step-by-step: ADE20K (SegFormer) via `run_segformer_image.py`, and OffSeg (clone repo, weights, pipeline). **SegFormer details:** [SEGFORMER_SETUP.md](SEGFORMER_SETUP.md).

### Off-road prototype stack (plug-and-play, no training)

**Road / drivable (dirt road matters most):**

- **SegFormer ADE20K** – For “dirt road” you don’t have a single class; treat these ADE20K classes as drivable / path: **earth** (13), **path** (52), **grass** (9), and optionally **road** (6). Mask pixels with those class IDs to get a drivable region. Fast (~20+ FPS with B0).
**Obstacles (cars, people, rocks):**

- **YOLOv8** – `pip install ultralytics`; pretrained on COCO. Strong for cars and people; large rocks may be hit-or-miss. Run in parallel with your segmentation model for a full stack (terrain + objects).

**Sim-native (CARLA-specific):**

- **nuCarla** – Dataset and pretrained BEV models (e.g. BEVFormer, BEVDet) built in CARLA for off-road/on-road. Good if you want bird’s-eye view for planning; search for “nuCarla” to find the repo and weights.

**Suggested combo for “road + obstacles”:** SegFormer ADE20K (mask earth/path/grass/road for drivable) + YOLOv8 (cars, people). “dirt road” .

---

## 6. Hooking segmentation models up to CARLA

- **Input:** Use the **ego RGB camera** from your CARLA client (e.g. the same feed as `autopilot.py`). Each frame from that camera is the image you run through the model.

- **Where:** In the same Python process as the CARLA client. In the camera callback (or a loop that reads the “latest” frame): convert CARLA image to numpy (e.g. RGB), resize/normalize to the model’s input size, run the model, get a per-pixel segmentation map (H×W of class IDs).

- **Display:**
  - **Overlay:** Map class IDs to a color palette (e.g. Cityscapes or the model’s dataset), resize the mask to camera resolution, blend with the camera image (e.g. 0.5×image + 0.5×colored_mask), show in the same OpenCV/Pygame window.
  - **Side-by-side:** One panel = camera image, one = colored segmentation mask.

- **Pipeline:** Extend (or copy) the autopilot-camera script: (1) Keep the part that receives camera frames. (2) Load the pretrained segmentation model once at startup. (3) For each new (or latest) frame: preprocess → model forward → colorize mask → show overlay or side-by-side.

- **Performance:** Use GPU if available; choose a fast model (e.g. SegFormer-B2, BiSeNet) or lower input resolution for near real-time. Sync vs async in CARLA only affects how often new frames arrive; the “run model on latest frame and display” logic is the same.

---

## 7. Telemetry and HUD

- The on-screen panel in CARLA’s manual control that shows speed, throttle, brake, etc. is the **HUD** (heads-up display).
- The **data** (speed, throttle, brake, steer, etc.) is **telemetry**.

---

## 8. Troubleshooting

| Issue | What to do |
|-------|------------|
| Connection failed | Ensure CARLA is running and a map is loaded. Check ports 2000 and 2001. |
| `Map 'Town07' not found` | Use a base map (e.g. `--map Town03`) or install CARLA’s additional maps package. Run `python scripts/list_maps.py` to see available maps. |
| Map load timeout | First load can take 30–90 s. The autopilot script uses a 90 s timeout; if it still times out, try a different map or restart CARLA. |
| `ImportError: No module named 'carla'` | Install the CARLA `.whl` from your CARLA install path (see section 1). |
| Wrong Python | CARLA 0.9.16 wheel is for Python 3.12. Use `python --version` and the correct venv. |

# carla-dev

Vision-based perception for soft road boundaries on unmarked dirt roads. This repo gets CARLA up and running with Python.

**Full guide:** [docs/CARLA_AND_PERCEPTION_GUIDE.md](docs/CARLA_AND_PERCEPTION_GUIDE.md) — scripts, camera/map options, maps, and prototyping with pre-trained segmentation models.

## Getting CARLA up and running (Python)

### 1. Start the CARLA server

From the project directory you can run:

```powershell
python scripts/start_carla.py
```

This launches `CarlaUE4.exe` from `C:\CARLA_0.9.16` (override with the `CARLA_ROOT` environment variable if your install is elsewhere). The script blocks until you close CARLA.

Wait until the Unreal Engine window opens and a town map is loaded. The server listens on `127.0.0.1:2000` by default.

### 2. Python environment

- **Python 3.12** is required for the CARLA 0.9.16 Windows wheel.
- Create and activate a virtual environment (recommended):

```powershell
cd c:\{path to carla-dev}\carla-dev
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the CARLA Python client from your CARLA install path, then the rest of the project deps:

```powershell
pip install "C:\CARLA_0.9.16\PythonAPI\carla\dist\carla-0.9.16-cp312-cp312-win_amd64.whl"
pip install -r requirements.txt
```

### 4. Ego camera in a separate window (autopilot)

To see exactly what the car’s camera sees while the car drives itself:

```powershell
python scripts/autopilot.py
```

A separate OpenCV window shows the RGB camera feed (defaults: 120° FOV, ~5 MP). The vehicle drives on autopilot (view only). Press **q** or **ESC** to exit. Optional: `--map`, `--fov`, `--width`, `--height`.

To drive the car yourself (manual control), use CARLA’s official manual control (full HUD, keybinds, camera angles, reverse, hand-brake, etc.):

```powershell
python scripts/manual_control.py
```

**Other CARLA launchers** (run from project with your venv):

- **Generate traffic** (NPC vehicles and pedestrians):
  `python scripts/generate_traffic.py`
- **Automatic control** (autopilot agent + camera + HUD):
  `python scripts/automatic_control.py`

### Troubleshooting

- **Connection failed**: Ensure `CarlaUE4.exe` is running and you’ve selected/loaded a map. Ports 2000 and 2001 must not be blocked.
- **ImportError: No module named 'carla'**: Install the `.whl` from step 3 (path must match your CARLA install).
- **Wrong Python**: The provided wheel is for **Python 3.12** (`cp312`). Use `python --version` and switch to 3.12 if needed.

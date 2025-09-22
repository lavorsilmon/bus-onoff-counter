# Bus ON/OFF Counter — Dual-Gate Baseline (YOLOv8 + BoT-SORT)

A lightweight, GPU-friendly counter that detects **people boarding (ON)** and **alighting (OFF)** from a **single 2D video**.
It uses YOLOv8 for person detection and dual-gate (two-line) pairing logic with tracking to robustly count entries/exits.

## Features
- **Two auto-placed gates** (LEFT & RIGHT) — no manual clicking.
- **Per-gate and total ON/OFF counts** + simple **unique-person tally**.
- **Real-time** on an RTX-class GPU with `--half` and modest `--imgsz`.
- Tunable knobs: `--gate_gap_px`, `--gate_window`, `--assoc_max_dist`, `--cooldown`, `--anchor`.

## How it works
1. Detect people (class 0) with YOLOv8.
2. Track them with BoT-SORT / ByteTrack to get stable IDs.
3. Auto-place two vertical lines per side (RIGHT=cyan, LEFT=magenta).
4. Pair L1/L2 crossings within a time+distance window:
   - L1→L2 = **ON** (boarding)
   - L2→L1 = **OFF** (alighting)

## Requirements
- Python 3.10+
- (Optional) NVIDIA GPU
- Packages: `ultralytics`, `opencv-python`, `numpy<2.3`

## Install
```bash
# New environment (recommended)
conda create -n bus-counter python=3.10 -y
conda activate bus-counter

# Install PyTorch (pick the right CUDA from pytorch.org)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Core deps
python -m pip install -U ultralytics opencv-python "numpy>=1.26,<2.3"
```

## Run (Windows example)
```bat
python src\bus_people_unique_onoff_v3_dualgate.py ^
  --source "videos\bus_sample.mp4" ^
  --model yolov8n.pt ^
  --tracker configs\botsort.yaml ^
  --device 0 --half ^
  --imgsz 512 --vid_stride 1 --max_det 20 ^
  --conf 0.45 ^
  --stable_frames 10 ^
  --dedup_frames 90 --dedup_iou 0.6 --dedup_center_alpha 0.55 ^
  --min_box_area 3000 ^
  --autogate both ^
  --gate_gap_px 70 ^
  --gate_min_gap 3 --gate_window 210 ^
  --assoc_window 210 --assoc_max_dist 110 ^
  --cooldown 45 ^
  --show --save
```

## Tuning
- `--gate_gap_px` (50–90): spacing between lines (bigger = stricter).
- `--gate_window` (180–240): max frames to complete L1↔L2.
- `--assoc_max_dist` (90–140): max px to pair hits (occlusion tolerance).
- `--cooldown` (45–60): frames to ignore a track after counting.
- `--anchor center|feet`: pick `center` when feet aren’t visible.

## Repo layout
```
bus-onoff-counter/
├─ src/
│  └─ bus_people_unique_onoff_v3_dualgate.py
├─ configs/
│  └─ botsort.yaml
├─ scripts/
│  └─ RUN_DUALGATE.bat
├─ videos/           # local test videos (gitignored)
├─ docs/
│  └─ diagram_dualgate.txt
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ LICENSE
```

## Troubleshooting
- **No counts**: verify `--source` path and visible people.
- **Over-counting**: increase `--cooldown`, reduce `--assoc_max_dist`, or increase `--gate_gap_px`.
- **Under-counting**: reduce `--gate_gap_px`, increase `--gate_window`.
- **Tracker errors**: `python -m pip install -U ultralytics` and ensure `botsort.yaml` matches your Ultralytics version.

## License
MIT

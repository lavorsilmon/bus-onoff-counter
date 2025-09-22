
@echo off
set "PY=%USERPROFILE%\anaconda3\envs\bus-counter\python.exe"

"%PY%" "%~dp0..\src\bus_people_unique_onoff_v3_dualgate.py" ^
  --source "%~dp0..\videos\bus_sample.mp4" ^
  --model yolov8n.pt ^
  --tracker "%~dp0..\configs\botsort.yaml" ^
  --device 0 --half ^
  --imgsz 512 --vid_stride 1 --max_det 20 ^
  --conf 0.45 ^
  --stable_frames 10 ^
  --dedup_frames 90 --dedup_iou 0.6 --dedup_center_alpha 0.55 ^
  --min_box_area 3000 ^
  --autogate both ^
  --gate_gap_px 80 ^
  --gate_min_gap 3 --gate_window 210 ^
  --assoc_window 210 --assoc_max_dist 100 ^
  --cooldown 45 ^
  --show --save
pause

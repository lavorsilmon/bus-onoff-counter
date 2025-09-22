
"""
Unique + ON/OFF (v3) with DUAL GATES (left + right)
- Keeps your v3 unique counter logic
- Adds TWO gate pairs: one near the LEFT edge and one near the RIGHT edge
- Each gate does outside->inside = ON, inside->outside = OFF
- No clicking; auto-places both gate pairs by sampling a short clip
- Shows total ON/OFF and per-gate ON/OFF

This file is **commented thoroughly** for clarity. No executable logic has been changed:
only comments and whitespace have been added to explain what each section does and
how key parameters influence behavior.
"""

# ---- Imports & basic setup ---------------------------------------------------
# argparse   : parse command-line arguments (e.g., --source, --conf, etc.)
# os, time   : filesystem paths and timing for FPS estimates
# collections: default dicts and deque for simple queues
# math       : geometry helpers (distances, diagonals)
# cv2        : OpenCV for video I/O and drawing
# numpy      : numeric ops for medians, arrays
# ultralytics: YOLO model (v8) for person detection + built-in tracker integration
import argparse, os, time, collections, math
import cv2, numpy as np
from collections import deque
from ultralytics import YOLO

# ---------------- v3 helpers --------------------------------------------------
# The following helpers are used by the "v3" unique-counting heuristic.
# They are orthogonal to ON/OFF and exist to 1) avoid double-counting the same
# person as "new" too often, and 2) provide a simple sanity metric alongside
# the ON/OFF counts.
def iou_xyxy(a, b):
    """
    Intersection-over-Union for two boxes in [x1,y1,x2,y2] format.
    Used in unique de-dup to decide if two boxes refer to the same person.
    """
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)       # overlap top-left
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)       # overlap bottom-right
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-6                  # add epsilon to avoid /0
    return inter / union

def center_of(bb):
    """Center point (cx, cy) of a bounding box."""
    x1,y1,x2,y2 = bb
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def feet_of(bb):
    """
    "Feet" anchor = midpoint of the bottom edge. This can be more stable for
    crossing vertical gate lines than the center point (especially in crowded scenes).
    """
    x1,y1,x2,y2 = bb
    return ((x1+x2)/2.0, y2)

def diag_len(bb):
    """Geometric diagonal length of a box. Used to scale center-distance thresholds."""
    x1,y1,x2,y2 = bb
    return math.hypot(x2-x1, y2-y1)

def is_dup(bb, bb_prev, alpha, iou_thresh):
    """
    Heuristic: two detections are considered the "same person" (for unique-counting)
    if either:
      - their IoU >= iou_thresh, or
      - the distance between centers is less than alpha * average box diagonal
    This suppresses immediate re-counts of the same person across nearby frames.
    """
    if iou_xyxy(bb, bb_prev) >= iou_thresh:
        return True
    c1 = center_of(bb); c2 = center_of(bb_prev)
    avg_diag = 0.5 * (diag_len(bb) + diag_len(bb_prev)) + 1e-6
    cdist = math.hypot(c1[0]-c2[0], c1[1]-c2[1])
    return cdist < (alpha * avg_diag)

# -------------- gate helpers --------------------------------------------------
# These helpers are for ON/OFF counting via pairs of vertical lines ("gates").
def line_side(p, a, b):
    """
    Signed area test: which side of oriented line AB does point p fall on?
    Positive or negative sign flips when p crosses the line.
    """
    return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])

def crossed(prev_s, curr_s, eps=0.0):
    """
    Detects a crossing event when the sign of line_side changes across frames.
    eps provides a small dead-zone to avoid jitter around exactly-on-the-line cases.
    """
    return (prev_s < -eps and curr_s > eps) or (prev_s > eps and curr_s < -eps)

def draw_boxes(im, xyxy, ids, confs, anchors=None):
    """
    Draw detections + track IDs + confidences for debugging.
    Optionally show the chosen "anchor" point (center or feet) as a small dot.
    """
    if xyxy is None: return
    for i, bb in enumerate(xyxy):
        x1,y1,x2,y2 = [int(v) for v in bb]
        cv2.rectangle(im, (x1,y1), (x2,y2), (0,180,255), 2)
        tid = int(ids[i]) if ids is not None and i < len(ids) else -1
        conf = float(confs[i]) if confs is not None and i < len(confs) else -1.0
        label = f"id:{tid if tid!=-1 else '?'}  {conf:.2f}" if conf >= 0 else f"id:{tid if tid!=-1 else '?'}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(y1 - 6, th + 6)
        cv2.rectangle(im, (x1, y_text - th - 6), (x1 + tw + 6, y_text + 4), (0,180,255), -1)
        cv2.putText(im, label, (x1 + 3, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        if anchors is not None and i < len(anchors):
            ax, ay = int(anchors[i][0]), int(anchors[i][1])
            cv2.circle(im, (ax, ay), 3, (255,255,255), -1)

def draw_overlay(im, unique_count, current_count, on_count, off_count, gates, fps=None):
    """
    Heads-up display (HUD) with unique/current totals + ON/OFF and per-gate stats.
    Also draws the two vertical lines for each gate side with labels.
    """
    cv2.rectangle(im, (10,10), (10+1100, 120), (0,0,0), -1)
    cv2.putText(im, f"UNIQUE: {unique_count}", (20, 48),  cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(im, f"CURRENT: {current_count}",(260, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(im, f"ON: {on_count}", (520, 48),        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(im, f"OFF: {off_count}", (680, 48),      cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0,0,255), 2, cv2.LINE_AA)
    if fps is not None:
        cv2.putText(im, f"FPS: {fps:.1f}", (860, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2, cv2.LINE_AA)
    # Per-gate mini counts (left/right)
    y0=85; x0=20; dx=220
    for i,g in enumerate(gates):
        cv2.putText(im, f"{g['name']}: ON {g['on']} | OFF {g['off']}", (x0 + i*dx, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2, cv2.LINE_AA)
    # Draw the vertical lines and labels for each gate
    for g in gates:
        (A1,B1) = g['A1'], g['B1']
        (A2,B2) = g['A2'], g['B2']
        color = (0,200,255) if g['name']=="RIGHT" else (200,0,255)
        cv2.line(im, A1, B1, color, 2); cv2.putText(im, f"{g['name']}-L1", (A1[0]+5, A1[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        cv2.line(im, A2, B2, color, 2); cv2.putText(im, f"{g['name']}-L2", (A2[0]+5, A2[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

# -------------- auto-gate placement per side ---------------------------------
def autogate_side(src, model, conf, imgsz, device, side="right", frames=150, step=2, edge_pct=0.25, gap_px=None):
    """
    Quickly samples ~'frames' frames (sparsely by 'step') and uses detected person
    centers near the selected image edge to estimate where the door band is.
    Then places two vertical lines (L1 outer, L2 inner) separated by 'gap_px'.
    Returns: A1,B1,A2,B2,(H,W)
    """
    cap = cv2.VideoCapture(src if not str(src).isdigit() else int(src))
    ok, frame = cap.read()
    if not ok:
        cap.release(); raise RuntimeError("Could not read first frame for autogate.")
    H, W = frame.shape[:2]
    xs, ys = [], []
    used = 0; idx = 0
    while used < frames:
        if idx % step == 0:
            ok, fr = (True, frame) if idx == 0 else cap.read()  # reuse first frame once to avoid initial lag
            if not ok: break
            # Run a quick person-only detection pass (class 0) to collect center positions
            res = model.predict(fr, conf=conf, verbose=False, classes=[0], imgsz=imgsz, device=device)
            if res and len(res[0].boxes) > 0:
                for bb in res[0].boxes.xyxy.cpu().numpy():
                    x1,y1,x2,y2 = bb.tolist()
                    cx, cy = (x1+x2)/2.0, (y1+y2)/2.0  # center anchor (no feet requirement here)
                    # Keep samples from the chosen edge band only
                    if side=="right" and cx > W*(1.0-edge_pct):
                        xs.append(cx); ys.append(cy)
                    if side=="left"  and cx < W*edge_pct:
                        xs.append(cx); ys.append(cy)
            used += 1
        idx += 1
    cap.release()

    if len(xs)==0:
        # Fallback: place gate near the edge if we collected no samples
        xs=[int(W*0.92) if side=="right" else int(W*0.08)]
        ys=[int(H*0.6)]
    x_med = int(np.median(xs))                              # robust center line in the edge band
    # Vertical extent of the gate clamped to a sensible range
    y_min = int(max(0.2*H, min(ys, default=int(0.2*H))))
    y_max = int(min(0.95*H, max(ys, default=int(0.95*H))))
    if gap_px is None:
        gap_px = max(30, int(0.04 * W))                     # default gap ~4% image width
    # Decide which vertical line is "outer" vs "inner" depending on the side
    if side=="right":
        x_outer = min(W-2, max(2, x_med + gap_px//2))
        x_inner = min(W-2, max(2, x_med - gap_px//2))
    else:
        x_outer = min(W-2, max(2, x_med - gap_px//2))
        x_inner = min(W-2, max(2, x_med + gap_px//2))
    A1,B1 = (x_outer, y_min), (x_outer, y_max)              # L1 (outer)
    A2,B2 = (x_inner, y_min), (x_inner, y_max)              # L2 (inner)
    return A1,B1,A2,B2,(H,W)

# ---------------------- main --------------------------------------------------
def main():
    # ---- CLI arguments -------------------------------------------------------
    ap = argparse.ArgumentParser("Unique + ON/OFF (v3 auto) with DUAL gates")
    ap.add_argument("--source", required=True)                   # path to video or webcam index (e.g., 0)
    ap.add_argument("--model", default="yolov8n.pt")             # YOLO model checkpoint
    ap.add_argument("--tracker", default="botsort.yaml")         # or 'bytetrack.yaml'
    ap.add_argument("--conf", type=float, default=0.45)          # detector confidence threshold
    ap.add_argument("--iou",  type=float, default=0.5)           # NMS IoU threshold
    ap.add_argument("--device", default=None)                    # GPU id like '0' or 'cpu'
    ap.add_argument("--imgsz", type=int, default=640)            # inference resolution
    ap.add_argument("--vid_stride", type=int, default=1)         # skip-rate for reading video frames
    ap.add_argument("--max_det", type=int, default=30)           # cap number of detections per frame
    ap.add_argument("--half", action="store_true")               # use FP16 if supported (speed)

    # v3 unique heuristic knobs (sanity unique count alongside ON/OFF)
    ap.add_argument("--stable_frames", type=int, default=10)     # require a track to exist this many frames first
    ap.add_argument("--dedup_frames",  type=int, default=60)     # lookback window to suppress re-entries
    ap.add_argument("--dedup_iou",     type=float, default=0.60) # IoU threshold for duplicate suppression
    ap.add_argument("--dedup_center_alpha", type=float, default=0.45)  # relative center distance threshold
    ap.add_argument("--min_box_area",  type=int, default=3000)   # ignore very small boxes

    # Gate placement + pairing logic
    ap.add_argument("--autogate", choices=["left","right","both"], default="both",
                    help="Place gate on left, right, or both sides")
    ap.add_argument("--calib_frames", type=int, default=150)     # how long to sample for autogate
    ap.add_argument("--gate_gap_px", type=int, default=70)       # horizontal gap between L1 and L2
    ap.add_argument("--gate_min_gap", type=int, default=3)       # min frames allowed between L1 and L2 hits
    ap.add_argument("--gate_window", type=int, default=150)      # max frames allowed between L1 and L2 hits
    ap.add_argument("--anchor", choices=["feet","center"], default="center")  # which anchor to test vs lines
    ap.add_argument("--min_move_px", type=float, default=12.0)   # reject tiny jitter crossings
    ap.add_argument("--assoc_window", type=int, default=210)     # how long to keep single hits waiting for pair
    ap.add_argument("--assoc_max_dist", type=float, default=80.0) # max pixels to match a pair (occlusion tolerance)
    ap.add_argument("--cooldown", type=int, default=45)          # after a count, ignore this track for N frames

    # Display / output
    ap.add_argument("--show", action="store_true")               # show live window
    ap.add_argument("--save", action="store_true")               # save annotated MP4 next to source
    args = ap.parse_args()

    # ---- Load detector -------------------------------------------------------
    model = YOLO(args.model)

    # ---- Build gates for requested sides ------------------------------------
    # We support "LEFT", "RIGHT", or both. Each gate has:
    # - A1,B1 (outer line), A2,B2 (inner line)
    # - prev_s1/prev_s2: last signed side value per track id (for crossing detection)
    # - unmatched: queue of single-line hits waiting to be paired with the other line
    # - on/off counters
    sides = ["right","left"] if args.autogate=="both" else [args.autogate]
    gates = []
    H=W=None
    for side in sides:
        A1,B1,A2,B2, (H,W) = autogate_side(
            args.source, model, args.conf, args.imgsz, args.device,
            side=side, frames=args.calib_frames, step=2, edge_pct=0.25, gap_px=args.gate_gap_px if args.gate_gap_px>0 else None
        )
        gates.append({
            "name": side.upper(),
            "A1": A1, "B1": B1, "A2": A2, "B2": B2,
            "prev_s1": {}, "prev_s2": {},
            "unmatched": deque(),   # pending single-line crossings to be paired
            "on": 0, "off": 0
        })
        print(f"[gate-{side}] L1={A1}->{B1}  L2={A2}->{B2}  (ON is L1->L2)")

    # ---- Tracking stream -----------------------------------------------------
    # Ultralytics .track(...) yields a live stream of per-frame results including
    # detection boxes, IDs (from the tracker), confidence scores, and original frame.
    stream = model.track(
        source=args.source, tracker=args.tracker, stream=True, conf=args.conf, iou=args.iou,
        device=args.device, classes=[0], verbose=False, persist=True,
        imgsz=args.imgsz, vid_stride=args.vid_stride, max_det=args.max_det, half=args.half
    )

    # ---- Global state for counting & display ---------------------------------
    writer=None; out_path=None
    seen_ids=set(); track_life=collections.defaultdict(int); frame_idx=0
    recent_counted=[]; fps_smooth=None; last_time=time.time()
    cooldown_until = {}  # global cooldown per tid (prevents double-taps after a count)

    # Choose anchor function (feet or center) for gate-cross tests
    get_anchor = feet_of if args.anchor=="feet" else center_of

    # Small helpers for gate queues and count validation
    def prune_unmatched(g, now_idx):
        """
        Keep the 'unmatched' queue bounded in time. If a single-line hit waits
        too long without finding its complement, drop it as stale.
        """
        dq = g["unmatched"]
        while dq and (now_idx - dq[0]["frame"]) > max(args.assoc_window, args.gate_window):
            dq.popleft()

    def maybe_count(g, prev_line, prev_frame, prev_pt, cur_line, cur_frame, cur_pt):
        """
        Validate a candidate pair (Lx at frame t0, Ly at frame t1) before counting:
        - respect min/max frame gap windows
        - require a minimum movement to avoid micro-jitter flips
        Decide ON (1->2) vs OFF (2->1) by the order of lines crossed.
        Return True if a count was registered.
        """
        gap = cur_frame - prev_frame
        if gap < args.gate_min_gap or gap > args.gate_window: return False
        move = max(abs(cur_pt[0]-prev_pt[0]), abs(cur_pt[1]-prev_pt[1]))
        if move < args.min_move_px: return False
        if f"{prev_line}->{cur_line}" == "1->2":
            g["on"]  += 1
        else:
            g["off"] += 1
        return True

    # ---- Main loop -----------------------------------------------------------
    for res in stream:
        frame_idx += 1
        im = res.orig_img.copy()  # BGR frame for drawing

        # Extract tracker outputs
        boxes = getattr(res, "boxes", None)
        ids = confs = xyxy = None
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy() if getattr(boxes, "xyxy", None) is not None else None
            ids  = boxes.id.cpu().numpy().astype(int) if getattr(boxes, "id", None) is not None and boxes.id is not None else None
            confs= boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else None

        # ---- Unique v3 logic (sanity unique count) ---------------------------
        if ids is not None:
            for tid in ids: track_life[int(tid)] += 1
        # decay recent unique entries beyond the lookback window
        recent_counted = [(f,bb) for (f,bb) in recent_counted if frame_idx - f <= args.dedup_frames]
        if ids is not None and xyxy is not None:
            for i, tid in enumerate(ids):
                bb = xyxy[i].tolist()
                if (bb[2]-bb[0])*(bb[3]-bb[1]) < args.min_box_area: continue   # drop tiny detections
                if track_life[tid] >= args.stable_frames and tid not in seen_ids:
                    dup=False
                    for f_prev, bb_prev in recent_counted:
                        if is_dup(bb, bb_prev, args.dedup_center_alpha, args.dedup_iou):
                            dup=True; break
                    if not dup:
                        seen_ids.add(int(tid))
                        recent_counted.append((frame_idx, bb))

        # ---- Anchor points for current frame ---------------------------------
        anchors=[]
        if xyxy is not None:
            anchors=[ get_anchor(bb) for bb in xyxy ]

        # ---- Gate logic per side ---------------------------------------------
        if xyxy is not None and ids is not None:
            for g in gates:
                prune_unmatched(g, frame_idx)  # keep queue fresh
                A1,B1,A2,B2 = g["A1"], g["B1"], g["A2"], g["B2"]
                prev_s1, prev_s2 = g["prev_s1"], g["prev_s2"]
                dq = g["unmatched"]

                for idx, (tid, bb) in enumerate(zip(ids, xyxy)):
                    # Optional cooldown after a count to avoid immediate double hits
                    if cooldown_until.get(tid, -1) > frame_idx:
                        continue
                    a = anchors[idx]                     # anchor point for this track
                    s1 = line_side(a, A1, B1)           # signed side relative to L1
                    s2 = line_side(a, A2, B2)           # signed side relative to L2
                    p1 = prev_s1.get(tid, None)         # previous signed values
                    p2 = prev_s2.get(tid, None)
                    c1 = p1 is not None and crossed(p1, s1)  # detect L1 crossing
                    c2 = p2 is not None and crossed(p2, s2)  # detect L2 crossing

                    event_line = None                    # which line was crossed this frame?
                    if c1: event_line = 1
                    if c2: event_line = 2 if event_line is None else event_line

                    if event_line is not None:
                        # We got one side of the pair; look backwards for the complement
                        wanted = 2 if event_line==1 else 1
                        best_k=-1; best_score=-1.0
                        for k in range(len(dq)-1, -1, -1):   # newest first
                            u = dq[k]
                            if u["line"] != wanted: continue
                            gap = frame_idx - u["frame"]
                            if gap < args.gate_min_gap or gap > args.gate_window: continue
                            dist = math.hypot(a[0]-u["pt"][0], a[1]-u["pt"][1])
                            if dist > args.assoc_max_dist: continue
                            score = (1.0 - min(1.0, dist/args.assoc_max_dist))  # nearer = better
                            if score > best_score:
                                best_score = score; best_k = k
                        if best_k >= 0:
                            # Found a plausible partner line hit; validate movement + count
                            u = dq[best_k]
                            if maybe_count(g, u["line"], u["frame"], u["pt"], event_line, frame_idx, a):
                                dq.remove(u)
                                # Cooldown both tracks so they don't immediately produce another count
                                cooldown_until[tid] = frame_idx + args.cooldown
                                cooldown_until[u["tid"]] = frame_idx + args.cooldown
                        else:
                            # No partner yet: remember this single hit for later pairing
                            dq.append({"line": event_line, "tid": tid, "frame": frame_idx, "pt": a})

                    # Save current signed side for next-frame crossing checks
                    prev_s1[tid] = s1
                    prev_s2[tid] = s2

        # ---- Draw / save / show ----------------------------------------------
        current_count = len(ids) if ids is not None else (len(xyxy) if xyxy is not None else 0)
        draw_boxes(im, xyxy, ids, confs, anchors)
        on_total = sum(g["on"] for g in gates)
        off_total= sum(g["off"] for g in gates)

        # FPS smoothing (lightweight EMA)
        now=time.time(); dt=now-last_time; last_time=now
        fps_smooth = (1.0/dt if dt>0 else 0.0) if fps_smooth is None else (0.9*fps_smooth + 0.1*(1.0/dt if dt>0 else 0.0))

        draw_overlay(im, unique_count=len(seen_ids), current_count=current_count,
                     on_count=on_total, off_count=off_total, gates=gates, fps=fps_smooth)

        # Lazy-init writer on first frame we need to save
        if writer is None and args.save and isinstance(args.source, str):
            out_path=(os.path.splitext(args.source)[0] if isinstance(args.source,str) else "webcam")+"_unique_onoff_v3dualgate.mp4"
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"); writer=cv2.VideoWriter(out_path,fourcc,30.0,(im.shape[1], im.shape[0]))
            print("Saving to:", out_path)
        if writer is not None: writer.write(im)

        if args.show:
            cv2.imshow("Unique+ON/OFF (v3 dual gates)", im)
            if cv2.waitKey(1) & 0xFF == 27: break   # ESC to quit

    # ---- Cleanup -------------------------------------------------------------
    if writer is not None: writer.release()
    if args.show: cv2.destroyAllWindows()
    print("FINAL -> UNIQUE:", len(seen_ids), "| ON total:", sum(g['on'] for g in gates), "| OFF total:", sum(g['off'] for g in gates))
    for g in gates:
        print(f"  {g['name']}: ON {g['on']} | OFF {g['off']}")

# Standard Python entry-point pattern
if __name__=="__main__":
    main()

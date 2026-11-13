"""
live.py - LIVE camera: run YOLO on a live video feed and update the dashboard in real time.

Sources you can use:
    --source 0                                   laptop webcam
    --source http://192.168.1.5:8080/video       phone camera (Android "IP Webcam" app)
    --source rtsp://user:pass@ip:554/stream      real CCTV / IP camera
    --source traffic.mp4                         a video file (plays like a live feed)

Example:
    python live.py --source http://192.168.1.5:8080/video --junction "Narayanaguda"

While it runs, every second it writes (used automatically by app.py):
    outputs/<junction>/frame.jpg     latest frame with detection boxes
    outputs/<junction>/counts.json   vehicles seen in the last 60 s, scaled to per hour
Press q in the preview window (or Ctrl + C in the terminal) to stop.
"""

import argparse
import json
import os
import time
from collections import deque

import cv2
from ultralytics import YOLO

from vision import PCU, category

WINDOW_S = 60  # counts are based on the last 60 seconds


def write_atomic(path, write_fn):
    """Write to a temp file then swap it in, so the dashboard never reads half a file."""
    root, ext = os.path.splitext(path)
    tmp = root + ".tmp" + ext  # keep the extension - cv2.imwrite needs it
    write_fn(tmp)
    try:
        os.replace(tmp, path)
    except PermissionError:  # Windows: file is being read right now - skip this update
        pass


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="0 for webcam, a URL, or a video file")
    ap.add_argument("--junction", required=True)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--no-window", action="store_true", help="don't show the preview window")
    args = ap.parse_args()

    out_dir = os.path.join("outputs", args.junction.lower().replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)
    # Remove an old recorded video so the dashboard shows the live frame instead
    old_video = os.path.join(out_dir, "annotated.mp4")
    if os.path.exists(old_video):
        os.rename(old_video, os.path.join(out_dir, f"annotated_recorded_{int(time.time())}.mp4"))

    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(args.model)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}\n"
                         "Phone camera? Check the phone and laptop are on the same Wi-Fi/hotspot "
                         "and the URL opens in your browser.")
    is_file = isinstance(source, str) and os.path.exists(source)
    fps_file = cap.get(cv2.CAP_PROP_FPS) or 25

    first_seen = {}                 # track id -> (time, category)
    recent = deque()                # (time, category) for vehicles first seen in the window
    start = last_write = time.time()
    frames = 0
    print(f"Live: {args.junction} <- {args.source}   (press q to stop)")

    while True:
        loop_start = time.time()
        ok, frame = cap.read()
        if not ok:
            if is_file:  # loop video files so they behave like a live feed
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("Stream ended or dropped - retrying in 2 s...")
            time.sleep(2)
            cap.release()
            cap = cv2.VideoCapture(source)
            continue

        result = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=args.conf,
                             imgsz=args.imgsz, verbose=False)[0]
        now = time.time()
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            for cls_id, tid in zip(boxes.cls.int().tolist(), boxes.id.int().tolist()):
                cat = category(result.names[cls_id])
                if cat and tid not in first_seen:
                    first_seen[tid] = (now, cat)
                    recent.append((now, cat))
        while recent and now - recent[0][0] > WINDOW_S:
            recent.popleft()

        annotated = result.plot(line_width=1, font_size=0.4)
        frames += 1

        if now - last_write >= 1.0:  # update dashboard files once per second
            window = min(WINDOW_S, max(now - start, 1.0))
            counts = {k: 0 for k in PCU}
            for _, cat in recent:
                counts[cat] += 1
            per_hour = {k: int(round(v * 3600 / window)) for k, v in counts.items()}
            data = {
                "junction": args.junction,
                "video": f"LIVE: {args.source}",
                "model": os.path.basename(args.model),
                "live": True,
                "updated": time.strftime("%H:%M:%S"),
                "duration_s": round(window, 1),
                "counts_in_clip": counts,
                "counts_per_hour": per_hour,
                "pcu_per_hour": round(sum(PCU[k] * v for k, v in per_hour.items())),
                "fps": round(frames / max(now - start, 1e-6), 1),
            }
            h, w = annotated.shape[:2]
            small = cv2.resize(annotated, (min(960, w), int(h * min(960, w) / w)))
            write_atomic(os.path.join(out_dir, "frame.jpg"),
                         lambda p: cv2.imwrite(p, small, [cv2.IMWRITE_JPEG_QUALITY, 80]))
            write_atomic(os.path.join(out_dir, "counts.json"), lambda p: save_json(p, data))
            last_write = now
            print(f"  {data['updated']}  last {int(window)}s: {sum(counts.values())} vehicles  "
                  f"({data['fps']} fps)")

        if not args.no_window:
            cv2.imshow(f"LIVE - {args.junction} (q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if is_file:  # play files at real speed
            time.sleep(max(0.0, 1 / fps_file - (time.time() - loop_start)))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

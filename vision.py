"""
vision.py - Deep learning: count vehicles in a traffic video with YOLO + ByteTrack.

Run this ONCE before the demo (live inference on a laptop can freeze on stage):

    python vision.py --video traffic.mp4 --junction "Hitech City"

Run it once per junction video - each junction gets its own folder:

    python vision.py --video panjagutta.mp4 --junction "Panjagutta"
    python vision.py --video narayanaguda.mp4 --junction "Narayanaguda"

Outputs (used automatically by app.py), e.g. for Panjagutta:
    outputs/panjagutta/counts.json     vehicle counts per class, scaled to vehicles/hour
    outputs/panjagutta/annotated.mp4   video with detection boxes (browser-playable)
    outputs/panjagutta/frame.jpg       one annotated frame (fallback image)

Model: defaults to yolo11n.pt (downloads automatically, COCO classes).
For better Indian results (autos, two-wheelers), pass UVH-26 weights from
https://huggingface.co/iisc-aim/UVH-26 with --model path/to/weights.pt
"""

import argparse
import json
import os
import shutil
import subprocess
from collections import defaultdict

import cv2
from ultralytics import YOLO

PCU = {"two_wheeler": 0.5, "auto": 1.2, "car": 1.0, "bus": 2.2, "truck": 2.2}


def category(class_name):
    """Map any model's class name (COCO or UVH-26) to our 5 categories."""
    n = class_name.lower().replace("-", "_").replace(" ", "_")
    if any(k in n for k in ["auto", "three", "3_wheel", "rickshaw"]):
        return "auto"
    if any(k in n for k in ["motor", "two", "2_wheel", "bike", "scooter", "bicycle", "cycle"]):
        return "two_wheeler"
    if any(k in n for k in ["bus"]):
        return "bus"
    if any(k in n for k in ["truck", "lcv", "tempo", "lorry", "tractor"]):
        return "truck"
    if any(k in n for k in ["car", "sedan", "suv", "hatch", "muv", "van", "jeep", "taxi"]):
        return "car"
    return None  # person, traffic light, etc. - ignored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--junction", default="Hitech City")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--max-seconds", type=float, default=60, help="only process the first N seconds")
    ap.add_argument("--stride", type=int, default=2, help="process every Nth frame (faster)")
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()

    out_dir = os.path.join("outputs", args.junction.lower().replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)
    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, 960 / max(w, 1))  # keep output video small
    out_size = (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2)
    tmp_path = os.path.join(out_dir, "_annotated_raw.mp4")
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps / args.stride, out_size)

    seen = defaultdict(set)          # category -> set of track ids
    max_frames = int(args.max_seconds * fps)
    frame_idx, processed, best_frame, best_count = 0, 0, None, -1

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % args.stride:
            continue
        result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                             conf=args.conf, verbose=False)[0]
        boxes = result.boxes
        n_vehicles = 0
        if boxes is not None and boxes.id is not None:
            for cls_id, track_id in zip(boxes.cls.int().tolist(), boxes.id.int().tolist()):
                cat = category(result.names[cls_id])
                if cat:
                    seen[cat].add(track_id)
                    n_vehicles += 1
        annotated = cv2.resize(result.plot(line_width=1, font_size=0.4), out_size)
        writer.write(annotated)
        if n_vehicles > best_count:
            best_count, best_frame = n_vehicles, annotated
        processed += 1
        if processed % 30 == 0:
            print(f"  {frame_idx / fps:5.1f}s  unique vehicles so far: {sum(len(s) for s in seen.values())}")

    cap.release()
    writer.release()
    duration_s = max(frame_idx / fps, 1e-6)

    counts = {k: len(seen.get(k, ())) for k in PCU}
    per_hour = {k: int(round(v * 3600 / duration_s)) for k, v in counts.items()}
    data = {
        "junction": args.junction,
        "video": os.path.basename(args.video),
        "model": os.path.basename(args.model),
        "duration_s": round(duration_s, 1),
        "counts_in_clip": counts,
        "counts_per_hour": per_hour,
        "pcu_per_hour": round(sum(PCU[k] * v for k, v in per_hour.items())),
    }
    with open(os.path.join(out_dir, "counts.json"), "w") as f:
        json.dump(data, f, indent=2)
    if best_frame is not None:
        cv2.imwrite(os.path.join(out_dir, "frame.jpg"), best_frame)

    # Browsers only play H.264 - convert with ffmpeg if available
    final = os.path.join(out_dir, "annotated.mp4")
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_path, "-c:v", "libx264",
                        "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an", final], check=False)
        if os.path.exists(final):
            os.remove(tmp_path)
    else:
        os.replace(tmp_path, final)
        print("ffmpeg not found - annotated.mp4 may not play in the browser; frame.jpg is shown instead.")

    print(json.dumps(data, indent=2))
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()

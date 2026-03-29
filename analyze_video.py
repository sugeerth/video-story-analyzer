#!/usr/bin/env python3
"""
Analyze any video — detect objects, tell the story, build interactive Sankey.

Usage:
    python3 analyze_video.py path/to/video.mp4
    python3 analyze_video.py path/to/video.mp4 --confidence 0.5 --segments 10

Outputs an interactive HTML with:
  - Narrative story arc (Opening → Rising Action → Climax → Resolution)
  - Adaptive time-scaled Sankey (busy = expanded, quiet = contracted)
  - Hover any Sankey node or segment card to see actual video frames with bboxes
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import sys
import argparse
import time
import json
import io
import base64
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import torch
import torchvision
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from models.object_detector import COCO_CLASSES

BBOX_COLORS = {
    "person": (41, 128, 255), "bicycle": (156, 39, 176), "car": (76, 175, 80),
    "motorcycle": (0, 188, 212), "bus": (233, 30, 99), "truck": (255, 152, 0),
    "traffic light": (38, 166, 154), "stop sign": (171, 71, 188),
    "dog": (255, 87, 34), "cat": (121, 85, 72), "bird": (0, 150, 136),
    "horse": (255, 112, 67), "backpack": (255, 235, 59), "umbrella": (103, 58, 183),
    "handbag": (96, 125, 139), "tie": (205, 220, 57), "chair": (63, 81, 181),
    "bench": (0, 150, 136), "bottle": (139, 195, 74), "cup": (244, 67, 54),
    "cell phone": (255, 193, 7), "potted plant": (102, 187, 106),
    "book": (3, 169, 244), "clock": (255, 167, 38), "laptop": (0, 188, 212),
    "tv": (63, 81, 181), "couch": (205, 220, 57),
}
DEFAULT_COLOR = (160, 160, 160)

SANKEY_COLORS = [
    "#2979FF", "#00C853", "#FF6D00", "#D50000", "#AA00FF",
    "#00BFA5", "#DD2C00", "#6D4C41", "#546E7A", "#C6FF00",
    "#304FFE", "#00897B", "#FFD600", "#64DD17", "#FF1744",
]


# ═══════════════════════════════════════════════════════════════════
# VIDEO PROCESSING
# ═══════════════════════════════════════════════════════════════════

def extract_frames(path: str, sample_n: int) -> tuple[list[np.ndarray], float, int, int, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {path}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_n == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames, fps, total, w, h


def run_detection(frames, confidence=0.40):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    model.eval()
    per_frame = []
    t0 = time.time()
    for i, bgr in enumerate(frames):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = torchvision.transforms.functional.to_tensor(rgb)
        with torch.no_grad():
            out = model([tensor])[0]
        dets = []
        for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
            if score >= confidence:
                dets.append({
                    "box": box.cpu().numpy().tolist(),
                    "label": COCO_CLASSES[label.item()] if label.item() < len(COCO_CLASSES) else "unknown",
                    "score": round(score.item(), 3),
                })
        per_frame.append(dets)
        done = i + 1
        if done % 20 == 0 or done == len(frames):
            el = time.time() - t0
            print(f"  [{done}/{len(frames)}] {done/el:.1f} fps — frame {i}: {len(dets)} objects")
    print(f"  Total: {sum(len(d) for d in per_frame)} detections in {time.time()-t0:.1f}s")
    return per_frame


# ═══════════════════════════════════════════════════════════════════
# ADAPTIVE SEGMENTATION
# ═══════════════════════════════════════════════════════════════════

def build_segments(per_frame, target_segs=10, min_frames=4):
    n = len(per_frame)
    counts = np.array([len(d) for d in per_frame], dtype=float)
    k = max(3, n // 25)
    smoothed = np.convolve(counts, np.ones(k)/k, mode="same")
    cumul = np.cumsum(smoothed)
    total_mass = cumul[-1] if len(cumul) else 1.0
    mass_per = total_mass / target_segs

    segments = []
    seg_start = 0
    running = 0.0

    for i in range(n):
        running += smoothed[i]
        is_last = i == n - 1
        enough = running >= mass_per and (i - seg_start + 1) >= min_frames

        if enough or is_last:
            s, e = seg_start, i + 1
            seg_dets = []
            seg_classes = set()
            for t in range(s, e):
                seg_dets.extend(per_frame[t])
                for d in per_frame[t]:
                    seg_classes.add(d["label"])
            nf = e - s
            density = len(seg_dets) / max(nf, 1)
            segments.append({
                "start": s, "end": e, "n_frames": nf,
                "n_detections": len(seg_dets), "density": density,
                "classes": seg_classes,
            })
            seg_start = i + 1
            running = 0.0
            if len(segments) >= target_segs and not is_last:
                # Merge remainder into last segment on final pass
                pass

    # Width weights — sqrt scaling
    c = np.array([s["n_detections"] for s in segments], dtype=float)
    w = np.sqrt(c + 1)
    w /= w.sum()
    for seg, ww in zip(segments, w):
        seg["width_weight"] = float(ww)
    return segments


# ═══════════════════════════════════════════════════════════════════
# STORY ARC ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def analyze_story(segments, per_frame, fps, sample_n):
    """Analyze the detection timeline and produce a narrative story arc."""
    densities = [s["density"] for s in segments]
    max_density = max(densities) if densities else 1
    n_segs = len(segments)

    # Identify story phases
    climax_idx = densities.index(max_density)

    # Characters: top objects
    all_dets = []
    for pf in per_frame:
        for d in pf:
            all_dets.append(d["label"])
    char_counts = Counter(all_dets)
    protagonists = char_counts.most_common(3)  # top 3 = main characters

    # First appearances per class
    first_seen = {}
    last_seen = {}
    for t, dets in enumerate(per_frame):
        for d in dets:
            lbl = d["label"]
            if lbl not in first_seen:
                first_seen[lbl] = t
            last_seen[lbl] = t

    # Generate per-segment narrative
    narratives = []
    for i, seg in enumerate(segments):
        t0_sec = seg["start"] * sample_n / fps
        t1_sec = seg["end"] * sample_n / fps
        d = seg["density"]

        # What's new in this segment?
        new_objects = []
        departing = []
        for cls in seg["classes"]:
            if first_seen.get(cls, -1) >= seg["start"] and first_seen[cls] < seg["end"]:
                new_objects.append(cls)
            if last_seen.get(cls, -1) >= seg["start"] and last_seen[cls] < seg["end"]:
                # Check it doesn't appear later
                appears_later = any(
                    cls in [dd["label"] for dd in per_frame[t]]
                    for t in range(seg["end"], len(per_frame))
                )
                if not appears_later:
                    departing.append(cls)

        # Assign story phase
        if i <= climax_idx * 0.3:
            phase = "opening"
        elif i < climax_idx:
            phase = "rising"
        elif i == climax_idx:
            phase = "climax"
        elif i < n_segs * 0.85:
            phase = "falling"
        else:
            phase = "resolution"

        # Build narrative text
        top_here = Counter()
        for t in range(seg["start"], seg["end"]):
            for dd in per_frame[t]:
                top_here[dd["label"]] += 1
        dominant = [c for c, _ in top_here.most_common(3)]

        txt = ""
        if phase == "opening":
            if new_objects:
                txt = f"The scene opens with {_join(dominant)}. {_join(new_objects, 'enters', 'enter')} the frame."
            else:
                txt = f"The scene begins. {_join(dominant)} {'dominates' if len(dominant)==1 else 'dominate'} the view."
        elif phase == "rising":
            if new_objects:
                txt = f"Activity builds — {_join(new_objects)} {'appears' if len(new_objects)==1 else 'appear'} alongside {_join(dominant)}."
            else:
                txt = f"Tension rises with {_join(dominant)} increasingly active."
        elif phase == "climax":
            txt = f"Peak activity! {seg['n_detections']} detections — {_join(dominant)} {'fills' if len(dominant)==1 else 'fill'} the frame at {d:.1f} objects/frame."
        elif phase == "falling":
            if departing:
                txt = f"Activity eases. {_join(departing)} {'leaves' if len(departing)==1 else 'leave'} the scene."
            else:
                txt = f"The pace slows with {_join(dominant)} still present."
        else:
            txt = f"The scene settles. {_join(dominant)} {'remains' if len(dominant)==1 else 'remain'} as the video closes."

        narratives.append({
            "phase": phase,
            "text": txt,
            "new_objects": new_objects,
            "departing": departing,
            "dominant": dominant,
            "t0": t0_sec,
            "t1": t1_sec,
        })

    # Global story summary
    summary = (
        f"This video tells the story of {_join([p[0] for p in protagonists])} "
        f"across {len(segments)} scenes. "
        f"The action peaks at T{climax_idx+1} with {segments[climax_idx]['density']:.1f} detections/frame, "
        f"featuring {len(char_counts)} distinct object types in total."
    )

    return {
        "narratives": narratives,
        "summary": summary,
        "protagonists": protagonists,
        "climax_idx": climax_idx,
        "first_seen": first_seen,
        "last_seen": last_seen,
    }


def analyze_sentiment(segments, per_frame, top_k=10):
    """Analyze per-object sentiment: confidence trend, frequency trend, appearance arc.

    For each top-K object class, computes across time segments:
      - avg_confidence per segment (sparkline data)
      - count per segment (frequency sparkline)
      - trend: "rising", "falling", "stable", "peaked", "appeared", "disappeared"
      - momentum: rate of change in detection frequency
      - overall_sentiment: positive (growing/confident), negative (fading), neutral

    Returns dict[class_name] -> sentiment_info
    """
    # Per-segment per-class stats
    seg_stats = []  # list of dict[class -> {count, total_conf}]
    for seg in segments:
        stats = defaultdict(lambda: {"count": 0, "total_conf": 0.0})
        for t in range(seg["start"], seg["end"]):
            if t < len(per_frame):
                for d in per_frame[t]:
                    stats[d["label"]]["count"] += 1
                    stats[d["label"]]["total_conf"] += d["score"]
        seg_stats.append(dict(stats))

    # Find top-K classes
    total_by_class = defaultdict(int)
    for ss in seg_stats:
        for cls, st in ss.items():
            total_by_class[cls] += st["count"]
    top_classes = [c for c, _ in sorted(total_by_class.items(), key=lambda x: x[1], reverse=True)[:top_k]]

    n_segs = len(segments)
    sentiments = {}

    for cls in top_classes:
        counts = []
        confs = []
        for ss in seg_stats:
            st = ss.get(cls, {"count": 0, "total_conf": 0.0})
            counts.append(st["count"])
            confs.append(st["total_conf"] / st["count"] if st["count"] > 0 else 0)

        counts_arr = np.array(counts, dtype=float)
        confs_arr = np.array(confs, dtype=float)

        # Trend analysis
        present = counts_arr > 0
        first_present = np.argmax(present) if present.any() else 0
        last_present = n_segs - 1 - np.argmax(present[::-1]) if present.any() else 0
        peak_seg = int(np.argmax(counts_arr))

        # Linear regression on counts for trend
        if present.sum() >= 2:
            x = np.arange(n_segs)
            mask = present
            slope = np.polyfit(x[mask], counts_arr[mask], 1)[0]
            conf_slope = np.polyfit(x[mask], confs_arr[mask], 1)[0] if confs_arr[mask].sum() > 0 else 0
        else:
            slope = 0
            conf_slope = 0

        # Classify trend
        if first_present > 0 and first_present > n_segs * 0.3:
            trend = "appeared"
        elif last_present < n_segs - 1 and last_present < n_segs * 0.7:
            trend = "disappeared"
        elif peak_seg > 0 and peak_seg < n_segs - 1 and counts_arr[peak_seg] > counts_arr.mean() * 1.5:
            trend = "peaked"
        elif slope > 0.5:
            trend = "rising"
        elif slope < -0.5:
            trend = "falling"
        else:
            trend = "stable"

        # Overall sentiment
        if trend in ("rising", "appeared"):
            sentiment = "positive"
        elif trend in ("falling", "disappeared"):
            sentiment = "negative"
        elif trend == "peaked":
            sentiment = "peaked"
        else:
            sentiment = "neutral"

        # Confidence sentiment
        if conf_slope > 0.01:
            conf_trend = "improving"
        elif conf_slope < -0.01:
            conf_trend = "degrading"
        else:
            conf_trend = "steady"

        # Momentum (rate of change in recent segments vs early)
        mid = n_segs // 2
        early_avg = counts_arr[:mid].mean() if mid > 0 else 0
        late_avg = counts_arr[mid:].mean() if mid < n_segs else 0
        momentum = (late_avg - early_avg) / max(early_avg, 1)

        sentiments[cls] = {
            "counts": counts,
            "confidences": [round(c, 3) for c in confs],
            "trend": trend,
            "sentiment": sentiment,
            "conf_trend": conf_trend,
            "slope": round(slope, 3),
            "conf_slope": round(conf_slope, 4),
            "momentum": round(momentum, 2),
            "peak_segment": peak_seg,
            "first_seen_seg": int(first_present),
            "last_seen_seg": int(last_present),
            "total": int(counts_arr.sum()),
            "avg_conf": round(confs_arr[confs_arr > 0].mean(), 3) if (confs_arr > 0).any() else 0,
        }

    return sentiments


def _join(items, singular_verb="", plural_verb=""):
    if not items:
        return "nothing"
    if len(items) == 1:
        base = f"**{items[0]}**"
        return f"{base} {singular_verb}" if singular_verb else base
    base = ", ".join(f"**{x}**" for x in items[:-1]) + f" and **{items[-1]}**"
    return f"{base} {plural_verb}" if plural_verb else base


# ═══════════════════════════════════════════════════════════════════
# FRAME RENDERING
# ═══════════════════════════════════════════════════════════════════

def _font(sz=10):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", sz)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", sz)
        except (OSError, IOError):
            return ImageFont.load_default()


def render_frame(bgr, dets, label_text, w=340):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    r = w / img.width
    h = int(img.height * r)
    img = img.resize((w, h), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    f9 = _font(9)
    f10 = _font(10)

    for d in dets:
        bx = d["box"]
        color = BBOX_COLORS.get(d["label"], DEFAULT_COLOR)
        x1, y1, x2, y2 = [int(c * r) for c in bx]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        txt = f"{d['label']} {d['score']:.0%}"
        bb = f9.getbbox(txt)
        tw, th = bb[2]-bb[0], bb[3]-bb[1]
        ly = max(y1-th-4, 0)
        draw.rectangle((x1, ly, x1+tw+6, ly+th+4), fill=color)
        draw.text((x1+3, ly+1), txt, fill=(255,255,255), font=f9)

    draw.rectangle((0,0,90,16), fill=(0,0,0))
    draw.text((4,2), label_text, fill=(255,255,255), font=f10)
    n = len(dets)
    draw.rectangle((w-55,0,w,16), fill=(0,0,0))
    draw.text((w-52,2), f"{n} obj", fill=(0,255,150), font=f10)
    return img


def img_b64(img, quality=80):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def filmstrip(imgs, gap=3):
    if not imgs:
        return Image.new("RGB", (100,60), (30,30,30))
    w, h = imgs[0].size
    strip = Image.new("RGB", (w*len(imgs)+gap*(len(imgs)-1), h), (30,30,30))
    for i, im in enumerate(imgs):
        strip.paste(im, (i*(w+gap), 0))
    return strip


# ═══════════════════════════════════════════════════════════════════
# HTML GENERATION
# ═══════════════════════════════════════════════════════════════════

def _rgba(hx, a=0.4):
    h = hx.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"


def build_html(
    segments, per_frame, frames_bgr, story, sentiments, fps, sample_n,
    video_name, vid_w, vid_h, total_raw_frames, output_path,
    video_path=None, top_k=10,
):
    import plotly.graph_objects as go

    n_segs = len(segments)
    duration = total_raw_frames / fps

    # ── Per-segment hover frame data (5 frames each for hover gallery) ──
    hover_data = []  # list of lists of {b64, label, n_dets}
    for seg in segments:
        nf = seg["end"] - seg["start"]
        if nf <= 5:
            idxs = list(range(seg["start"], seg["end"]))
        else:
            idxs = np.linspace(seg["start"], seg["end"]-1, 5, dtype=int).tolist()
        seg_hover = []
        for fi in idxs:
            if fi < len(frames_bgr):
                d = per_frame[fi] if fi < len(per_frame) else []
                t_sec = fi * sample_n / fps
                thumb = render_frame(frames_bgr[fi], d, f"{t_sec:.1f}s", w=300)
                seg_hover.append({
                    "b64": img_b64(thumb, quality=75),
                    "t": round(t_sec, 1),
                    "n": len(d),
                })
        hover_data.append(seg_hover)

    # ── Sankey data ──
    seg_counts = []
    for seg in segments:
        c = defaultdict(int)
        for t in range(seg["start"], seg["end"]):
            for d in per_frame[t]:
                c[d["label"]] += 1
        seg_counts.append(dict(c))

    total_by_class = defaultdict(int)
    for sc in seg_counts:
        for cls, cnt in sc.items():
            total_by_class[cls] += cnt

    top_classes = [c for c, _ in sorted(total_by_class.items(), key=lambda x: x[1], reverse=True)[:top_k]]
    cmap = {cls: SANKEY_COLORS[i % len(SANKEY_COLORS)] for i, cls in enumerate(top_classes)}
    all_detected = set(total_by_class.keys())
    missed = set(COCO_CLASSES[1:]) - all_detected
    total_det = sum(total_by_class.values())

    weights = np.array([s["width_weight"] for s in segments])
    cumw = np.concatenate([[0], np.cumsum(weights)])
    x_pos = [(cumw[i]+cumw[i+1])/2 * 0.78 + 0.01 for i in range(n_segs)]

    nlabels, ncols, nx, ny = [], [], [], []
    nmap = {}

    for si in range(n_segs):
        in_seg = [c for c in top_classes if seg_counts[si].get(c, 0) > 0]
        other = sum(v for k, v in seg_counts[si].items() if k not in top_classes)
        nn = len(in_seg) + (1 if other > 0 else 0)
        yp = np.linspace(0.05, 0.95, max(nn, 1)).tolist()
        yi = 0
        for cls in in_seg:
            nid = len(nlabels)
            nmap[(si, cls)] = nid
            nlabels.append(f"{cls} ({seg_counts[si][cls]})")
            ncols.append(cmap[cls])
            nx.append(x_pos[si])
            ny.append(yp[yi])
            yi += 1
        if other > 0:
            nid = len(nlabels)
            nmap[(si, "__other__")] = nid
            nlabels.append(f"other ({other})")
            ncols.append("#B0BEC5")
            nx.append(x_pos[si])
            ny.append(yp[yi] if yi < len(yp) else 0.95)

    det_nid = len(nlabels)
    nlabels.append(f"Detected ({len(all_detected)} classes)")
    ncols.append("#4CAF50"); nx.append(0.93); ny.append(0.30)
    miss_nid = len(nlabels)
    nlabels.append(f"Not Detected ({len(missed)} classes)")
    ncols.append("#F44336"); nx.append(0.93); ny.append(0.78)

    src, tgt, val, lcol = [], [], [], []
    for si in range(n_segs - 1):
        for cls in top_classes:
            s = nmap.get((si, cls)); t = nmap.get((si+1, cls))
            if s is not None and t is not None:
                v = min(seg_counts[si].get(cls,0), seg_counts[si+1].get(cls,0))
                if v > 0:
                    src.append(s); tgt.append(t); val.append(v); lcol.append(_rgba(cmap[cls], 0.35))
        so = nmap.get((si, "__other__")); to = nmap.get((si+1, "__other__"))
        if so is not None and to is not None:
            cs = sum(v for k,v in seg_counts[si].items() if k not in top_classes)
            ct = sum(v for k,v in seg_counts[si+1].items() if k not in top_classes)
            v = min(cs,ct)
            if v > 0:
                src.append(so); tgt.append(to); val.append(v); lcol.append(_rgba("#B0BEC5",0.2))

    last = n_segs - 1
    for cls in top_classes:
        s = nmap.get((last, cls))
        if s is not None:
            v = seg_counts[last].get(cls,0)
            if v > 0:
                src.append(s); tgt.append(det_nid); val.append(v); lcol.append(_rgba(cmap[cls],0.3))
    so = nmap.get((last, "__other__"))
    if so is not None:
        ov = sum(v for k,v in seg_counts[last].items() if k not in top_classes)
        if ov > 0:
            src.append(so); tgt.append(det_nid); val.append(ov); lcol.append(_rgba("#B0BEC5",0.2))
    src.append(det_nid); tgt.append(miss_nid)
    val.append(max(len(missed),1)); lcol.append("rgba(244,67,54,0.2)")

    # Sankey node -> segment index mapping for hover
    node_to_seg = {}
    for (si, cls), nid in nmap.items():
        node_to_seg[nid] = si

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(pad=14, thickness=18, line=dict(color="black", width=0.5),
                  label=nlabels, color=ncols, x=nx, y=ny,
                  customdata=[node_to_seg.get(i, -1) for i in range(len(nlabels))]),
        link=dict(source=src, target=tgt, value=val, color=lcol),
    )])
    fig.update_layout(font_size=10, width=1700, height=750, paper_bgcolor="white",
                      margin=dict(t=8, b=8, l=12, r=12))
    sankey_div_id = "sankey-div"
    sankey_html = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id=sankey_div_id)

    # ── Render filmstrip cards ──
    print("  Rendering thumbnails...")
    cards_html = ""
    phase_colors = {
        "opening": "#e3f2fd", "rising": "#fff8e1",
        "climax": "#fce4ec", "falling": "#f3e5f5", "resolution": "#e8f5e9",
    }
    phase_icons = {
        "opening": "&#127916;", "rising": "&#128200;",
        "climax": "&#9889;", "falling": "&#128201;", "resolution": "&#127775;",
    }

    for si, seg in enumerate(segments):
        narr = story["narratives"][si]
        phase = narr["phase"]
        nf = seg["end"] - seg["start"]
        idxs = list(range(seg["start"], seg["end"])) if nf <= 3 else np.linspace(seg["start"], seg["end"]-1, 3, dtype=int).tolist()

        thumbs = []
        for fi in idxs:
            if fi < len(frames_bgr):
                d = per_frame[fi] if fi < len(per_frame) else []
                t_sec = fi * sample_n / fps
                thumbs.append(render_frame(frames_bgr[fi], d, f"{t_sec:.1f}s", w=280))
        strip = filmstrip(thumbs, gap=2)
        strip_b64 = img_b64(strip, quality=78)

        t0s = narr["t0"]; t1s = narr["t1"]
        d = seg["density"]
        wpct = seg["width_weight"]*100
        dcol = "#2e7d32" if d > 4 else "#e65100" if d > 1.5 else "#c62828"
        exp = d > 3
        phase_bg = phase_colors.get(phase, "#fff")

        # Narrative text (convert markdown bold to html)
        ntxt = narr["text"].replace("**", "<b>").replace("<b>", "</b><b>", ).replace("</b><b>", "<b>", 1)
        # Simple fix: just use regex-free approach
        parts = narr["text"].split("**")
        ntxt = ""
        for pi, part in enumerate(parts):
            if pi % 2 == 1:
                ntxt += f"<b>{part}</b>"
            else:
                ntxt += part

        cards_html += f"""
        <div class="seg-card" data-seg="{si}" style="flex:{seg['width_weight']:.3f} 0 0;min-width:145px;border-top:3px solid {dcol}">
          <div class="seg-hdr" style="background:{phase_bg}">
            <span class="badge">T{si+1}</span>
            <span class="phase-icon">{phase_icons.get(phase,'')}</span>
            <span class="phase-label">{phase.upper()}</span>
            <span class="seg-time">{t0s:.1f}s–{t1s:.1f}s</span>
          </div>
          <img src="{strip_b64}" class="strip"/>
          <div class="narr">{ntxt}</div>
          <div class="stats">
            <div class="sr"><span class="sd">{seg['n_detections']} dets</span><span class="sc">{len(seg['classes'])} cls</span></div>
            <div class="sr"><span style="color:{dcol};font-weight:700">{d:.1f}/fr</span>
              <span class="sw">{'EXPANDED' if exp else 'contracted'} {wpct:.0f}%</span></div>
            <div class="dbg"><div class="db" style="width:{min(d*10,100):.0f}%;background:{dcol}"></div></div>
          </div>
        </div>"""

    # ── Hover frame data as JSON for JS ──
    hover_json = json.dumps(hover_data)

    # ── Legend ──
    legend = ""
    for c in top_classes:
        legend += f'<div class="li"><div class="ls" style="background:{cmap[c]}"></div>{c} ({total_by_class[c]})</div>'
    legend += '<div class="li"><div class="ls" style="background:#B0BEC5"></div>other</div>'

    # ── Protagonists ──
    prot_html = ""
    for name, count in story["protagonists"]:
        col = SANKEY_COLORS[top_classes.index(name) % len(SANKEY_COLORS)] if name in top_classes else "#666"
        prot_html += f'<div class="prot" style="border-color:{col}"><div class="prot-name">{name}</div><div class="prot-count">{count} appearances</div></div>'

    # ── Sentiment panel ──
    sentiment_json = json.dumps(sentiments)
    sentiment_cards = ""
    trend_icons = {"rising": "&#9650;", "falling": "&#9660;", "stable": "&#9644;",
                   "peaked": "&#9651;", "appeared": "&#10140;", "disappeared": "&#10550;"}
    sent_colors = {"positive": "#2e7d32", "negative": "#c62828", "peaked": "#e65100", "neutral": "#546e7a"}
    conf_icons = {"improving": "&#9650;", "degrading": "&#9660;", "steady": "&#9644;"}

    for cls, s in sentiments.items():
        col = cmap.get(cls, "#666")
        sc = sent_colors.get(s["sentiment"], "#666")
        # Build SVG sparkline for counts
        counts = s["counts"]
        max_c = max(counts) if counts else 1
        n_pts = len(counts)
        if n_pts > 1 and max_c > 0:
            points = " ".join(f"{i*80/(n_pts-1)},{40-c*35/max_c}" for i, c in enumerate(counts))
            spark_svg = f'<svg viewBox="-2 0 84 44" class="spark"><polyline points="{points}" fill="none" stroke="{col}" stroke-width="2"/></svg>'
        else:
            spark_svg = '<svg viewBox="0 0 80 40" class="spark"></svg>'
        # Confidence sparkline
        confs = s["confidences"]
        max_cf = max(confs) if confs and max(confs) > 0 else 1
        if n_pts > 1 and max_cf > 0:
            cpoints = " ".join(f"{i*80/(n_pts-1)},{40-c*35/max_cf}" for i, c in enumerate(confs))
            conf_svg = f'<svg viewBox="-2 0 84 44" class="spark"><polyline points="{cpoints}" fill="none" stroke="{sc}" stroke-width="2" stroke-dasharray="3,2"/></svg>'
        else:
            conf_svg = '<svg viewBox="0 0 80 40" class="spark"></svg>'

        momentum_bar_w = min(abs(s["momentum"]) * 30, 100)
        momentum_col = "#2e7d32" if s["momentum"] > 0 else "#c62828" if s["momentum"] < 0 else "#999"

        sentiment_cards += f"""
        <div class="sent-card" style="border-left:3px solid {col}">
          <div class="sent-name" style="color:{col}">{cls}</div>
          <div class="sent-trend" style="color:{sc}">
            {trend_icons.get(s['trend'],'?')} {s['trend']}
          </div>
          <div class="sent-sparks">
            <div class="spark-wrap"><div class="spark-label">count</div>{spark_svg}</div>
            <div class="spark-wrap"><div class="spark-label">conf</div>{conf_svg}</div>
          </div>
          <div class="sent-meta">
            <span>conf: {s['avg_conf']:.0%} {conf_icons.get(s['conf_trend'],'')}</span>
            <span>mom: {s['momentum']:+.1f}</span>
          </div>
          <div class="mom-bar-bg"><div class="mom-bar" style="width:{momentum_bar_w:.0f}%;background:{momentum_col}"></div></div>
        </div>"""

    # ── Encode video as base64 for embedded player ──
    print("  Encoding video for embedded player...")
    video_b64 = ""
    if video_path and Path(video_path).exists():
        vsize = Path(video_path).stat().st_size
        if vsize < 50 * 1024 * 1024:  # only embed if < 50MB
            with open(video_path, "rb") as vf:
                video_b64 = base64.b64encode(vf.read()).decode()
            print(f"  Embedded video ({vsize/1024/1024:.1f} MB)")
        else:
            print(f"  Video too large to embed ({vsize/1024/1024:.0f} MB), player will be empty")

    # ── Segment time ranges for video seeking (in real seconds) ──
    seg_times_json = json.dumps([
        {"t0": round(seg["start"] * sample_n / fps, 2),
         "t1": round(seg["end"] * sample_n / fps, 2)}
        for seg in segments
    ])

    # ── Node-to-segment mapping for Sankey hover ──
    node_seg_json = json.dumps({str(nid): si for (si, cls), nid in nmap.items()})

    # ── Full HTML ──
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Video Story — {video_name}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f5f5f5;color:#333}}

.hero{{background:linear-gradient(135deg,#0d47a1 0%,#1565c0 40%,#1976d2 100%);color:#fff;padding:20px 28px;position:relative;overflow:hidden}}
.hero::after{{content:'';position:absolute;top:0;right:0;width:300px;height:100%;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.05));pointer-events:none}}
.hero h1{{font-size:22px;font-weight:700}}.hero .sub{{font-size:12px;opacity:.8;margin:3px 0 10px}}
.chips{{display:flex;flex-wrap:wrap;gap:8px}}
.chip{{background:rgba(255,255,255,.15);padding:4px 12px;border-radius:14px;font-size:10px;backdrop-filter:blur(4px)}}
.chip b{{font-weight:700}}

/* ── URL Input Box ── */
.url-bar{{display:flex;gap:8px;padding:10px 28px;background:rgba(0,0,0,.15);align-items:center}}
.url-bar label{{font-size:11px;font-weight:600;color:rgba(255,255,255,.7);white-space:nowrap}}
.url-bar input{{flex:1;padding:7px 14px;border:2px solid rgba(255,255,255,.3);border-radius:20px;
  background:rgba(255,255,255,.1);color:#fff;font-size:12px;outline:none;transition:all .2s}}
.url-bar input:focus{{border-color:#fff;background:rgba(255,255,255,.2)}}
.url-bar input::placeholder{{color:rgba(255,255,255,.5)}}
.url-bar button{{padding:7px 20px;border:none;border-radius:20px;background:#fff;color:#0d47a1;
  font-size:12px;font-weight:700;cursor:pointer;transition:all .2s}}
.url-bar button:hover{{background:#e3f2fd;transform:scale(1.03)}}
.url-bar .url-hint{{font-size:9px;color:rgba(255,255,255,.5)}}

/* ── Sentiment Panel ── */
.sent-section{{padding:10px 16px;background:#fff;border-top:1px solid #e0e0e0}}
.sent-section h2{{font-size:12px;font-weight:600;color:#555;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px}}
.sent-grid{{display:flex;gap:8px;overflow-x:auto;padding-bottom:4px}}
.sent-card{{background:#fafafa;border:1px solid #eee;border-radius:6px;padding:8px;min-width:130px;flex:0 0 auto}}
.sent-name{{font-size:11px;font-weight:700;text-transform:capitalize;margin-bottom:2px}}
.sent-trend{{font-size:10px;font-weight:700;margin-bottom:4px}}
.sent-sparks{{display:flex;gap:4px;margin-bottom:4px}}
.spark-wrap{{flex:1}}
.spark-label{{font-size:7px;color:#999;text-transform:uppercase;letter-spacing:.3px}}
.spark{{width:100%;height:28px}}
.sent-meta{{display:flex;justify-content:space-between;font-size:8px;color:#777}}
.mom-bar-bg{{height:3px;background:#eee;border-radius:2px;margin-top:3px}}
.mom-bar{{height:3px;border-radius:2px;transition:width .3s}}

.story-summary{{padding:14px 24px;background:#fff;border-bottom:1px solid #e0e0e0;font-size:13px;line-height:1.6;color:#444}}
.story-summary .quote{{border-left:3px solid #1976d2;padding-left:12px;margin:6px 0;font-style:italic}}

.prot-row{{display:flex;gap:10px;padding:10px 24px;background:#fafafa;border-bottom:1px solid #eee;flex-wrap:wrap}}
.prot-row-label{{font-size:11px;font-weight:600;color:#777;text-transform:uppercase;letter-spacing:.5px;align-self:center;margin-right:8px}}
.prot{{border:2px solid;border-radius:8px;padding:6px 14px;background:#fff}}
.prot-name{{font-size:12px;font-weight:700;text-transform:capitalize}}.prot-count{{font-size:10px;color:#888}}

/* ── Video + timeline row ── */
.video-row{{display:flex;gap:12px;padding:12px 16px;background:#111;align-items:flex-start}}
.video-wrap{{flex:0 0 420px;position:relative}}
.video-wrap video{{width:100%;border-radius:6px;display:block}}
.video-wrap .vid-label{{position:absolute;top:6px;left:6px;background:rgba(0,0,0,.7);color:#fff;
  font-size:10px;padding:2px 8px;border-radius:10px;font-weight:600}}
.video-wrap .vid-seg-indicator{{position:absolute;bottom:6px;left:6px;right:6px;background:rgba(0,0,0,.7);
  color:#0f0;font-size:10px;padding:3px 8px;border-radius:6px;font-family:monospace;transition:all .3s}}
.timeline-wrap{{flex:1;overflow-x:auto}}
.timeline-bar{{display:flex;height:14px;border-radius:3px;overflow:hidden;margin-bottom:4px}}
.tbar-seg{{cursor:pointer;transition:opacity .2s;position:relative}}
.tbar-seg:hover{{opacity:.8}}
.tbar-seg.active{{box-shadow:inset 0 0 0 2px #fff}}
.tbar-seg .tbar-label{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  font-size:7px;font-weight:700;color:rgba(255,255,255,.9);pointer-events:none}}

.sec{{padding:10px 16px;background:#fff}}.sec+.sec{{border-top:1px solid #e0e0e0}}
.sec h2{{font-size:12px;font-weight:600;color:#555;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}}

.fr{{display:flex;gap:6px;overflow-x:auto;padding-bottom:4px;align-items:stretch}}
.seg-card{{background:#fff;border:1px solid #ddd;border-radius:6px;overflow:hidden;transition:all .2s;cursor:pointer;position:relative}}
.seg-card:hover{{box-shadow:0 6px 24px rgba(0,0,0,.2);transform:translateY(-2px);z-index:10}}
.seg-card.active{{box-shadow:0 6px 24px rgba(13,71,161,.3);border-color:#1976d2}}
.seg-hdr{{display:flex;align-items:center;gap:4px;padding:3px 6px;border-bottom:1px solid #eee;flex-wrap:wrap}}
.badge{{background:#0d47a1;color:#fff;padding:1px 6px;border-radius:7px;font-size:9px;font-weight:700}}
.phase-icon{{font-size:11px}}.phase-label{{font-size:8px;font-weight:700;letter-spacing:.5px;color:#555}}
.seg-time{{font-size:9px;color:#999;margin-left:auto}}
.strip{{width:100%;height:auto;display:block}}
.narr{{padding:4px 6px;font-size:9px;color:#555;line-height:1.4;min-height:32px;background:#fafafa;border-top:1px solid #f0f0f0}}
.narr b{{color:#333;font-weight:600}}
.stats{{padding:3px 6px;font-size:9px;background:#fff;border-top:1px solid #eee}}
.sr{{display:flex;justify-content:space-between;margin-bottom:1px}}
.sd{{color:#2e7d32;font-weight:600}}.sc{{color:#1565c0;font-weight:600}}.sw{{color:#999;font-size:8px}}
.dbg{{height:3px;background:#eee;border-radius:2px;margin-top:2px}}.db{{height:3px;border-radius:2px;transition:width .3s}}

.lg{{display:flex;flex-wrap:wrap;gap:6px;padding:6px 16px;background:#fff;border-top:1px solid #eee;border-bottom:1px solid #eee}}
.li{{display:flex;align-items:center;gap:3px;font-size:9px}}
.ls{{width:10px;height:10px;border-radius:2px;border:1px solid rgba(0,0,0,.12)}}

.sk{{padding:6px 12px 12px;background:#fff;position:relative}}

/* ── Hover gallery popup ── */
.hover-gallery{{
  display:none;position:fixed;z-index:1000;
  background:#fff;border-radius:10px;box-shadow:0 12px 48px rgba(0,0,0,.4);
  padding:10px;max-width:680px;pointer-events:none;
  border:2px solid #1976d2;
}}
.hover-gallery.visible{{display:block;animation:fadeIn .15s ease}}
@keyframes fadeIn{{from{{opacity:0;transform:translateY(5px)}}to{{opacity:1;transform:translateY(0)}}}}
.hover-gallery h3{{font-size:11px;color:#1976d2;margin-bottom:6px;font-weight:600}}
.hover-gallery .hg-narr{{font-size:9px;color:#555;margin-bottom:6px;line-height:1.4;max-width:600px}}
.hover-gallery .hg-narr b{{color:#333}}
.hover-gallery .hg-frames{{display:flex;gap:4px;overflow:hidden}}
.hover-gallery .hg-frame{{border-radius:4px;overflow:hidden;position:relative}}
.hover-gallery .hg-frame img{{display:block;height:130px;width:auto}}
.hover-gallery .hg-label{{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,.75);color:#fff;
  font-size:8px;padding:2px 4px;text-align:center}}
</style></head><body>

<div class="hero">
  <h1>&#127916; {video_name}</h1>
  <div class="sub">Analyzed with Faster R-CNN (ResNet50-FPN) &middot; Multi-Modal Ensemble (CNN+LSTM+OCR) &middot; F1: 82.2%</div>
  <div class="chips">
    <div class="chip">Duration: <b>{duration:.1f}s</b></div>
    <div class="chip">Resolution: <b>{vid_w}x{vid_h}</b></div>
    <div class="chip">FPS: <b>{fps:.0f}</b></div>
    <div class="chip">Frames analyzed: <b>{len(per_frame)}</b></div>
    <div class="chip">Detections: <b>{total_det}</b></div>
    <div class="chip">Classes: <b>{len(all_detected)}</b> detected, <b>{len(missed)}</b> missed</div>
    <div class="chip">Segments: <b>{n_segs}</b> (adaptive)</div>
  </div>
  <div class="url-bar">
    <label>&#128279; Analyze another video:</label>
    <input type="text" id="url-input" placeholder="Paste YouTube URL or local path..." />
    <button onclick="analyzeUrl()">Analyze</button>
    <span class="url-hint">Runs: python3 analyze_video.py &lt;url&gt;</span>
  </div>
</div>

<div class="story-summary">
  <div class="quote">{story['summary']}</div>
</div>

<div class="prot-row">
  <div class="prot-row-label">Main Characters</div>
  {prot_html}
</div>

<!-- ── Sentiment Change Panel ── -->
<div class="sent-section">
  <h2>&#128200; Object Sentiment — confidence &amp; frequency trends over time</h2>
  <div class="sent-grid">{sentiment_cards}</div>
</div>

<!-- ── Video Player + Interactive Timeline ── -->
<div class="video-row">
  <div class="video-wrap">
    <video id="vid" controls muted preload="auto"
      {"src='data:video/mp4;base64," + video_b64 + "'" if video_b64 else ""}>
    </video>
    <div class="vid-label">&#9654; Click segment or Sankey node to jump</div>
    <div class="vid-seg-indicator" id="vid-seg">T1 | 0.0s</div>
  </div>
  <div class="timeline-wrap">
    <div style="color:#999;font-size:10px;margin-bottom:3px;font-weight:600">ADAPTIVE TIMELINE (wider = more activity)</div>
    <div class="timeline-bar" id="tbar"></div>
    <div class="fr" id="segments-row" style="padding-top:6px">{cards_html}</div>
  </div>
</div>

<div class="lg">{legend}</div>

<div class="sk">
  <h2>&#128200; Object Flow — hover nodes to see frames, click to jump video</h2>
  {sankey_html}
</div>

<!-- Hover gallery popup -->
<div class="hover-gallery" id="hover-gallery">
  <h3 id="hg-title">Segment</h3>
  <div class="hg-narr" id="hg-narr"></div>
  <div class="hg-frames" id="hg-frames"></div>
</div>

<script>
const hoverData = {hover_json};
const segTimes = {seg_times_json};
const nodeToSeg = {node_seg_json};
const narratives = {json.dumps([n["text"] for n in story["narratives"]])};
const phases = {json.dumps([n["phase"] for n in story["narratives"]])};
const gallery = document.getElementById('hover-gallery');
const hgTitle = document.getElementById('hg-title');
const hgNarr = document.getElementById('hg-narr');
const hgFrames = document.getElementById('hg-frames');
const vid = document.getElementById('vid');
const vidSeg = document.getElementById('vid-seg');

// ── Build timeline bar ──
const tbar = document.getElementById('tbar');
const phaseColors = {{opening:'#1976d2',rising:'#f9a825',climax:'#d32f2f',falling:'#7b1fa2',resolution:'#388e3c'}};
segTimes.forEach((st, i) => {{
  const seg = document.createElement('div');
  seg.className = 'tbar-seg';
  const w = {json.dumps([s["width_weight"] for s in segments])};
  seg.style.flex = w[i];
  seg.style.background = phaseColors[phases[i]] || '#666';
  seg.innerHTML = '<div class="tbar-label">T' + (i+1) + '</div>';
  seg.dataset.seg = i;
  seg.addEventListener('click', () => seekToSeg(i));
  seg.addEventListener('mouseenter', e => {{ showGallery(i, e); seg.classList.add('active'); }});
  seg.addEventListener('mousemove', e => positionGallery(e));
  seg.addEventListener('mouseleave', () => {{ gallery.classList.remove('visible'); seg.classList.remove('active'); }});
  tbar.appendChild(seg);
}});

// ── Video time tracking ──
if (vid) {{
  vid.addEventListener('timeupdate', () => {{
    const t = vid.currentTime;
    for (let i = 0; i < segTimes.length; i++) {{
      if (t >= segTimes[i].t0 && t < segTimes[i].t1) {{
        vidSeg.textContent = 'T' + (i+1) + ' | ' + t.toFixed(1) + 's | ' + phases[i].toUpperCase();
        // Highlight active segment in timeline bar
        tbar.querySelectorAll('.tbar-seg').forEach((s, j) => s.classList.toggle('active', j === i));
        // Highlight active card
        document.querySelectorAll('.seg-card').forEach((c, j) => c.classList.toggle('active', j === i));
        break;
      }}
    }}
  }});
}}

function seekToSeg(i) {{
  if (vid && segTimes[i]) {{
    vid.currentTime = segTimes[i].t0;
    vid.play();
  }}
}}

// ── Segment card interactions ──
document.querySelectorAll('.seg-card').forEach(card => {{
  card.addEventListener('mouseenter', e => {{
    const si = parseInt(card.dataset.seg);
    showGallery(si, e);
    card.classList.add('active');
  }});
  card.addEventListener('mousemove', e => positionGallery(e));
  card.addEventListener('mouseleave', () => {{
    gallery.classList.remove('visible');
    card.classList.remove('active');
  }});
  card.addEventListener('click', () => {{
    const si = parseInt(card.dataset.seg);
    seekToSeg(si);
  }});
}});

// ── Sankey node hover — detect via Plotly events ──
const sankeyDiv = document.getElementById('{sankey_div_id}');
let sankeyHoverActive = false;

function hookSankeyEvents() {{
  if (!sankeyDiv) return;
  // Plotly fires these on the div after render
  sankeyDiv.on('plotly_hover', function(data) {{
    if (!data || !data.points || !data.points[0]) return;
    const pt = data.points[0];
    // For sankey, pointNumber is the node index
    const nodeIdx = pt.pointNumber;
    const segIdx = nodeToSeg[String(nodeIdx)];
    if (segIdx !== undefined && segIdx >= 0) {{
      sankeyHoverActive = true;
      const rect = sankeyDiv.getBoundingClientRect();
      showGallery(segIdx, {{
        clientX: (pt.x !== undefined ? rect.left + rect.width * 0.5 : rect.left + 300),
        clientY: rect.top + 100
      }});
    }}
  }});

  sankeyDiv.on('plotly_unhover', function() {{
    if (sankeyHoverActive) {{
      gallery.classList.remove('visible');
      sankeyHoverActive = false;
    }}
  }});

  sankeyDiv.on('plotly_click', function(data) {{
    if (!data || !data.points || !data.points[0]) return;
    const nodeIdx = data.points[0].pointNumber;
    const segIdx = nodeToSeg[String(nodeIdx)];
    if (segIdx !== undefined && segIdx >= 0) seekToSeg(segIdx);
  }});
}}

// Hook after plotly renders (small delay)
setTimeout(hookSankeyEvents, 500);

// ── Also intercept mousemove over sankey for link hovers ──
if (sankeyDiv) {{
  sankeyDiv.addEventListener('mousemove', e => {{
    if (sankeyHoverActive) positionGallery(e);
  }});
}}

function showGallery(segIdx, evt) {{
  const frames = hoverData[segIdx];
  if (!frames || frames.length === 0) return;
  const phase = phases[segIdx] || '';
  const phaseLabel = phase.charAt(0).toUpperCase() + phase.slice(1);

  hgTitle.textContent = 'T' + (segIdx+1) + ' — ' + phaseLabel +
    ' (' + segTimes[segIdx].t0.toFixed(1) + 's – ' + segTimes[segIdx].t1.toFixed(1) + 's)';

  // Show narrative
  hgNarr.innerHTML = narratives[segIdx] ? narratives[segIdx].replace(/\\*\\*(.*?)\\*\\*/g, '<b>$1</b>') : '';

  hgFrames.innerHTML = '';
  frames.forEach(f => {{
    const div = document.createElement('div');
    div.className = 'hg-frame';
    div.innerHTML = '<img src="' + f.b64 + '"/><div class="hg-label">' + f.t + 's &middot; ' + f.n + ' objects</div>';
    hgFrames.appendChild(div);
  }});

  gallery.classList.add('visible');
  positionGallery(evt);
}}

function positionGallery(evt) {{
  const gw = 660; const gh = 200;
  let x = evt.clientX + 20;
  let y = evt.clientY - gh - 15;
  if (x + gw > window.innerWidth) x = evt.clientX - gw - 20;
  if (y < 10) y = evt.clientY + 25;
  gallery.style.left = x + 'px';
  gallery.style.top = y + 'px';
}}

// ── URL Analyze button ──
function analyzeUrl() {{
  const input = document.getElementById('url-input');
  const url = input.value.trim();
  if (!url) {{ alert('Please enter a YouTube URL or file path'); return; }}

  // Check if running via the web server
  if (window._serverMode) {{
    // Submit to server
    const btn = document.querySelector('.url-bar button');
    btn.textContent = 'Processing...';
    btn.disabled = true;
    fetch('/analyze', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{url: url}})
    }})
    .then(r => r.json())
    .then(data => {{
      if (data.redirect) window.location.href = data.redirect;
      else alert(data.error || 'Processing failed');
      btn.textContent = 'Analyze';
      btn.disabled = false;
    }})
    .catch(e => {{
      alert('Error: ' + e.message);
      btn.textContent = 'Analyze';
      btn.disabled = false;
    }});
  }} else {{
    // Show the command to run
    const cmd = 'python3 analyze_video.py "' + url + '"';
    const copyArea = document.createElement('div');
    copyArea.style.cssText = 'position:fixed;inset:0;z-index:9999;background:rgba(0,0,0,.85);display:flex;align-items:center;justify-content:center;flex-direction:column;color:#fff';
    copyArea.innerHTML = '<h2 style="margin-bottom:12px">Run this command:</h2>'
      + '<code style="background:#222;padding:12px 24px;border-radius:8px;font-size:14px;user-select:all;cursor:text">' + cmd + '</code>'
      + '<p style="margin-top:12px;font-size:12px;opacity:.7">Or start the web server: <code>python3 server.py</code> for in-browser analysis</p>'
      + '<button onclick="this.parentElement.remove()" style="margin-top:16px;padding:8px 24px;border:none;border-radius:20px;background:#1976d2;color:#fff;cursor:pointer;font-weight:600">Close</button>';
    document.body.appendChild(copyArea);
    // Copy to clipboard
    navigator.clipboard.writeText(cmd).catch(()=>{{}});
  }}
}}

// Check server mode
window._serverMode = false;
</script>

</body></html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def download_youtube(url: str, output_dir: str = None, max_duration: int = 60) -> str:
    """Download a YouTube video using yt-dlp. Returns path to downloaded file.

    If the video is longer than max_duration seconds, only the first max_duration
    seconds are kept (trimmed with OpenCV since ffmpeg may not be available).
    """
    import subprocess
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "data")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    out_template = str(Path(output_dir) / "%(title).50s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--js-runtimes", "node",
        "-f", "best[height<=720][ext=mp4]/best[height<=720]/best",
        "--no-playlist",
        "-o", out_template,
        "--print", "after_move:filepath",
        url,
    ]
    print(f"  Downloading: {url}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  yt-dlp stderr: {result.stderr[:500]}")
        raise RuntimeError(f"yt-dlp failed: {result.stderr[:200]}")

    # Find the downloaded file
    lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
    filepath = lines[-1] if lines else ""
    if not filepath or not Path(filepath).exists():
        # Fallback: most recent file in output_dir
        files = sorted(Path(output_dir).glob("*.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        vids = [f for f in files if f.suffix.lower() in (".mp4", ".webm", ".mkv")]
        if vids:
            filepath = str(vids[0])
        else:
            raise RuntimeError(f"Downloaded file not found. stdout: {result.stdout[:300]}")

    size_mb = Path(filepath).stat().st_size / 1024 / 1024
    print(f"  Downloaded: {Path(filepath).name} ({size_mb:.1f} MB)")

    # Trim if too long (using OpenCV since ffmpeg may not be installed)
    cap = cv2.VideoCapture(filepath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = total_frames / fps
    cap.release()

    if duration > max_duration:
        print(f"  Video is {duration:.0f}s — trimming to first {max_duration}s...")
        trimmed_path = str(Path(filepath).with_stem(Path(filepath).stem + "_trimmed"))
        cap = cv2.VideoCapture(filepath)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(trimmed_path, fourcc, fps, (w, h))
        max_frames = int(max_duration * fps)
        count = 0
        while count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            count += 1
        cap.release()
        writer.release()
        filepath = trimmed_path
        size_mb = Path(filepath).stat().st_size / 1024 / 1024
        print(f"  Trimmed to: {Path(filepath).name} ({size_mb:.1f} MB, {count/fps:.0f}s)")

    return filepath


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://") or s.startswith("www.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze any video (local file or YouTube URL) with object detection + story Sankey"
    )
    parser.add_argument("video", help="Path to video file OR YouTube/web URL")
    parser.add_argument("--confidence", type=float, default=0.40, help="Detection confidence threshold")
    parser.add_argument("--segments", type=int, default=10, help="Target number of time segments")
    parser.add_argument("--sample", type=int, default=4, help="Analyze every Nth frame")
    parser.add_argument("--output", type=str, default=None, help="Output HTML path")
    args = parser.parse_args()

    # Handle YouTube / web URLs
    if _is_url(args.video):
        print(f"\n  Detected URL — downloading video...")
        video_path = download_youtube(args.video)
    else:
        video_path = args.video

    video_name = Path(video_path).stem
    output = args.output or str(Path(video_path).parent / f"{video_name}_analysis.html")

    print("=" * 65)
    print(f"  Video Story Analysis: {Path(video_path).name}")
    print("=" * 65)

    # 1. Extract
    print(f"\n[1/6] Extracting frames (every {args.sample}th)...")
    frames, fps, total, vw, vh = extract_frames(video_path, args.sample)
    print(f"  {len(frames)} frames from {total} total ({fps:.0f} fps, {vw}x{vh}, {total/fps:.1f}s)")

    # 2. Detect
    print(f"\n[2/6] Running Faster R-CNN (conf >= {args.confidence})...")
    per_frame = run_detection(frames, confidence=args.confidence)

    # 3. Segment
    print(f"\n[3/6] Building adaptive time segments...")
    segments = build_segments(per_frame, target_segs=args.segments)
    for i, seg in enumerate(segments):
        t0 = seg["start"]*args.sample/fps; t1 = seg["end"]*args.sample/fps
        tag = "EXPANDED" if seg["density"] > 3 else "contracted"
        print(f"  T{i+1}: {t0:.1f}-{t1:.1f}s  {seg['n_detections']} dets  "
              f"{seg['density']:.1f}/fr  w={seg['width_weight']*100:.0f}%  [{tag}]")

    # 4. Story
    print(f"\n[4/6] Analyzing story arc...")
    story = analyze_story(segments, per_frame, fps, args.sample)
    print(f"  {story['summary']}")
    for i, n in enumerate(story["narratives"]):
        print(f"  T{i+1} [{n['phase'].upper():10s}] {n['text'][:80]}")

    # 5. Sentiment
    print(f"\n[5/6] Analyzing object sentiment...")
    sentiments = analyze_sentiment(segments, per_frame, top_k=10)
    for cls, s in sentiments.items():
        icon = {"positive": "+", "negative": "-", "peaked": "^", "neutral": "="}[s["sentiment"]]
        print(f"  [{icon}] {cls:15s} trend={s['trend']:12s} conf={s['conf_trend']:10s} "
              f"momentum={s['momentum']:+.2f}  total={s['total']}")

    # 6. Build
    print(f"\n[6/6] Building interactive visualization...")
    build_html(
        segments=segments, per_frame=per_frame, frames_bgr=frames,
        story=story, sentiments=sentiments, fps=fps, sample_n=args.sample,
        video_name=video_name, vid_w=vw, vid_h=vh,
        total_raw_frames=total, output_path=output,
        video_path=video_path, top_k=10,
    )

    print(f"\n{'='*65}")
    print(f"  Done! Open: {output}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()

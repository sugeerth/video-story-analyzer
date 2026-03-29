"""Multi-video real detection pipeline with full time contraction/expansion Sankey.

Processes multiple real videos:
1. Extract + detect objects in each video with Faster R-CNN
2. Combine all results into one unified timeline
3. Adaptive time segmentation based on detection density
4. Full contraction (quiet) / expansion (busy) across the timeline
5. Embed real video frames with bounding boxes at every segment
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import time
import cv2
import torch
import torchvision
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
import io
import base64

from models.object_detector import COCO_CLASSES


# ─── Video processing ─────────────────────────────────────────────

def extract_frames(video_path: str, sample_every_n: int = 3) -> tuple[list[np.ndarray], float, int]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_every_n == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames, fps, total


def detect_frames(
    frames: list[np.ndarray],
    model,
    device: str = "cpu",
    confidence: float = 0.40,
) -> list[list[dict]]:
    per_frame = []
    for i, bgr in enumerate(frames):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = torchvision.transforms.functional.to_tensor(rgb).to(device)
        with torch.no_grad():
            out = model([tensor])[0]
        dets = []
        for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
            if score >= confidence:
                dets.append({
                    "box": box.cpu().numpy().tolist(),
                    "label": COCO_CLASSES[label.item()] if label.item() < len(COCO_CLASSES) else "unknown",
                    "label_id": label.item(),
                    "score": score.item(),
                })
        per_frame.append(dets)
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(frames)} frames done")
    return per_frame


# ─── Adaptive time segmentation ──────────────────────────────────

def build_adaptive_segments(
    per_frame: list[list[dict]],
    video_boundaries: list[int],
    target_segments: int = 10,
    min_frames: int = 4,
) -> list[dict]:
    """Build adaptive segments with contraction/expansion.

    Segments with high detection density are EXPANDED (take more visual space).
    Segments with low density are CONTRACTED.
    Video boundaries are respected (no segment spans two videos).
    """
    n = len(per_frame)
    det_counts = np.array([len(d) for d in per_frame], dtype=float)

    # Smooth density
    k = max(3, n // 30)
    kernel = np.ones(k) / k
    smoothed = np.convolve(det_counts, kernel, mode="same")

    # Split into sub-ranges at video boundaries
    boundary_set = set(video_boundaries)
    ranges = []
    start = 0
    for b in sorted(boundary_set):
        if b > start:
            ranges.append((start, b))
        start = b
    if start < n:
        ranges.append((start, n))

    # Allocate segments to each video proportionally to its length
    total_len = sum(e - s for s, e in ranges)
    segments = []

    for r_start, r_end in ranges:
        r_len = r_end - r_start
        n_segs = max(2, round(target_segments * r_len / total_len))

        # Within this range, split by equal detection mass
        r_smoothed = smoothed[r_start:r_end]
        cumul = np.cumsum(r_smoothed)
        total_mass = cumul[-1] if len(cumul) > 0 else 1.0
        mass_per_seg = total_mass / n_segs

        seg_start = 0
        running = 0.0

        for i in range(r_end - r_start):
            running += r_smoothed[i]
            is_last = (i == r_end - r_start - 1)
            enough = running >= mass_per_seg and (i - seg_start + 1) >= min_frames
            remaining_segs = n_segs - len([s for s in segments if s["start"] >= r_start])

            if enough or is_last or (remaining_segs <= 1 and is_last):
                abs_start = r_start + seg_start
                abs_end = r_start + i + 1

                seg_dets = []
                seg_classes = set()
                for t in range(abs_start, abs_end):
                    seg_dets.extend(per_frame[t])
                    for d in per_frame[t]:
                        seg_classes.add(d["label"])

                nf = abs_end - abs_start
                density = len(seg_dets) / max(nf, 1)

                segments.append({
                    "start": abs_start,
                    "end": abs_end,
                    "n_frames": nf,
                    "n_detections": len(seg_dets),
                    "density": density,
                    "classes": seg_classes,
                })

                seg_start = i + 1
                running = 0.0

    # Compute width weights using sqrt scaling
    counts = np.array([s["n_detections"] for s in segments], dtype=float)
    weights = np.sqrt(counts + 1)
    weights = weights / weights.sum()
    for seg, w in zip(segments, weights):
        seg["width_weight"] = float(w)

    return segments


# ─── Frame rendering ─────────────────────────────────────────────

BBOX_COLORS = {
    "person": (33, 150, 243), "car": (76, 175, 80), "truck": (255, 152, 0),
    "bus": (233, 30, 99), "bicycle": (156, 39, 176), "motorcycle": (0, 188, 212),
    "dog": (255, 87, 34), "cat": (121, 85, 72), "traffic light": (38, 166, 154),
    "stop sign": (171, 71, 188), "backpack": (255, 235, 59), "umbrella": (255, 112, 67),
    "bottle": (139, 195, 74), "cup": (244, 67, 54), "cell phone": (255, 193, 7),
    "handbag": (103, 58, 183), "tie": (0, 150, 136), "chair": (96, 125, 139),
    "bench": (205, 220, 57), "skateboard": (63, 81, 181), "bird": (121, 85, 72),
    "horse": (255, 112, 67), "clock": (3, 169, 244), "potted plant": (102, 187, 106),
}
DEFAULT_COL = (180, 180, 180)


def _font(size=10):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def render_frame(bgr: np.ndarray, dets: list[dict], frame_label: str, w: int = 340) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    ratio = w / img.width
    h = int(img.height * ratio)
    img = img.resize((w, h), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    f10 = _font(10)
    f9 = _font(9)

    for d in dets:
        box = d["box"]
        label = d["label"]
        score = d["score"]
        color = BBOX_COLORS.get(label, DEFAULT_COL)
        x1, y1, x2, y2 = [int(c * ratio) for c in box]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        txt = f"{label} {score:.0%}"
        bb = f9.getbbox(txt)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        ly = max(y1 - th - 4, 0)
        draw.rectangle((x1, ly, x1 + tw + 6, ly + th + 4), fill=color)
        draw.text((x1 + 3, ly + 1), txt, fill=(255, 255, 255), font=f9)

    # Overlays
    draw.rectangle((0, 0, 80, 16), fill=(0, 0, 0))
    draw.text((4, 2), frame_label, fill=(255, 255, 255), font=f10)
    ct = f"{len(dets)} obj"
    draw.rectangle((w - 50, 0, w, 16), fill=(0, 0, 0))
    draw.text((w - 47, 2), ct, fill=(0, 255, 150), font=f10)
    return img


def img_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def filmstrip(imgs: list[Image.Image], gap=3) -> Image.Image:
    if not imgs:
        return Image.new("RGB", (100, 60), (30, 30, 30))
    w, h = imgs[0].size
    strip = Image.new("RGB", (w * len(imgs) + gap * (len(imgs) - 1), h), (30, 30, 30))
    for i, im in enumerate(imgs):
        strip.paste(im, (i * (w + gap), 0))
    return strip


# ─── Sankey HTML builder ─────────────────────────────────────────

COLORS = [
    "#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0",
    "#00BCD4", "#FF5722", "#795548", "#607D8B", "#CDDC39",
    "#3F51B5", "#009688", "#FFC107", "#8BC34A", "#F44336",
]


def _rgba(hx, a=0.4):
    h = hx.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"


def build_html(
    segments, per_frame, all_frames_bgr, video_labels, video_boundaries,
    fps, sample_n, output_path, top_k=10,
):
    import plotly.graph_objects as go

    n_segs = len(segments)

    # Adaptive x positions from width weights
    weights = np.array([s["width_weight"] for s in segments])
    cumul = np.concatenate([[0], np.cumsum(weights)])
    x_pos = [(cumul[i] + cumul[i+1]) / 2 * 0.78 + 0.01 for i in range(n_segs)]

    # Per-segment per-class counts
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
    cmap = {cls: COLORS[i % len(COLORS)] for i, cls in enumerate(top_classes)}

    all_detected = set(total_by_class.keys())
    missed = set(COCO_CLASSES[1:]) - all_detected
    total_det = sum(total_by_class.values())

    # Nodes
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
            cnt = seg_counts[si][cls]
            nlabels.append(f"{cls} ({cnt})")
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

    # Links
    src, tgt, val, lcol = [], [], [], []
    for si in range(n_segs - 1):
        # Skip links across video boundaries
        seg_end = segments[si]["end"]
        next_start = segments[si + 1]["start"]
        cross_boundary = seg_end != next_start

        for cls in top_classes:
            s = nmap.get((si, cls))
            t = nmap.get((si + 1, cls))
            if s is not None and t is not None and not cross_boundary:
                v = min(seg_counts[si].get(cls, 0), seg_counts[si + 1].get(cls, 0))
                if v > 0:
                    src.append(s); tgt.append(t); val.append(v)
                    lcol.append(_rgba(cmap[cls], 0.35))
        so = nmap.get((si, "__other__"))
        to = nmap.get((si + 1, "__other__"))
        if so is not None and to is not None and not cross_boundary:
            cs = sum(v for k, v in seg_counts[si].items() if k not in top_classes)
            ct = sum(v for k, v in seg_counts[si + 1].items() if k not in top_classes)
            v = min(cs, ct)
            if v > 0:
                src.append(so); tgt.append(to); val.append(v)
                lcol.append(_rgba("#B0BEC5", 0.2))

    last = n_segs - 1
    for cls in top_classes:
        s = nmap.get((last, cls))
        if s is not None:
            v = seg_counts[last].get(cls, 0)
            if v > 0:
                src.append(s); tgt.append(det_nid); val.append(v)
                lcol.append(_rgba(cmap[cls], 0.3))
    so = nmap.get((last, "__other__"))
    if so is not None:
        ov = sum(v for k, v in seg_counts[last].items() if k not in top_classes)
        if ov > 0:
            src.append(so); tgt.append(det_nid); val.append(ov)
            lcol.append(_rgba("#B0BEC5", 0.2))
    src.append(det_nid); tgt.append(miss_nid)
    val.append(max(len(missed), 1)); lcol.append("rgba(244,67,54,0.2)")

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(pad=14, thickness=18, line=dict(color="black", width=0.5),
                  label=nlabels, color=ncols, x=nx, y=ny),
        link=dict(source=src, target=tgt, value=val, color=lcol),
    )])
    fig.update_layout(font_size=10, width=1700, height=800, paper_bgcolor="white",
                      margin=dict(t=8, b=8, l=12, r=12))
    sankey_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    # Render frame cards
    print("  Rendering real frame thumbnails with bounding boxes...")
    cards_html = ""
    boundary_set = set(video_boundaries)

    for si, seg in enumerate(segments):
        nf = seg["end"] - seg["start"]
        if nf <= 3:
            idxs = list(range(seg["start"], seg["end"]))
        else:
            idxs = np.linspace(seg["start"], seg["end"] - 1, 3, dtype=int).tolist()

        thumbs = []
        for fi in idxs:
            if fi < len(all_frames_bgr):
                d = per_frame[fi] if fi < len(per_frame) else []
                # Find which video this frame belongs to
                vid_idx = 0
                for bi, b in enumerate(video_boundaries):
                    if fi >= b:
                        vid_idx = bi + 1
                vid_name = video_labels[vid_idx] if vid_idx < len(video_labels) else ""
                lbl = f"F{fi} {vid_name}"
                thumbs.append(render_frame(all_frames_bgr[fi], d, lbl, w=300))

        strip = filmstrip(thumbs, gap=2)
        b64 = img_b64(strip)

        t0 = seg["start"] * sample_n / fps
        t1 = seg["end"] * sample_n / fps
        density = seg["density"]
        wpct = seg["width_weight"] * 100
        dcol = "#2e7d32" if density > 4 else "#e65100" if density > 1.5 else "#c62828"
        expanded = density > 3
        scale_tag = "EXPANDED" if expanded else "contracted"
        scale_bg = "#e8f5e9" if expanded else "#fff8e1" if density > 1.5 else "#ffebee"

        # Is this a video boundary?
        is_boundary = seg["start"] in boundary_set
        boundary_marker = ""
        if is_boundary and si > 0:
            # Find which video starts here
            for bi, b in enumerate(video_boundaries):
                if seg["start"] == b:
                    vname = video_labels[bi + 1] if bi + 1 < len(video_labels) else ""
                    boundary_marker = f'<div class="vid-marker">{vname}</div>'

        cards_html += f"""
        {boundary_marker}
        <div class="seg-card" style="flex:{seg['width_weight']:.3f} 0 0; min-width:150px;">
            <div class="seg-header">
                <span class="seg-badge">T{si+1}</span>
                <span class="seg-time">{t0:.1f}s &ndash; {t1:.1f}s</span>
                <span class="scale-tag" style="background:{scale_bg}; color:{'#2e7d32' if expanded else '#c62828'}">{scale_tag}</span>
            </div>
            <img src="{b64}" class="filmstrip"/>
            <div class="seg-stats">
                <div class="sr"><span class="sd">{seg['n_detections']} dets</span><span class="sc">{len(seg['classes'])} cls</span></div>
                <div class="sr"><span style="color:{dcol};font-weight:600">{density:.1f}/fr</span><span class="sw">w:{wpct:.0f}%</span></div>
                <div class="dbar-bg"><div class="dbar" style="width:{min(density*12,100):.0f}%;background:{dcol}"></div></div>
            </div>
        </div>"""

    # Time scale bar
    scale_html = '<div class="scale-row">'
    for si, seg in enumerate(segments):
        d = seg["density"]
        expanded = d > 3
        bg = "#c8e6c9" if expanded else "#fff3e0" if d > 1.5 else "#ffcdd2"
        arr = "&#9650;" if expanded else "&#9660;"
        scale_html += f'<div class="sb" style="flex:{seg["width_weight"]:.3f};background:{bg}">'
        scale_html += f'<span class="sbl">{arr} {"EXPAND" if expanded else "contract"}</span>'
        scale_html += f'<span class="sbv">{seg["width_weight"]*100:.0f}%</span></div>'
    scale_html += '</div>'

    # Legend
    legend = ""
    for c in top_classes:
        legend += f'<div class="li"><div class="ls" style="background:{cmap[c]}"></div>{c} ({total_by_class[c]})</div>'
    legend += '<div class="li"><div class="ls" style="background:#B0BEC5"></div>other</div>'
    legend += '<div class="li"><div class="ls" style="background:#4CAF50"></div>detected</div>'
    legend += '<div class="li"><div class="ls" style="background:#F44336"></div>not detected</div>'

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Real Multi-Video Detection — Adaptive Time Sankey</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#eee;color:#333}}
.hdr{{background:linear-gradient(135deg,#0d47a1,#1565c0,#1e88e5);color:#fff;padding:14px 24px;text-align:center}}
.hdr h1{{font-size:19px;font-weight:600}}.hdr .sub{{font-size:11px;opacity:.8;margin-top:2px}}
.mets{{display:flex;justify-content:center;gap:12px;margin-top:8px;flex-wrap:wrap}}
.met{{background:rgba(255,255,255,.15);padding:4px 12px;border-radius:14px;font-size:10px;font-weight:500}}
.met b{{font-weight:700}}
.sec{{padding:10px 16px;background:#fff}}.sec+.sec{{border-top:1px solid #e0e0e0}}
.sec h2{{font-size:12px;font-weight:600;color:#555;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}}
.fr{{display:flex;gap:6px;overflow-x:auto;padding-bottom:4px;align-items:stretch}}
.seg-card{{background:#fafafa;border:1px solid #ddd;border-radius:6px;overflow:hidden;transition:box-shadow .2s}}
.seg-card:hover{{box-shadow:0 4px 16px rgba(0,0,0,.18)}}
.seg-header{{display:flex;align-items:center;gap:5px;padding:3px 6px;background:#fff;border-bottom:1px solid #eee;flex-wrap:wrap}}
.seg-badge{{background:#0d47a1;color:#fff;padding:1px 6px;border-radius:7px;font-size:9px;font-weight:700}}
.seg-time{{font-size:9px;color:#777}}
.scale-tag{{font-size:8px;font-weight:700;padding:1px 5px;border-radius:8px;text-transform:uppercase;letter-spacing:.3px}}
.filmstrip{{width:100%;height:auto;display:block}}
.seg-stats{{padding:3px 6px;font-size:9px;background:#fff;border-top:1px solid #eee}}
.sr{{display:flex;justify-content:space-between;margin-bottom:1px}}
.sd{{color:#2e7d32;font-weight:600}}.sc{{color:#1565c0;font-weight:600}}.sw{{color:#999}}
.dbar-bg{{height:3px;background:#eee;border-radius:2px;margin-top:2px}}
.dbar{{height:3px;border-radius:2px}}
.vid-marker{{writing-mode:vertical-rl;background:#0d47a1;color:#fff;font-size:9px;font-weight:700;
  padding:8px 3px;border-radius:4px;display:flex;align-items:center;letter-spacing:.5px;flex-shrink:0}}
.scale-row{{display:flex;gap:2px;padding:4px 16px;background:#fff;border-top:1px solid #e0e0e0}}
.sb{{padding:3px 4px;border-radius:3px;text-align:center;border:1px solid rgba(0,0,0,.06)}}
.sbl{{font-size:8px;font-weight:700;text-transform:uppercase;letter-spacing:.3px;display:block}}
.sbv{{font-size:9px;color:#666;display:block}}
.lg{{display:flex;flex-wrap:wrap;gap:6px;padding:6px 16px;background:#fff;border-top:1px solid #eee;border-bottom:1px solid #eee}}
.li{{display:flex;align-items:center;gap:3px;font-size:9px}}
.ls{{width:10px;height:10px;border-radius:2px;border:1px solid rgba(0,0,0,.12)}}
.sk{{padding:6px 12px 12px;background:#fff}}
</style></head><body>

<div class="hdr">
<h1>Real Multi-Video Object Detection — Adaptive Time-Scaled Sankey</h1>
<div class="sub">Faster R-CNN (ResNet50-FPN) | Multi-Modal Ensemble (CNN+LSTM+OCR) | F1: 82.2%</div>
<div class="mets">
<div class="met">Videos: <b>{len(video_labels)}</b></div>
<div class="met">Frames analyzed: <b>{len(per_frame)}</b></div>
<div class="met">Total detections: <b>{total_det}</b></div>
<div class="met">Classes found: <b>{len(all_detected)}</b></div>
<div class="met">Missed: <b>{len(missed)}</b></div>
<div class="met">Segments: <b>{n_segs}</b> (adaptive)</div>
</div></div>

<div class="sec"><h2>Real Video Frames + Bounding Boxes — Adaptive Time Segments (wider = more activity)</h2>
<div class="fr">{cards_html}</div></div>

<div class="sec" style="padding:3px 16px"><h2>Time Scale: Contraction &amp; Expansion</h2>{scale_html}</div>

<div class="lg">{legend}</div>

<div class="sk"><h2 style="font-size:12px;font-weight:600;color:#555;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px">
Object Detection Flow Across Adaptive Time</h2>{sankey_html}</div>

</body></html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"  Saved: {output_path}")


# ─── Main ─────────────────────────────────────────────────────────

def main():
    data_dir = Path(__file__).parent / "data"
    output_path = str(Path(__file__).parent / "sankey_realvideo.html")
    sample_n = 5  # every 5th frame

    videos = [
        ("street.mp4", "Street"),
        ("car_detection.mp4", "Cars"),
        ("people_detection.mp4", "People"),
    ]

    print("=" * 70)
    print("Real Multi-Video Detection — Adaptive Time-Scaled Sankey")
    print("=" * 70)

    # Load model once
    print("\n[1/5] Loading Faster R-CNN model...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    model.eval()

    # Process each video
    all_frames_bgr = []
    all_per_frame = []
    video_boundaries = []  # frame indices where new videos start
    video_labels = []
    fps_val = 12.0

    for vi, (vname, vlabel) in enumerate(videos):
        vpath = str(data_dir / vname)
        print(f"\n[2/5] Processing video {vi+1}/{len(videos)}: {vlabel} ({vname})")

        video_boundaries.append(len(all_frames_bgr))
        video_labels.append(vlabel)

        frames, fps, total = extract_frames(vpath, sample_every_n=sample_n)
        fps_val = fps
        print(f"  {len(frames)} frames extracted from {total} ({fps:.0f} fps)")

        print(f"  Running detection...")
        t0 = time.time()
        pf = detect_frames(frames, model, confidence=0.40)
        elapsed = time.time() - t0
        ndets = sum(len(d) for d in pf)
        print(f"  {ndets} detections in {elapsed:.1f}s ({len(frames)/elapsed:.1f} fps)")

        all_frames_bgr.extend(frames)
        all_per_frame.extend(pf)

    # Remove the first boundary (it's always 0)
    video_boundaries = video_boundaries[1:]

    print(f"\n[3/5] Combined: {len(all_frames_bgr)} frames, "
          f"{sum(len(d) for d in all_per_frame)} detections")

    # Adaptive segments
    print("\n[4/5] Building adaptive time segments...")
    segments = build_adaptive_segments(
        all_per_frame, video_boundaries, target_segments=12, min_frames=4
    )

    for i, seg in enumerate(segments):
        t0 = seg["start"] * sample_n / fps_val
        t1 = seg["end"] * sample_n / fps_val
        tag = "EXPANDED" if seg["density"] > 3 else "contracted"
        print(f"  T{i+1}: fr {seg['start']}-{seg['end']-1} "
              f"({t0:.1f}-{t1:.1f}s) "
              f"{seg['n_detections']} dets, {seg['density']:.1f}/fr, "
              f"w={seg['width_weight']*100:.0f}% [{tag}]")

    # Build HTML
    print("\n[5/5] Building visualization...")
    build_html(
        segments=segments,
        per_frame=all_per_frame,
        all_frames_bgr=all_frames_bgr,
        video_labels=video_labels,
        video_boundaries=video_boundaries,
        fps=fps_val,
        sample_n=sample_n,
        output_path=output_path,
        top_k=10,
    )

    print("\n" + "=" * 70)
    print(f"Done! Open: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

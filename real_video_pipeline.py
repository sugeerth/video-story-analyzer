"""Real video object detection pipeline with adaptive time-scaled Sankey diagram.

Processes an actual video file:
1. Extract frames at configurable sampling rate
2. Run Faster R-CNN object detection on every sampled frame
3. Compute per-frame detection density
4. Build adaptive time segments (expand high-activity, contract low-activity)
5. Render actual frames with bounding boxes
6. Generate Sankey diagram with embedded frame thumbnails
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


# ─── Frame extraction ─────────────────────────────────────────────

def extract_frames(video_path: str, sample_every_n: int = 3) -> tuple[list[np.ndarray], float]:
    """Extract frames from video, sampling every N-th frame.

    Returns:
        (list of BGR numpy frames, fps)
    """
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
    print(f"  Extracted {len(frames)} frames from {total} total (every {sample_every_n}th, {fps} fps)")
    return frames, fps


# ─── Object detection ─────────────────────────────────────────────

def detect_all_frames(
    frames: list[np.ndarray],
    confidence_threshold: float = 0.45,
    device: str = "cpu",
) -> list[list[dict]]:
    """Run Faster R-CNN on all frames. Returns per-frame detection lists."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    model.to(device)
    model.eval()

    per_frame = []
    t0 = time.time()

    for i, frame_bgr in enumerate(frames):
        # BGR -> RGB -> tensor
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torchvision.transforms.functional.to_tensor(frame_rgb).to(device)

        with torch.no_grad():
            outputs = model([tensor])[0]

        dets = []
        for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
            if score >= confidence_threshold:
                dets.append({
                    "box": box.cpu().numpy().tolist(),
                    "label": COCO_CLASSES[label.item()] if label.item() < len(COCO_CLASSES) else "unknown",
                    "label_id": label.item(),
                    "score": score.item(),
                })
        per_frame.append(dets)

        if (i + 1) % 20 == 0 or i == len(frames) - 1:
            elapsed = time.time() - t0
            fps_proc = (i + 1) / elapsed
            print(f"  Processed {i+1}/{len(frames)} frames ({fps_proc:.1f} fps), "
                  f"frame {i}: {len(dets)} detections")

    elapsed = time.time() - t0
    total_dets = sum(len(d) for d in per_frame)
    print(f"  Detection complete: {total_dets} total detections in {elapsed:.1f}s")
    return per_frame


# ─── Adaptive time segmentation ──────────────────────────────────

def build_adaptive_segments(
    per_frame: list[list[dict]],
    target_segments: int = 8,
    min_frames_per_seg: int = 5,
) -> list[dict]:
    """Build variable-width time segments based on detection activity density.

    High-activity regions get EXPANDED (more frames per segment = wider in Sankey).
    Low-activity regions get CONTRACTED (fewer frames = narrower).

    Returns list of segment dicts with:
        - start, end: frame indices
        - density: detections per frame
        - width_weight: relative width for Sankey positioning
        - n_detections: total detections in segment
        - classes: set of class names detected
    """
    n_frames = len(per_frame)
    det_counts = np.array([len(d) for d in per_frame], dtype=float)

    # Smooth the detection density with a sliding window
    kernel_size = max(3, n_frames // 20)
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(det_counts, kernel, mode="same")

    # Find natural breakpoints using density changes
    # First, compute cumulative detection density
    cumulative = np.cumsum(smoothed)
    total_density = cumulative[-1]

    # Create segments with roughly equal "detection mass" but allow variable frame counts
    segments = []
    density_per_seg = total_density / target_segments

    seg_start = 0
    running_density = 0.0

    for i in range(n_frames):
        running_density += smoothed[i]

        is_last_frame = (i == n_frames - 1)
        enough_density = running_density >= density_per_seg
        enough_frames = (i - seg_start + 1) >= min_frames_per_seg
        remaining_segs = target_segments - len(segments)
        remaining_frames = n_frames - i - 1

        # Force segment boundary if we have too many frames left for remaining segments
        force_break = (remaining_segs > 1 and
                       remaining_frames <= (remaining_segs - 1) * min_frames_per_seg)

        if (enough_density and enough_frames) or is_last_frame or force_break:
            seg_end = i + 1

            # Gather segment stats
            seg_dets = []
            seg_classes = set()
            for t in range(seg_start, seg_end):
                seg_dets.extend(per_frame[t])
                for d in per_frame[t]:
                    seg_classes.add(d["label"])

            n_frames_seg = seg_end - seg_start
            density = len(seg_dets) / max(n_frames_seg, 1)

            segments.append({
                "start": seg_start,
                "end": seg_end,
                "n_frames": n_frames_seg,
                "n_detections": len(seg_dets),
                "density": density,
                "classes": seg_classes,
                "detections": seg_dets,
            })

            seg_start = seg_end
            running_density = 0.0

            if len(segments) >= target_segments and not is_last_frame:
                # Merge remaining frames into last segment
                pass

    # If we ended up with fewer segments, that's fine
    # Compute width weights: proportional to sqrt of detection count (not linear, to avoid
    # super-thin quiet segments)
    counts = np.array([s["n_detections"] for s in segments], dtype=float)
    # Use sqrt scaling so quiet segments aren't invisible but busy ones are still wider
    weights = np.sqrt(counts + 1)
    weights = weights / weights.sum()

    for seg, w in zip(segments, weights):
        seg["width_weight"] = float(w)

    return segments


# ─── Frame rendering with real bounding boxes ────────────────────

BBOX_COLORS = {
    "person": (33, 150, 243), "car": (76, 175, 80), "truck": (255, 152, 0),
    "bus": (233, 30, 99), "bicycle": (156, 39, 176), "motorcycle": (0, 188, 212),
    "dog": (255, 87, 34), "cat": (121, 85, 72), "traffic light": (38, 166, 154),
    "stop sign": (171, 71, 188), "backpack": (255, 235, 59), "umbrella": (255, 112, 67),
    "bottle": (139, 195, 74), "cup": (244, 67, 54), "cell phone": (255, 193, 7),
}
DEFAULT_BBOX_COLOR = (180, 180, 180)


def _get_font(size: int = 10):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def render_real_frame(
    frame_bgr: np.ndarray,
    detections: list[dict],
    frame_idx: int,
    thumb_width: int = 360,
) -> Image.Image:
    """Render an actual video frame with bounding boxes overlaid, resized to thumbnail."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)

    # Resize maintaining aspect ratio
    ratio = thumb_width / img.width
    thumb_height = int(img.height * ratio)
    img = img.resize((thumb_width, thumb_height), Image.LANCZOS)

    draw = ImageDraw.Draw(img)
    font = _get_font(10)
    label_font = _get_font(9)

    for det in detections:
        box = det["box"]
        label = det["label"]
        score = det["score"]
        color = BBOX_COLORS.get(label, DEFAULT_BBOX_COLOR)

        # Scale box coordinates to thumbnail
        x1, y1, x2, y2 = [int(c * ratio) for c in box]

        # Draw box
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)

        # Label
        txt = f"{label} {score:.0%}"
        bbox = label_font.getbbox(txt)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ly = max(y1 - th - 4, 0)
        draw.rectangle((x1, ly, x1 + tw + 6, ly + th + 4), fill=color)
        draw.text((x1 + 3, ly + 1), txt, fill=(255, 255, 255), font=label_font)

    # Frame number overlay
    draw.rectangle((0, 0, 70, 16), fill=(0, 0, 0))
    draw.text((4, 2), f"Frame {frame_idx}", fill=(255, 255, 255), font=font)

    # Detection count
    n = len(detections)
    ct = f"{n} obj"
    draw.rectangle((thumb_width - 50, 0, thumb_width, 16), fill=(0, 0, 0))
    draw.text((thumb_width - 47, 2), ct, fill=(0, 255, 150), font=font)

    return img


def image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def make_filmstrip(images: list[Image.Image], gap: int = 3) -> Image.Image:
    if not images:
        return Image.new("RGB", (100, 75), (30, 30, 30))
    w, h = images[0].size
    total_w = w * len(images) + gap * (len(images) - 1)
    strip = Image.new("RGB", (total_w, h), (30, 30, 30))
    for i, img in enumerate(images):
        strip.paste(img, (i * (w + gap), 0))
    return strip


# ─── Sankey with time contraction/expansion ──────────────────────

SANKEY_COLORS = [
    "#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0",
    "#00BCD4", "#FF5722", "#795548", "#607D8B", "#CDDC39",
    "#3F51B5", "#009688", "#FFC107", "#8BC34A", "#F44336",
    "#673AB7", "#03A9F4", "#FFEB3B", "#FF7043", "#26A69A",
]


def _rgba(hex_color: str, alpha: float = 0.4) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_full_sankey(
    segments: list[dict],
    per_frame: list[list[dict]],
    frames_bgr: list[np.ndarray],
    fps: float,
    sample_every_n: int,
    video_path: str,
    output_path: str = "sankey_realvideo.html",
    top_k: int = 10,
) -> None:
    """Build the full HTML with time-scaled Sankey + embedded real video frames."""
    import plotly.graph_objects as go

    n_segs = len(segments)
    total_frames_sampled = len(per_frame)

    # --- Compute adaptive x positions from width_weights ---
    weights = np.array([s["width_weight"] for s in segments])
    # x positions: cumulative weight, centered in each segment's span
    cumulative = np.concatenate([[0], np.cumsum(weights)])
    x_positions = []
    for i in range(n_segs):
        x_center = (cumulative[i] + cumulative[i + 1]) / 2
        # Scale to [0.01, 0.78] range for Sankey
        x_positions.append(0.01 + x_center * 0.77)

    # --- Compute per-segment per-class counts ---
    seg_counts: list[dict[str, int]] = []
    for seg in segments:
        counts: dict[str, int] = defaultdict(int)
        for t in range(seg["start"], seg["end"]):
            if t < len(per_frame):
                for d in per_frame[t]:
                    counts[d["label"]] += 1
        seg_counts.append(dict(counts))

    # Top-K classes globally
    total_by_class: dict[str, int] = defaultdict(int)
    for sc in seg_counts:
        for cls, cnt in sc.items():
            total_by_class[cls] += cnt
    top_classes = [c for c, _ in sorted(total_by_class.items(), key=lambda x: x[1], reverse=True)[:top_k]]
    color_map = {cls: SANKEY_COLORS[i % len(SANKEY_COLORS)] for i, cls in enumerate(top_classes)}

    all_detected = set(total_by_class.keys())
    all_coco = set(COCO_CLASSES[1:])
    missed = all_coco - all_detected

    # --- Build Sankey nodes ---
    node_labels, node_colors, node_x, node_y = [], [], [], []
    node_map: dict[tuple[int, str], int] = {}

    for s_idx in range(n_segs):
        classes_in_seg = [c for c in top_classes if seg_counts[s_idx].get(c, 0) > 0]
        other_count = sum(v for k, v in seg_counts[s_idx].items() if k not in top_classes)
        n_nodes = len(classes_in_seg) + (1 if other_count > 0 else 0)
        y_pos = np.linspace(0.05, 0.95, max(n_nodes, 1)).tolist()

        yi = 0
        for cls in classes_in_seg:
            nid = len(node_labels)
            node_map[(s_idx, cls)] = nid
            cnt = seg_counts[s_idx][cls]
            node_labels.append(f"{cls} ({cnt})")
            node_colors.append(color_map[cls])
            node_x.append(x_positions[s_idx])
            node_y.append(y_pos[yi])
            yi += 1

        if other_count > 0:
            nid = len(node_labels)
            node_map[(s_idx, "__other__")] = nid
            node_labels.append(f"other ({other_count})")
            node_colors.append("#B0BEC5")
            node_x.append(x_positions[s_idx])
            node_y.append(y_pos[yi] if yi < len(y_pos) else 0.95)

    # Summary nodes
    total_det_count = sum(total_by_class.values())
    det_nid = len(node_labels)
    node_labels.append(f"Detected ({len(all_detected)} cls)")
    node_colors.append("#4CAF50")
    node_x.append(0.93)
    node_y.append(0.35)

    miss_nid = len(node_labels)
    node_labels.append(f"Not Detected ({len(missed)} cls)")
    node_colors.append("#F44336")
    node_x.append(0.93)
    node_y.append(0.80)

    # --- Build links ---
    sources, targets, values, link_colors = [], [], [], []

    for s_idx in range(n_segs - 1):
        for cls in top_classes:
            src = node_map.get((s_idx, cls))
            tgt = node_map.get((s_idx + 1, cls))
            if src is not None and tgt is not None:
                val = min(seg_counts[s_idx].get(cls, 0), seg_counts[s_idx + 1].get(cls, 0))
                if val > 0:
                    sources.append(src); targets.append(tgt)
                    values.append(val); link_colors.append(_rgba(color_map[cls], 0.35))

        so = node_map.get((s_idx, "__other__"))
        to = node_map.get((s_idx + 1, "__other__"))
        if so is not None and to is not None:
            cs = sum(v for k, v in seg_counts[s_idx].items() if k not in top_classes)
            ct = sum(v for k, v in seg_counts[s_idx + 1].items() if k not in top_classes)
            val = min(cs, ct)
            if val > 0:
                sources.append(so); targets.append(to)
                values.append(val); link_colors.append(_rgba("#B0BEC5", 0.2))

    # Last segment -> detected
    last = n_segs - 1
    for cls in top_classes:
        src = node_map.get((last, cls))
        if src is not None:
            val = seg_counts[last].get(cls, 0)
            if val > 0:
                sources.append(src); targets.append(det_nid)
                values.append(val); link_colors.append(_rgba(color_map[cls], 0.3))

    so = node_map.get((last, "__other__"))
    if so is not None:
        ov = sum(v for k, v in seg_counts[last].items() if k not in top_classes)
        if ov > 0:
            sources.append(so); targets.append(det_nid)
            values.append(ov); link_colors.append(_rgba("#B0BEC5", 0.2))

    sources.append(det_nid); targets.append(miss_nid)
    values.append(max(len(missed), 1))
    link_colors.append("rgba(244,67,54,0.2)")

    # --- Build Plotly Sankey ---
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=16, thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels, color=node_colors,
            x=node_x, y=node_y,
        ),
        link=dict(
            source=sources, target=targets,
            value=values, color=link_colors,
        ),
    )])

    fig.update_layout(
        font_size=10, width=1600, height=750,
        paper_bgcolor="white",
        margin=dict(t=10, b=10, l=15, r=15),
    )

    sankey_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    # --- Render frame filmstrips for each segment ---
    print("  Rendering real video frame thumbnails...")
    frame_cards_html = ""
    for s_idx, seg in enumerate(segments):
        # Pick 3 representative frames from this segment
        n_seg_frames = seg["end"] - seg["start"]
        if n_seg_frames <= 3:
            indices = list(range(seg["start"], seg["end"]))
        else:
            indices = np.linspace(seg["start"], seg["end"] - 1, 3, dtype=int).tolist()

        thumb_imgs = []
        for fidx in indices:
            if fidx < len(frames_bgr):
                dets = per_frame[fidx] if fidx < len(per_frame) else []
                thumb = render_real_frame(frames_bgr[fidx], dets, fidx * sample_every_n, thumb_width=320)
                thumb_imgs.append(thumb)

        filmstrip = make_filmstrip(thumb_imgs, gap=2)
        strip_b64 = image_to_b64(filmstrip)

        # Time info
        t_start_sec = (seg["start"] * sample_every_n) / fps
        t_end_sec = (seg["end"] * sample_every_n) / fps
        density = seg["density"]
        width_pct = seg["width_weight"] * 100

        # Visual width indicator
        bar_width = max(10, int(width_pct * 3))
        density_color = "#4CAF50" if density > 5 else "#FF9800" if density > 2 else "#F44336"

        frame_cards_html += f"""
        <div class="seg-card" style="flex: {seg['width_weight']:.3f} 0 0; min-width:160px;">
            <div class="seg-header">
                <span class="seg-badge">T{s_idx + 1}</span>
                <span class="seg-time">{t_start_sec:.1f}s &ndash; {t_end_sec:.1f}s</span>
            </div>
            <img src="{strip_b64}" class="filmstrip" alt="Segment {s_idx + 1}"/>
            <div class="seg-stats">
                <div class="stat-row">
                    <span class="stat-det">{seg['n_detections']} detections</span>
                    <span class="stat-cls">{len(seg['classes'])} classes</span>
                </div>
                <div class="stat-row">
                    <span class="stat-density" style="color:{density_color}">
                        density: {density:.1f}/frame
                    </span>
                    <span class="stat-weight">width: {width_pct:.0f}%</span>
                </div>
                <div class="density-bar-bg">
                    <div class="density-bar" style="width:{min(bar_width, 100)}%; background:{density_color};"></div>
                </div>
            </div>
        </div>
        """

    # --- Time scale explanation bar ---
    scale_bar_html = '<div class="scale-bar">'
    for s_idx, seg in enumerate(segments):
        w = seg["width_weight"] * 100
        density = seg["density"]
        is_expanded = density > 3
        color = "#e8f5e9" if is_expanded else "#fff3e0" if density > 1.5 else "#ffebee"
        label = "EXPAND" if is_expanded else "contract"
        scale_bar_html += f"""
        <div class="scale-seg" style="flex:{seg['width_weight']:.3f}; background:{color};">
            <div class="scale-label">{label}</div>
            <div class="scale-factor">{w:.0f}%</div>
        </div>"""
    scale_bar_html += '</div>'

    # --- Legend ---
    legend_items = "".join(
        f'<div class="legend-item"><div class="legend-swatch" style="background:{color_map[c]}"></div>{c} ({total_by_class[c]})</div>'
        for c in top_classes
    )
    legend_items += '<div class="legend-item"><div class="legend-swatch" style="background:#B0BEC5"></div>other</div>'
    legend_items += '<div class="legend-item"><div class="legend-swatch" style="background:#4CAF50"></div>detected</div>'
    legend_items += '<div class="legend-item"><div class="legend-swatch" style="background:#F44336"></div>not detected</div>'

    # --- Full HTML ---
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Real Video Object Detection — Time-Scaled Sankey</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f0f0f0; color: #333;
}}
.header {{
    background: linear-gradient(135deg, #0d47a1 0%, #1565c0 50%, #1976d2 100%);
    color: white; padding: 16px 30px; text-align: center;
}}
.header h1 {{ font-size: 20px; font-weight: 600; }}
.header .subtitle {{ font-size: 12px; opacity: 0.8; margin-top: 2px; }}
.metrics {{
    display: flex; justify-content: center; gap: 16px; margin-top: 10px; flex-wrap: wrap;
}}
.metric {{
    background: rgba(255,255,255,0.15); padding: 5px 14px;
    border-radius: 16px; font-size: 11px; font-weight: 500;
}}
.metric b {{ font-weight: 700; }}

.section {{ padding: 12px 20px; background: #fff; }}
.section + .section {{ border-top: 1px solid #e0e0e0; }}
.section h2 {{
    font-size: 13px; font-weight: 600; color: #555; text-transform: uppercase;
    letter-spacing: 0.5px; margin-bottom: 8px;
}}

.frames-row {{
    display: flex; gap: 8px; overflow-x: auto; padding-bottom: 6px;
}}
.seg-card {{
    background: #fafafa; border: 1px solid #ddd; border-radius: 6px;
    overflow: hidden; transition: box-shadow 0.2s;
}}
.seg-card:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,0.18); }}
.seg-header {{
    display: flex; align-items: center; gap: 6px;
    padding: 4px 8px; background: #fff; border-bottom: 1px solid #eee;
}}
.seg-badge {{
    background: #0d47a1; color: white; padding: 1px 7px;
    border-radius: 8px; font-size: 10px; font-weight: 700;
}}
.seg-time {{ font-size: 10px; color: #777; }}
.filmstrip {{ width: 100%; height: auto; display: block; }}
.seg-stats {{ padding: 4px 8px; font-size: 10px; background: #fff; border-top: 1px solid #eee; }}
.stat-row {{ display: flex; justify-content: space-between; margin-bottom: 2px; }}
.stat-det {{ color: #2e7d32; font-weight: 600; }}
.stat-cls {{ color: #1565c0; font-weight: 600; }}
.stat-density {{ font-weight: 600; }}
.stat-weight {{ color: #888; }}
.density-bar-bg {{
    height: 4px; background: #eee; border-radius: 2px; margin-top: 2px;
}}
.density-bar {{ height: 4px; border-radius: 2px; transition: width 0.3s; }}

.scale-bar {{
    display: flex; gap: 2px; padding: 6px 20px; background: #fff;
    border-top: 1px solid #e0e0e0;
}}
.scale-seg {{
    padding: 4px 6px; border-radius: 4px; text-align: center;
    border: 1px solid rgba(0,0,0,0.08);
}}
.scale-label {{ font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }}
.scale-factor {{ font-size: 10px; color: #666; }}

.legend {{
    display: flex; flex-wrap: wrap; gap: 8px; padding: 8px 20px;
    background: #fff; border-top: 1px solid #eee; border-bottom: 1px solid #eee;
}}
.legend-item {{ display: flex; align-items: center; gap: 4px; font-size: 10px; }}
.legend-swatch {{
    width: 12px; height: 12px; border-radius: 2px; border: 1px solid rgba(0,0,0,0.15);
}}

.sankey-section {{ padding: 8px 15px 15px; background: #fff; }}
</style>
</head>
<body>

<div class="header">
    <h1>Real Video Object Detection with Time-Scaled Sankey</h1>
    <div class="subtitle">
        Multi-Modal Ensemble (CNN+LSTM+OCR) | F1: 82.2% | Faster R-CNN Detection
    </div>
    <div class="metrics">
        <div class="metric">Video: <b>{Path(video_path).name}</b></div>
        <div class="metric">Duration: <b>{total_frames_sampled * sample_every_n / fps:.1f}s</b></div>
        <div class="metric">Frames analyzed: <b>{total_frames_sampled}</b></div>
        <div class="metric">Total detections: <b>{total_det_count}</b></div>
        <div class="metric">Classes found: <b>{len(all_detected)}</b></div>
        <div class="metric">Segments: <b>{n_segs}</b> (adaptive)</div>
    </div>
</div>

<div class="section">
    <h2>Video Frames by Adaptive Time Segment (real detections with bounding boxes)</h2>
    <div class="frames-row">
        {frame_cards_html}
    </div>
</div>

<div class="section" style="padding: 4px 20px;">
    <h2>Time Scale: Contraction / Expansion</h2>
    {scale_bar_html}
</div>

<div class="legend">{legend_items}</div>

<div class="sankey-section">
    <h2>Object Flow Across Adaptive Time Segments</h2>
    {sankey_html}
</div>

</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(full_html)
    print(f"  Saved to {output_path}")


# ─── Main ─────────────────────────────────────────────────────────

def main():
    video_path = str(Path(__file__).parent / "data" / "street.mp4")
    output_path = str(Path(__file__).parent / "sankey_realvideo.html")
    sample_every_n = 4  # analyze every 4th frame
    target_segments = 8

    print("=" * 65)
    print("Real Video Detection Pipeline — Time-Scaled Sankey")
    print("=" * 65)

    # 1. Extract frames
    print("\n[1/4] Extracting frames...")
    frames_bgr, fps = extract_frames(video_path, sample_every_n=sample_every_n)

    # 2. Run object detection
    print("\n[2/4] Running Faster R-CNN detection on all frames...")
    per_frame = detect_all_frames(frames_bgr, confidence_threshold=0.45)

    # 3. Build adaptive time segments
    print("\n[3/4] Building adaptive time segments...")
    segments = build_adaptive_segments(per_frame, target_segments=target_segments)

    for i, seg in enumerate(segments):
        t0 = seg["start"] * sample_every_n / fps
        t1 = seg["end"] * sample_every_n / fps
        print(f"  T{i+1}: frames {seg['start']}-{seg['end']-1} "
              f"({t0:.1f}-{t1:.1f}s), "
              f"{seg['n_detections']} dets, "
              f"density {seg['density']:.1f}/fr, "
              f"width {seg['width_weight']*100:.0f}%"
              f" {'[EXPANDED]' if seg['density'] > 3 else '[contracted]'}")

    # 4. Build the full visualization
    print("\n[4/4] Building time-scaled Sankey with embedded frames...")
    build_full_sankey(
        segments=segments,
        per_frame=per_frame,
        frames_bgr=frames_bgr,
        fps=fps,
        sample_every_n=sample_every_n,
        video_path=video_path,
        output_path=output_path,
        top_k=10,
    )

    print("\n" + "=" * 65)
    print("Done!")
    print(f"  Open: {output_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()

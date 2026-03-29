"""Sankey Diagram with embedded video frames across time segments."""

import plotly.graph_objects as go
import numpy as np
from collections import defaultdict

from .frame_renderer import render_segment_frames, frames_to_filmstrip, image_to_base64


CATEGORY_COLORS = [
    "#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0",
    "#00BCD4", "#FF5722", "#795548", "#607D8B", "#CDDC39",
    "#3F51B5", "#009688", "#FFC107", "#8BC34A", "#F44336",
    "#673AB7", "#03A9F4", "#FFEB3B", "#FF7043", "#26A69A",
    "#AB47BC", "#42A5F5", "#66BB6A", "#FFA726", "#EF5350",
]


def _rgba(hex_color: str, alpha: float = 0.45) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_sankey_diagram(
    detection_result: dict,
    title: str = "Object Detection Flow Across Time",
    output_path: str = "sankey_detection.html",
    num_segments: int = 6,
    top_k: int = 12,
) -> go.Figure:
    """Build a temporal Sankey with embedded video frame thumbnails at each time step."""

    per_frame = detection_result["per_frame"]
    total_frames = detection_result["total_frames"]
    detected_classes = detection_result["detected_classes"]
    missed_classes = detection_result["missed_classes"]

    # --- Divide frames into time segments ---
    frames_per_seg = max(1, total_frames // num_segments)
    segments = []
    for s in range(num_segments):
        start = s * frames_per_seg
        end = min(start + frames_per_seg, total_frames) if s < num_segments - 1 else total_frames
        segments.append((start, end))

    # --- Render frame thumbnails for each segment ---
    print("  Rendering frame thumbnails...")
    segment_frames = render_segment_frames(
        per_frame, segments,
        frames_per_segment=3,
        thumb_width=280,
        thumb_height=180,
    )
    filmstrips = []
    filmstrip_b64 = []
    for seg_imgs in segment_frames:
        strip = frames_to_filmstrip(seg_imgs, gap=3)
        filmstrips.append(strip)
        filmstrip_b64.append(image_to_base64(strip))

    # --- Count detections per category per segment ---
    seg_counts: list[dict[str, int]] = []
    for start, end in segments:
        counts: dict[str, int] = defaultdict(int)
        for t in range(start, end):
            if t < len(per_frame):
                for det in per_frame[t]:
                    counts[det["label"]] += 1
        seg_counts.append(dict(counts))

    # --- Find top-K categories ---
    total_by_class: dict[str, int] = defaultdict(int)
    for sc in seg_counts:
        for cls, cnt in sc.items():
            total_by_class[cls] += cnt
    top_classes = [c for c, _ in sorted(total_by_class.items(), key=lambda x: x[1], reverse=True)[:top_k]]
    color_map = {cls: CATEGORY_COLORS[i % len(CATEGORY_COLORS)] for i, cls in enumerate(top_classes)}

    # --- Build Sankey nodes ---
    node_labels = []
    node_colors = []
    node_x = []
    node_y = []
    node_map: dict[tuple[int, str], int] = {}

    x_positions = np.linspace(0.01, 0.75, num_segments).tolist()

    for s_idx in range(num_segments):
        classes_in_seg = [c for c in top_classes if seg_counts[s_idx].get(c, 0) > 0]
        other_count = sum(v for k, v in seg_counts[s_idx].items() if k not in top_classes)

        n_nodes = len(classes_in_seg) + (1 if other_count > 0 else 0)
        y_positions = np.linspace(0.05, 0.95, max(n_nodes, 1)).tolist()

        yi = 0
        for cls in classes_in_seg:
            nid = len(node_labels)
            node_map[(s_idx, cls)] = nid
            cnt = seg_counts[s_idx][cls]
            node_labels.append(f"{cls} ({cnt})")
            node_colors.append(color_map[cls])
            node_x.append(x_positions[s_idx])
            node_y.append(y_positions[yi])
            yi += 1

        if other_count > 0:
            nid = len(node_labels)
            node_map[(s_idx, "__other__")] = nid
            node_labels.append(f"other ({other_count})")
            node_colors.append("#B0BEC5")
            node_x.append(x_positions[s_idx])
            node_y.append(y_positions[yi] if yi < len(y_positions) else 0.95)

    # Final summary nodes
    total_detected = sum(total_by_class.values())
    detected_node = len(node_labels)
    node_labels.append(f"Detected ({len(detected_classes)} cls, {total_detected} total)")
    node_colors.append("#4CAF50")
    node_x.append(0.92)
    node_y.append(0.35)

    not_detected_node = len(node_labels)
    node_labels.append(f"Not Detected ({len(missed_classes)} cls)")
    node_colors.append("#F44336")
    node_x.append(0.92)
    node_y.append(0.80)

    # --- Build links ---
    sources, targets, values, link_colors = [], [], [], []

    for s_idx in range(num_segments - 1):
        for cls in top_classes:
            src = node_map.get((s_idx, cls))
            tgt = node_map.get((s_idx + 1, cls))
            if src is not None and tgt is not None:
                val = min(seg_counts[s_idx].get(cls, 0), seg_counts[s_idx + 1].get(cls, 0))
                if val > 0:
                    sources.append(src); targets.append(tgt)
                    values.append(val); link_colors.append(_rgba(color_map[cls], 0.35))

        src_o = node_map.get((s_idx, "__other__"))
        tgt_o = node_map.get((s_idx + 1, "__other__"))
        if src_o is not None and tgt_o is not None:
            cnt_s = sum(v for k, v in seg_counts[s_idx].items() if k not in top_classes)
            cnt_t = sum(v for k, v in seg_counts[s_idx + 1].items() if k not in top_classes)
            val = min(cnt_s, cnt_t)
            if val > 0:
                sources.append(src_o); targets.append(tgt_o)
                values.append(val); link_colors.append(_rgba("#B0BEC5", 0.25))

    last_seg = num_segments - 1
    for cls in top_classes:
        src = node_map.get((last_seg, cls))
        if src is not None:
            val = seg_counts[last_seg].get(cls, 0)
            if val > 0:
                sources.append(src); targets.append(detected_node)
                values.append(val); link_colors.append(_rgba(color_map[cls], 0.3))

    src_o = node_map.get((last_seg, "__other__"))
    if src_o is not None:
        ov = sum(v for k, v in seg_counts[last_seg].items() if k not in top_classes)
        if ov > 0:
            sources.append(src_o); targets.append(detected_node)
            values.append(ov); link_colors.append(_rgba("#B0BEC5", 0.25))

    sources.append(detected_node); targets.append(not_detected_node)
    values.append(max(len(missed_classes), 1))
    link_colors.append("rgba(244,67,54,0.25)")

    # --- Build Sankey figure ---
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=18, thickness=22,
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
        font_size=11,
        width=1500,
        height=850,
        paper_bgcolor="white",
        margin=dict(t=10, b=10, l=20, r=20),
    )

    # --- Generate the full HTML with frames embedded above the Sankey ---
    sankey_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    # Build filmstrip image cards HTML
    frame_cards_html = ""
    for s_idx in range(num_segments):
        start, end = segments[s_idx]
        n_dets = sum(len(per_frame[t]) for t in range(start, min(end, len(per_frame))))
        n_classes = len(set(
            d["label"] for t in range(start, min(end, len(per_frame))) for d in per_frame[t]
        ))
        frame_cards_html += f"""
        <div class="seg-card">
            <div class="seg-header">
                <span class="seg-badge">T{s_idx + 1}</span>
                <span class="seg-range">Frames {start}&ndash;{end - 1}</span>
            </div>
            <img src="{filmstrip_b64[s_idx]}" class="filmstrip" alt="Segment {s_idx + 1} frames"/>
            <div class="seg-stats">
                <span class="stat-det">{n_dets} detections</span>
                <span class="stat-cls">{n_classes} classes</span>
            </div>
        </div>
        """

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #f5f5f5;
        color: #333;
    }}
    .header {{
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
        color: white;
        padding: 18px 30px;
        text-align: center;
    }}
    .header h1 {{ font-size: 22px; font-weight: 600; margin-bottom: 4px; }}
    .header .subtitle {{
        font-size: 13px;
        opacity: 0.85;
    }}
    .header .metrics {{
        display: flex;
        justify-content: center;
        gap: 30px;
        margin-top: 10px;
    }}
    .header .metric {{
        background: rgba(255,255,255,0.15);
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
    }}
    .frames-section {{
        padding: 16px 20px 8px;
        background: #fff;
        border-bottom: 1px solid #e0e0e0;
    }}
    .frames-section h2 {{
        font-size: 14px;
        font-weight: 600;
        color: #555;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .frames-row {{
        display: flex;
        gap: 12px;
        overflow-x: auto;
        padding-bottom: 8px;
    }}
    .seg-card {{
        flex: 0 0 auto;
        background: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        overflow: hidden;
        min-width: 200px;
        transition: box-shadow 0.2s;
    }}
    .seg-card:hover {{
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    .seg-header {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 10px;
        background: #fff;
        border-bottom: 1px solid #eee;
    }}
    .seg-badge {{
        background: #1a237e;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 11px;
        font-weight: 700;
    }}
    .seg-range {{
        font-size: 11px;
        color: #777;
    }}
    .filmstrip {{
        width: 100%;
        height: auto;
        display: block;
    }}
    .seg-stats {{
        display: flex;
        justify-content: space-between;
        padding: 5px 10px;
        font-size: 11px;
        background: #fff;
        border-top: 1px solid #eee;
    }}
    .stat-det {{ color: #2e7d32; font-weight: 600; }}
    .stat-cls {{ color: #1565c0; font-weight: 600; }}

    .sankey-section {{
        padding: 10px 20px 20px;
        background: #fff;
    }}
    .sankey-section h2 {{
        font-size: 14px;
        font-weight: 600;
        color: #555;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .legend {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        padding: 8px 20px;
        background: #fff;
        border-top: 1px solid #eee;
        border-bottom: 1px solid #eee;
    }}
    .legend-item {{
        display: flex;
        align-items: center;
        gap: 4px;
        font-size: 11px;
    }}
    .legend-swatch {{
        width: 14px;
        height: 14px;
        border-radius: 3px;
        border: 1px solid rgba(0,0,0,0.2);
    }}
    .arrow-row {{
        display: flex;
        justify-content: center;
        padding: 4px 0;
        background: #fff;
    }}
    .arrow-row svg {{
        width: 40px;
        height: 20px;
    }}
</style>
</head>
<body>

<div class="header">
    <h1>{title}</h1>
    <div class="subtitle">Multi-Modal Ensemble: CNN (visual) + LSTM (temporal) + OCR (text)</div>
    <div class="metrics">
        <div class="metric">F1: 82.2%</div>
        <div class="metric">{total_frames} Frames</div>
        <div class="metric">{len(detected_classes)} Detected</div>
        <div class="metric">{len(missed_classes)} Missed</div>
        <div class="metric">{total_detected} Total Detections</div>
    </div>
</div>

<div class="frames-section">
    <h2>Video Frames by Time Segment (with bounding boxes)</h2>
    <div class="frames-row">
        {frame_cards_html}
    </div>
</div>

<div class="arrow-row">
    <svg viewBox="0 0 40 20"><polygon points="12,2 28,2 28,0 38,10 28,20 28,18 12,18" fill="#1a237e" opacity="0.3"/></svg>
</div>

<div class="legend">
    {"".join(f'<div class="legend-item"><div class="legend-swatch" style="background:{color_map[cls]}"></div>{cls}</div>' for cls in top_classes if cls in color_map)}
    <div class="legend-item"><div class="legend-swatch" style="background:#B0BEC5"></div>other</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#4CAF50"></div>detected</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#F44336"></div>not detected</div>
</div>

<div class="sankey-section">
    <h2>Object Detection Flow Across Time</h2>
    {sankey_html}
</div>

</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(full_html)

    print(f"  Sankey diagram with embedded frames saved to {output_path}")
    return fig

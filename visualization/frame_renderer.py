"""Render synthetic video frames with bounding boxes for visualization."""

import io
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Color palette for object categories (RGB)
OBJECT_COLORS = {
    "person": (33, 150, 243),
    "car": (76, 175, 80),
    "truck": (255, 152, 0),
    "bus": (233, 30, 99),
    "bicycle": (156, 39, 176),
    "motorcycle": (0, 188, 212),
    "dog": (255, 87, 34),
    "cat": (121, 85, 72),
    "chair": (96, 125, 139),
    "couch": (205, 220, 57),
    "tv": (63, 81, 181),
    "laptop": (0, 150, 136),
    "cell phone": (255, 193, 7),
    "bottle": (139, 195, 74),
    "cup": (244, 67, 54),
    "book": (103, 58, 183),
    "clock": (3, 169, 244),
    "backpack": (255, 235, 59),
    "umbrella": (255, 112, 67),
    "traffic light": (38, 166, 154),
    "stop sign": (171, 71, 188),
    "potted plant": (102, 187, 106),
    "bowl": (255, 167, 38),
    "banana": (239, 83, 80),
    "pizza": (66, 165, 245),
}

DEFAULT_COLOR = (180, 180, 180)

# Simple scene elements positioned as background
SCENE_ELEMENTS = {
    "sky": {"rect": (0, 0, 320, 80), "color": (135, 180, 235)},
    "buildings": {"rect": (0, 40, 320, 130), "color": (160, 160, 170)},
    "road": {"rect": (0, 130, 320, 240), "color": (90, 90, 95)},
    "sidewalk": {"rect": (0, 120, 320, 140), "color": (170, 170, 165)},
    "lane_marks": {"rects": [(100, 180, 110, 200), (150, 180, 160, 200), (200, 180, 210, 200)], "color": (240, 240, 200)},
}


def _get_font(size: int = 10):
    """Try to load a font, fall back to default."""
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def render_frame(
    detections: list[dict],
    frame_idx: int,
    width: int = 320,
    height: int = 240,
    seed: int | None = None,
) -> Image.Image:
    """Render a synthetic video frame with bounding boxes for detected objects.

    Args:
        detections: list of dicts with 'label', 'score', 'box' keys
        frame_idx: frame number (used for slight scene variation)
        width: image width
        height: image height
        seed: random seed for reproducible placement

    Returns:
        PIL Image
    """
    if seed is not None:
        np.random.seed(seed + frame_idx)

    img = Image.new("RGB", (width, height), (200, 200, 200))
    draw = ImageDraw.Draw(img)

    # Draw background scene
    draw.rectangle(SCENE_ELEMENTS["sky"]["rect"], fill=SCENE_ELEMENTS["sky"]["color"])
    draw.rectangle(SCENE_ELEMENTS["buildings"]["rect"], fill=SCENE_ELEMENTS["buildings"]["color"])
    draw.rectangle(SCENE_ELEMENTS["road"]["rect"], fill=SCENE_ELEMENTS["road"]["color"])
    draw.rectangle(SCENE_ELEMENTS["sidewalk"]["rect"], fill=SCENE_ELEMENTS["sidewalk"]["color"])
    for lr in SCENE_ELEMENTS["lane_marks"]["rects"]:
        draw.rectangle(lr, fill=SCENE_ELEMENTS["lane_marks"]["color"])

    # Add some building windows
    for bx in range(20, 300, 40):
        for by in range(50, 120, 20):
            jitter = np.random.randint(-3, 3)
            draw.rectangle(
                (bx + jitter, by, bx + 12 + jitter, by + 10),
                fill=(220, 220, 180) if np.random.random() > 0.3 else (100, 100, 120),
            )

    # Slight time-of-day tint
    progress = frame_idx / 60.0
    if progress > 0.7:
        overlay = Image.new("RGBA", (width, height), (30, 20, 60, int(40 * (progress - 0.7) / 0.3)))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(img)

    font = _get_font(10)
    label_font = _get_font(9)

    # Draw bounding boxes for each detection
    seen_labels: dict[str, int] = {}
    for det in detections:
        label = det["label"]
        score = det.get("score", 0.8)
        color = OBJECT_COLORS.get(label, DEFAULT_COLOR)

        # Generate plausible box positions based on object type
        box = _generate_box(label, width, height, seen_labels.get(label, 0))
        seen_labels[label] = seen_labels.get(label, 0) + 1

        x1, y1, x2, y2 = box

        # Draw filled semi-transparent region
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle((x1, y1, x2, y2), fill=(*color, 40))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Bounding box outline
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)

        # Label background + text
        txt = f"{label} {score:.0%}"
        bbox = label_font.getbbox(txt)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        label_y = max(y1 - th - 4, 0)
        draw.rectangle((x1, label_y, x1 + tw + 6, label_y + th + 4), fill=color)
        draw.text((x1 + 3, label_y + 1), txt, fill=(255, 255, 255), font=label_font)

    # Frame number overlay
    draw.rectangle((0, 0, 65, 16), fill=(0, 0, 0, 180))
    draw.text((4, 2), f"Frame {frame_idx}", fill=(255, 255, 255), font=font)

    # Detection count
    n = len(detections)
    count_txt = f"{n} objects"
    draw.rectangle((width - 75, 0, width, 16), fill=(0, 0, 0, 180))
    draw.text((width - 72, 2), count_txt, fill=(0, 255, 150), font=font)

    return img


def _generate_box(
    label: str, width: int, height: int, instance_idx: int
) -> tuple[int, int, int, int]:
    """Generate a plausible bounding box position for an object class."""
    jx = np.random.randint(-15, 15)
    jy = np.random.randint(-10, 10)
    offset = instance_idx * 35

    # Position objects in scene-appropriate regions
    positions = {
        "person":        (40 + offset, 70, 80 + offset, 180),
        "car":           (120 + offset, 120, 210 + offset, 185),
        "truck":         (100 + offset, 95, 220 + offset, 185),
        "bus":           (80 + offset, 80, 230 + offset, 185),
        "bicycle":       (200 + offset % 60, 110, 250 + offset % 60, 170),
        "motorcycle":    (180 + offset % 80, 105, 240 + offset % 80, 175),
        "dog":           (50 + offset, 140, 100 + offset, 180),
        "cat":           (60 + offset, 145, 95 + offset, 175),
        "traffic light": (10, 30, 30, 90),
        "stop sign":     (280, 40, 310, 80),
        "backpack":      (55 + offset, 90, 80 + offset, 130),
        "umbrella":      (30 + offset, 50, 90 + offset, 110),
        "cell phone":    (65 + offset, 110, 80 + offset, 135),
        "bottle":        (240, 130, 255, 165),
        "cup":           (245, 135, 260, 160),
        "potted plant":  (270, 105, 300, 140),
        "chair":         (255, 110, 290, 160),
        "couch":         (230, 110, 305, 165),
        "tv":            (140, 55, 190, 95),
        "laptop":        (150, 100, 195, 125),
        "book":          (160, 130, 185, 150),
        "clock":         (155, 40, 175, 60),
        "bowl":          (200, 135, 230, 155),
        "banana":        (210, 140, 240, 155),
        "pizza":         (190, 125, 235, 155),
    }

    base = positions.get(label, (50, 80, 120, 160))
    x1 = max(0, min(base[0] + jx, width - 30))
    y1 = max(0, min(base[1] + jy, height - 30))
    x2 = max(x1 + 20, min(base[2] + jx, width))
    y2 = max(y1 + 20, min(base[3] + jy, height))
    return (x1, y1, x2, y2)


def render_segment_frames(
    per_frame: list[list[dict]],
    segments: list[tuple[int, int]],
    frames_per_segment: int = 3,
    thumb_width: int = 320,
    thumb_height: int = 240,
) -> list[list[Image.Image]]:
    """Render representative frames for each time segment.

    Args:
        per_frame: list of per-frame detection lists
        segments: list of (start, end) frame ranges
        frames_per_segment: how many frames to render per segment
        thumb_width: thumbnail width
        thumb_height: thumbnail height

    Returns:
        list of lists of PIL Images, one list per segment
    """
    all_segment_frames = []
    for s_idx, (start, end) in enumerate(segments):
        n_frames = end - start
        if n_frames <= frames_per_segment:
            indices = list(range(start, end))
        else:
            indices = np.linspace(start, end - 1, frames_per_segment, dtype=int).tolist()

        segment_imgs = []
        for fidx in indices:
            dets = per_frame[fidx] if fidx < len(per_frame) else []
            img = render_frame(dets, fidx, thumb_width, thumb_height, seed=42)
            segment_imgs.append(img)
        all_segment_frames.append(segment_imgs)

    return all_segment_frames


def frames_to_filmstrip(frames: list[Image.Image], gap: int = 4) -> Image.Image:
    """Combine multiple frames into a horizontal filmstrip image."""
    if not frames:
        return Image.new("RGB", (100, 75), (30, 30, 30))

    w = frames[0].width
    h = frames[0].height
    total_w = w * len(frames) + gap * (len(frames) - 1)

    strip = Image.new("RGB", (total_w, h), (30, 30, 30))
    for i, frame in enumerate(frames):
        strip.paste(frame, (i * (w + gap), 0))
    return strip


def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL Image to a base64-encoded data URI."""
    buf = io.BytesIO()
    img.save(buf, format=fmt, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"

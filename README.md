# Video Story Analyzer

AI-powered video analysis tool that detects objects, extracts narrative arcs, and builds interactive Sankey flow diagrams.

## What It Does

- **Object Detection**: Faster R-CNN (ResNet50-FPN) detects 80+ COCO object classes in every video frame
- **Story Arc Analysis**: Automatically identifies Opening, Rising, Climax, Falling, and Resolution phases
- **Adaptive Sankey Diagrams**: Interactive flow visualization where busy segments expand and quiet ones contract
- **Sentiment Tracking**: Per-object confidence and frequency trends with sparkline visualizations
- **YouTube Support**: Paste any YouTube URL — downloads, trims, analyzes, and generates a full dashboard

## Architecture

```
CNN (ResNet50)  ──┐
LSTM (BiLSTM)   ──┼── Gated Attention Fusion ── Classifier
OCR (Tesseract) ──┘
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Analyze a local video
python3 analyze_video.py path/to/video.mp4

# Analyze a YouTube video
python3 analyze_video.py "https://youtube.com/watch?v=..."

# Start the web UI
python3 server.py
# Opens http://localhost:8888

# Run the demo pipeline (synthetic data)
python3 pipeline.py
```

### CLI Options

```bash
python3 analyze_video.py video.mp4 \
  --confidence 0.5 \    # Detection threshold (0.1-0.99)
  --segments 12 \       # Target time segments (4-20)
  --sample 3            # Analyze every Nth frame
```

## Output

Each analysis produces a self-contained HTML dashboard with:
- Embedded video player with clickable timeline
- Interactive Sankey diagram (hover for frame galleries, click to jump video)
- Object sentiment panel with sparklines
- Segment cards with filmstrip thumbnails and narrative text

## Requirements

- Python 3.10+
- PyTorch, TorchVision, OpenCV, Plotly, Pillow, NumPy
- `yt-dlp` (for YouTube downloads)
- Optional: Tesseract (for OCR text detection)

## Project Structure

```
analyze_video.py          # Main CLI analyzer + HTML generation
server.py                 # Web server (port 8888)
pipeline.py               # Demo pipeline with synthetic data
models/
  cnn_visual.py           # ResNet50 visual feature extractor
  lstm_temporal.py        # BiLSTM with attention
  ocr_text.py             # Tesseract OCR + embedding encoder
  fusion.py               # Gated attention fusion layer
  object_detector.py      # Faster R-CNN wrapper (80 COCO classes)
visualization/
  frame_renderer.py       # Synthetic frame rendering + filmstrips
  sankey.py               # Plotly Sankey diagram generation
docs/
  index.html              # GitHub Pages demo site
```

#!/usr/bin/env python3
"""
Web server for Video Story Analysis.

Usage:
    python3 server.py              # starts on port 8888
    python3 server.py --port 9000  # custom port

Open http://localhost:8888 in your browser, paste any YouTube URL, and click Analyze.
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import http.server
import json
import subprocess
import sys
import threading
import urllib.parse
import webbrowser
from pathlib import Path

PORT = 8888
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

LANDING_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Video Story Analyzer</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#0d47a1;min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;color:#fff}

.container{max-width:700px;width:90%;text-align:center}
h1{font-size:36px;font-weight:800;margin-bottom:8px}
.sub{font-size:14px;opacity:.7;margin-bottom:30px}

.input-group{display:flex;gap:8px;margin-bottom:16px}
.input-group input{flex:1;padding:14px 20px;border:2px solid rgba(255,255,255,.3);border-radius:30px;
  background:rgba(255,255,255,.1);color:#fff;font-size:15px;outline:none;transition:all .2s}
.input-group input:focus{border-color:#fff;background:rgba(255,255,255,.2)}
.input-group input::placeholder{color:rgba(255,255,255,.5)}
.input-group button{padding:14px 32px;border:none;border-radius:30px;background:#fff;color:#0d47a1;
  font-size:15px;font-weight:700;cursor:pointer;transition:all .2s;white-space:nowrap}
.input-group button:hover{background:#e3f2fd;transform:scale(1.02)}
.input-group button:disabled{opacity:.6;cursor:wait}

.options{display:flex;gap:12px;justify-content:center;margin-bottom:20px;flex-wrap:wrap}
.opt{display:flex;align-items:center;gap:4px;font-size:12px;opacity:.8}
.opt label{font-weight:600}
.opt input,.opt select{padding:4px 8px;border:1px solid rgba(255,255,255,.3);border-radius:8px;
  background:rgba(255,255,255,.1);color:#fff;font-size:11px;outline:none}

.status{margin-top:20px;padding:14px;border-radius:10px;background:rgba(0,0,0,.3);display:none;text-align:left;font-size:12px;max-height:300px;overflow-y:auto}
.status.visible{display:block}
.status .line{margin-bottom:2px;font-family:monospace;line-height:1.5}
.status .done{color:#69f0ae;font-weight:700}
.status .error{color:#ff5252;font-weight:700}

.recent{margin-top:30px;text-align:left}
.recent h3{font-size:13px;font-weight:600;opacity:.7;margin-bottom:8px}
.recent a{display:block;color:#82b1ff;font-size:12px;text-decoration:none;padding:4px 0;transition:color .2s}
.recent a:hover{color:#fff}

.features{display:flex;gap:16px;margin-top:30px;flex-wrap:wrap;justify-content:center}
.feat{background:rgba(255,255,255,.08);padding:10px 16px;border-radius:10px;text-align:left;max-width:200px}
.feat h4{font-size:11px;font-weight:700;margin-bottom:4px}.feat p{font-size:10px;opacity:.7;line-height:1.4}
</style>
</head><body>

<div class="container">
  <h1>&#127916; Video Story Analyzer</h1>
  <div class="sub">Paste any YouTube URL — AI detects objects, tells the story, builds an interactive Sankey</div>

  <div class="input-group">
    <input type="text" id="url" placeholder="https://www.youtube.com/watch?v=... or /path/to/video.mp4" autofocus />
    <button id="btn" onclick="analyze()">&#9654; Analyze</button>
  </div>

  <div class="options">
    <div class="opt"><label>Segments:</label><input type="number" id="segments" value="8" min="4" max="20" style="width:50px"/></div>
    <div class="opt"><label>Confidence:</label><input type="number" id="conf" value="0.40" min="0.1" max="0.9" step="0.05" style="width:60px"/></div>
    <div class="opt"><label>Sample:</label>
      <select id="sample">
        <option value="3">Every 3rd frame (slow)</option>
        <option value="5" selected>Every 5th frame</option>
        <option value="8">Every 8th frame (fast)</option>
      </select>
    </div>
  </div>

  <div class="status" id="status"></div>

  <div class="recent" id="recent"></div>

  <div class="features">
    <div class="feat"><h4>&#128270; Object Detection</h4><p>Faster R-CNN detects 80+ COCO object types in every frame</p></div>
    <div class="feat"><h4>&#128200; Sentiment Tracking</h4><p>Per-object confidence & frequency trends with sparklines</p></div>
    <div class="feat"><h4>&#9889; Time Scaling</h4><p>Busy moments expand, quiet moments contract in the Sankey</p></div>
    <div class="feat"><h4>&#127916; Story Arc</h4><p>Opening → Rising → Climax → Falling → Resolution narrative</p></div>
  </div>
</div>

<script>
const statusDiv = document.getElementById('status');
const btn = document.getElementById('btn');

function analyze() {
  const url = document.getElementById('url').value.trim();
  if (!url) { alert('Enter a URL or file path'); return; }
  const segs = document.getElementById('segments').value;
  const conf = document.getElementById('conf').value;
  const sample = document.getElementById('sample').value;

  btn.textContent = 'Processing...';
  btn.disabled = true;
  statusDiv.className = 'status visible';
  statusDiv.innerHTML = '<div class="line">Starting analysis...</div>';

  const es = new EventSource('/analyze-stream?' + new URLSearchParams({url, segments: segs, confidence: conf, sample}));

  es.onmessage = function(e) {
    const data = JSON.parse(e.data);
    if (data.line) {
      statusDiv.innerHTML += '<div class="line">' + data.line + '</div>';
      statusDiv.scrollTop = statusDiv.scrollHeight;
    }
    if (data.done) {
      statusDiv.innerHTML += '<div class="line done">&#10004; Done! Redirecting...</div>';
      es.close();
      btn.textContent = '&#9654; Analyze';
      btn.disabled = false;
      setTimeout(() => window.location.href = data.redirect, 500);
    }
    if (data.error) {
      statusDiv.innerHTML += '<div class="line error">&#10008; ' + data.error + '</div>';
      es.close();
      btn.textContent = '&#9654; Analyze';
      btn.disabled = false;
    }
  };

  es.onerror = function() {
    statusDiv.innerHTML += '<div class="line error">Connection lost. Check terminal.</div>';
    es.close();
    btn.textContent = '&#9654; Analyze';
    btn.disabled = false;
  };
}

// Load recent analyses
fetch('/recent').then(r=>r.json()).then(files => {
  if (files.length === 0) return;
  const div = document.getElementById('recent');
  div.innerHTML = '<h3>Recent Analyses</h3>' + files.map(f =>
    '<a href="/results/' + encodeURIComponent(f.name) + '">' + f.name.replace('_analysis.html','') + ' (' + f.size + ')</a>'
  ).join('');
});

// Enter key submits
document.getElementById('url').addEventListener('keydown', e => { if (e.key === 'Enter') analyze(); });
</script>
</body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "":
            self._respond(200, "text/html", LANDING_HTML)

        elif path == "/recent":
            files = []
            for f in sorted(DATA_DIR.glob("*_analysis.html"), key=lambda p: p.stat().st_mtime, reverse=True)[:10]:
                size_mb = f.stat().st_size / 1024 / 1024
                files.append({"name": f.name, "size": f"{size_mb:.1f} MB"})
            self._respond(200, "application/json", json.dumps(files))

        elif path.startswith("/results/"):
            fname = urllib.parse.unquote(path[len("/results/"):])
            fpath = DATA_DIR / fname
            if fpath.exists() and fpath.suffix == ".html":
                content = fpath.read_text()
                # Inject server mode flag
                content = content.replace("window._serverMode = false;", "window._serverMode = true;")
                self._respond(200, "text/html", content)
            else:
                self._respond(404, "text/plain", "Not found")

        elif path == "/analyze-stream":
            params = urllib.parse.parse_qs(parsed.query)
            url = params.get("url", [""])[0]
            segs = params.get("segments", ["8"])[0]
            conf = params.get("confidence", ["0.40"])[0]
            sample = params.get("sample", ["5"])[0]

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            def send_event(data):
                try:
                    msg = f"data: {json.dumps(data)}\n\n"
                    self.wfile.write(msg.encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

            send_event({"line": f"Analyzing: {url}"})
            send_event({"line": f"Settings: segments={segs}, confidence={conf}, sample={sample}"})

            script = str(Path(__file__).parent / "analyze_video.py")
            cmd = [sys.executable, script, url,
                   "--segments", segs, "--confidence", conf, "--sample", sample]

            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, bufsize=1)
                output_file = None
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        send_event({"line": line})
                        if "Done! Open:" in line:
                            output_file = line.split("Open:")[-1].strip()

                proc.wait()

                if proc.returncode == 0 and output_file:
                    fname = Path(output_file).name
                    send_event({"done": True, "redirect": f"/results/{fname}"})
                else:
                    send_event({"error": f"Process exited with code {proc.returncode}"})
            except Exception as e:
                send_event({"error": str(e)})

        else:
            self._respond(404, "text/plain", "Not found")

    def do_POST(self):
        if self.path == "/analyze":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            url = body.get("url", "")
            # Redirect to stream-based approach
            self._respond(200, "application/json", json.dumps({"redirect": f"/?url={url}"}))
        else:
            self._respond(404, "text/plain", "Not found")

    def _respond(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        if isinstance(body, str):
            body = body.encode()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Cleaner logging
        print(f"  [{self.address_string()}] {args[0]}" if args else "")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Video Story Analyzer — Web Server")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port (default: {PORT})")
    args = parser.parse_args()

    server = http.server.HTTPServer(("0.0.0.0", args.port), Handler)
    url = f"http://localhost:{args.port}"

    print("=" * 55)
    print(f"  Video Story Analyzer Server")
    print(f"  Running at: {url}")
    print(f"  Ctrl+C to stop")
    print("=" * 55)

    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()

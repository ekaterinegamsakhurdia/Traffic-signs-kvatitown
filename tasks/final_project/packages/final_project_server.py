"""
Final-project server for the Duckiebot (real robot).

Changes from the original
──────────────────────────
* The YOLO / ObjectDetectionAgent detection pipeline has been removed entirely:
  - No separate detection thread, frame queue, or shared detections list.
  - No GPU / TensorRT dependency for obstacle detection.
  - draw_detections() is no longer called.

* Obstacle detection is now purely colour-based (HSV corridor masks) and runs
  inside agent.compute_commands() on every frame, synchronously with the lane-
  following and traffic-rule logic.

* The web overlay has been updated to show corridor bounds and active threats
  from the colour-based detector in place of YOLO bounding boxes.

Everything else (camera, wheels, manual mode, AprilTags, red-line, crossroad
state machine) is unchanged.
"""

import sys
import os
import signal
import threading
import time
import socket

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..', '..')
sys.path.insert(0, project_root)

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify, request

from tasks.final_project.packages.agent import FinalProjectAgent
from tasks.final_project.packages.apriltag_activity import draw_tags

from duckiebot.camera_driver import CameraDriver
from duckiebot.wheel_driver import DaguWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from launcher.ports import find_available_port
from servers.common import make_frame_generator, shutdown_cleanup, suppress_http_logs


# ── Minimal web UI ─────────────────────────────────────────────────────────────
_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Duckiebot — Final Project</title>
<style>
  body { margin:0; background:#111; color:#eee; font-family:monospace; }
  h2   { margin:8px 12px; font-size:1rem; color:#aef; }
  #feed    { display:block; max-width:100%; border:2px solid #333; }
  #controls{ padding:8px 12px; display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
  button   { padding:6px 14px; border:none; border-radius:4px; cursor:pointer;
             font-size:.85rem; background:#444; color:#eee; }
  button:hover { background:#666; }
  #btnStart { background:#2a7; } #btnStop { background:#a33; }
  #status  { padding:6px 12px; font-size:.8rem; color:#bbb; }
  #mode-row{ padding:4px 12px; }
  label    { margin-right:12px; cursor:pointer; }
</style>
</head>
<body>
<h2>Duckiebot — Final Project ({{ hostname }})</h2>
<img id="feed" src="/video">
<div id="controls">
  <button id="btnStart" onclick="post('/start')">▶ Start</button>
  <button id="btnStop"  onclick="post('/stop')">■ Stop</button>
  <button onclick="post('/reset')">↺ Reset</button>
</div>
<div id="mode-row">
  <label><input type="radio" name="mode" value="auto"   onchange="setMode('auto')"   checked> Auto</label>
  <label><input type="radio" name="mode" value="manual" onchange="setMode('manual')">         Manual (WASD)</label>
</div>
<div id="status">Loading…</div>
<script>
function post(url, data){
  fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data||{})});
}
function setMode(m){ post('/set_mode',{mode:m}); }

// WASD manual control
const keys = {w:false,a:false,s:false,d:false};
document.addEventListener('keydown', e => {
  if(e.key in keys){ keys[e.key]=true; sendKeys(); }
});
document.addEventListener('keyup', e => {
  if(e.key in keys){ keys[e.key]=false; sendKeys(); }
});
function sendKeys(){
  post('/keys',{up:keys.w, down:keys.s, left:keys.a, right:keys.d});
}

// Status polling
setInterval(() => {
  fetch('/status').then(r=>r.json()).then(d=>{
    const el = document.getElementById('status');
    const threats = (d.threats||[]).map(t=>t.label+'@('+t.cx.toFixed(2)+','+t.cy.toFixed(2)+')['+t.zone+']').join('  ');
    el.textContent = [
      d.running ? '🟢 running' : '🔴 stopped',
      'state='+(d.behavior_state||'?'),
      'reason='+(d.behavior_reason||'?'),
      threats ? '⚠ '+threats : '',
    ].filter(Boolean).join('  |  ');
  }).catch(()=>{});
}, 500);
</script>
</body>
</html>"""


# ── Application globals ───────────────────────────────────────────────────────
app = Flask(__name__)

agent:   FinalProjectAgent | None = None
camera:  CameraDriver       | None = None
wheels:  DaguWheelsDriver   | None = None

running     = False
manual_mode = False
stop_event  = threading.Event()

keys_pressed      = {'up': False, 'down': False, 'left': False, 'right': False}
_keys_lock        = threading.Lock()
_keys_last_update = time.time()


# ── Manual control thread ─────────────────────────────────────────────────────

def manual_control_loop():
    global _keys_last_update
    while not stop_event.is_set():
        if not manual_mode or not wheels:
            time.sleep(0.05)
            continue

        # Safety: zero wheels if keys haven't been refreshed for 0.5 s.
        if time.time() - _keys_last_update > 0.5:
            with _keys_lock:
                for k in keys_pressed:
                    keys_pressed[k] = False

        with _keys_lock:
            kc = keys_pressed.copy()

        left = right = 0.0
        if kc['up']:    left, right = 0.5,  0.5
        if kc['down']:  left, right = -0.5, -0.5
        if kc['up']   and kc['left']:  left, right = 0.2, 0.5
        elif kc['up'] and kc['right']: left, right = 0.5, 0.2
        elif kc['left']:  left, right = -0.3,  0.3
        elif kc['right']: left, right =  0.3, -0.3

        wheels.set_wheels_speed(left, right)
        time.sleep(0.05)


# ── Frame visualiser ──────────────────────────────────────────────────────────

def _draw_corridor_overlay(frame_bgr: np.ndarray) -> None:
    """
    Draw the estimated driving corridor and any active threats directly onto
    *frame_bgr* (in-place).  Replaces the old YOLO bounding-box overlay.
    """
    if agent is None:
        return

    h, w = frame_bgr.shape[:2]

    # Corridor boundaries
    left_x, right_x = agent.obj_detector.last_corridor
    if right_x > left_x:
        cv2.line(frame_bgr, (left_x,  0), (left_x,  h), (0, 200, 200), 1)
        cv2.line(frame_bgr, (right_x, 0), (right_x, h), (200, 200, 0), 1)

    # Active threats
    for t in agent.last_threats:
        cx_px = int(t.cx_norm * w)
        cy_px = int(t.cy_norm * h)
        colour = (0, 80, 255) if t.cls_id == 0 else (255, 120, 0)   # red=duck, orange=truck
        cv2.circle(frame_bgr, (cx_px, cy_px), 18, colour, 3)
        cv2.putText(
            frame_bgr,
            f"{t.label} [{t.proximity_zone}]",
            (cx_px - 40, cy_px - 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA,
        )


def visualize(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Called by make_frame_generator() for every camera frame.

    Obstacle detection now runs inside agent.compute_commands() itself, so
    this function only needs to:
      1. Drive the wheels based on the agent's decision.
      2. Draw the debug overlay.
    """
    if wheels is None:
        return frame_bgr

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if not manual_mode and agent is not None:
        # All perception (lane follow + colour obstacles + AprilTags + red line)
        # happens inside compute_commands().  No separate detection pipeline.
        pwm_left, pwm_right = agent.compute_commands(frame_rgb)

        if running:
            wheels.set_wheels_speed(pwm_left, pwm_right)
        else:
            wheels.set_wheels_speed(0.0, 0.0)

    # ── Overlay ───────────────────────────────────────────────────────────────
    if agent is not None:
        # AprilTag outlines
        draw_tags(frame_bgr, agent.last_tags)

        # Corridor bounds + threat circles (replaces YOLO boxes)
        _draw_corridor_overlay(frame_bgr)

        # Behaviour state banner
        decision = agent.last_decision
        if decision:
            label = (
                f"{decision.state.value}  {decision.reason}"
                + (f"  turn={decision.chosen_turn}" if decision.chosen_turn else "")
            )
            cv2.putText(
                frame_bgr, label,
                (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                (0, 255, 255), 2, cv2.LINE_AA,
            )

    return frame_bgr


generate_frames = make_frame_generator(lambda: camera, visualize, quality=50, rgb=False)


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(_HTML, hostname=socket.gethostname())


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start', methods=['POST'])
def start():
    global running
    running = True
    return jsonify({'status': 'running'})


@app.route('/stop', methods=['POST'])
def stop():
    global running
    running = False
    if wheels:
        wheels.set_wheels_speed(0.0, 0.0)
    return jsonify({'status': 'stopped'})


@app.route('/reset', methods=['POST'])
def reset():
    global running
    running = False
    if wheels:
        wheels.set_wheels_speed(0.0, 0.0)
    if agent:
        agent.rules._reset_state()
        agent.last_tags     = []
        agent.last_decision = None
        agent.last_threats  = []
    return jsonify({'status': 'reset'})


@app.route('/set_mode', methods=['POST'])
def set_mode():
    global manual_mode
    mode = request.json.get('mode', 'auto') if request.json else 'auto'
    manual_mode = (mode == 'manual')
    if wheels and not manual_mode:
        wheels.set_wheels_speed(0.0, 0.0)
    return jsonify({'mode': 'manual' if manual_mode else 'auto'})


@app.route('/keys', methods=['POST'])
def update_keys():
    global _keys_last_update
    data = request.json or {}
    with _keys_lock:
        for k in keys_pressed:
            keys_pressed[k] = bool(data.get(k, False))
    _keys_last_update = time.time()
    return jsonify({'status': 'ok'})


@app.route('/status')
def status():
    decision = agent.last_decision if agent else None
    return jsonify({
        'running':         running,
        'manual_mode':     manual_mode,

        'apriltag_ready':  agent.tag_detector.ready   if agent else False,
        'apriltag_backend':agent.tag_detector.backend if agent else None,
        'tags':            agent.last_tags             if agent else [],

        'behavior_state':  decision.state.value   if decision else None,
        'behavior_reason': decision.reason        if decision else None,
        'active_tag_id':   decision.active_tag_id if decision else None,
        'chosen_turn':     decision.chosen_turn   if decision else None,

        'threats': [
            {
                'label': t.label,
                'cx':    round(t.cx_norm,   3),
                'cy':    round(t.cy_norm,   3),
                'area':  round(t.area_frac, 4),
                'zone':  t.proximity_zone,
                'side':  t.side,
            }
            for t in (agent.last_threats if agent else [])
        ],
    })


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global agent, camera, wheels

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=5000)
    args = ap.parse_args()

    suppress_http_logs()
    print('=' * 60)
    print('FINAL PROJECT — TRAFFIC SIGNS + APRILTAGS (REAL ROBOT)')
    print('Obstacle detection: colour-based corridor masks (no YOLO)')
    print('=' * 60)

    def _init_wheels():
        global wheels
        wheels = DaguWheelsDriver(WheelPWMConfiguration(), WheelPWMConfiguration())
        print('[Init] Wheels ready')

    def _init_camera():
        global camera
        cam = CameraDriver()
        cam.start()
        camera = cam
        print('[Init] Camera ready')

    def _init_agent():
        global agent
        agent = FinalProjectAgent()
        print(f'[Init] FinalProjectAgent ready — lane speed: {agent.lane_agent.base_speed}')
        print(f'[Init] AprilTag ready: {agent.tag_detector.ready} ({agent.tag_detector.backend})')
        print(
            f'[Init] Obstacle detector: '
            f'duck_area≥{agent.obj_detector.duck_min_area_frac:.3f} '
            f'truck_area≥{agent.obj_detector.truck_min_area_frac:.3f} '
            f'confirm={agent.obj_detector.confirm_frames}fr '
            f'clear={agent.obj_detector.clear_frames}fr'
        )

    threading.Thread(target=_init_wheels,         daemon=True).start()
    threading.Thread(target=_init_camera,         daemon=True).start()
    threading.Thread(target=_init_agent,          daemon=True).start()
    threading.Thread(target=manual_control_loop,  daemon=True).start()

    def _shutdown(signum, frame):
        shutdown_cleanup(wheels, camera, stop_event)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    web_port = find_available_port(args.port)
    print(f'\nWeb Interface: http://{socket.gethostname()}.local:{web_port}')
    print('=' * 60 + '\n')

    try:
        app.run(host='0.0.0.0', port=web_port, debug=False, threaded=True)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        shutdown_cleanup(wheels, camera, stop_event)


if __name__ == '__main__':
    sys.exit(main())
import sys
import os
import threading
import time
import socket

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify, request

from tasks.final_project.packages.agent import FinalProjectAgent
from tasks.final_project.packages.apriltag_activity import draw_tags

# ── Inline template — no dependency on the old object_detection package ───────
_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Duckiebot — Final Project Virtual ({{ hostname }})</title>
<style>
  body  { margin:0; background:#111; color:#eee; font-family:monospace; }
  h2    { margin:8px 12px; font-size:1rem; color:#aef; }
  #feed { display:block; max-width:100%; border:2px solid #333; }
  #controls { padding:8px 12px; display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
  button { padding:6px 14px; border:none; border-radius:4px; cursor:pointer;
           font-size:.85rem; background:#444; color:#eee; }
  button:hover { background:#666; }
  #btnStart { background:#2a7; } #btnStop { background:#a33; }
  #scenes   { padding:4px 12px; display:flex; gap:6px; flex-wrap:wrap; }
  .scene-btn { background:#336; font-size:.8rem; }
  #mode-row { padding:4px 12px; }
  label { margin-right:12px; cursor:pointer; }
  #status { padding:6px 12px; font-size:.8rem; color:#bbb; min-height:1.4em; }
  #gameover { display:none; padding:4px 12px; color:#f88; font-weight:bold; }
</style>
</head>
<body>
<h2>Duckiebot — Final Project (Godot) &mdash; {{ hostname }}</h2>
<img id="feed" src="/video">
<div id="controls">
  <button id="btnStart" onclick="post('/start')">&#9654; Start</button>
  <button id="btnStop"  onclick="post('/stop')">&#9632; Stop</button>
  <button onclick="post('/reset')">&#8635; Reset</button>
</div>
<div id="scenes">
  <span style="color:#aaa;font-size:.8rem;align-self:center">Scenes:</span>
  <button class="scene-btn" onclick="switchScene('final_project')">Final Project</button>
  <button class="scene-btn" onclick="switchScene('lane_following')">Lane Follow</button>
  <button class="scene-btn" onclick="switchScene('introduction')">Intro (manual)</button>
</div>
<div id="mode-row">
  <label><input type="radio" name="mode" value="auto"   onchange="setMode('auto')"   checked> Auto</label>
  <label><input type="radio" name="mode" value="manual" onchange="setMode('manual')"> Manual (WASD)</label>
</div>
<div id="gameover">&#9888; GAME OVER — click Reset to restart</div>
<div id="status">—</div>
<script>
function post(url, data) {
  fetch(url, {method:'POST', headers:{'Content-Type':'application/json'},
              body: JSON.stringify(data||{})});
}
function setMode(m)       { post('/set_mode',    {mode:m}); }
function switchScene(s)   { post('/switch_scene', {scene:s}); }

const keys = {w:false, a:false, s:false, d:false};
document.addEventListener('keydown', e => { if(e.key in keys){ keys[e.key]=true;  sendKeys(); }});
document.addEventListener('keyup',   e => { if(e.key in keys){ keys[e.key]=false; sendKeys(); }});
function sendKeys() {
  post('/keys', {up:keys.w, down:keys.s, left:keys.a, right:keys.d});
}

setInterval(() => {
  fetch('/status').then(r => r.json()).then(d => {
    document.getElementById('gameover').style.display = d.game_over ? 'block' : 'none';
    const threats = (d.threats||[])
      .map(t => t.label + '@(' + t.cx.toFixed(2) + ',' + t.cy.toFixed(2)
               + ') ' + t.zone).join('  ');
    document.getElementById('status').textContent = [
      d.running ? '🟢 running' : '🔴 stopped',
      'scene='  + (d.current_scene   || '?'),
      'state='  + (d.behavior_state  || '?'),
      'reason=' + (d.behavior_reason || '?'),
      threats ? '⚠ ' + threats : '',
    ].filter(Boolean).join('  |  ');
  }).catch(() => {});
}, 500);
</script>
</body>
</html>"""

from duckiebot.camera_driver.godot_camera_driver import GodotCameraDriver, GodotCameraConfig
from duckiebot.wheel_driver.godot_wheels_driver import GodotWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from launcher.ports import find_available_port
from launcher.config import GODOT_SCENES
from servers.common import make_frame_generator, shutdown_cleanup, suppress_http_logs


app = Flask(__name__)

agent = None
camera = None
wheels = None

running = False
manual_mode = False
stop_event = threading.Event()

keys_pressed = {'up': False, 'down': False, 'left': False, 'right': False}
_keys_lock = threading.Lock()
_keys_last_update = time.time()

_current_scene = 'final_project'


def manual_control_loop():
    global _keys_last_update
    while not stop_event.is_set():
        if not manual_mode or not wheels:
            time.sleep(0.05)
            continue

        if time.time() - _keys_last_update > 0.5:
            with _keys_lock:
                for k in keys_pressed:
                    keys_pressed[k] = False

        with _keys_lock:
            kc = keys_pressed.copy()

        left = right = 0.0
        if kc['up']:
            left, right = 0.5, 0.5
        if kc['down']:
            left, right = -0.5, -0.5
        if kc['up'] and kc['left']:
            left, right = 0.2, 0.5
        elif kc['up'] and kc['right']:
            left, right = 0.5, 0.2
        elif kc['left']:
            left, right = -0.3, 0.3
        elif kc['right']:
            left, right = 0.3, -0.3

        if not wheels.is_game_over():
            wheels.set_wheels_speed(left, right)

        time.sleep(0.05)


def _draw_corridor_overlay(frame_bgr: np.ndarray) -> None:
    """
    Draw the curved driving corridor and any active threats onto *frame_bgr*.

    The corridor is drawn as two polylines — one tracing the yellow centre-line
    (left boundary, drawn in yellow) and one tracing the white right boundary
    (drawn in cyan).  These follow road curvature rather than being vertical
    straight lines.  Active threats are shown as labelled circles.
    """
    if agent is None:
        return
    h, w = frame_bgr.shape[:2]

    poly = agent.obj_detector.last_corridor_poly   # [(y, left_x, right_x), ...]
    if poly and len(poly) >= 2:
        left_pts  = np.array([[lx, y] for y, lx, _  in poly], dtype=np.int32)
        right_pts = np.array([[rx, y] for y, _,  rx in poly], dtype=np.int32)
        # Yellow lane boundary → yellow (BGR 0, 200, 200)
        cv2.polylines(frame_bgr, [left_pts.reshape(-1, 1, 2)],  False, (0, 200, 200), 2)
        # White lane boundary  → cyan   (BGR 200, 200, 0)
        cv2.polylines(frame_bgr, [right_pts.reshape(-1, 1, 2)], False, (200, 200, 0), 2)

    for t in agent.last_threats:
        cx_px = int(t.cx_norm * w)
        cy_px = int(t.cy_norm * h)
        colour = (0, 80, 255) if t.cls_id == 0 else (255, 120, 0)
        cv2.circle(frame_bgr, (cx_px, cy_px), 18, colour, 3)
        cv2.putText(frame_bgr, f"{t.label}[{t.proximity_zone}]",
                    (cx_px - 40, cy_px - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, colour, 2, cv2.LINE_AA)


def visualize(frame_rgb):
    """
    Receives an RGB frame from the Godot camera.
    Obstacle detection is handled inside FinalProjectAgent (corridor-based,
    no external YOLO model required).
    """
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if wheels is None:
        return bgr

    if manual_mode:
        pass
    elif agent is not None:
        pwm_left, pwm_right = agent.compute_commands(frame_rgb)

        if running and not wheels.is_game_over():
            wheels.set_wheels_speed(pwm_left, pwm_right)
        else:
            wheels.set_wheels_speed(0.0, 0.0)

    # Draw AprilTags, corridor overlay, and behavior state.
    if agent is not None:
        draw_tags(bgr, agent.last_tags)
        _draw_corridor_overlay(bgr)
        decision = agent.last_decision
        if decision:
            cv2.putText(
                bgr,
                f"{decision.state.value} {decision.reason} turn={decision.chosen_turn}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return bgr


generate_frames = make_frame_generator(lambda: camera, visualize, quality=50)


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
    if wheels:
        wheels.reset_game()
    running = True
    if agent:
        agent.rules._reset_state()
        agent.last_tags     = []
        agent.last_decision = None
        agent.last_threats  = []
    return jsonify({'status': 'reset', 'running': running})


@app.route('/set_mode', methods=['POST'])
def set_mode():
    global manual_mode
    mode = request.json.get('mode', 'auto') if request.json else 'auto'
    manual_mode = (mode == 'manual')
    if wheels and not manual_mode:
        wheels.set_wheels_speed(0.0, 0.0)
    return jsonify({'mode': 'manual' if manual_mode else 'auto'})


@app.route('/switch_scene', methods=['POST'])
def switch_scene():
    global manual_mode, _current_scene
    target = request.json.get('scene', '') if request.json else ''
    if target not in GODOT_SCENES:
        return jsonify({'error': f'unknown scene {target!r}'}), 400
    if wheels:
        wheels.change_scene(GODOT_SCENES[target])
    _current_scene = target
    manual_mode = (target == 'introduction')
    if wheels and not manual_mode:
        wheels.set_wheels_speed(0.0, 0.0)
    return jsonify({'scene': target, 'manual_mode': manual_mode})


@app.route('/keys', methods=['POST'])
def update_keys():
    global _keys_last_update
    data = request.json or {}
    with _keys_lock:
        for k in keys_pressed:
            keys_pressed[k] = bool(data.get(k, False))
    _keys_last_update = time.time()
    return jsonify({'status': 'ok'})


@app.route('/remove_objects', methods=['POST'])
def remove_objects():
    name_filter = request.json.get('filter', '') if request.json else ''
    if wheels and name_filter:
        wheels.remove_objects(name_filter)
    return jsonify({'status': 'ok', 'filter': name_filter})


@app.route('/status')
def status():
    decision = agent.last_decision if agent else None

    return jsonify({
        'running':      running,
        'manual_mode':  manual_mode,
        'current_scene': _current_scene,
        'game_over':    wheels.is_game_over() if wheels else False,

        'apriltag_ready':   agent.tag_detector.ready if agent else False,
        'apriltag_backend': agent.tag_detector.backend if agent else None,
        'tags':             agent.last_tags if agent else [],

        'behavior_state':  decision.state.value if decision else None,
        'behavior_reason': decision.reason if decision else None,
        'active_tag_id':   decision.active_tag_id if decision else None,
        'chosen_turn':     decision.chosen_turn if decision else None,

        'threats': [
            {
                'label': t.label,
                'side':  t.side,
                'area':  round(t.area_frac, 4),
                'cx':    round(t.cx_norm,   3),
                'cy':    round(t.cy_norm,   3),
                'zone':  t.proximity_zone,
            }
            for t in (agent.last_threats if agent else [])
        ],
    })


def main():
    global agent, camera, wheels

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=5000)
    ap.add_argument('--frame-port', type=int, default=5001)
    ap.add_argument('--wheel-port', type=int, default=5002)
    ap.add_argument('--godot-host', type=str, default='localhost')
    args = ap.parse_args()

    suppress_http_logs()
    print('=' * 60)
    print('FINAL PROJECT — TRAFFIC SIGNS + APRILTAGS + GODOT')
    print('=' * 60)

    print('\n[1/3] Creating final project agent...')
    agent = FinalProjectAgent()
    print(f'  Lane speed: {agent.lane_agent.base_speed}')
    print(f'  AprilTag ready: {agent.tag_detector.ready} ({agent.tag_detector.backend})')

    print('\n[2/3] Initializing wheels...')
    wheels = GodotWheelsDriver(
        WheelPWMConfiguration(pwm_min=0), WheelPWMConfiguration(pwm_min=0),
        godot_host=args.godot_host, godot_port=args.wheel_port,
    )

    print('\n[3/3] Initializing Godot camera...')
    camera = GodotCameraDriver(godot_config=GodotCameraConfig(host='0.0.0.0', port=args.frame_port))
    camera.start()

    threading.Thread(target=manual_control_loop, daemon=True).start()

    web_port = find_available_port(args.port)
    print(f'\nWeb Interface: http://localhost:{web_port}')
    print('=' * 60 + '\n')

    try:
        app.run(host='127.0.0.1', port=web_port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print('\nShutting down...')
    finally:
        shutdown_cleanup(wheels, camera, stop_event)


if __name__ == '__main__':
    sys.exit(main())
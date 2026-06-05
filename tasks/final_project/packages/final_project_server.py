import sys
import os
import signal
import threading
import time
import queue
import socket

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..', '..')
sys.path.insert(0, project_root)

import cv2
from flask import Flask, Response, render_template_string, jsonify, request

from tasks.final_project.packages.agent import FinalProjectAgent
from tasks.final_project.packages.apriltag_activity import draw_tags
from tasks.object_detection.packages.agent import ObjectDetectionAgent, CLASS_NAMES
from servers.object_detection.visualization import draw_detections
from servers.templates.object_detection import OBJECT_DETECTION_TEMPLATE as HTML_TEMPLATE

from duckiebot.camera_driver import CameraDriver
from duckiebot.wheel_driver import DaguWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from launcher.ports import find_available_port
from servers.common import make_frame_generator, shutdown_cleanup, suppress_http_logs


app = Flask(__name__)

agent     = None
det_agent = None
camera    = None
wheels    = None

running     = False
manual_mode = False
stop_event  = threading.Event()

_frame_queue     = queue.Queue(maxsize=1)
_last_detections = []
_detection_lock  = threading.Lock()

keys_pressed      = {'up': False, 'down': False, 'left': False, 'right': False}
_keys_lock        = threading.Lock()
_keys_last_update = time.time()

# Only send every Nth frame to the object detector.
# Camera runs at ~30 fps; N=3 gives ~10 fps detection which is plenty
# for a slow-moving robot and keeps CPU/GPU load low.
DETECTION_EVERY_N_FRAMES = 3
_det_frame_counter = 0


def detection_loop():
    global _last_detections
    while not stop_event.is_set():
        if det_agent is None or not det_agent.model_loaded:
            time.sleep(0.1)
            continue

        try:
            frame_rgb = _frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        result = det_agent.detect(frame_rgb)
        if result is not None:
            with _detection_lock:
                _last_detections = result


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

        wheels.set_wheels_speed(left, right)
        time.sleep(0.05)


def visualize(frame_bgr):
    """
    Receives a BGR frame from the real camera (CameraDriver).
    Mirrors virtual.server.py logic but works in BGR space throughout.
    """
    global _det_frame_counter

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Object detector runs in a side thread.
    # FIX 1: queue frame BEFORE the wheels guard so detection starts immediately.
    # FIX 2: pre-resize to model input size to reduce queue bandwidth.
    # FIX 3: only queue every DETECTION_EVERY_N_FRAMES frames to limit GPU load.
    _det_frame_counter += 1
    if (det_agent is not None and det_agent.model_loaded
            and _det_frame_counter % DETECTION_EVERY_N_FRAMES == 0):
        try:
            small = cv2.resize(frame_rgb, (det_agent.img_size, det_agent.img_size))
            _frame_queue.put_nowait(small)
        except queue.Full:
            pass

    if wheels is None:
        return frame_bgr

    with _detection_lock:
        detections = list(_last_detections)

    if manual_mode:
        pass
    elif agent is not None:
        pwm_left, pwm_right = agent.compute_commands(frame_rgb, detections=detections)

        if running:
            wheels.set_wheels_speed(pwm_left, pwm_right)
        else:
            wheels.set_wheels_speed(0.0, 0.0)

    # Draw YOLO detections.
    if det_agent is not None and det_agent.model_loaded and detections:
        draw_detections(frame_bgr, detections)

    # Draw AprilTags and behavior state.
    if agent is not None:
        draw_tags(frame_bgr, agent.last_tags)
        decision = agent.last_decision
        if decision:
            cv2.putText(
                frame_bgr,
                f"{decision.state.value} {decision.reason} turn={decision.chosen_turn}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return frame_bgr


generate_frames = make_frame_generator(lambda: camera, visualize, quality=50, rgb=False)


@app.route('/')
def index():
    return render_template_string(
        HTML_TEMPLATE,
        config=det_agent,
        hostname=socket.gethostname(),
        virtual=False,
    )


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
    global _last_detections, running
    running = False
    if wheels:
        wheels.set_wheels_speed(0.0, 0.0)
    if agent:
        agent.rules._reset_state()
        agent.last_tags     = []
        agent.last_decision = None
    with _detection_lock:
        _last_detections = []
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


@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    value = request.json.get('value') if request.json else None
    if det_agent and value is not None:
        det_agent.conf_threshold = float(value)
    return jsonify({'conf_threshold': det_agent.conf_threshold if det_agent else None})


@app.route('/status')
def status():
    with _detection_lock:
        dets = list(_last_detections)

    decision = agent.last_decision if agent else None

    return jsonify({
        'running':      running,
        'manual_mode':  manual_mode,

        'model_loaded': det_agent.model_loaded if det_agent else False,
        'load_error':   det_agent.load_error if det_agent else None,
        'trt_building': getattr(det_agent, 'trt_building', False) if det_agent else False,
        'conf_threshold': det_agent.conf_threshold if det_agent else 0.5,

        'apriltag_ready':   agent.tag_detector.ready if agent else False,
        'apriltag_backend': agent.tag_detector.backend if agent else None,
        'tags':             agent.last_tags if agent else [],

        'behavior_state':  decision.state.value if decision else None,
        'behavior_reason': decision.reason if decision else None,
        'active_tag_id':   decision.active_tag_id if decision else None,
        'chosen_turn':     decision.chosen_turn if decision else None,

        'detections': [
            {'class': CLASS_NAMES.get(c, str(c)), 'score': round(s, 3), 'bbox': list(b)}
            for b, s, c in dets
        ],
    })


def main():
    global agent, det_agent, camera, wheels

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=5000)
    args = ap.parse_args()

    suppress_http_logs()
    print('=' * 60)
    print('FINAL PROJECT — TRAFFIC SIGNS + APRILTAGS (REAL ROBOT)')
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

    def _init_agents():
        global agent, det_agent
        agent = FinalProjectAgent()
        print(f'[Init] FinalProjectAgent ready — lane speed: {agent.lane_agent.base_speed}')
        print(f'[Init] AprilTag ready: {agent.tag_detector.ready} ({agent.tag_detector.backend})')

        det_agent = ObjectDetectionAgent()
        if det_agent.model_loaded:
            print(f'[Init] YOLO ready: {det_agent.img_size}px')
        else:
            print(f'[Init] YOLO WARNING: {det_agent.load_error}')

    threading.Thread(target=_init_wheels,        daemon=True).start()
    threading.Thread(target=_init_camera,        daemon=True).start()
    threading.Thread(target=_init_agents,        daemon=True).start()
    threading.Thread(target=detection_loop,      daemon=True).start()
    threading.Thread(target=manual_control_loop, daemon=True).start()

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
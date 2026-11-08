import os
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

if os.name == 'nt':
    os.system('')
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass

import time
import socket
import struct
import threading
import json
import math
from collections import deque
import numpy as np
import pygame
from scipy.interpolate import interp1d
import joblib
import torch
import torch.nn as nn

from ahrs_utils import SensorCalibration, MadgwickAHRS
from ui_theme import (
    BG_VOID, BG_PANEL, SURFACE_CARD, SURFACE_CARD_ALT, SURFACE_PILL, SURFACE_TRACK,
    BORDER_BASE, BORDER_FOCUS, BORDER_MUTED,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    ACCENT_CYAN, ACCENT_EMERALD, ACCENT_AMBER, ACCENT_CORAL, ACCENT_VIOLET, ACCENT_GOLD,
    WAND_GRIP, WAND_SHAFT, WAND_TIP_AURA, WAND_TIP_CORE,
    GRID_LINE, GRID_AXIS_X, GRID_AXIS_Y, GRID_AXIS_Z,
    draw_card, draw_pill_badge, draw_progress_bar, draw_glow_sphere
)
from train_models import (
    Gesture1DCNN,
    GestureConvLSTM,
    extract_rf_features,
    resample_sequence,
    GESTURE_NAMES,
    FEATURE_COLS,
    TARGET_LEN
)

# Dźwięki na Windows
try:
    import winsound
    def beep(freq, duration):
        winsound.Beep(freq, duration)
except ImportError:
    def beep(freq, duration):
        pass

MODELS_DIR = "models"
META_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
RF_PATH = os.path.join(MODELS_DIR, "random_forest_model.joblib")
CNN_PATH = os.path.join(MODELS_DIR, "gesture_1d_cnn.pth")
LSTM_PATH = os.path.join(MODELS_DIR, "gesture_conv_lstm.pth")

PORT = 2137
DATA_FORMAT = "<I9f?"
PACKET_SIZE = struct.calcsize(DATA_FORMAT)

# English gesture display names
GESTURE_EN = {
    "circle":   "CIRCLE",
    "go_up":    "SWIPE UP",
    "go_down":  "SWIPE DOWN",
    "go_left":  "SWIPE LEFT",
    "go_right": "SWIPE RIGHT",
    "iks":      "X-MARK",
    "idle":     "IDLE"
}

# Continuous Sliding Window & Stationarity Gating Parameters
MOTION_THRESHOLD = 0.38        # Angular velocity threshold for motion onset [rad/s]
REST_THRESHOLD = 0.28          # Angular velocity threshold for rest / stationary [rad/s]
MIN_GESTURE_SAMPLES = 25       # Minimum gesture duration (250 ms at 100 Hz)
MAX_GESTURE_SAMPLES = 220      # Maximum gesture duration (2.2 s at 100 Hz)
STATIONARY_HOLD_MS = 150       # Time below REST_THRESHOLD to confirm completion (150 ms)
COOLDOWN_MS = 320              # Refractory period to avoid secondary bounce triggers (320 ms)


class MultiModelClassifier:
    """
    Supports models: Conv-LSTM (1D-CNN + BiLSTM), 1D-CNN, and Random Forest.
    """
    def __init__(self):
        if not os.path.exists(META_PATH):
            raise FileNotFoundError(f"Missing metadata file: {META_PATH}. Run train_models.py first!")

        with open(META_PATH, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.class_names = self.meta["gesture_names"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Random Forest
        self.rf_model = joblib.load(RF_PATH)

        # 2. 1D-CNN
        self.cnn_model = Gesture1DCNN(in_channels=6, num_classes=len(self.class_names)).to(self.device)
        self.cnn_model.load_state_dict(torch.load(CNN_PATH, map_location=self.device))
        self.cnn_model.eval()

        # 3. Conv-LSTM (BiLSTM)
        self.has_lstm = os.path.exists(LSTM_PATH)
        if self.has_lstm:
            self.lstm_model = GestureConvLSTM(in_channels=6, num_classes=len(self.class_names)).to(self.device)
            self.lstm_model.load_state_dict(torch.load(LSTM_PATH, map_location=self.device))
            self.lstm_model.eval()
            print("[OK] Loaded Conv-LSTM model (1D-CNN + BiLSTM).")
        else:
            self.lstm_model = None

        self.active_model_name = "Conv-LSTM (BiLSTM)" if self.has_lstm else "1D-CNN"
        print(f"[OK] Models loaded. Active engine: {self.active_model_name}")

    def toggle_model(self):
        if self.has_lstm:
            if self.active_model_name == "Conv-LSTM (BiLSTM)":
                self.active_model_name = "1D-CNN"
            elif self.active_model_name == "1D-CNN":
                self.active_model_name = "Random Forest"
            else:
                self.active_model_name = "Conv-LSTM (BiLSTM)"
        else:
            self.active_model_name = "Random Forest" if self.active_model_name == "1D-CNN" else "1D-CNN"
        return self.active_model_name

    def predict_sequence(self, raw_mat: np.ndarray):
        """
        Przyjmuje sekwencję (N, 6), interpoluje do TARGET_LEN i wykonuje inferencję.
        """
        resampled = resample_sequence(raw_mat, TARGET_LEN) # (100, 6)

        if self.active_model_name == "Conv-LSTM (BiLSTM)" and self.lstm_model is not None:
            tensor_in = torch.from_numpy(resampled.T.astype(np.float32)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.lstm_model(tensor_in)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        elif self.active_model_name == "1D-CNN":
            tensor_in = torch.from_numpy(resampled.T.astype(np.float32)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.cnn_model(tensor_in)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        else: # Random Forest
            feats = extract_rf_features(resampled).reshape(1, -1)
            probs = self.rf_model.predict_proba(feats)[0]

        best_idx = int(np.argmax(probs))
        return probs, self.class_names[best_idx], float(probs[best_idx])


class LiveDemoVisualizer:
    def __init__(self, classifier: MultiModelClassifier, existing_server=None):
        self.clf = classifier
        self.server = existing_server
        self.running = True
        self.calib = SensorCalibration()
        self.ahrs = MadgwickAHRS(sample_rate=100.0, gain=0.1, use_mag=False, filter_type="ekf")

        self.lock = threading.Lock()
        self.packet_queue = []
        self.tcp_client = None
        self.total_packets = 0
        self.auto_tared = False

        # Angular velocity dynamics (Omega)
        self.current_omega = 0.0
        self.smooth_omega = 0.0
        self.peak_omega = 0.0

        # Continuous sliding window and stationarity gating state
        self.in_gesture = False
        self.pre_roll = deque(maxlen=8)        # 80 ms pre-motion buffer
        self.stroke_buffer = []                # Active motion trajectory samples
        self.rolling_window = deque(maxlen=100) # Continuous 1.0 s rolling window
        self.last_motion_time_ms = 0
        self.stroke_start_time = 0.0
        self.cooldown_until_ms = 0

        # Smooth class probabilities for HUD
        self.smooth_probs = np.zeros(len(self.clf.class_names), dtype=np.float32)
        idle_idx = self.clf.class_names.index("idle") if "idle" in self.clf.class_names else 0
        self.smooth_probs[idle_idx] = 1.0

        # Last recognized gesture
        self.latest_result = None
        self.result_display_timer = 0.0
        self.gesture_history = []

        # Network sockets (only if standalone without existing WandDataServer)
        if self.server is None:
            self.tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.tcp_server.bind(("0.0.0.0", PORT))
            self.tcp_server.listen(1)

            self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
            self.udp_sock.bind(("0.0.0.0", PORT))

            self.tcp_thread = threading.Thread(target=self._tcp_loop, daemon=True)
            self.udp_thread = threading.Thread(target=self._udp_loop, daemon=True)
            self.tcp_thread.start()
            self.udp_thread.start()
        else:
            self.tcp_server = None
            self.udp_sock = None
            self.tcp_thread = None
            self.udp_thread = None

        # Pygame
        pygame.init()
        self.width, self.height = 1260, 780
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("IMU LIVE GESTURE RECOGNITION - Sliding Window & Stationarity Gating")
        self.clock = pygame.time.Clock()
        self.dt = 0.016

        sans_fonts = ["segoe ui", "trebuchet ms", "arial", "helvetica", "dejavu sans"]
        mono_fonts = ["consolas", "fira code", "cascadia code", "courier new"]
        self.font_title   = pygame.font.SysFont(sans_fonts, 20, bold=True)
        self.font_main    = pygame.font.SysFont(sans_fonts, 15, bold=True)
        self.font_small   = pygame.font.SysFont(sans_fonts, 13)
        self.font_badge   = pygame.font.SysFont(sans_fonts, 11, bold=True)
        self.font_mono    = pygame.font.SysFont(mono_fonts, 12, bold=True)
        self.font_mono_sm = pygame.font.SysFont(mono_fonts, 11)
        self.font_tiny    = pygame.font.SysFont(mono_fonts, 11)

        self.flip_y = -1.0
        self.flip_z = -1.0
        self.cam_yaw = math.radians(-125.0)
        self.cam_pitch = math.radians(22.0)
        self.cam_dist = 3.2

        self.live_trail = deque(maxlen=85)

        # Geometria pałeczki 3D (wzdłuż osi X)
        self.wand_spine = np.array([
            [-0.35, 0.0, 0.0],
            [ 0.05, 0.0, 0.0],
            [ 1.10, 0.0, 0.0],
        ], dtype=np.float64)

        n_sides = 6
        angles = [i * (2.0 * math.pi / n_sides) for i in range(n_sides)]
        self.ring_back = np.array([[-0.35, 0.065 * math.cos(a), 0.065 * math.sin(a)] for a in angles])
        self.ring_mid = np.array([[0.05, 0.065 * math.cos(a), 0.065 * math.sin(a)] for a in angles])
        self.ring_shaft = np.array([[0.05, 0.032 * math.cos(a), 0.032 * math.sin(a)] for a in angles])
        self.ring_tip = np.array([[1.05, 0.012 * math.cos(a), 0.012 * math.sin(a)] for a in angles])

        self.axes_local = {
            "X": (np.array([0.5, 0.0, 0.0]), (255, 75, 75)),
            "Y": (np.array([0.0, 0.5, 0.0]), (75, 240, 110)),
            "Z": (np.array([0.0, 0.0, 0.5]), (80, 160, 255)),
        }

    def _tcp_loop(self):
        print(f"[TCP] Listening for ESP32 connection on port {PORT}...")
        while self.running:
            try:
                client, addr = self.tcp_server.accept()
                with self.lock:
                    if self.tcp_client:
                        try:
                            self.tcp_client.close()
                        except Exception:
                            pass
                    self.tcp_client = client
                print(f"[TCP] Connected to ESP32: {addr[0]}. Sending START ('0')...")
                client.sendall(b"0")
            except Exception:
                if self.running:
                    time.sleep(0.5)

    def _udp_loop(self):
        while self.running:
            try:
                raw_data, _ = self.udp_sock.recvfrom(2048)
                if len(raw_data) >= PACKET_SIZE and (len(raw_data) % PACKET_SIZE == 0):
                    batch = []
                    for fields in struct.iter_unpack(DATA_FORMAT, raw_data):
                        batch.append({
                            "timestamp_ms": fields[0],
                            "accel_x": fields[1], "accel_y": fields[2], "accel_z": fields[3],
                            "gyro_x":  fields[4], "gyro_y":  fields[5], "gyro_z":  fields[6],
                            "mag_x":   fields[7], "mag_y":   fields[8], "mag_z":   fields[9],
                            "wand_id": 1 if fields[10] else 0
                        })
                    with self.lock:
                        self.packet_queue.extend(batch)
            except Exception:
                pass

    def _transform(self, pts_local, R):
        pts_w = pts_local @ R.T
        pts_w = pts_w.copy()
        if pts_w.ndim == 1:
            pts_w[1] *= self.flip_y
            pts_w[2] *= self.flip_z
        else:
            pts_w[:, 1] *= self.flip_y
            pts_w[:, 2] *= self.flip_z
        return pts_w

    def _project(self, pt3d, center_x, center_y):
        x, y, z = pt3d
        cy, sy = math.cos(self.cam_yaw), math.sin(self.cam_yaw)
        x1 = x * cy - y * sy
        y1 = x * sy + y * cy
        z1 = z

        cp, sp = math.cos(self.cam_pitch), math.sin(self.cam_pitch)
        cam_x = y1
        cam_y = z1 * cp - x1 * sp
        cam_z = self.cam_dist + z1 * sp + x1 * cp
        if cam_z < 0.2:
            cam_z = 0.2

        fov = min(self.width, self.height) * 0.78
        px = int(center_x + (cam_x * fov) / cam_z)
        py = int(center_y - (cam_y * fov) / cam_z)
        return (px, py), cam_z

    def _draw_prism(self, r1_w, r2_w, col, cx, cy):
        n = len(r1_w)
        p1 = [self._project(p, cx, cy)[0] for p in r1_w]
        p2 = [self._project(p, cx, cy)[0] for p in r2_w]
        for i in range(n):
            nxt = (i + 1) % n
            pygame.draw.line(self.screen, col, p1[i], p1[nxt], 2)
            pygame.draw.line(self.screen, col, p2[i], p2[nxt], 2)
            pygame.draw.line(self.screen, col, p1[i], p2[i], 2)

    def run(self):
        print("\n" + "="*60)
        print("LIVE GESTURE RECOGNITION (Sliding Window + Stationarity Gating)")
        print("Perform a gesture in the air and hold still / rest to confirm.")
        print(f" -> Active Model: {self.clf.active_model_name}")
        print(" -> 3D Orientation: Quaternion SLERP (60 FPS)")
        print(" -> Key [N]: Toggle Model (1D-CNN <-> RF <-> Conv-LSTM)")
        print(" -> Key [R / SPACE]: Tare 3D orientation")
        print(" -> Key [C]: Clear gesture history")
        print(" -> Key [ESC]: Exit to menu")
        print("="*60)

        while self.running:
            self.dt = max(0.001, min(0.1, self.clock.tick(60) / 1000.0))
            self.width, self.height = self.screen.get_size()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_r, pygame.K_SPACE):
                        self.ahrs.reset()
                        self.live_trail.clear()
                        if self.server is not None:
                            self.server.start_streaming()
                        elif self.tcp_client is not None:
                            try:
                                self.tcp_client.sendall(b"0")
                            except Exception:
                                pass
                    elif event.key == pygame.K_n:
                        new_name = self.clf.toggle_model()
                        print(f"[MODEL] Switched model to: {new_name}")
                    elif event.key == pygame.K_c:
                        self.gesture_history.clear()

            # 1. Fetch new packets from shared server or network queue
            if self.server is not None:
                new_samples = self.server.get_live_samples()
            else:
                with self.lock:
                    new_samples = self.packet_queue[:]
                    self.packet_queue.clear()

            # 2. Continuous Sliding Window Processing & Gyroscope Stationarity Gating
            if new_samples:
                for s in new_samples:
                    cal_acc, cal_gyro = self.calib.apply(s, adaptive_gyro=False)
                    s["cal_accel_x"], s["cal_accel_y"], s["cal_accel_z"] = cal_acc
                    s["cal_gyro_x"],  s["cal_gyro_y"],  s["cal_gyro_z"]  = cal_gyro

                    mag = self.calib.align_mag_to_lsm6ds3(s) if getattr(self.ahrs, 'use_mag', False) else None
                    self.ahrs.update(cal_gyro, cal_acc, mag=mag, timestamp_ms=s["timestamp_ms"])
                    self.total_packets += 1
                    if not self.auto_tared and self.total_packets >= 15:
                        self.ahrs.reset()
                        self.auto_tared = True

                    # Angular velocity magnitude ||ω||
                    g_norm = float(np.linalg.norm(cal_gyro))
                    self.current_omega = g_norm
                    self.smooth_omega = 0.70 * self.smooth_omega + 0.30 * g_norm
                    self.rolling_window.append(s)

                    now_ms = s["timestamp_ms"]

                    # Skip triggers during refractory cooldown
                    if now_ms < self.cooldown_until_ms:
                        self.in_gesture = False
                        self.stroke_buffer.clear()
                        continue

                    # Stationarity-Gated State Machine (No return-to-center required!)
                    if not self.in_gesture:
                        self.pre_roll.append(s)
                        if self.smooth_omega >= MOTION_THRESHOLD:
                            self.in_gesture = True
                            self.stroke_buffer = list(self.pre_roll)
                            self.stroke_buffer.append(s)
                            self.stroke_start_time = time.time()
                            self.peak_omega = g_norm
                            self.last_motion_time_ms = now_ms
                    else:
                        self.stroke_buffer.append(s)
                        if g_norm > self.peak_omega:
                            self.peak_omega = g_norm

                        if self.smooth_omega >= REST_THRESHOLD:
                            # Still actively moving
                            self.last_motion_time_ms = now_ms
                        else:
                            # Motion energy dropped below REST_THRESHOLD: check stationary hold duration
                            time_stationary = now_ms - self.last_motion_time_ms
                            if time_stationary >= STATIONARY_HOLD_MS:
                                # Stationarity reached at gesture completion!
                                if len(self.stroke_buffer) >= MIN_GESTURE_SAMPLES and self.peak_omega >= 0.50:
                                    mat = np.array([
                                        [x["cal_accel_x"], x["cal_accel_y"], x["cal_accel_z"],
                                         x["cal_gyro_x"],  x["cal_gyro_y"],  x["cal_gyro_z"]]
                                        for x in self.stroke_buffer
                                    ], dtype=np.float32)

                                    probs, best_name, conf = self.clf.predict_sequence(mat)

                                    if best_name != "idle" and conf >= 0.45:
                                        self.latest_result = {
                                            "name": best_name,
                                            "conf": conf,
                                            "n_samples": len(self.stroke_buffer),
                                            "duration_s": len(self.stroke_buffer) / 100.0,
                                            "model": self.clf.active_model_name
                                        }
                                        self.result_display_timer = time.time() + 2.6
                                        self.gesture_history.insert(0, (best_name, conf, time.strftime("%H:%M:%S")))
                                        if len(self.gesture_history) > 6:
                                            self.gesture_history.pop()

                                        beep(1500, 110)
                                        self.cooldown_until_ms = now_ms + COOLDOWN_MS
                                        disp_name = GESTURE_EN.get(best_name, best_name)
                                        print(f" >>> [RECOGNIZED] {disp_name:<16} ({conf*100:5.1f}%) | Samples: {len(self.stroke_buffer):3d} ({len(self.stroke_buffer)/100:.2f} s) | Peak ||ω||: {self.peak_omega:.2f} rad/s")

                                self.stroke_buffer.clear()
                                self.in_gesture = False
                                self.peak_omega = 0.0

                        # Guard against prolonged motion
                        if len(self.stroke_buffer) > MAX_GESTURE_SAMPLES:
                            self.stroke_buffer.clear()
                            self.in_gesture = False
                            self.peak_omega = 0.0
            else:
                self.smooth_omega *= 0.94

            # 3. Continuous Live Visual Feedforward (Dynamic Probability Bars)
            live_best_name = "idle"
            live_best_conf = 1.0

            if self.in_gesture and len(self.stroke_buffer) >= 15:
                mat_ongoing = np.array([
                    [x["cal_accel_x"], x["cal_accel_y"], x["cal_accel_z"],
                     x["cal_gyro_x"],  x["cal_gyro_y"],  x["cal_gyro_z"]]
                    for x in self.stroke_buffer
                ], dtype=np.float32)

                live_probs, live_best_name, live_best_conf = self.clf.predict_sequence(mat_ongoing)
                self.smooth_probs = 0.65 * self.smooth_probs + 0.35 * live_probs
            else:
                if not self.in_gesture:
                    target_p = np.zeros_like(self.smooth_probs)
                    idle_i = self.clf.class_names.index("idle") if "idle" in self.clf.class_names else 0
                    target_p[idle_i] = 1.0
                    self.smooth_probs = 0.85 * self.smooth_probs + 0.15 * target_p
                    live_best_name = "idle"
                    live_best_conf = float(self.smooth_probs[idle_i])

            # 4. Render 3D Scene (SLERP Quaternion Smoothing 60 FPS)
            self.screen.fill(BG_VOID)

            panel_w = 450
            viewport_w = self.width - panel_w
            center_3d_x = int(viewport_w / 2)
            center_3d_y = int(self.height / 2) + 25

            # Smooth rotation matrix with network packet batching decoupling
            R = self.ahrs.get_smooth_rotation_matrix(dt=self.dt, smoothing_speed=24.0)
            spine_w = self._transform(self.wand_spine, R)
            tip_world = spine_w[2]
            self.live_trail.append(tip_world.copy())

            # Horizon Floor Grid
            for g in np.linspace(-1.2, 1.2, 7):
                p1, _ = self._project((g, -1.2, -0.9), center_3d_x, center_3d_y)
                p2, _ = self._project((g,  1.2, -0.9), center_3d_x, center_3d_y)
                pygame.draw.line(self.screen, GRID_LINE, p1, p2, 1)
                p3, _ = self._project((-1.2, g, -0.9), center_3d_x, center_3d_y)
                p4, _ = self._project(( 1.2, g, -0.9), center_3d_x, center_3d_y)
                pygame.draw.line(self.screen, GRID_LINE, p3, p4, 1)

            # Tip motion plasma trail
            if len(self.live_trail) > 1:
                trail_2d = [self._project(pt, center_3d_x, center_3d_y)[0] for pt in self.live_trail]
                for i in range(1, len(trail_2d)):
                    r = i / len(trail_2d)
                    col = (
                        int(ACCENT_GOLD[0] * r + ACCENT_CYAN[0] * (1.0 - r)),
                        int(ACCENT_GOLD[1] * r + ACCENT_CYAN[1] * (1.0 - r)),
                        int(ACCENT_GOLD[2] * r + ACCENT_CYAN[2] * (1.0 - r))
                    ) if self.smooth_omega >= MOTION_THRESHOLD else (
                        int(ACCENT_CYAN[0] * r + 20 * (1.0 - r)),
                        int(ACCENT_CYAN[1] * r + 40 * (1.0 - r)),
                        int(ACCENT_CYAN[2] * r + 80 * (1.0 - r))
                    )
                    pygame.draw.line(self.screen, col, trail_2d[i-1], trail_2d[i], max(1, int(4 * r)))

            # 3D Wand Mesh (Aligned along X-axis)
            self._draw_prism(self._transform(self.ring_back, R), self._transform(self.ring_mid, R), WAND_GRIP, center_3d_x, center_3d_y)
            self._draw_prism(self._transform(self.ring_shaft, R), self._transform(self.ring_tip, R), WAND_SHAFT, center_3d_x, center_3d_y)

            p_back, _ = self._project(spine_w[0], center_3d_x, center_3d_y)
            p_mid, _  = self._project(spine_w[1], center_3d_x, center_3d_y)
            p_tip, _  = self._project(spine_w[2], center_3d_x, center_3d_y)
            pygame.draw.line(self.screen, (130, 150, 180), p_back, p_mid, 8)
            pygame.draw.line(self.screen, (245, 150, 45), p_mid, p_tip, 5)
            draw_glow_sphere(self.screen, p_tip, base_radius=8, color_rgb=(255, 130, 45))

            # Local coordinate axes
            orig_2d, _ = self._project((0.0, 0.0, 0.0), center_3d_x, center_3d_y)
            for lbl, (vec_l, col) in self.axes_local.items():
                v_w = self._transform(vec_l, R)
                e_2d, _ = self._project(v_w, center_3d_x, center_3d_y)
                pygame.draw.line(self.screen, col, orig_2d, e_2d, 3)

            # Floating Euler Angles Card (Bottom-Left Viewport)
            r_deg, p_deg, y_deg = self.ahrs.get_euler_angles_deg(smooth=True)
            euler_w, euler_h = 360, 36
            euler_rect = pygame.Rect(18, self.height - euler_h - 16, euler_w, euler_h)
            draw_card(self.screen, euler_rect, bg_color=SURFACE_CARD, border_color=BORDER_BASE, radius=18)
            rot_str = f"Roll: {r_deg:+6.1f}°  Pitch: {p_deg:+6.1f}°  Yaw: {y_deg:+6.1f}°"
            self.screen.blit(self.font_mono.render(rot_str, True, ACCENT_CYAN), (euler_rect.x + 18, euler_rect.y + 10))

            # =========================================================================
            # FLOATING TOP STATUS CAPSULE (LEFT VIEWPORT)
            # =========================================================================
            top_w = viewport_w - 36
            top_h = 72
            top_rect = pygame.Rect(18, 14, top_w, top_h)
            now_sec = time.time()

            if now_sec < self.result_display_timer and self.latest_result:
                gname = self.latest_result["name"]
                gen = GESTURE_EN.get(gname, gname)
                draw_card(self.screen, top_rect, bg_color=(18, 54, 38), border_color=ACCENT_EMERALD, radius=12, border_w=2)
                pygame.draw.circle(self.screen, ACCENT_EMERALD, (top_rect.x + 24, top_rect.y + 24), 7)
                t1 = self.font_title.render(f"★ GESTURE RECOGNIZED: {gen}", True, TEXT_PRIMARY)
                sub_txt = f"Confidence: {self.latest_result['conf']*100:5.1f}%  |  Duration: {self.latest_result['duration_s']:.2f} s ({self.latest_result['n_samples']} samples)  |  Model: {self.latest_result['model']}"
                t2 = self.font_small.render(sub_txt, True, (215, 255, 230))
            elif self.in_gesture:
                lead_name = live_best_name
                lead_en = GESTURE_EN.get(lead_name, lead_name)
                lead_pct = live_best_conf * 100
                draw_card(self.screen, top_rect, bg_color=(48, 34, 12), border_color=ACCENT_AMBER, radius=12, border_w=2)
                pygame.draw.circle(self.screen, ACCENT_AMBER, (top_rect.x + 24, top_rect.y + 24), 7)
                t1 = self.font_title.render(f"GESTURE IN PROGRESS: {lead_en} ({lead_pct:4.1f}%)", True, TEXT_PRIMARY)
                dur_s = len(self.stroke_buffer) / 100.0
                t2 = self.font_small.render(f"Samples: {len(self.stroke_buffer)} ({dur_s:.2f} s)  |  Peak ||ω||: {self.peak_omega:4.2f} rad/s  |  Hold still to confirm", True, (255, 235, 185))
            else:
                draw_card(self.screen, top_rect, bg_color=SURFACE_CARD, border_color=BORDER_BASE, radius=12)
                pygame.draw.circle(self.screen, ACCENT_CYAN, (top_rect.x + 24, top_rect.y + 24), 7)
                t1 = self.font_title.render("STATUS: READY (IDLE) — Perform gesture in the air", True, TEXT_PRIMARY)
                t2 = self.font_small.render(f"Dynamics ||ω||: {self.smooth_omega:4.2f} rad/s (Threshold: {MOTION_THRESHOLD:.2f} rad/s)  |  Key [N]: Switch engine", True, TEXT_SECONDARY)

            self.screen.blit(t1, (top_rect.x + 42, top_rect.y + 12))
            self.screen.blit(t2, (top_rect.x + 42, top_rect.y + 42))

            # =========================================================================
            # RIGHT COMMAND CENTER PANEL (BENTO DASHBOARD)
            # =========================================================================
            px = viewport_w
            side_rect = pygame.Rect(px, 0, panel_w, self.height)
            pygame.draw.rect(self.screen, BG_PANEL, side_rect)
            pygame.draw.line(self.screen, BORDER_BASE, (px, 0), (px, self.height), 2)

            # 1. Header Card (Model Engine)
            hdr_rect = pygame.Rect(px + 16, 14, panel_w - 32, 54)
            draw_card(self.screen, hdr_rect, bg_color=SURFACE_CARD, border_color=BORDER_BASE, radius=10)
            pygame.draw.circle(self.screen, ACCENT_VIOLET, (hdr_rect.x + 20, hdr_rect.y + 27), 6)

            model_lbl = f"{self.clf.active_model_name}"
            self.screen.blit(self.font_main.render(model_lbl, True, TEXT_PRIMARY), (hdr_rect.x + 36, hdr_rect.y + 11))
            self.screen.blit(self.font_tiny.render("AI CLASSIFICATION ENGINE", True, TEXT_MUTED), (hdr_rect.x + 36, hdr_rect.y + 32))

            draw_pill_badge(self.screen, "[N] Switch", self.font_badge, hdr_rect.right - 96, hdr_rect.y + 14,
                            bg_color=SURFACE_PILL, text_color=ACCENT_CYAN, border_color=BORDER_BASE, padding_x=8, padding_y=3)

            # 2. Probability Telemetry Bento Card
            probs_card = pygame.Rect(px + 16, 76, panel_w - 32, 290)
            draw_card(self.screen, probs_card, bg_color=SURFACE_CARD, border_color=BORDER_BASE, radius=10)

            self.screen.blit(self.font_badge.render("REAL-TIME CLASS PROBABILITIES", True, TEXT_MUTED), (probs_card.x + 18, probs_card.y + 14))

            bar_y = probs_card.y + 38
            for i, name in enumerate(self.clf.class_names):
                prob = float(self.smooth_probs[i])
                en_label = GESTURE_EN.get(name, name)
                is_top = (prob >= 0.40) and (i == int(np.argmax(self.smooth_probs)))

                # Item row container with responsive padding
                row_rect = pygame.Rect(probs_card.x + 10, bar_y - 4, probs_card.w - 20, 28)
                if is_top:
                    draw_card(self.screen, row_rect, bg_color=SURFACE_CARD_ALT, border_color=ACCENT_CYAN, radius=6, border_w=1)
                    txt_col = TEXT_PRIMARY
                    bar_col = ACCENT_EMERALD if prob >= 0.70 else ACCENT_AMBER
                else:
                    txt_col = TEXT_SECONDARY
                    bar_col = SURFACE_TRACK

                # Class Name Label
                self.screen.blit(self.font_small.render(en_label, True, txt_col), (row_rect.x + 10, bar_y + 2))

                # Percentage Readout (Right Aligned inside row_rect)
                pct_str = f"{prob * 100:5.1f}%"
                pct_surf = self.font_mono.render(pct_str, True, txt_col)
                pct_x = row_rect.right - pct_surf.get_width() - 10
                self.screen.blit(pct_surf, (pct_x, bar_y + 2))

                # Adaptive Progress Bar (Fills gap between label and percentage, zero overflow)
                bar_x = row_rect.x + 115
                bar_w = max(20, pct_x - bar_x - 12)
                draw_progress_bar(self.screen, bar_x, bar_y + 6, bar_w, 12, prob,
                                  fill_color=bar_col, bg_color=SURFACE_TRACK, border_color=None, radius=4)

                bar_y += 34

            # 3. Kinetic Dynamics Gauge Card (Omega)
            dyn_card = pygame.Rect(px + 16, 374, panel_w - 32, 116)
            draw_card(self.screen, dyn_card, bg_color=SURFACE_CARD, border_color=BORDER_BASE, radius=10)

            self.screen.blit(self.font_badge.render("KINETIC DYNAMICS (ANGULAR RATE)", True, TEXT_MUTED), (dyn_card.x + 18, dyn_card.y + 14))

            # Omega Numerical Readout (Right Aligned inside dyn_card)
            omega_val_str = f"{self.smooth_omega:4.2f} rad/s"
            val_col = ACCENT_CORAL if self.smooth_omega >= MOTION_THRESHOLD else (ACCENT_AMBER if self.smooth_omega >= REST_THRESHOLD else ACCENT_CYAN)
            val_surf = self.font_main.render(omega_val_str, True, val_col)
            self.screen.blit(val_surf, (dyn_card.right - val_surf.get_width() - 18, dyn_card.y + 12))

            state_badge = "IN MOTION" if self.smooth_omega >= MOTION_THRESHOLD else ("ARMED" if self.smooth_omega >= REST_THRESHOLD else "RESTING")
            state_col = ACCENT_CORAL if self.smooth_omega >= MOTION_THRESHOLD else (ACCENT_AMBER if self.smooth_omega >= REST_THRESHOLD else ACCENT_CYAN)
            draw_pill_badge(self.screen, state_badge, self.font_tiny, dyn_card.x + 18, dyn_card.y + 36,
                            bg_color=SURFACE_PILL, text_color=state_col, border_color=BORDER_BASE, dot_color=state_col, padding_x=8, padding_y=2)

            # Gauge Track
            omega_bar_x = dyn_card.x + 18
            omega_bar_w = dyn_card.w - 36
            omega_bar_h = 16
            bar_top_y = dyn_card.y + 66

            fill_ratio = min(1.0, max(0.0, self.smooth_omega / 4.0))
            draw_progress_bar(self.screen, omega_bar_x, bar_top_y, omega_bar_w, omega_bar_h,
                              fill_ratio, fill_color=val_col, bg_color=SURFACE_TRACK, border_color=None, radius=5)

            # Threshold Marker Flag
            thresh_x = int(omega_bar_x + (MOTION_THRESHOLD / 4.0) * omega_bar_w)
            pygame.draw.line(self.screen, TEXT_PRIMARY, (thresh_x, bar_top_y - 4), (thresh_x, bar_top_y + omega_bar_h + 4), 2)
            self.screen.blit(self.font_tiny.render(f"Threshold {MOTION_THRESHOLD:.2f}", True, TEXT_SECONDARY), (thresh_x - 30, bar_top_y + omega_bar_h + 5))

            # 4. History Bento Card
            hist_card = pygame.Rect(px + 16, 498, panel_w - 32, self.height - 498 - 60)
            draw_card(self.screen, hist_card, bg_color=SURFACE_CARD, border_color=BORDER_BASE, radius=10)

            self.screen.blit(self.font_badge.render("RECENT DETECTIONS", True, TEXT_MUTED), (hist_card.x + 18, hist_card.y + 14))
            draw_pill_badge(self.screen, "[C] Clear", self.font_tiny, hist_card.right - 85, hist_card.y + 10,
                            bg_color=SURFACE_PILL, text_color=TEXT_SECONDARY, border_color=BORDER_BASE, padding_x=6, padding_y=2)

            hist_y = hist_card.y + 40
            if not self.gesture_history:
                self.screen.blit(self.font_small.render("Perform gestures in the air...", True, TEXT_MUTED), (hist_card.x + 18, hist_y + 10))
            else:
                for idx, (h_name, h_conf, h_time) in enumerate(self.gesture_history[:6]):
                    h_en = GESTURE_EN.get(h_name, h_name)
                    item_col = TEXT_PRIMARY if idx == 0 else TEXT_SECONDARY
                    dot_col = ACCENT_EMERALD if idx == 0 else TEXT_MUTED

                    row_y = hist_y + idx * 24
                    pygame.draw.circle(self.screen, dot_col, (hist_card.x + 22, row_y + 8), 4)
                    item_txt = f"{h_time}  {h_en:<12} ({h_conf*100:4.1f}%)"
                    self.screen.blit(self.font_mono.render(item_txt, True, item_col), (hist_card.x + 34, row_y))

            # 5. Bottom Navigation / Shortcut Pill (Centered)
            foot_rect = pygame.Rect(px + 16, self.height - 48, panel_w - 32, 36)
            draw_card(self.screen, foot_rect, bg_color=SURFACE_CARD, border_color=BORDER_BASE, radius=18)
            foot_str = "[SPACE] Tare   [N] Model   [C] Clear   [ESC] Exit"
            foot_surf = self.font_mono_sm.render(foot_str, True, TEXT_MUTED)
            foot_x = foot_rect.x + (foot_rect.w - foot_surf.get_width()) // 2
            self.screen.blit(foot_surf, (foot_x, foot_rect.y + 10))

            pygame.display.flip()

        pygame.display.quit()
        self.running = False
        if self.server is None:
            try:
                if self.tcp_client:
                    self.tcp_client.sendall(b"1")
                    self.tcp_client.close()
                if self.tcp_server:
                    self.tcp_server.close()
                if self.udp_sock:
                    self.udp_sock.close()
            except Exception:
                pass


def main():
    print("Initializing Multi-Model Classifier...")
    clf = MultiModelClassifier()
    vis = LiveDemoVisualizer(clf)
    vis.run()


if __name__ == "__main__":
    main()

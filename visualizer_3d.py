import math
import time
import socket
import struct
import threading
from collections import deque
import numpy as np
import pygame

from ahrs_utils import SensorCalibration, MadgwickAHRS, HAS_AHRS_LIB
from ui_theme import (
    BG_VOID, BG_PANEL, SURFACE_CARD, SURFACE_CARD_ALT, SURFACE_PILL, SURFACE_TRACK,
    BORDER_BASE, BORDER_FOCUS, BORDER_MUTED,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    ACCENT_CYAN, ACCENT_EMERALD, ACCENT_AMBER, ACCENT_CORAL, ACCENT_VIOLET, ACCENT_GOLD,
    WAND_GRIP, WAND_SHAFT, WAND_TIP_AURA, WAND_TIP_CORE,
    GRID_LINE, GRID_AXIS_X, GRID_AXIS_Y, GRID_AXIS_Z,
    Typography, draw_card, draw_pill_badge, draw_progress_bar, draw_glow_sphere
)

# struct __attribute__((packed)) SensorData: uint32_t (I) + 9x float (9f) + 1x bool (?) = 41 bajtów
DATA_FORMAT = "<I9f?"
PACKET_SIZE = struct.calcsize(DATA_FORMAT)
PORT = 2137


class Wand3DVisualizer:
    def __init__(self, existing_server=None, use_mag=True, filter_type="ekf"):
        self.server = existing_server
        self.running = True
        self.calib = SensorCalibration()
        # Domyślnie 9-DOF (use_mag=True) oraz Extended Kalman Filter (filter_type="ekf")
        self.ahrs = MadgwickAHRS(sample_rate=100.0, gain=0.1, use_mag=use_mag, filter_type=filter_type)

        self.processed_count = 0
        self.total_packets = 0
        self.latest_gyro_norm = 0.0
        self.wand_id = 0
        self.auto_tared = False
        self.lock = threading.Lock()
        self._standalone_queue = []

        # Odwrócenie osi Y (lewo/prawo) i Z (góra/dół) względem poprzedniej wersji,
        # tak aby ruch w górę szedł w górę, a w lewo szedł w lewo (można też przełączać klawiszami [Y] i [Z])
        self.flip_y = -1.0
        self.flip_z = -1.0

        # Ślad końcówki pałeczki (ostatnie 80 punktów w 3D)
        self.tip_trail = deque(maxlen=80)

        # Parametry kamery 3D (orbitowanie myszką lub strzałkami)
        self.cam_yaw = math.radians(-125.0)
        self.cam_pitch = math.radians(22.0)
        self.cam_dist = 3.2

        # Jeśli uruchamiamy standalone (bez recorder_app.py):
        if self.server is None:
            self.tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.tcp_server.bind(("0.0.0.0", PORT))
            self.tcp_server.listen(1)

            self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
            self.udp_sock.bind(("0.0.0.0", PORT))

            self.tcp_thread = threading.Thread(target=self._standalone_tcp_loop, daemon=True)
            self.udp_thread = threading.Thread(target=self._standalone_udp_loop, daemon=True)
            self.tcp_thread.start()
            self.udp_thread.start()

    def _standalone_tcp_loop(self):
        print(f"[TCP] Oczekiwanie na połączenie z ESP32 na porcie {PORT}...")
        while self.running:
            try:
                client, addr = self.tcp_server.accept()
                print(f"[TCP] Połączono z ESP32 ({addr[0]}). Wysyłanie komendy START ('0')...")
                client.sendall(b"0")
            except Exception:
                time.sleep(0.5)

    def _standalone_udp_loop(self):
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
                        self._standalone_queue.extend(batch)
            except Exception:
                pass

    def _process_sample(self, sample):
        cal_acc, cal_gyro = self.calib.apply(sample)
        self.latest_gyro_norm = float(np.linalg.norm(cal_gyro))
        self.wand_id = sample.get("wand_id", 0)
        # Sprowadzenie osi magnetometru MMC56x3 do układu osi LSM6DS3 (obrót o 90° na PCB)
        mag = self.calib.align_mag_to_lsm6ds3(sample)
        self.ahrs.update(cal_gyro, cal_acc, mag=mag, timestamp_ms=sample["timestamp_ms"])
        self.total_packets += 1
        if not self.auto_tared and self.total_packets == 15:
            self.ahrs.reset()
            self.auto_tared = True

    def _consume_pending_samples(self):
        if self.server is not None:
            with self.server.lock:
                buf_len = len(self.server.current_buffer)
                if buf_len > self.processed_count:
                    new_samples = self.server.current_buffer[self.processed_count:buf_len]
                    self.processed_count = buf_len
                else:
                    new_samples = []
        else:
            with self.lock:
                new_samples = self._standalone_queue[:]
                self._standalone_queue.clear()

        for s in new_samples:
            self._process_sample(s)

    def _transform_points(self, pts_local, R):
        """Przekształca punkty z układu lokalnego pałeczki do świata z korektą zwrotu osi Y i Z."""
        pts_w = pts_local @ R.T
        pts_w = pts_w.copy()
        if pts_w.ndim == 1:
            pts_w[1] *= self.flip_y
            pts_w[2] *= self.flip_z
        else:
            pts_w[:, 1] *= self.flip_y
            pts_w[:, 2] *= self.flip_z
        return pts_w

    def _project_3d_to_2d(self, pt3d, width, height):
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

        fov = min(width, height) * 0.85
        px = int(width / 2 + (cam_x * fov) / cam_z)
        py = int(height / 2 - (cam_y * fov) / cam_z)
        return (px, py), cam_z

    def _build_wand_geometry_along_x(self):
        n_sides = 6
        r_handle = 0.065
        r_shaft = 0.032
        r_tip = 0.012

        angles = [i * (2.0 * math.pi / n_sides) for i in range(n_sides)]

        ring_back = np.array([[-0.35, r_handle * math.cos(a), r_handle * math.sin(a)] for a in angles])
        ring_mid = np.array([[0.05, r_handle * math.cos(a), r_handle * math.sin(a)] for a in angles])
        ring_shaft_start = np.array([[0.05, r_shaft * math.cos(a), r_shaft * math.sin(a)] for a in angles])
        ring_tip = np.array([[1.05, r_tip * math.cos(a), r_tip * math.sin(a)] for a in angles])

        return ring_back, ring_mid, ring_shaft_start, ring_tip

    def _draw_ring_prism(self, screen, ring1_world, ring2_world, color_edge, width, height):
        n = len(ring1_world)
        pts1 = [self._project_3d_to_2d(p, width, height)[0] for p in ring1_world]
        pts2 = [self._project_3d_to_2d(p, width, height)[0] for p in ring2_world]

        for i in range(n):
            nxt = (i + 1) % n
            pygame.draw.line(screen, color_edge, pts1[i], pts1[nxt], 2)
            pygame.draw.line(screen, color_edge, pts2[i], pts2[nxt], 2)
            pygame.draw.line(screen, color_edge, pts1[i], pts2[i], 2)

    def run(self):
        pygame.init()
        width, height = 1080, 750
        screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("Wizualizacja Pałeczki 3D (9-DOF MARG, Oś X = długość) - AHRS Madgwick")
        clock = pygame.time.Clock()

        font_main = pygame.font.SysFont("consolas", 18, bold=True)
        font_small = pygame.font.SysFont("consolas", 15)

        ring_back, ring_mid, ring_shaft_start, ring_tip = self._build_wand_geometry_along_x()

        wand_spine_local = np.array([
            [-0.35, 0.0, 0.0],
            [ 0.05, 0.0, 0.0],
            [ 1.10, 0.0, 0.0],
        ], dtype=np.float64)

        axis_len = 0.55
        axes_local = {
            "X (Wand)":     (np.array([axis_len, 0.0, 0.0]), (255, 75, 75)),
            "Y (Side)":     (np.array([0.0, axis_len, 0.0]), (75, 240, 110)),
            "Z (PCB Up)":   (np.array([0.0, 0.0, axis_len]), (80, 160, 255)),
        }

        dragging = False
        last_mouse = (0, 0)

        print("\n[OK] Opened Pygame 3D window (Default: 9-DOF EKF Kalman, Wand aligned along X-axis).")
        print(" -> [R] or [SPACE]: Reset orientation (tare current position as zero)")
        print(" -> [K]: Toggle filter: EKF (Kalman) / Madgwick")
        print(" -> [M]: Toggle 9-DOF (MARG) / 6-DOF (IMU)")
        print(" -> [Y] / [Z]: Invert Left-Right [Y] or Up-Down [Z]")
        print(" -> [C]: Clear wand tip trail")
        print(" -> [ESC]: Close window and return to menu")

        dt = 0.016
        while self.running:
            width, height = screen.get_size()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key in (pygame.K_r, pygame.K_SPACE):
                        self.ahrs.reset()
                        self.tip_trail.clear()
                    elif event.key == pygame.K_k:
                        self.ahrs.toggle_filter()
                        print(f"[AHRS] Switched filter to: {self.ahrs.filter_name}")
                    elif event.key == pygame.K_m:
                        self.ahrs.use_mag = not self.ahrs.use_mag
                    elif event.key == pygame.K_y:
                        self.flip_y *= -1.0
                        self.tip_trail.clear()
                    elif event.key == pygame.K_z:
                        self.flip_z *= -1.0
                        self.tip_trail.clear()
                    elif event.key == pygame.K_c:
                        self.tip_trail.clear()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        dragging = True
                        last_mouse = event.pos
                    elif event.button == 4:
                        self.cam_dist = max(1.4, self.cam_dist - 0.2)
                    elif event.button == 5:
                        self.cam_dist = min(6.5, self.cam_dist + 0.2)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        dragging = False
                elif event.type == pygame.MOUSEMOTION and dragging:
                    dx = event.pos[0] - last_mouse[0]
                    dy = event.pos[1] - last_mouse[1]
                    last_mouse = event.pos
                    self.cam_yaw += dx * 0.007
                    self.cam_pitch = max(math.radians(-80), min(math.radians(80), self.cam_pitch + dy * 0.007))

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.cam_yaw -= 0.03
            if keys[pygame.K_RIGHT]:
                self.cam_yaw += 0.03
            if keys[pygame.K_UP]:
                self.cam_pitch = min(math.radians(80), self.cam_pitch + 0.025)
            if keys[pygame.K_DOWN]:
                self.cam_pitch = max(math.radians(-80), self.cam_pitch - 0.025)

            self._consume_pending_samples()

            R = self.ahrs.get_smooth_rotation_matrix(dt=dt, smoothing_speed=24.0)
            roll, pitch, yaw = self.ahrs.get_euler_angles_deg(smooth=True)
            pitch *= self.flip_z
            yaw *= self.flip_y

            screen.fill(BG_VOID)

            # 1. Floor Grid (Z = -0.9)
            grid_z = -0.9
            for g in np.linspace(-1.2, 1.2, 7):
                p1, _ = self._project_3d_to_2d((g, -1.2, grid_z), width, height)
                p2, _ = self._project_3d_to_2d((g,  1.2, grid_z), width, height)
                pygame.draw.line(screen, GRID_LINE, p1, p2, 1)

                p3, _ = self._project_3d_to_2d((-1.2, g, grid_z), width, height)
                p4, _ = self._project_3d_to_2d(( 1.2, g, grid_z), width, height)
                pygame.draw.line(screen, GRID_LINE, p3, p4, 1)

            # 2. Wand Geometry aligned along X-axis
            spine_w = self._transform_points(wand_spine_local, R)
            rb_w = self._transform_points(ring_back, R)
            rm_w = self._transform_points(ring_mid, R)
            rs_w = self._transform_points(ring_shaft_start, R)
            rt_w = self._transform_points(ring_tip, R)

            tip_world = spine_w[2]
            self.tip_trail.append(tip_world.copy())

            # 3. Plasma Glow Motion Trail
            if len(self.tip_trail) > 2:
                trail_pts_2d = [self._project_3d_to_2d(pt, width, height)[0] for pt in self.tip_trail]
                for i in range(1, len(trail_pts_2d)):
                    ratio = i / len(trail_pts_2d)
                    col = (
                        int(ACCENT_GOLD[0] * ratio + ACCENT_CYAN[0] * (1 - ratio)),
                        int(ACCENT_GOLD[1] * ratio + ACCENT_CYAN[1] * (1 - ratio)),
                        int(ACCENT_GOLD[2] * ratio + ACCENT_CYAN[2] * (1 - ratio))
                    )
                    pygame.draw.line(screen, col, trail_pts_2d[i - 1], trail_pts_2d[i], max(1, int(4 * ratio)))

            # 4. 3D Wand Mesh
            self._draw_ring_prism(screen, rb_w, rm_w, WAND_GRIP, width, height)
            self._draw_ring_prism(screen, rs_w, rt_w, WAND_SHAFT, width, height)

            p_back, _ = self._project_3d_to_2d(spine_w[0], width, height)
            p_mid, _  = self._project_3d_to_2d(spine_w[1], width, height)
            p_tip, _  = self._project_3d_to_2d(spine_w[2], width, height)

            pygame.draw.line(screen, (130, 150, 180), p_back, p_mid, 8)
            pygame.draw.line(screen, (245, 150, 45), p_mid, p_tip, 5)
            draw_glow_sphere(screen, p_tip, base_radius=8, color_rgb=(255, 130, 45))

            # 5. Local XYZ Coordinate Axes
            origin_2d, _ = self._project_3d_to_2d((0.0, 0.0, 0.0), width, height)
            for label, (vec_local, color) in axes_local.items():
                vec_w = self._transform_points(vec_local, R)
                end_2d, _ = self._project_3d_to_2d(vec_w, width, height)
                pygame.draw.line(screen, color, origin_2d, end_2d, 3)
                pygame.draw.circle(screen, color, end_2d, 4)
                lbl_surf = font_small.render(label, True, color)
                screen.blit(lbl_surf, (end_2d[0] + 6, end_2d[1] - 8))

            # 6. Bento Telemetry Card (HUD)
            hud_w, hud_h = 490, 116
            draw_card(screen, pygame.Rect(16, 14, hud_w, hud_h), bg_color=SURFACE_CARD, border_color=BORDER_BASE, radius=10)

            # Card Header
            title_txt = f"3D ORIENTATION [EKF KALMAN]" if "Kalman" in self.ahrs.filter_name else "3D ORIENTATION [MADGWICK]"
            screen.blit(font_main.render(title_txt, True, TEXT_PRIMARY), (28, 22))

            calib_dot = ACCENT_EMERALD if self.calib.is_calibrated else ACCENT_AMBER
            calib_str = "CALIBRATED" if self.calib.is_calibrated else "NOT CALIBRATED"
            draw_pill_badge(screen, calib_str, font_small, 340, 20, bg_color=SURFACE_PILL, text_color=TEXT_PRIMARY,
                            border_color=BORDER_BASE, dot_color=calib_dot, padding_x=8, padding_y=2)

            # Euler Angles
            euler_txt = f"Roll: {roll:+6.1f}°   Pitch: {pitch:+6.1f}°   Yaw: {yaw:+6.1f}°"
            screen.blit(font_main.render(euler_txt, True, ACCENT_CYAN), (28, 52))

            # Gyro Dynamics & Mode
            dof_str = "9-DOF MARG" if self.ahrs.use_mag else "6-DOF IMU"
            sub_hud = f"||ω||: {self.latest_gyro_norm:4.2f} rad/s   |   Packets: {self.total_packets}   |   Mode: {dof_str}"
            screen.blit(font_small.render(sub_hud, True, TEXT_SECONDARY), (28, 82))

            # Floating Footer Pill
            foot_w, foot_h = min(width - 32, 940), 36
            foot_rect = pygame.Rect(16, height - foot_h - 12, foot_w, foot_h)
            draw_card(screen, foot_rect, bg_color=SURFACE_CARD, border_color=BORDER_BASE, radius=18)

            help_text = "[R/SPACE] Tare   [K] Kalman/Madgwick   [M] 9/6-DOF   [Y] Flip L/R   [Z] Flip U/D   [C] Clear   [ESC] Exit"
            screen.blit(font_small.render(help_text, True, TEXT_SECONDARY), (foot_rect.x + 20, foot_rect.y + 10))

            pygame.display.flip()
            dt = clock.tick(60) / 1000.0

        pygame.display.quit()
        self.running = False


# --- SUPPORTED GESTURES & HOTKEYS ---
GESTURES = {
    "1": ("circle",    "Circle"),
    "2": ("go_up",     "Swipe Up"),
    "3": ("go_down",   "Swipe Down"),
    "4": ("go_left",   "Swipe Left"),
    "5": ("go_right",  "Swipe Right"),
    "6": ("iks",       "Cross (X)"),
    "7": ("idle",      "Rest (Idle)"),
}

# Adaptive capture window per gesture class (seconds)
# Gives compound motions like 'X' ample time without cutting off, while fast swipes remain brisk
DEFAULT_GESTURE_CAPTURE_DURATIONS = {
    "circle":   2.2,   # Circle: ~1.1 - 1.9s execution -> 2.2s capture window
    "go_up":    1.5,   # Swipe Up: ~0.3 - 1.0s execution -> 1.5s capture window
    "go_down":  1.6,   # Swipe Down: ~0.4 - 1.2s execution -> 1.6s capture window
    "go_left":  1.5,   # Swipe Left: ~0.5 - 0.9s execution -> 1.5s capture window
    "go_right": 1.5,   # Swipe Right: ~0.5 - 0.9s execution -> 1.5s capture window
    "iks":      3.2,   # Cross (X): two strokes, ~2.0 - 2.9s execution -> 3.2s capture window!
    "idle":     5.0,   # Rest (Idle): 5.0s stationary baseline
}

KEY_TO_GESTURE = {
    pygame.K_1: "1", pygame.K_KP1: "1",
    pygame.K_2: "2", pygame.K_KP2: "2",
    pygame.K_3: "3", pygame.K_KP3: "3",
    pygame.K_4: "4", pygame.K_KP4: "4",
    pygame.K_5: "5", pygame.K_KP5: "5",
    pygame.K_6: "6", pygame.K_KP6: "6",
    pygame.K_7: "7", pygame.K_KP7: "7",
}


class GestureRecordingSessionUI:
    """
    Persistent Pygame 3D (6-DOF) window for gesture recording session:
    - Provides continuous 3D feedforward orientation during countdown (no freeze, no sudden jumps).
    - Smoothly records and renders live wand motion during gesture execution.
    - Loops a 6-DOF trajectory replay with full 3D ghost trail for immediate visual review.
    - Allows single-key accept ([ENTER]/[SPACE]), reject ([D]/[BACKSPACE]), or abort ([Q]/[ESC]).
    """
    def __init__(self, server, calib: SensorCalibration, beep_fn=None):
        self.server = server
        self.calib = calib
        self.beep = beep_fn if beep_fn is not None else (lambda f, d: None)

        self.flip_y = -1.0
        self.flip_z = -1.0
        self.cam_yaw = math.radians(-125.0)
        self.cam_pitch = math.radians(22.0)
        self.cam_dist = 3.2

        # Start continuous ESP32 data streaming so live tracking works immediately
        self.server.start_streaming()
        self.live_ahrs = MadgwickAHRS(sample_rate=100.0, gain=0.1, use_mag=False, filter_type="ekf")
        self.processed_count = 0
        self.live_trail = deque(maxlen=80)

        pygame.init()
        self.width, self.height = 1260, 780
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("IMU Gesture Recording Session - 6-DOF Live Preview & Review")
        self.clock = pygame.time.Clock()

        sans_fonts = ["segoe ui", "trebuchet ms", "arial", "helvetica", "dejavu sans"]
        mono_fonts = ["consolas", "fira code", "cascadia code", "courier new"]
        self.font_title   = pygame.font.SysFont(sans_fonts, 20, bold=True)
        self.font_main    = pygame.font.SysFont(sans_fonts, 15, bold=True)
        self.font_small   = pygame.font.SysFont(sans_fonts, 13)
        self.font_badge   = pygame.font.SysFont(sans_fonts, 11, bold=True)
        self.font_mono    = pygame.font.SysFont(mono_fonts, 12, bold=True)
        self.font_mono_sm = pygame.font.SysFont(mono_fonts, 11)
        self.font_tiny    = pygame.font.SysFont(mono_fonts, 11)

        # Dataset Class & Speed Balancer state
        self.dataset_counts = {}
        self.speed_analysis = {}
        self.current_gesture_key = "1"
        self.current_gesture_name = "circle"
        self.current_gesture_label = "Circle"
        self.session_saved_count = 0
        self.show_dataset_sidebar = True

        self._helper = Wand3DVisualizer(existing_server=server, use_mag=False, filter_type="ekf")
        self.ring_back, self.ring_mid, self.ring_shaft_start, self.ring_tip = (
            self._helper._build_wand_geometry_along_x()
        )
        self.wand_spine_local = np.array([
            [-0.35, 0.0, 0.0],
            [ 0.05, 0.0, 0.0],
            [ 1.10, 0.0, 0.0],
        ], dtype=np.float64)
        axis_len = 0.50
        self.axes_local = {
            "X": (np.array([axis_len, 0.0, 0.0]), (255, 75, 75)),
            "Y": (np.array([0.0, axis_len, 0.0]), (75, 240, 110)),
            "Z": (np.array([0.0, 0.0, axis_len]), (80, 160, 255)),
        }

    def close(self):
        try:
            self.server.stop_streaming()
        except Exception:
            pass
        pygame.display.quit()

    def _transform_points(self, pts_local, R):
        pts_w = pts_local @ R.T
        pts_w = pts_w.copy()
        if pts_w.ndim == 1:
            pts_w[1] *= self.flip_y
            pts_w[2] *= self.flip_z
        else:
            pts_w[:, 1] *= self.flip_y
            pts_w[:, 2] *= self.flip_z
        return pts_w

    def _project(self, pt3d):
        self._helper.cam_yaw = self.cam_yaw
        self._helper.cam_pitch = self.cam_pitch
        self._helper.cam_dist = self.cam_dist
        return self._project_inner(pt3d)

    def _project_inner(self, pt3d):
        vp_w = self.width - 340 if getattr(self, 'show_dataset_sidebar', False) else self.width
        return self._helper._project_3d_to_2d(pt3d, vp_w, self.height)

    def _draw_sidebar(self, sidebar_x, sidebar_w):
        side_rect = pygame.Rect(sidebar_x + 10, 14, sidebar_w - 20, self.height - 28)
        draw_card(self.screen, side_rect, bg_color=SURFACE_CARD, border_color=BORDER_BASE, radius=12)

        # Header
        pygame.draw.circle(self.screen, ACCENT_EMERALD, (side_rect.x + 22, side_rect.y + 24), 6)
        self.screen.blit(self.font_main.render("DATASET & SPEED BALANCER", True, TEXT_PRIMARY), (side_rect.x + 36, side_rect.y + 12))
        total_samples = sum(self.dataset_counts.values()) if self.dataset_counts else 0
        self.screen.blit(self.font_tiny.render(f"Total: {total_samples} samples  (+{self.session_saved_count} session)", True, TEXT_MUTED), (side_rect.x + 36, side_rect.y + 34))

        # Rows
        max_val = max(max(self.dataset_counts.values(), default=1), 25) if self.dataset_counts else 25
        row_y = side_rect.y + 58
        for k in ["1", "2", "3", "4", "5", "6", "7"]:
            gname, glbl = GESTURES[k]
            cnt = self.dataset_counts.get(gname, 0)
            is_active = (gname == self.current_gesture_name)

            s_info = self.speed_analysis.get(gname, {})
            rec_tag = s_info.get("rec_tag", "BALANCED")
            rec_str = s_info.get("rec_str", "Balanced")

            row_rect = pygame.Rect(side_rect.x + 10, row_y, side_rect.w - 20, 52)
            if is_active:
                draw_card(self.screen, row_rect, bg_color=SURFACE_CARD_ALT, border_color=ACCENT_CYAN, radius=8, border_w=2)
                txt_col = TEXT_PRIMARY
            else:
                draw_card(self.screen, row_rect, bg_color=SURFACE_PILL, border_color=BORDER_MUTED, radius=6, border_w=1)
                txt_col = TEXT_SECONDARY

            # Key pill
            draw_pill_badge(self.screen, f"[{k}]", self.font_badge, row_rect.x + 8, row_rect.y + 9,
                            bg_color=SURFACE_CARD if is_active else SURFACE_TRACK,
                            text_color=ACCENT_CYAN if is_active else TEXT_MUTED,
                            border_color=BORDER_BASE, padding_x=4, padding_y=2)

            # Name
            self.screen.blit(self.font_small.render(f"{glbl:<13}", True, txt_col), (row_rect.x + 38, row_rect.y + 6))

            # Count
            bar_col = ACCENT_EMERALD if cnt >= 30 else (ACCENT_AMBER if cnt >= 15 else ACCENT_CORAL)
            cnt_str = f"{cnt:2d}"
            self.screen.blit(self.font_mono.render(cnt_str, True, bar_col), (row_rect.right - 36, row_rect.y + 6))

            # Speed recommendation tag
            if rec_tag == "NEED_FAST":
                tag_col = ACCENT_AMBER
                tag_text = f"⚡ {rec_str}"
            elif rec_tag == "NEED_SLOW":
                tag_col = ACCENT_CYAN
                tag_text = f"🐢 {rec_str}"
            elif rec_tag == "NO_SAMPLES":
                tag_col = TEXT_MUTED
                tag_text = "(No samples)"
            else:
                tag_col = ACCENT_EMERALD
                tag_text = f"✓ {rec_str}"

            self.screen.blit(self.font_tiny.render(tag_text, True, tag_col), (row_rect.x + 38, row_rect.y + 28))

            # Progress mini bar
            ratio = min(1.0, cnt / max_val)
            draw_progress_bar(self.screen, row_rect.right - 88, row_rect.y + 32, 50, 5, ratio,
                              fill_color=bar_col, bg_color=SURFACE_TRACK, radius=2)

            row_y += 56

        # Bottom Controls & Guidance Card
        bot_rect = pygame.Rect(side_rect.x + 10, row_y + 6, side_rect.w - 20, side_rect.bottom - (row_y + 6) - 10)
        draw_card(self.screen, bot_rect, bg_color=SURFACE_CARD_ALT, border_color=BORDER_BASE, radius=8)

        self.screen.blit(self.font_badge.render("TARGET SPEED GUIDANCE", True, TEXT_MUTED), (bot_rect.x + 12, bot_rect.y + 8))
        active_info = self.speed_analysis.get(self.current_gesture_name, {})
        rec_t = active_info.get("rec_tag", "BALANCED")
        if rec_t == "NEED_FAST":
            prompt_str = f"⚡ Fast missing (<{active_info.get('t_fast', 1.0):.2f}s)!"
            prompt_col = ACCENT_AMBER
        elif rec_t == "NEED_SLOW":
            prompt_str = f"🐢 Slow missing (>{active_info.get('t_slow', 1.0):.2f}s)!"
            prompt_col = ACCENT_CYAN
        elif rec_t == "NO_SAMPLES":
            prompt_str = "Perform first sample"
            prompt_col = TEXT_SECONDARY
        else:
            prompt_str = "✓ Speeds well balanced!"
            prompt_col = ACCENT_EMERALD

        self.screen.blit(self.font_main.render(prompt_str, True, prompt_col), (bot_rect.x + 12, bot_rect.y + 26))
        n_f = active_info.get("fast", 0)
        n_m = active_info.get("normal", active_info.get("med_count", active_info.get("med", 0)))
        n_s = active_info.get("slow", 0)
        self.screen.blit(self.font_tiny.render(f"Class Profile: {n_f} Fast  •  {n_m} Med  •  {n_s} Slow", True, TEXT_SECONDARY), (bot_rect.x + 12, bot_rect.y + 48))

        self.screen.blit(self.font_badge.render("HOTKEYS", True, TEXT_MUTED), (bot_rect.x + 12, bot_rect.y + 68))
        self.screen.blit(self.font_mono_sm.render("[1-7]   Switch Class", True, TEXT_PRIMARY), (bot_rect.x + 12, bot_rect.y + 84))
        self.screen.blit(self.font_mono_sm.render("[SPACE] Accept (+1)", True, ACCENT_EMERALD), (bot_rect.x + 12, bot_rect.y + 102))
        self.screen.blit(self.font_mono_sm.render("[D]     Retry", True, ACCENT_CORAL), (bot_rect.x + 150, bot_rect.y + 102))
        self.screen.blit(self.font_mono_sm.render("[Q/ESC] Exit Session", True, TEXT_MUTED), (bot_rect.x + 12, bot_rect.y + 120))

    def _handle_camera_events(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_y:
                self.flip_y *= -1.0
            elif event.key == pygame.K_z:
                self.flip_z *= -1.0
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4:
                self.cam_dist = max(1.4, self.cam_dist - 0.2)
            elif event.button == 5:
                self.cam_dist = min(6.5, self.cam_dist + 0.2)

    def _draw_scene(self, R, active_trail, full_ghost_trail=None, banner_top="", banner_color=(255, 220, 100),
                    sub_info="", footer_info="", progress_ratio=None):
        self.width, self.height = self.screen.get_size()
        sidebar_w = 340 if getattr(self, 'show_dataset_sidebar', False) else 0
        viewport_w = self.width - sidebar_w

        self._helper.cam_yaw = self.cam_yaw
        self._helper.cam_pitch = self.cam_pitch
        self._helper.cam_dist = self.cam_dist

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.cam_yaw -= 0.03
        if keys[pygame.K_RIGHT]:
            self.cam_yaw += 0.03
        if keys[pygame.K_UP]:
            self.cam_pitch = min(math.radians(80), self.cam_pitch + 0.025)
        if keys[pygame.K_DOWN]:
            self.cam_pitch = max(math.radians(-80), self.cam_pitch - 0.025)

        self.screen.fill(BG_VOID)

        # 1. Horizon Floor Grid (Z = -0.9)
        grid_z = -0.9
        for g in np.linspace(-1.2, 1.2, 7):
            p1, _ = self._project_inner((g, -1.2, grid_z))
            p2, _ = self._project_inner((g,  1.2, grid_z))
            pygame.draw.line(self.screen, GRID_LINE, p1, p2, 1)
            p3, _ = self._project_inner((-1.2, g, grid_z))
            p4, _ = self._project_inner(( 1.2, g, grid_z))
            pygame.draw.line(self.screen, GRID_LINE, p3, p4, 1)

        # 2. Ghost Trail (Full gesture trajectory during review)
        if full_ghost_trail is not None and len(full_ghost_trail) > 1:
            ghost_2d = [self._project_inner(pt)[0] for pt in full_ghost_trail]
            pygame.draw.lines(self.screen, (45, 65, 95), False, ghost_2d, 2)
            # Start landmark (Emerald) & End landmark (Coral)
            pygame.draw.circle(self.screen, ACCENT_EMERALD, ghost_2d[0], 6)
            pygame.draw.circle(self.screen, ACCENT_CORAL, ghost_2d[-1], 6)

        # 3. Dynamic Plasma Motion Trail
        if len(active_trail) > 1:
            trail_2d = [self._project_inner(pt)[0] for pt in active_trail]
            for i in range(1, len(trail_2d)):
                ratio = i / len(trail_2d)
                col = (
                    int(ACCENT_GOLD[0] * ratio + ACCENT_CYAN[0] * (1.0 - ratio)),
                    int(ACCENT_GOLD[1] * ratio + ACCENT_CYAN[1] * (1.0 - ratio)),
                    int(ACCENT_GOLD[2] * ratio + ACCENT_CYAN[2] * (1.0 - ratio))
                )
                pygame.draw.line(self.screen, col, trail_2d[i - 1], trail_2d[i], max(2, int(5 * ratio)))

        # 4. 3D Wand Mesh (Aligned along X-axis)
        spine_w = self._transform_points(self.wand_spine_local, R)
        rb_w = self._transform_points(self.ring_back, R)
        rm_w = self._transform_points(self.ring_mid, R)
        rs_w = self._transform_points(self.ring_shaft_start, R)
        rt_w = self._transform_points(self.ring_tip, R)

        self._helper._draw_ring_prism(self.screen, rb_w, rm_w, WAND_GRIP, viewport_w, self.height)
        self._helper._draw_ring_prism(self.screen, rs_w, rt_w, WAND_SHAFT, viewport_w, self.height)

        p_back, _ = self._project_inner(spine_w[0])
        p_mid, _  = self._project_inner(spine_w[1])
        p_tip, _  = self._project_inner(spine_w[2])

        pygame.draw.line(self.screen, (130, 150, 180), p_back, p_mid, 8)
        pygame.draw.line(self.screen, (245, 150, 45), p_mid, p_tip, 5)
        draw_glow_sphere(self.screen, p_tip, base_radius=8, color_rgb=(255, 130, 45))

        # 5. Local XYZ Coordinate Axes
        origin_2d, _ = self._project_inner((0.0, 0.0, 0.0))
        for label, (vec_local, color) in self.axes_local.items():
            vec_w = self._transform_points(vec_local, R)
            end_2d, _ = self._project_inner(vec_w)
            pygame.draw.line(self.screen, color, origin_2d, end_2d, 3)
            lbl_surf = self.font_small.render(label, True, color)
            self.screen.blit(lbl_surf, (end_2d[0] + 5, end_2d[1] - 8))

        # 6. Floating Cyber Header Capsule (Left Viewport)
        top_w = viewport_w - 32
        top_h = 70
        top_rect = pygame.Rect(16, 14, top_w, top_h)
        draw_card(self.screen, top_rect, bg_color=SURFACE_CARD, border_color=BORDER_BASE, radius=10)
        pygame.draw.circle(self.screen, banner_color, (top_rect.x + 24, top_rect.y + 24), 6)

        s1 = self.font_title.render(banner_top, True, TEXT_PRIMARY)
        self.screen.blit(s1, (top_rect.x + 42, top_rect.y + 11))
        if sub_info:
            s2 = self.font_main.render(sub_info, True, TEXT_SECONDARY)
            self.screen.blit(s2, (top_rect.x + 42, top_rect.y + 40))

        # Floating Scrubber Progress Track (Review Mode)
        if progress_ratio is not None:
            draw_progress_bar(self.screen, 16, 88, top_w, 4, progress_ratio,
                              fill_color=ACCENT_CYAN, bg_color=SURFACE_TRACK, border_color=None)

        # 7. Floating Action Footer Capsule (Left Viewport)
        if footer_info:
            foot_w = viewport_w - 32
            foot_h = 38
            foot_rect = pygame.Rect(16, self.height - foot_h - 14, foot_w, foot_h)
            draw_card(self.screen, foot_rect, bg_color=SURFACE_CARD, border_color=BORDER_BASE, radius=19)
            sf = self.font_main.render(footer_info, True, TEXT_SECONDARY)
            self.screen.blit(sf, (foot_rect.x + 20, foot_rect.y + 10))

        # 8. Right Sidebar: DATASET BALANCER (All Class Counts)
        if getattr(self, 'show_dataset_sidebar', False):
            self._draw_sidebar(viewport_w, sidebar_w)

        pygame.display.flip()
        return spine_w[2].copy()

    def run_countdown(self, gesture_label, trial_num, total_trials, countdown_sec=2.0):
        """Displays countdown before gesture with live continuous 6-DOF tracking. Returns False if user presses Q/ESC."""
        self.server.prepare_signal()
        t_start = time.time()
        last_beep_sec = -1
        last_t = time.time()

        while True:
            now = time.time()
            elapsed = now - t_start
            rem = countdown_sec - elapsed
            if rem <= 0:
                break

            sec_int = int(math.ceil(rem))
            if sec_int != last_beep_sec:
                last_beep_sec = sec_int
                self.beep(600 if sec_int > 1 else 800, 120)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                self._handle_camera_events(event)

            # Consume live samples arriving during preparation
            live_samples = self.server.get_live_samples()
            for s in live_samples:
                cal_acc, cal_gyro = self.calib.apply(s, adaptive_gyro=False)
                self.live_ahrs.update(cal_gyro, cal_acc, mag=None, timestamp_ms=s["timestamp_ms"])

            dt = max(0.001, min(0.05, now - last_t))
            last_t = now
            R = self.live_ahrs.get_smooth_rotation_matrix(dt=dt, smoothing_speed=20.0)

            self._draw_scene(
                R, [],
                banner_top=f"[{trial_num}/{total_trials}] GESTURE: {gesture_label.upper()} — GET READY: {rem:0.1f} s",
                banner_color=(255, 200, 70),
                sub_info="Hold wand in starting position (live tracking active)...",
                footer_info="[Q / ESC] Abort Session   |   [Arrow Keys] Camera   |   [Y / Z] Flip Axes"
            )
            self.clock.tick(60)
        return True

    def run_live_capture(self, gesture_label, trial_num, total_trials, duration_sec=1.5, manual_stop=False):
        """
        Records gesture with live 6-DOF tracking in Pygame.
        Preserves existing AHRS state to avoid sudden orientation snapping.
        Returns (samples, aborted_bool).
        """
        # Clear live trail and reset current_buffer read index
        self.live_trail.clear()
        self.processed_count = 0

        self.beep(1200, 200)
        self.server.start_recording()
        t_start = time.time()
        last_t = time.time()

        while True:
            now = time.time()
            elapsed = now - t_start
            if not manual_stop and elapsed >= duration_sec:
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.server.stop_recording(stop_esp_stream=False)
                    return [], True
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        self.server.stop_recording(stop_esp_stream=False)
                        return [], True
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        if manual_stop or elapsed >= 0.35:
                            samples = self.server.stop_recording(stop_esp_stream=False)
                            self.beep(450, 150)
                            return samples, False
                self._handle_camera_events(event)

            # Process new recorded samples from server buffer
            with self.server.lock:
                buf_len = len(self.server.current_buffer)
                new_samples = self.server.current_buffer[self.processed_count:buf_len]
                self.processed_count = buf_len

            for s in new_samples:
                cal_acc, cal_gyro = self.calib.apply(s, adaptive_gyro=False)
                self.live_ahrs.update(cal_gyro, cal_acc, mag=None, timestamp_ms=s["timestamp_ms"])

            dt = max(0.001, min(0.05, now - last_t))
            last_t = now
            R = self.live_ahrs.get_smooth_rotation_matrix(dt=dt, smoothing_speed=24.0)
            tip_pt = self._transform_points(self.wand_spine_local, R)[2]
            if self.processed_count > 0:
                self.live_trail.append(tip_pt.copy())

            if manual_stop:
                top_txt = f"[{trial_num}] RECORDING: {gesture_label.upper()} ({elapsed:0.1f} s) — PERFORM GESTURE!"
                foot_txt = "[ENTER / SPACE] Stop Recording   |   [Q / ESC] Abort"
            else:
                rem = max(0.0, duration_sec - elapsed)
                top_txt = f"[{trial_num}/{total_trials}] >>> PERFORM GESTURE NOW: {gesture_label.upper()} ({rem:0.1f} s) <<<"
                foot_txt = "[SPACE / ENTER] Finish Early   |   [Arrow Keys] Camera   |   [Q / ESC] Abort"

            self._draw_scene(
                R, self.live_trail,
                banner_top=top_txt,
                banner_color=(80, 250, 130),
                sub_info=f"Samples received: {self.processed_count}",
                footer_info=foot_txt
            )
            self.clock.tick(60)

        samples = self.server.stop_recording(stop_esp_stream=False)
        self.beep(450, 150)
        return samples, False

    def review_and_confirm_gesture(self, trimmed_samples, raw_count, gesture_label, trial_num, total_trials):
        """
        Loops a 6-DOF trajectory replay for the recorded & trimmed gesture.
        Returns:
          'accept' -> user confirmed ([ENTER] or [SPACE])
          'reject' -> user rejected ([D] or [BACKSPACE])
          'quit'   -> user aborted session ([Q] or [ESC])
        """
        if not trimmed_samples:
            return "reject"

        # Pre-calculate 6-DOF orientation and tip positions for replay loop
        replay_ahrs = MadgwickAHRS(sample_rate=100.0, gain=0.1, use_mag=False, filter_type="ekf")
        matrices = []
        tip_positions = []
        max_gyro = 0.0

        for idx, s in enumerate(trimmed_samples):
            cal_acc, cal_gyro = self.calib.apply(s, adaptive_gyro=False)
            g_norm = float(np.linalg.norm(cal_gyro))
            if g_norm > max_gyro:
                max_gyro = g_norm
            replay_ahrs.update(cal_gyro, cal_acc, mag=None, timestamp_ms=s["timestamp_ms"])
            if idx == 0:
                replay_ahrs.reset()
            R_k = replay_ahrs.get_rotation_matrix().copy()
            matrices.append(R_k)
            tip_k = self._transform_points(self.wand_spine_local, R_k)[2]
            tip_positions.append(tip_k.copy())

        n_frames = len(trimmed_samples)
        duration_ms = max(10, trimmed_samples[-1]["timestamp_ms"] - trimmed_samples[0]["timestamp_ms"])
        pause_after_ms = 500  # 500 ms pause at completion before looping

        loop_start_time = time.time()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        return "accept"
                    elif event.key in (pygame.K_d, pygame.K_BACKSPACE, pygame.K_DELETE):
                        return "reject"
                    elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                        return "quit"
                    elif event.key == pygame.K_r:
                        loop_start_time = time.time()
                self._handle_camera_events(event)

            elapsed_ms = (time.time() - loop_start_time) * 1000.0
            total_cycle_ms = duration_ms + pause_after_ms
            if elapsed_ms >= total_cycle_ms:
                loop_start_time = time.time()
                elapsed_ms = 0.0

            # Find matching frame for elapsed replay time
            t0 = trimmed_samples[0]["timestamp_ms"]
            frame_idx = n_frames - 1
            for k, s in enumerate(trimmed_samples):
                if (s["timestamp_ms"] - t0) >= elapsed_ms:
                    frame_idx = k
                    break

            R_curr = matrices[frame_idx]
            active_trail = tip_positions[:frame_idx + 1]
            progress_ratio = min(1.0, (frame_idx + 1) / max(1, n_frames))

            self._draw_scene(
                R_curr,
                active_trail=active_trail,
                full_ghost_trail=tip_positions,
                banner_top=f"6-DOF REPLAY [{trial_num}/{total_trials}]: {gesture_label.upper()} (Frame {frame_idx+1}/{n_frames})",
                banner_color=ACCENT_CYAN,
                sub_info=f"Motion: {n_frames} samples ({duration_ms} ms) | Max ||ω||: {max_gyro:.2f} rad/s",
                footer_info="[ENTER / SPACE] Accept & Save   |   [D / BACKSPACE] Reject & Retry   |   [Q / ESC] Abort",
                progress_ratio=progress_ratio
            )
            self.clock.tick(60)

    def run_countdown_interactive(self, countdown_sec=1.8):
        """
        Displays preparation countdown before gesture with live continuous 6-DOF tracking.
        Allows instant gesture switching via [1-7].
        Returns (ok: bool, switched: bool).
        """
        self.server.prepare_signal()
        t_start = time.time()
        last_beep_sec = -1
        last_t = time.time()

        while True:
            now = time.time()
            elapsed = now - t_start
            rem = countdown_sec - elapsed
            if rem <= 0:
                break

            sec_int = int(math.ceil(rem))
            if sec_int != last_beep_sec:
                last_beep_sec = sec_int
                self.beep(600 if sec_int > 1 else 800, 120)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False, False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        return False, False
                    elif event.key in KEY_TO_GESTURE:
                        k = KEY_TO_GESTURE[event.key]
                        self.current_gesture_key = k
                        self.current_gesture_name, self.current_gesture_label = GESTURES[k]
                        self.beep(900, 100)
                        return True, True
                self._handle_camera_events(event)

            # Consume live samples arriving during preparation
            live_samples = self.server.get_live_samples()
            for s in live_samples:
                cal_acc, cal_gyro = self.calib.apply(s, adaptive_gyro=False)
                self.live_ahrs.update(cal_gyro, cal_acc, mag=None, timestamp_ms=s["timestamp_ms"])

            dt = max(0.001, min(0.05, now - last_t))
            last_t = now
            R = self.live_ahrs.get_smooth_rotation_matrix(dt=dt, smoothing_speed=20.0)

            # Speed guidance prompt
            s_info = self.speed_analysis.get(self.current_gesture_name, {})
            rec_tag = s_info.get("rec_tag", "BALANCED")
            if rec_tag == "NEED_FAST":
                tip_str = f"⚡ PERFORM FAST STROKE (<{s_info.get('t_fast', 1.0):.2f}s)!"
            elif rec_tag == "NEED_SLOW":
                tip_str = f"🐢 PERFORM SLOW STROKE (>{s_info.get('t_slow', 1.0):.2f}s)!"
            else:
                tip_str = "✓ Normal speed (distribution balanced)"

            self._draw_scene(
                R, [],
                banner_top=f"★ INFINITE MODE: TARGET [{self.current_gesture_label.upper()}] — GET READY: {rem:0.1f} s",
                banner_color=(255, 200, 70),
                sub_info=f"Tip: {tip_str} | Profile: {s_info.get('fast', 0)}F / {s_info.get('normal', s_info.get('med_count', 0))}M / {s_info.get('slow', 0)}S",
                footer_info="[1-7] Change Gesture Class   |   [Arrow Keys] Camera   |   [Q / ESC] Finish Infinite Session"
            )
            self.clock.tick(60)
        return True, False

    def review_and_confirm_interactive(self, trimmed_samples, raw_count, gesture_label, trial_num):
        """
        Loops a 6-DOF trajectory replay for review in infinite mode.
        Allows instant hotkey class switching ([1-7]), accept, reject, or quit.
        Returns (decision, switch_key_str):
          ('accept', None)
          ('reject', None)
          ('switch', key_str)
          ('quit', None)
        """
        if not trimmed_samples:
            return "reject", None

        replay_ahrs = MadgwickAHRS(sample_rate=100.0, gain=0.1, use_mag=False, filter_type="ekf")
        matrices = []
        tip_positions = []
        max_gyro = 0.0

        for idx, s in enumerate(trimmed_samples):
            cal_acc, cal_gyro = self.calib.apply(s, adaptive_gyro=False)
            g_norm = float(np.linalg.norm(cal_gyro))
            if g_norm > max_gyro:
                max_gyro = g_norm
            replay_ahrs.update(cal_gyro, cal_acc, mag=None, timestamp_ms=s["timestamp_ms"])
            if idx == 0:
                replay_ahrs.reset()
            R_k = replay_ahrs.get_rotation_matrix().copy()
            matrices.append(R_k)
            tip_k = self._transform_points(self.wand_spine_local, R_k)[2]
            tip_positions.append(tip_k.copy())

        n_frames = len(trimmed_samples)
        duration_ms = max(10, trimmed_samples[-1]["timestamp_ms"] - trimmed_samples[0]["timestamp_ms"])
        pause_after_ms = 450
        loop_start_time = time.time()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit", None
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        return "accept", None
                    elif event.key in (pygame.K_d, pygame.K_BACKSPACE, pygame.K_DELETE):
                        return "reject", None
                    elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                        return "quit", None
                    elif event.key in KEY_TO_GESTURE:
                        k = KEY_TO_GESTURE[event.key]
                        return "switch", k
                    elif event.key == pygame.K_r:
                        loop_start_time = time.time()
                self._handle_camera_events(event)

            elapsed_ms = (time.time() - loop_start_time) * 1000.0
            total_cycle_ms = duration_ms + pause_after_ms
            if elapsed_ms >= total_cycle_ms:
                loop_start_time = time.time()
                elapsed_ms = 0.0

            t0 = trimmed_samples[0]["timestamp_ms"]
            frame_idx = n_frames - 1
            for k_idx, s in enumerate(trimmed_samples):
                if (s["timestamp_ms"] - t0) >= elapsed_ms:
                    frame_idx = k_idx
                    break

            R_curr = matrices[frame_idx]
            active_trail = tip_positions[:frame_idx + 1]
            progress_ratio = min(1.0, (frame_idx + 1) / max(1, n_frames))

            dur_s = duration_ms / 1000.0
            s_info = self.speed_analysis.get(self.current_gesture_name, {})
            t_f = s_info.get("t_fast", 1.0)
            t_s = s_info.get("t_slow", 1.5)
            rec_tag = s_info.get("rec_tag", "BALANCED")

            if dur_s <= t_f:
                spd_lbl = "FAST ⚡"
                gap_note = " — FILLS FAST GAP! ★" if rec_tag == "NEED_FAST" else ""
            elif dur_s >= t_s:
                spd_lbl = "SLOW 🐢"
                gap_note = " — FILLS SLOW GAP! ★" if rec_tag == "NEED_SLOW" else ""
            else:
                spd_lbl = "MEDIUM"
                gap_note = ""

            sub_txt = f"Trial: {dur_s:.2f} s ({spd_lbl}){gap_note} | {n_frames} samples | Max ||ω||: {max_gyro:.2f} rad/s"

            self._draw_scene(
                R_curr,
                active_trail=active_trail,
                full_ghost_trail=tip_positions,
                banner_top=f"6-DOF REPLAY: [{gesture_label.upper()}] Trial #{trial_num} (Frame {frame_idx+1}/{n_frames})",
                banner_color=ACCENT_CYAN,
                sub_info=sub_txt,
                footer_info="[SPACE / ENTER] Accept & Save (+1)   |   [D] Reject   |   [1-7] Switch Class   |   [ESC] Exit",
                progress_ratio=progress_ratio
            )
            self.clock.tick(60)

    def run_infinite_session(self, initial_gesture="circle", duration_sec=None, trim_fn=None, save_fn=None, speed_analysis=None):
        """
        Continuous Infinite Gesture Recording Session.
        Loops indefinitely until [ESC] or [Q] is pressed:
          - Real-time preparation countdown (switch gesture anytime via [1-7]).
          - Live speed gap guidance (shows whether fast or slow variations are missing).
          - Live gesture motion capture with adaptive per-gesture durations (e.g. 3.2s for X, 5.0s for Idle).
          - Early finish support with [SPACE] / [ENTER] if gesture completes before timeout.
          - 6-DOF loop replay review with speed classification feedback.
          - Single-key confirmation, automatic save, and dataset counter increment.
        Returns total samples saved in this session.
        """
        self.show_dataset_sidebar = True
        if speed_analysis is not None:
            self.speed_analysis = speed_analysis

        for k, (name, lbl) in GESTURES.items():
            if name == initial_gesture:
                self.current_gesture_key = k
                self.current_gesture_name = name
                self.current_gesture_label = lbl
                break

        trial_num = 1
        session_saved = 0

        while True:
            # 1. Preparation countdown with live 3D wand tracking
            while True:
                ok, switched = self.run_countdown_interactive(countdown_sec=1.8)
                if not ok:
                    return session_saved
                if not switched:
                    break

            # 2. Live capture (gesture execution with adaptive duration)
            active_duration = duration_sec if duration_sec is not None else DEFAULT_GESTURE_CAPTURE_DURATIONS.get(self.current_gesture_name, 2.0)
            raw_samples, aborted = self.run_live_capture(
                self.current_gesture_label, trial_num, total_trials="INF", duration_sec=active_duration, manual_stop=False
            )
            if aborted:
                return session_saved

            if len(raw_samples) < 10:
                time.sleep(0.3)
                continue

            # 3. Trim silence
            trimmed_samples, _, _ = trim_fn(raw_samples, self.current_gesture_name) if trim_fn else (raw_samples, 0, len(raw_samples))

            # 4. Review & Confirm
            while True:
                decision, switch_k = self.review_and_confirm_interactive(
                    trimmed_samples, len(raw_samples), self.current_gesture_label, trial_num
                )
                if decision == "quit":
                    return session_saved
                elif decision == "switch":
                    self.current_gesture_key = switch_k
                    self.current_gesture_name, self.current_gesture_label = GESTURES[switch_k]
                    self.beep(900, 100)
                    break
                elif decision == "reject":
                    break
                elif decision == "accept":
                    if save_fn:
                        save_fn(self.current_gesture_name, trimmed_samples, already_trimmed=True)
                    self.dataset_counts[self.current_gesture_name] = self.dataset_counts.get(self.current_gesture_name, 0) + 1

                    # Update speed analysis counts and recompute recommendation in place
                    duration_ms = max(10, trimmed_samples[-1]["timestamp_ms"] - trimmed_samples[0]["timestamp_ms"])
                    dur_s = duration_ms / 1000.0
                    if self.current_gesture_name in self.speed_analysis:
                        s_info = self.speed_analysis[self.current_gesture_name]
                        if dur_s <= s_info.get("t_fast", 1.0):
                            s_info["fast"] = s_info.get("fast", 0) + 1
                        elif dur_s >= s_info.get("t_slow", 1.5):
                            s_info["slow"] = s_info.get("slow", 0) + 1
                        else:
                            s_info["normal"] = s_info.get("normal", s_info.get("med_count", 0)) + 1
                            s_info["med_count"] = s_info["normal"]
                        s_info["n"] = s_info.get("n", 0) + 1

                        n_tot = s_info["n"]
                        if self.current_gesture_name == "idle":
                            s_info["rec_tag"] = "BALANCED"
                            s_info["rec_str"] = "Resting Baseline"
                        elif s_info["fast"] < max(2, int(0.25 * n_tot)):
                            s_info["rec_tag"] = "NEED_FAST"
                            s_info["rec_str"] = f"Need Fast (<{s_info['t_fast']:.2f}s)"
                        elif s_info["slow"] < max(2, int(0.25 * n_tot)):
                            s_info["rec_tag"] = "NEED_SLOW"
                            s_info["rec_str"] = f"Need Slow (>{s_info['t_slow']:.2f}s)"
                        else:
                            s_info["rec_tag"] = "BALANCED"
                            s_info["rec_str"] = "Well balanced"

                    session_saved += 1
                    self.session_saved_count += 1
                    trial_num += 1
                    self.beep(1400, 250)
                    break



if __name__ == "__main__":
    print("Starting 3D Orientation Visualizer in Pygame (standalone mode, 9-DOF)...")
    vis = Wand3DVisualizer(use_mag=True)
    vis.run()

"""
WandSense IMU - Space Invaders (Gesture Arcade Defense)
Defend Earth against the alien armada using Magic Wand IMU Gestures:
  - [Swipe Left]  (go_left)  -> Dash / Move Left
  - [Swipe Right] (go_right) -> Dash / Move Right
  - [Swipe Up]    (go_up)    -> Advance Upward
  - [Swipe Down]  (go_down)  -> Retreat Downward
  - [Circle]      (circle)   -> Fire Dual Plasma Cannons
  - [X-Mark]      (iks)      -> Charged Orbital Ion Cannon (Screen-clearing beam)
  - Key [N]                  -> Switch AI Model (Conv-LSTM <-> 1D-CNN <-> RF)
Keyboard fallbacks: Arrows / WASD, SPACE (Shoot), X (Ulti), N (Model), ESC (Exit).
"""
import os
import sys
import time
import socket
import struct
import threading
import math
import random
from collections import deque
import numpy as np
import pygame

# UTF-8 and ANSI output
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from ahrs_utils import SensorCalibration, MadgwickAHRS
from ui_theme import (
    BG_VOID, BG_PANEL, SURFACE_CARD, SURFACE_CARD_ALT, SURFACE_PILL, SURFACE_TRACK,
    BORDER_BASE, BORDER_FOCUS, BORDER_MUTED,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    ACCENT_CYAN, ACCENT_EMERALD, ACCENT_AMBER, ACCENT_CORAL, ACCENT_VIOLET, ACCENT_GOLD,
    WAND_GRIP, WAND_SHAFT, WAND_TIP_AURA, WAND_TIP_CORE,
    draw_card, draw_pill_badge, draw_progress_bar, draw_glow_sphere
)
from train_models import (
    Gesture1DCNN,
    GestureConvLSTM,
    extract_rf_features,
    resample_sequence,
    TARGET_LEN
)
from live_recognition import MultiModelClassifier

# --- SOUND SYSTEM (Non-blocking async worker on Windows) ---
try:
    import winsound
    def play_sound(freq: int, duration_ms: int):
        threading.Thread(target=winsound.Beep, args=(freq, duration_ms), daemon=True).start()
except ImportError:
    def play_sound(freq: int, duration_ms: int):
        pass

PORT = 2137
DATA_FORMAT = "<I9f?"
PACKET_SIZE = struct.calcsize(DATA_FORMAT)

# Recognition thresholds tuned for responsive arcade gaming
MOTION_THRESHOLD = 0.35
REST_THRESHOLD   = 0.25
MIN_GESTURE_SAMPLES = 18       # 180 ms minimum
MAX_GESTURE_SAMPLES = 200      # 2.0 s maximum
STATIONARY_HOLD_MS = 110       # 110 ms confirmation hold
COOLDOWN_MS = 220              # 220 ms refractory period

GESTURE_ICONS = {
    "circle":   ("⭕", "LASER CANNON", ACCENT_CYAN),
    "iks":      ("❌", "ORBITAL ION BEAM", ACCENT_CORAL),
    "go_up":    ("▲", "ADVANCE UP", ACCENT_EMERALD),
    "go_down":  ("▼", "RETREAT DOWN", ACCENT_AMBER),
    "go_left":  ("◄", "DASH LEFT", ACCENT_VIOLET),
    "go_right": ("►", "DASH RIGHT", ACCENT_VIOLET),
    "idle":     ("●", "READY", TEXT_MUTED),
}


class Particle:
    """Explosion / Spark / Ion Beam Particle"""
    def __init__(self, x, y, vx, vy, color, lifetime, size=3.0):
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        self.color = color
        self.lifetime = float(lifetime)
        self.max_life = float(lifetime)
        self.size = float(size)

    def update(self, dt):
        self.x += self.vx * dt * 60.0
        self.y += self.vy * dt * 60.0
        self.vx *= 0.95
        self.vy *= 0.95
        self.lifetime -= dt

    def draw(self, surface):
        if self.lifetime <= 0:
            return
        ratio = max(0.0, self.lifetime / self.max_life)
        cur_size = max(1.0, self.size * ratio)
        col = (int(self.color[0] * ratio), int(self.color[1] * ratio), int(self.color[2] * ratio))
        pygame.draw.circle(surface, col, (int(self.x), int(self.y)), int(cur_size))


class FloatingText:
    """Floating combat score/event text"""
    def __init__(self, text, x, y, color=ACCENT_GOLD, lifetime=1.2, font=None):
        self.text = text
        self.x = float(x)
        self.y = float(y)
        self.color = color
        self.lifetime = float(lifetime)
        self.max_life = float(lifetime)
        self.font = font

    def update(self, dt):
        self.y -= 36.0 * dt
        self.lifetime -= dt

    def draw(self, surface):
        if self.lifetime <= 0 or not self.font:
            return
        ratio = max(0.0, self.lifetime / self.max_life)
        col = (int(self.color[0] * ratio), int(self.color[1] * ratio), int(self.color[2] * ratio))
        surf = self.font.render(self.text, True, col)
        surface.blit(surf, (int(self.x - surf.get_width() / 2), int(self.y)))


class Projectile:
    """Player or Alien Laser Bolt"""
    def __init__(self, x, y, vy, color=ACCENT_CYAN, is_enemy=False, width=3, height=12):
        self.x = float(x)
        self.y = float(y)
        self.vy = float(vy)
        self.color = color
        self.is_enemy = is_enemy
        self.width = width
        self.height = height
        self.alive = True

    def update(self, dt):
        self.y += self.vy * dt

    def draw(self, surface):
        r = pygame.Rect(int(self.x - self.width / 2), int(self.y), self.width, self.height)
        pygame.draw.rect(surface, self.color, r, border_radius=2)
        # Inner bright core
        core = pygame.Rect(int(self.x - self.width / 4), int(self.y + 2), max(1, self.width // 2), self.height - 4)
        pygame.draw.rect(surface, (255, 255, 255), core)


class Bunker:
    """Destructible Defense Barrier with 4 health blocks"""
    def __init__(self, x, y, width=70, height=44):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        # 4 segments per bunker
        self.blocks = [
            pygame.Rect(x, y, width // 2 - 2, height // 2 - 2),
            pygame.Rect(x + width // 2 + 2, y, width // 2 - 2, height // 2 - 2),
            pygame.Rect(x, y + height // 2 + 2, width // 2 - 4, height // 2 - 2),
            pygame.Rect(x + width // 2 + 4, y + height // 2 + 2, width // 2 - 4, height // 2 - 2),
        ]
        self.healths = [3, 3, 3, 3]

    def hit_test(self, px, py):
        for idx, (b, hp) in enumerate(zip(self.blocks, self.healths)):
            if hp > 0 and b.collidepoint(px, py):
                self.healths[idx] -= 1
                return True
        return False

    def draw(self, surface):
        cols = [None, (90, 40, 40), (180, 100, 20), (0, 216, 246)]
        for b, hp in zip(self.blocks, self.healths):
            if hp > 0:
                col = cols[min(3, hp)]
                pygame.draw.rect(surface, col, b, border_radius=4)
                pygame.draw.rect(surface, (255, 255, 255), b, width=1, border_radius=4)


class Invader:
    """Cyber Alien Invader with animated procedural frames"""
    def __init__(self, x, y, row, col_idx):
        self.x = float(x)
        self.y = float(y)
        self.row = row          # 0 = Boss/UFO, 1-2 = Crab, 3-4 = Squid
        self.col_idx = col_idx
        self.width = 38
        self.height = 26
        self.alive = True

        if row == 0:
            self.score_val = 300
            self.color = ACCENT_CORAL
            self.type_name = "COMMAND"
        elif row in (1, 2):
            self.score_val = 200
            self.color = ACCENT_CYAN
            self.type_name = "CRAB"
        else:
            self.score_val = 100
            self.color = ACCENT_EMERALD
            self.type_name = "SQUID"

    def get_rect(self):
        return pygame.Rect(int(self.x - self.width / 2), int(self.y - self.height / 2), self.width, self.height)

    def draw(self, surface, anim_phase):
        cx, cy = int(self.x), int(self.y)
        w2, h2 = self.width // 2, self.height // 2

        # Retro cyber alien geometric vector design
        if self.row == 0:  # Command Cruiser
            pts = [(cx - w2, cy), (cx, cy - h2), (cx + w2, cy), (cx, cy + h2)]
            pygame.draw.polygon(surface, self.color, pts)
            pygame.draw.polygon(surface, (255, 255, 255), pts, 1)
            pygame.draw.circle(surface, ACCENT_GOLD, (cx, cy), 4)
        elif self.row in (1, 2):  # Cyber Crab
            leg_y = h2 if anim_phase == 0 else h2 - 4
            pts = [
                (cx - w2, cy - 2), (cx - w2 + 6, cy - h2),
                (cx + w2 - 6, cy - h2), (cx + w2, cy - 2),
                (cx + w2 - 4, cy + leg_y), (cx + w2 - 10, cy + 4),
                (cx - w2 + 10, cy + 4), (cx - w2 + 4, cy + leg_y)
            ]
            pygame.draw.polygon(surface, self.color, pts)
            pygame.draw.polygon(surface, (255, 255, 255), pts, 1)
            # Eyes
            pygame.draw.circle(surface, (255, 255, 255), (cx - 7, cy - 3), 2)
            pygame.draw.circle(surface, (255, 255, 255), (cx + 7, cy - 3), 2)
        else:  # Squid Drone
            tentacle_y = h2 if anim_phase == 0 else h2 - 6
            pts = [
                (cx - w2 + 4, cy - h2), (cx + w2 - 4, cy - h2),
                (cx + w2, cy), (cx + w2 - 6, cy + tentacle_y),
                (cx, cy + 2), (cx - w2 + 6, cy + tentacle_y),
                (cx - w2, cy)
            ]
            pygame.draw.polygon(surface, self.color, pts)
            pygame.draw.polygon(surface, (255, 255, 255), pts, 1)
            pygame.draw.circle(surface, (255, 255, 255), (cx - 6, cy - 2), 2)
            pygame.draw.circle(surface, (255, 255, 255), (cx + 6, cy - 2), 2)


class MysteryUFO:
    """High-value Red Saucer flying across top of screen"""
    def __init__(self, y=68, screen_w=1260):
        self.screen_w = screen_w
        self.dir = random.choice([-1, 1])
        self.x = -40 if self.dir == 1 else screen_w + 40
        self.y = float(y)
        self.vx = 140.0 * self.dir
        self.width = 54
        self.height = 20
        self.alive = True
        self.score_val = random.choice([500, 800, 1000])

    def update(self, dt):
        self.x += self.vx * dt
        if (self.dir == 1 and self.x > self.screen_w + 60) or (self.dir == -1 and self.x < -60):
            self.alive = False

    def get_rect(self):
        return pygame.Rect(int(self.x - self.width / 2), int(self.y - self.height / 2), self.width, self.height)

    def draw(self, surface):
        cx, cy = int(self.x), int(self.y)
        w2, h2 = self.width // 2, self.height // 2
        pts = [(cx - w2, cy), (cx - 12, cy - h2), (cx + 12, cy - h2), (cx + w2, cy), (cx + 14, cy + h2), (cx - 14, cy + h2)]
        pygame.draw.polygon(surface, ACCENT_CORAL, pts)
        pygame.draw.polygon(surface, ACCENT_GOLD, pts, 2)
        pygame.draw.ellipse(surface, (255, 255, 255), (cx - 10, cy - 6, 20, 8))


class DefenderCannon:
    """Player Mobile Defense Vessel"""
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.width = 46
        self.height = 26
        self.shields = 3
        self.max_shields = 3
        self.invulnerable_timer = 0.0

        # Ion Cannon Ulti
        self.ulti_charge = 100.0
        self.is_charging_ulti = False
        self.ulti_charge_timer = 0.0
        self.active_beam_timer = 0.0

    def update(self, dt, min_x, max_x, min_y, max_y):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vx *= 0.86
        self.vy *= 0.86

        self.x = max(min_x, min(max_x, self.x))
        self.y = max(min_y, min(max_y, self.y))

        if self.invulnerable_timer > 0:
            self.invulnerable_timer -= dt

        if self.active_beam_timer > 0:
            self.active_beam_timer -= dt

    def get_rect(self):
        return pygame.Rect(int(self.x - self.width / 2), int(self.y - self.height / 2), self.width, self.height)

    def draw(self, surface):
        if self.invulnerable_timer > 0 and int(self.invulnerable_timer * 14) % 2 == 0:
            return

        cx, cy = int(self.x), int(self.y)
        w2, h2 = self.width // 2, self.height // 2

        # Charging Ulti Vortex
        if self.is_charging_ulti:
            aura_r = int(32 + 10 * math.sin(time.time() * 25))
            pygame.draw.circle(surface, ACCENT_CORAL, (cx, cy), aura_r, 2)
            pygame.draw.circle(surface, ACCENT_GOLD, (cx, cy), int(aura_r * 0.6), 1)

        # Active Orbital Ion Beam
        if self.active_beam_timer > 0:
            beam_w = int(64 * (self.active_beam_timer / 0.45))
            beam_r = pygame.Rect(cx - beam_w // 2, 0, beam_w, cy)
            pygame.draw.rect(surface, (255, 60, 60, 180), beam_r)
            inner_r = pygame.Rect(cx - beam_w // 4, 0, beam_w // 2, cy)
            pygame.draw.rect(surface, (255, 255, 255), inner_r)

        # Hull Geometry
        pts = [
            (cx - w2, cy + h2), (cx - w2 + 6, cy - 2),
            (cx - 5, cy - 2), (cx - 5, cy - h2),
            (cx + 5, cy - h2), (cx + 5, cy - 2),
            (cx + w2 - 6, cy - 2), (cx + w2, cy + h2)
        ]
        pygame.draw.polygon(surface, (18, 32, 54), pts)
        pygame.draw.polygon(surface, ACCENT_CYAN, pts, 2)

        # Cockpit Energy Core
        core_col = ACCENT_EMERALD if self.shields == 3 else (ACCENT_AMBER if self.shields == 2 else ACCENT_CORAL)
        pygame.draw.circle(surface, core_col, (cx, cy + 3), 5)
        pygame.draw.line(surface, ACCENT_CYAN, (cx - 3, cy - h2), (cx - 3, cy - h2 - 6), 2)
        pygame.draw.line(surface, ACCENT_CYAN, (cx + 3, cy - h2), (cx + 3, cy - h2 - 6), 2)


class SpaceInvadersGame:
    """Master Game Controller with integrated Gesture Recognition and Live Model Switcher"""
    def __init__(self, existing_server=None):
        self.server = existing_server
        self.running = True
        self.width = 1260
        self.height = 780

        # Pygame init
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("WAND SPACE INVADERS • Edge AI Gesture Arcade Defense")
        self.clock = pygame.time.Clock()
        self.dt = 0.016

        # Fonts
        sans_fonts = ["segoe ui", "trebuchet ms", "arial", "helvetica", "dejavu sans"]
        mono_fonts = ["consolas", "fira code", "cascadia code", "courier new"]
        self.font_title  = pygame.font.SysFont(sans_fonts, 24, bold=True)
        self.font_main   = pygame.font.SysFont(sans_fonts, 16, bold=True)
        self.font_badge  = pygame.font.SysFont(sans_fonts, 12, bold=True)
        self.font_mono   = pygame.font.SysFont(mono_fonts, 13, bold=True)
        self.font_big    = pygame.font.SysFont(sans_fonts, 42, bold=True)
        self.font_huge   = pygame.font.SysFont(sans_fonts, 64, bold=True)

        # AI Classifier & Telemetry
        print("[AI] Initializing Multi-Model Classifier...")
        self.clf = MultiModelClassifier()
        self.calib = SensorCalibration()
        self.ahrs = MadgwickAHRS(sample_rate=100.0, gain=0.1, use_mag=False, filter_type="ekf")

        self.lock = threading.Lock()
        self.packet_queue = []
        self.tcp_client = None
        self.total_packets = 0
        self.auto_tared = False

        # Continuous Gesture Detection State
        self.current_omega = 0.0
        self.smooth_omega = 0.0
        self.peak_omega = 0.0
        self.in_gesture = False
        self.pre_roll = deque(maxlen=8)
        self.stroke_buffer = []
        self.last_motion_time_ms = 0
        self.cooldown_until_ms = 0

        # HUD feedback
        self.last_gesture_name = "idle"
        self.last_gesture_conf = 1.0
        self.last_gesture_time = 0.0

        # Standalone Sockets (if not running under recorder_app)
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
        else:
            self.tcp_server = None
            self.udp_sock = None
            self.tcp_thread = None
            self.udp_thread = None

        # Game Entities
        self.cannon = DefenderCannon(self.width / 2, self.height - 75)
        self.invaders = []
        self.player_lasers = []
        self.enemy_bombs = []
        self.bunkers = []
        self.ufo = None
        self.particles = []
        self.floating_texts = []

        # Swarm motion & pacing
        self.swarm_dir = 1
        self.swarm_speed = 38.0
        self.swarm_step_timer = 0.0
        self.anim_phase = 0
        self.enemy_fire_timer = 1.2
        self.ufo_spawn_timer = random.uniform(14.0, 22.0)

        # Game State
        self.score = 0
        self.wave = 1
        self.game_over = False
        self.victory = False
        self.screen_shake = 0.0

        # 3D Wand Preview in HUD
        self.flip_y = -1.0
        self.flip_z = -1.0
        self.cam_yaw = math.radians(-125.0)
        self.cam_pitch = math.radians(22.0)
        self.cam_dist = 3.2
        self.wand_spine = np.array([[-0.35, 0, 0], [0.05, 0, 0], [1.10, 0, 0]], dtype=np.float64)

        # Starfield
        self.stars = [(random.randint(0, 1920), random.randint(0, 1080), random.choice([1, 2]), random.uniform(0.3, 0.9)) for _ in range(110)]

        self._spawn_bunkers()
        self._spawn_wave(self.wave)

    def _standalone_tcp_loop(self):
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
                print(f"[TCP] Space Invaders connected to ESP32: {addr[0]}. Starting telemetry...")
                client.sendall(b"0")
            except Exception:
                if self.running:
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
                        self.packet_queue.extend(batch)
            except Exception:
                pass

    def _spawn_bunkers(self):
        self.bunkers.clear()
        n_bunkers = 4
        b_width = 72
        spacing = (self.width - 240) / (n_bunkers + 1)
        y_pos = self.height - 165
        for i in range(n_bunkers):
            bx = 120 + spacing * (i + 1) - b_width / 2
            self.bunkers.append(Bunker(bx, y_pos, width=b_width, height=40))

    def _spawn_wave(self, wave_num):
        """Builds classic 5x9 Invader Armada grid"""
        self.invaders.clear()
        self.player_lasers.clear()
        self.enemy_bombs.clear()

        rows = 5
        cols = 9
        grid_start_x = (self.width - (cols * 56)) / 2 + 28
        grid_start_y = 135 + min(110, (wave_num - 1) * 20)

        for r in range(rows):
            for c in range(cols):
                ix = grid_start_x + c * 56
                iy = grid_start_y + r * 40
                self.invaders.append(Invader(ix, iy, row=r, col_idx=c))

        self.swarm_speed = 36.0 + min(60.0, (wave_num - 1) * 12.0)
        self.swarm_dir = 1
        self.floating_texts.append(
            FloatingText(f"★ WAVE {wave_num}: INVADERS APPROACHING ★", self.width / 2, self.height / 2 - 40,
                         color=ACCENT_CYAN, lifetime=2.2, font=self.font_big)
        )
        play_sound(750, 140)

    # --- WEAPONRY & ACTIONS ---
    def action_shoot(self):
        """Fires dual plasma cannons"""
        if len(self.player_lasers) >= 4:  # Max 4 concurrent bolts
            return
        cx, cy = self.cannon.x, self.cannon.y - self.cannon.height / 2
        self.player_lasers.append(Projectile(cx - 10, cy, vy=-950.0, color=ACCENT_CYAN))
        self.player_lasers.append(Projectile(cx + 10, cy, vy=-950.0, color=ACCENT_CYAN))
        play_sound(1450, 35)

        for _ in range(4):
            self.particles.append(
                Particle(cx + random.uniform(-10, 10), cy, random.uniform(-30, 30), random.uniform(-90, -10),
                         ACCENT_CYAN, lifetime=0.18, size=3.0)
            )

    def action_ultimate(self):
        """Charges and unleashes Orbital Ion Cannon"""
        if self.cannon.is_charging_ulti or self.cannon.active_beam_timer > 0:
            return

        self.cannon.is_charging_ulti = True
        self.cannon.ulti_charge_timer = 0.50  # 500 ms charging phase
        play_sound(500, 120)
        self.screen_shake = 10.0

        self.floating_texts.append(
            FloatingText("⚡ CHARGING ORBITAL ION CANNON! ⚡", self.cannon.x, self.cannon.y - 45,
                         color=ACCENT_CORAL, lifetime=1.2, font=self.font_main)
        )

    def _detonate_ultimate(self):
        """Fires the massive vertical Ion Beam obliterating columns of invaders"""
        self.cannon.is_charging_ulti = False
        self.cannon.active_beam_timer = 0.45
        self.screen_shake = 22.0
        play_sound(1750, 200)

        cx = self.cannon.x
        beam_w = 70.0
        hit_count = 0

        # Incinerate all invaders in beam's column
        for inv in self.invaders[:]:
            if abs(inv.x - cx) < (beam_w / 2 + inv.width / 2):
                hit_count += 1
                self.score += inv.score_val * 2
                self._explode_invader(inv, is_ulti=True)
                self.invaders.remove(inv)

        # Vaporize enemy bombs in beam
        self.enemy_bombs = [b for b in self.enemy_bombs if abs(b.x - cx) >= beam_w / 2]

        # UFO check
        if self.ufo and self.ufo.alive and abs(self.ufo.x - cx) < (beam_w / 2 + self.ufo.width / 2):
            self.score += self.ufo.score_val * 2
            self.floating_texts.append(FloatingText(f"★ MOTHER SHIP VAPORIZED (+{self.ufo.score_val*2})! ★", cx, self.ufo.y, color=ACCENT_GOLD, font=self.font_main))
            self.ufo.alive = False

        self.cannon.ulti_charge = 0.0
        if hit_count > 0:
            bonus = hit_count * 200
            self.score += bonus
            self.floating_texts.append(
                FloatingText(f"💥 ION STRIKE: {hit_count} INVADERS (+{bonus} PTS)! 💥", self.width / 2, self.height / 2 - 20,
                             color=ACCENT_GOLD, lifetime=2.2, font=self.font_big)
            )

    def _explode_invader(self, inv, is_ulti=False):
        play_sound(420, 50)
        n = 16 if is_ulti else 10
        for _ in range(n):
            ang = random.uniform(0, 2 * math.pi)
            spd = random.uniform(70, 220 if is_ulti else 140)
            col = ACCENT_GOLD if random.random() < 0.4 else inv.color
            self.particles.append(
                Particle(inv.x, inv.y, math.cos(ang) * spd, math.sin(ang) * spd, col, lifetime=random.uniform(0.3, 0.7), size=random.uniform(2.5, 4.5))
            )

    def _process_gesture(self, gesture_name, conf):
        """Maps gestures to Defender Cannon combat moves"""
        self.last_gesture_name = gesture_name
        self.last_gesture_conf = conf
        self.last_gesture_time = time.time()

        if gesture_name == "circle":
            self.action_shoot()
            self.floating_texts.append(FloatingText("⭕ LASER SHOT!", self.cannon.x, self.cannon.y - 30, color=ACCENT_CYAN, font=self.font_badge))
        elif gesture_name == "iks":
            self.action_ultimate()
        elif gesture_name == "go_left":
            self.cannon.vx = -460.0
            self.floating_texts.append(FloatingText("◄ DASH LEFT", self.cannon.x, self.cannon.y - 30, color=ACCENT_VIOLET, font=self.font_badge))
            play_sound(620, 25)
        elif gesture_name == "go_right":
            self.cannon.vx = 460.0
            self.floating_texts.append(FloatingText("► DASH RIGHT", self.cannon.x, self.cannon.y - 30, color=ACCENT_VIOLET, font=self.font_badge))
            play_sound(620, 25)
        elif gesture_name == "go_up":
            self.cannon.vy = -300.0
            self.floating_texts.append(FloatingText("▲ ADVANCE", self.cannon.x, self.cannon.y - 30, color=ACCENT_EMERALD, font=self.font_badge))
            play_sound(720, 25)
        elif gesture_name == "go_down":
            self.cannon.vy = 300.0
            self.floating_texts.append(FloatingText("▼ RETREAT", self.cannon.x, self.cannon.y - 30, color=ACCENT_AMBER, font=self.font_badge))
            play_sound(380, 30)

    def _render_mini_wand(self, surface, center_x, center_y):
        """Draws live 3D wand in HUD showing real-time player orientation"""
        R = self.ahrs.get_smooth_rotation_matrix(dt=self.dt, smoothing_speed=24.0)

        def proj(pt3d):
            x, y, z = pt3d
            cy, sy = math.cos(self.cam_yaw), math.sin(self.cam_yaw)
            x1 = x * cy - y * sy
            y1 = x * sy + y * cy
            z1 = z
            cp, sp = math.cos(self.cam_pitch), math.sin(self.cam_pitch)
            cam_x = y1
            cam_y = z1 * cp - x1 * sp
            cam_z = self.cam_dist + z1 * sp + x1 * cp
            fov = 75.0
            px = int(center_x + (cam_x * fov) / max(0.2, cam_z))
            py = int(center_y - (cam_y * fov) / max(0.2, cam_z))
            return px, py

        pts_w = self.wand_spine @ R.T
        pts_w[:, 1] *= self.flip_y
        pts_w[:, 2] *= self.flip_z
        p0 = proj(pts_w[0])
        p1 = proj(pts_w[1])
        p2 = proj(pts_w[2])
        pygame.draw.line(surface, WAND_GRIP, p0, p1, 5)
        pygame.draw.line(surface, WAND_SHAFT, p1, p2, 4)
        draw_glow_sphere(surface, p2, base_radius=5, color_rgb=WAND_TIP_AURA)

    def run(self):
        print("\n" + "="*60)
        print("WAND SPACE INVADERS • GESTURE ARCADE DEFENSE READY")
        print(f" -> Active AI Engine: {self.clf.active_model_name}")
        print(" -> Key [N]: Toggle Model (Conv-LSTM <-> 1D-CNN <-> Random Forest)")
        print(" -> [Swipe Left/Right] : Dash Left / Right")
        print(" -> [Swipe Up/Down]    : Advance / Retreat")
        print(" -> [Circle]           : Fire Dual Lasers")
        print(" -> [X-Mark]           : Charged Orbital Ion Cannon")
        print("="*60)

        while self.running:
            self.dt = max(0.001, min(0.06, self.clock.tick(60) / 1000.0))
            self.width, self.height = self.screen.get_size()

            # --- 1. USER INPUT & EVENTS ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key in (pygame.K_SPACE, pygame.K_RETURN) and (self.game_over or self.victory):
                        # Restart Game
                        self.cannon = DefenderCannon(self.width / 2, self.height - 75)
                        self.score = 0
                        self.wave = 1
                        self.game_over = False
                        self.victory = False
                        self._spawn_bunkers()
                        self._spawn_wave(self.wave)
                    elif event.key == pygame.K_SPACE:
                        self.action_shoot()
                    elif event.key == pygame.K_x:
                        self.action_ultimate()
                    elif event.key == pygame.K_n:
                        # Switch AI Model
                        new_model = self.clf.toggle_model()
                        self.floating_texts.append(
                            FloatingText(f"AI ENGINE: {new_model}", self.width / 2, 70, color=ACCENT_VIOLET, font=self.font_main)
                        )
                        play_sound(950, 60)
                    elif event.key in (pygame.K_r, pygame.K_t):
                        self.ahrs.reset()
                        if self.server is not None:
                            self.server.start_streaming()
                        elif self.tcp_client is not None:
                            try:
                                self.tcp_client.sendall(b"0")
                            except Exception:
                                pass

            # Keyboard Fallback Movement
            keys = pygame.key.get_pressed()
            if not self.game_over and not self.victory:
                if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                    self.cannon.vx = -420.0
                if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                    self.cannon.vx = 420.0
                if keys[pygame.K_UP] or keys[pygame.K_w]:
                    self.cannon.vy = -260.0
                if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                    self.cannon.vy = 260.0

            # --- 2. INGEST TELEMETRY & CLASSIFY GESTURES ---
            if self.server is not None:
                new_samples = self.server.get_live_samples()
            else:
                with self.lock:
                    new_samples = self.packet_queue[:]
                    self.packet_queue.clear()

            if new_samples:
                for s in new_samples:
                    cal_acc, cal_gyro = self.calib.apply(s, adaptive_gyro=False)
                    s["cal_accel_x"], s["cal_accel_y"], s["cal_accel_z"] = cal_acc
                    s["cal_gyro_x"],  s["cal_gyro_y"],  s["cal_gyro_z"]  = cal_gyro

                    self.ahrs.update(cal_gyro, cal_acc, mag=None, timestamp_ms=s["timestamp_ms"])
                    self.total_packets += 1
                    if not self.auto_tared and self.total_packets >= 15:
                        self.ahrs.reset()
                        self.auto_tared = True

                    g_norm = float(np.linalg.norm(cal_gyro))
                    self.current_omega = g_norm
                    self.smooth_omega = 0.70 * self.smooth_omega + 0.30 * g_norm
                    now_ms = s["timestamp_ms"]

                    if now_ms < self.cooldown_until_ms:
                        self.in_gesture = False
                        self.stroke_buffer.clear()
                        continue

                    if not self.in_gesture:
                        self.pre_roll.append(s)
                        if self.smooth_omega >= MOTION_THRESHOLD:
                            self.in_gesture = True
                            self.stroke_buffer = list(self.pre_roll)
                            self.stroke_buffer.append(s)
                            self.peak_omega = g_norm
                            self.last_motion_time_ms = now_ms
                    else:
                        self.stroke_buffer.append(s)
                        if g_norm > self.peak_omega:
                            self.peak_omega = g_norm

                        if self.smooth_omega >= REST_THRESHOLD:
                            self.last_motion_time_ms = now_ms
                        else:
                            time_stationary = now_ms - self.last_motion_time_ms
                            if time_stationary >= STATIONARY_HOLD_MS:
                                if len(self.stroke_buffer) >= MIN_GESTURE_SAMPLES and self.peak_omega >= 0.44:
                                    mat = np.array([
                                        [x["cal_accel_x"], x["cal_accel_y"], x["cal_accel_z"],
                                         x["cal_gyro_x"],  x["cal_gyro_y"],  x["cal_gyro_z"]]
                                        for x in self.stroke_buffer
                                    ], dtype=np.float32)

                                    probs, best_name, conf = self.clf.predict_sequence(mat)
                                    if best_name != "idle" and conf >= 0.40:
                                        self._process_gesture(best_name, conf)
                                        self.cooldown_until_ms = now_ms + COOLDOWN_MS

                                self.stroke_buffer.clear()
                                self.in_gesture = False
                                self.peak_omega = 0.0

                        if len(self.stroke_buffer) > MAX_GESTURE_SAMPLES:
                            self.stroke_buffer.clear()
                            self.in_gesture = False
            else:
                self.smooth_omega *= 0.94

            # --- 3. GAME SIMULATION & COMBAT LOGIC ---
            if not self.game_over and not self.victory:
                # Handle Ulti charging countdown
                if self.cannon.is_charging_ulti:
                    self.cannon.ulti_charge_timer -= self.dt
                    if self.cannon.ulti_charge_timer <= 0:
                        self._detonate_ultimate()

                # Cannon bounds (lower arena)
                self.cannon.update(self.dt, min_x=50, max_x=self.width - 50, min_y=self.height - 180, max_y=self.height - 50)
                self.cannon.ulti_charge = min(100.0, self.cannon.ulti_charge + 4.5 * self.dt)

                # Marching Invaders
                if self.invaders:
                    hit_wall = False
                    # Speed scaling as swarm thins
                    alive_ratio = len(self.invaders) / 45.0
                    cur_speed = self.swarm_speed * (1.0 + (1.0 - alive_ratio) * 1.8)

                    for inv in self.invaders:
                        inv.x += self.swarm_dir * cur_speed * self.dt
                        if (self.swarm_dir == 1 and inv.x >= self.width - 60) or (self.swarm_dir == -1 and inv.x <= 60):
                            hit_wall = True

                    if hit_wall:
                        self.swarm_dir *= -1
                        self.anim_phase = 1 - self.anim_phase
                        for inv in self.invaders:
                            inv.y += 20.0
                            # Check if invaders breached bottom defense line
                            if inv.y >= self.height - 130:
                                self.game_over = True
                                play_sound(200, 500)
                                break

                    # Alien Bomb Dropping
                    self.enemy_fire_timer -= self.dt
                    if self.enemy_fire_timer <= 0:
                        shooter = random.choice(self.invaders)
                        self.enemy_bombs.append(Projectile(shooter.x, shooter.y + 14, vy=340.0, color=ACCENT_AMBER, is_enemy=True, width=4, height=14))
                        self.enemy_fire_timer = random.uniform(0.65, 1.35)

                # Mystery UFO update & spawn
                self.ufo_spawn_timer -= self.dt
                if self.ufo_spawn_timer <= 0 and (self.ufo is None or not self.ufo.alive):
                    self.ufo = MysteryUFO(y=75, screen_w=self.width)
                    self.ufo_spawn_timer = random.uniform(18.0, 28.0)
                    play_sound(1200, 60)

                if self.ufo and self.ufo.alive:
                    self.ufo.update(self.dt)

                # Update Lasers
                for l in self.player_lasers[:]:
                    l.update(self.dt)
                    if l.y < 30:
                        self.player_lasers.remove(l)

                # Update Enemy Bombs
                for b in self.enemy_bombs[:]:
                    b.update(self.dt)
                    if b.y > self.height:
                        self.enemy_bombs.remove(b)

                # Collision: Player Laser vs Invader
                for l in self.player_lasers[:]:
                    hit = False
                    for inv in self.invaders[:]:
                        if inv.get_rect().collidepoint(l.x, l.y):
                            if l in self.player_lasers:
                                self.player_lasers.remove(l)
                            self.invaders.remove(inv)
                            self.score += inv.score_val
                            self.floating_texts.append(FloatingText(f"+{inv.score_val}", inv.x, inv.y, color=inv.color, font=self.font_badge))
                            self._explode_invader(inv)
                            hit = True
                            break
                    if not hit and self.ufo and self.ufo.alive and self.ufo.get_rect().collidepoint(l.x, l.y):
                        if l in self.player_lasers:
                            self.player_lasers.remove(l)
                        self.score += self.ufo.score_val
                        self.floating_texts.append(FloatingText(f"★ UFO DESTROYED +{self.ufo.score_val}! ★", self.ufo.x, self.ufo.y, color=ACCENT_GOLD, font=self.font_main))
                        self.ufo.alive = False
                        play_sound(1600, 100)

                # Collision: Lasers vs Bunkers
                for l in self.player_lasers[:]:
                    for bk in self.bunkers:
                        if bk.hit_test(l.x, l.y):
                            if l in self.player_lasers:
                                self.player_lasers.remove(l)
                            break

                for b in self.enemy_bombs[:]:
                    for bk in self.bunkers:
                        if bk.hit_test(b.x, b.y):
                            if b in self.enemy_bombs:
                                self.enemy_bombs.remove(b)
                            break

                # Collision: Enemy Bomb vs Defender Cannon
                if self.cannon.invulnerable_timer <= 0:
                    c_rect = self.cannon.get_rect()
                    for b in self.enemy_bombs[:]:
                        if c_rect.collidepoint(b.x, b.y):
                            self.enemy_bombs.remove(b)
                            self.cannon.shields -= 1
                            self.cannon.invulnerable_timer = 2.0
                            self.screen_shake = 16.0
                            play_sound(260, 160)
                            self.floating_texts.append(
                                FloatingText("⚠️ HULL HIT! -1 SHIELD", self.cannon.x, self.cannon.y - 45, color=ACCENT_CORAL, font=self.font_main)
                            )
                            if self.cannon.shields <= 0:
                                self.game_over = True
                                play_sound(160, 450)
                            break

                # Wave Complete Check
                if not self.invaders:
                    self.wave += 1
                    self.score += 1000
                    self._spawn_wave(self.wave)

            # Update Particles & Floating Text
            for p in self.particles[:]:
                p.update(self.dt)
                if p.lifetime <= 0:
                    self.particles.remove(p)

            for ft in self.floating_texts[:]:
                ft.update(self.dt)
                if ft.lifetime <= 0:
                    self.floating_texts.remove(ft)

            # --- 4. RENDERING PIPELINE ---
            self.screen.fill(BG_VOID)

            # Screen shake offset
            shake_x, shake_y = 0, 0
            if self.screen_shake > 0:
                shake_x = random.uniform(-self.screen_shake, self.screen_shake)
                shake_y = random.uniform(-self.screen_shake, self.screen_shake)
                self.screen_shake = max(0.0, self.screen_shake - self.dt * 45.0)

            # Starfield
            for sx, sy, sz, br in self.stars:
                col = (int(160 * br), int(190 * br), int(220 * br))
                pygame.draw.circle(self.screen, col, (int((sx + shake_x) % self.width), int((sy + shake_y) % self.height)), sz)

            # Defense Line Divider
            line_y = self.height - 180
            pygame.draw.line(self.screen, (28, 38, 56), (0, line_y), (self.width, line_y), 1)

            # Render Bunkers
            for bk in self.bunkers:
                bk.draw(self.screen)

            # Render Invaders
            for inv in self.invaders:
                inv.draw(self.screen, self.anim_phase)

            # Render Mystery UFO
            if self.ufo and self.ufo.alive:
                self.ufo.draw(self.screen)

            # Render Lasers & Bombs
            for l in self.player_lasers:
                l.draw(self.screen)
            for b in self.enemy_bombs:
                b.draw(self.screen)

            # Render Defender Cannon
            if not self.game_over:
                self.cannon.draw(self.screen)

            # Render Particles & Texts
            for p in self.particles:
                p.draw(self.screen)
            for ft in self.floating_texts:
                ft.draw(self.screen)

            # --- 5. BENTO HUD OVERLAY ---
            # Top-Left Card: Mission Score & Fleet Health
            card_tl = pygame.Rect(18, 14, 270, 95)
            draw_card(self.screen, card_tl, bg_color=SURFACE_CARD, border_color=BORDER_BASE, radius=12)

            self.screen.blit(self.font_title.render(f"{self.score:06d} PTS", True, TEXT_PRIMARY), (card_tl.x + 16, card_tl.y + 10))
            self.screen.blit(self.font_badge.render(f"WAVE {self.wave:02d}", True, ACCENT_CYAN), (card_tl.right - 75, card_tl.y + 14))

            # Shield Bar
            sh_ratio = max(0.0, self.cannon.shields / self.cannon.max_shields)
            sh_col = ACCENT_EMERALD if self.cannon.shields == 3 else (ACCENT_AMBER if self.cannon.shields == 2 else ACCENT_CORAL)
            draw_progress_bar(self.screen, card_tl.x + 16, card_tl.y + 44, card_tl.w - 32, 10, sh_ratio, fill_color=sh_col)
            self.screen.blit(self.font_mono.render(f"SHIELDS: {self.cannon.shields}/{self.cannon.max_shields}", True, TEXT_SECONDARY), (card_tl.x + 16, card_tl.y + 58))

            # Ulti Bar
            u_ratio = self.cannon.ulti_charge / 100.0
            draw_progress_bar(self.screen, card_tl.x + 16, card_tl.y + 76, card_tl.w - 32, 7, u_ratio, fill_color=ACCENT_CORAL)

            # Top-Right Card: Live Telemetry, Model Switcher & 3D Wand Preview
            card_tr = pygame.Rect(self.width - 360, 14, 342, 104)
            draw_card(self.screen, card_tr, bg_color=SURFACE_CARD, border_color=BORDER_BASE, radius=12)

            # Mini 3D Wand Viewport
            self._render_mini_wand(self.screen, card_tr.right - 55, card_tr.y + 52)

            # Last Gesture Pill
            g_icon, g_label, g_col = GESTURE_ICONS.get(self.last_gesture_name, ("●", "READY", TEXT_MUTED))
            is_recent = (time.time() - self.last_gesture_time) < 1.4
            badge_bg = (34, 48, 72) if is_recent else SURFACE_PILL
            draw_pill_badge(self.screen, f"{g_icon} {g_label}", self.font_badge, card_tr.x + 14, card_tr.y + 10,
                            bg_color=badge_bg, text_color=g_col if is_recent else TEXT_SECONDARY, border_color=BORDER_BASE)

            # Live Model Engine Pill with [N] Shortcut
            draw_pill_badge(self.screen, f"[N] {self.clf.active_model_name}", self.font_badge, card_tr.x + 14, card_tr.y + 42,
                            bg_color=(28, 22, 48), text_color=ACCENT_VIOLET, border_color=ACCENT_VIOLET, padding_x=8, padding_y=3)

            # Dynamics Gauge
            dyn_str = f"||ω||: {self.smooth_omega:4.2f} rad/s"
            self.screen.blit(self.font_mono.render(dyn_str, True, ACCENT_AMBER if self.smooth_omega >= MOTION_THRESHOLD else TEXT_MUTED), (card_tr.x + 16, card_tr.y + 72))
            dyn_ratio = min(1.0, self.smooth_omega / 1.5)
            draw_progress_bar(self.screen, card_tr.x + 16, card_tr.y + 88, 175, 6, dyn_ratio,
                              fill_color=ACCENT_CORAL if self.smooth_omega >= MOTION_THRESHOLD else ACCENT_CYAN)

            # Bottom Controls Pill
            foot_w = min(840, self.width - 36)
            foot_rect = pygame.Rect((self.width - foot_w) // 2, self.height - 40, foot_w, 30)
            draw_card(self.screen, foot_rect, bg_color=SURFACE_CARD, border_color=BORDER_BASE, radius=15)
            ctrl_str = "[◄/► Left/Right] Dash   [▲/▼ Up/Down] Move   [⭕ Circle] Shoot   [❌ X-Mark] Ulti   [N] Switch Model   [ESC] Exit"
            foot_surf = self.font_mono.render(ctrl_str, True, TEXT_MUTED)
            self.screen.blit(foot_surf, (foot_rect.x + (foot_rect.w - foot_surf.get_width()) // 2, foot_rect.y + 7))

            # Game Over Overlay
            if self.game_over:
                overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                overlay.fill((10, 14, 23, 220))
                self.screen.blit(overlay, (0, 0))

                t_gov = self.font_huge.render("EARTH HAS FALLEN", True, ACCENT_CORAL)
                t_sub = self.font_title.render(f"Final Score: {self.score} PTS  |  Wave {self.wave}", True, TEXT_PRIMARY)
                t_rst = self.font_main.render("Press [SPACE] or [ENTER] to Restart Defense", True, ACCENT_GOLD)

                self.screen.blit(t_gov, ((self.width - t_gov.get_width()) // 2, self.height // 2 - 80))
                self.screen.blit(t_sub, ((self.width - t_sub.get_width()) // 2, self.height // 2))
                self.screen.blit(t_rst, ((self.width - t_rst.get_width()) // 2, self.height // 2 + 55))

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
    game = SpaceInvadersGame()
    game.run()


if __name__ == "__main__":
    main()

import os
import sys
import re

# Ensure UTF-8 output encoding across Windows consoles
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

if os.name == 'nt':
    os.system('')  # Enable ANSI in console
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
import csv
import math
from collections import deque
from datetime import datetime
from ahrs_utils import SensorCalibration, MadgwickAHRS
from visualizer_3d import Wand3DVisualizer, GestureRecordingSessionUI

# Global calibration instance
calibration = SensorCalibration()

# --- ANSI TERMINAL STYLING TOKENS (HUD / Dark Tech Dashboard) ---
C_CYAN    = "\033[96m"
C_BLUE    = "\033[38;5;39m"
C_GREEN   = "\033[92m"
C_EMERALD = "\033[38;5;49m"
C_AMBER   = "\033[93m"
C_RED     = "\033[91m"
C_PURPLE  = "\033[95m"
C_VIOLET  = "\033[38;5;141m"
C_BOLD    = "\033[1m"
C_DIM     = "\033[90m"
C_WHITE   = "\033[97m"
C_RESET   = "\033[0m"

# --- MOTION DETECTION & TRIMMING PARAMETERS ---
GYRO_ENERGY_THRESHOLD = 0.35   # Motion detection threshold [rad/s]
PADDING_BEFORE_SAMPLES = 8     # Pre-roll padding (~80 ms at 100 Hz)
PADDING_AFTER_SAMPLES  = 10    # Post-roll padding (~100 ms at 100 Hz)
AUTO_TRIM_ENABLED      = True   # Whether auto-trimming silence is active

# Sound alerts on Windows
try:
    import winsound
    def beep(freq, duration):
        winsound.Beep(freq, duration)
except ImportError:
    def beep(freq, duration):
        pass

# --- NETWORK CONFIGURATION ---
HOST = "0.0.0.0"
PORT = 2137

# --- ESP32 TELEMETRY PROTOCOL ---
DATA_FORMAT = "<I9f?"
PACKET_SIZE = struct.calcsize(DATA_FORMAT)

CMD_START_STREAMING = '0'  # In ESP32: start_streaming = '0'
CMD_STOP_STREAMING  = '1'  # In ESP32: stop_streaming  = '1'
CMD_PREPARE         = '2'  # Short prepare flash

# --- SUPPORTED GESTURES ---
GESTURES = {
    "1": ("circle",    "Circle"),
    "2": ("go_up",     "Swipe Up"),
    "3": ("go_down",   "Swipe Down"),
    "4": ("go_left",   "Swipe Left"),
    "5": ("go_right",  "Swipe Right"),
    "6": ("iks",       "Cross (X)"),
    "7": ("idle",      "Rest (Idle)"),
}

DEFAULT_GESTURE_CAPTURE_DURATIONS = {
    "circle":   2.2,   # Circle: 2.2s (actual motion is ~1.1 - 1.9s)
    "go_up":    1.5,   # Swipe Up: 1.5s (actual motion is ~0.3 - 1.0s)
    "go_down":  1.6,   # Swipe Down: 1.6s (actual motion is ~0.4 - 1.2s)
    "go_left":  1.5,   # Swipe Left: 1.5s (actual motion is ~0.5 - 0.9s)
    "go_right": 1.5,   # Swipe Right: 1.5s (actual motion is ~0.5 - 0.9s)
    "iks":      3.2,   # Cross (X): 3.2s (two diagonal strokes take 2.0 - 2.9s!)
    "idle":     5.0,   # Rest (Idle): 5.0s stationary baseline
}

DATASET_DIR = "dataset"


# =========================================================================
# CLAUDE CODE IN-PLACE CONSOLE ENGINE
# =========================================================================
class ClaudeConsole:
    @staticmethod
    def strip_ansi(s):
        return re.sub(r'\033\[[0-9;]*m', '', s)

    @staticmethod
    def box_row(styled_content, width=74):
        plain_len = len(ClaudeConsole.strip_ansi(styled_content))
        spaces = " " * max(0, width - 4 - plain_len)
        return f" │  {styled_content}{spaces}│"
    @staticmethod
    def setup():
        if sys.stdout.encoding.lower() != 'utf-8':
            try:
                sys.stdout.reconfigure(encoding='utf-8')
            except Exception:
                pass
        if os.name == 'nt':
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except Exception:
                pass

    @staticmethod
    def clear():
        """Full clear and reset to home (row 1, col 1)"""
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

    @staticmethod
    def hide_cursor():
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()

    @staticmethod
    def show_cursor():
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()

    @staticmethod
    def render_frame(lines):
        """
        Renders a multi-line buffer in-place from row 1, col 1 without scrolling or flicker.
        \\033[H repositions cursor to top-left.
        \\033[K clears from line text to end of row.
        \\033[J clears any remaining lines below the buffer.
        """
        buf = "\033[H" + "".join(f"{line}\033[K\n" for line in lines) + "\033[J"
        sys.stdout.write(buf)
        sys.stdout.flush()

    @staticmethod
    def poll_key():
        """Non-blocking check for keypress on Windows. Returns char or None."""
        try:
            import msvcrt
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in (b'\x00', b'\xe0'):
                    msvcrt.getch()  # consume scan code for arrow/function keys
                    return None
                if ch == b'\x1b':   # ESC
                    return "ESC"
                if ch in (b'\r', b'\n'):
                    return "ENTER"
                return ch.decode("utf-8", errors="ignore")
        except Exception:
            pass
        return None

    @staticmethod
    def get_key(prompt=None):
        """Blocking wait for a single key."""
        if prompt:
            sys.stdout.write(prompt)
            sys.stdout.flush()
        try:
            import msvcrt
            while True:
                ch = msvcrt.getch()
                if ch in (b'\x00', b'\xe0'):
                    msvcrt.getch()
                    continue
                if ch == b'\x1b':
                    return "ESC"
                if ch in (b'\r', b'\n'):
                    return "ENTER"
                char = ch.decode("utf-8", errors="ignore")
                return char
        except Exception:
            return input().strip()


# =========================================================================
# WAND DATA SERVER (TCP COMMANDS & UDP TELEMETRY)
# =========================================================================
class WandDataServer:
    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = port
        self.tcp_client = None
        self.tcp_addr = None
        self.is_connected = False
        self.is_recording = False
        self.is_streaming = False
        self.running = True
        self.batch_size = 10  # Domyślny rozmiar batcha ESP32 (1-32 próbek na pakiet)

        self.current_buffer = []
        self.live_stream_queue = []
        self.lock = threading.Lock()

        # Telemetry metrics
        self.total_packets = 0
        self.last_packet_time = 0.0
        self.latest_sample = None
        self.estimated_hz = 0.0
        self._hz_counter = 0
        self._last_hz_time = time.time()

        # Network sockets
        self.tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tcp_server.bind((self.host, self.port))
        self.tcp_server.listen(1)

        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        self.udp_sock.bind((self.host, self.port))

        self.tcp_thread = threading.Thread(target=self._listen_tcp, daemon=True)
        self.udp_thread = threading.Thread(target=self._listen_udp, daemon=True)

    def start(self):
        self.tcp_thread.start()
        self.udp_thread.start()

    def _listen_tcp(self):
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
                    self.tcp_addr = addr
                    self.is_connected = True

                    # Synchronizuj niestandardowy rozmiar batcha przy nowym połączeniu
                    if self.batch_size != 10:
                        try:
                            self.tcp_client.sendall(f"B{self.batch_size}\n".encode("ascii"))
                        except Exception:
                            pass
            except Exception:
                if self.running:
                    time.sleep(0.5)

    def set_batch_size(self, size: int) -> bool:
        """Sends command to ESP32 to change UDP packet batch size dynamically at runtime."""
        size = max(1, min(32, int(size)))
        with self.lock:
            self.batch_size = size
            if self.tcp_client and self.is_connected:
                try:
                    self.tcp_client.sendall(f"B{size}\n".encode("ascii"))
                    return True
                except Exception:
                    self.is_connected = False
                    self.tcp_client = None
                    return False
        return False

    def _listen_udp(self):
        while self.running:
            try:
                raw_data, _ = self.udp_sock.recvfrom(2048)
                if len(raw_data) >= PACKET_SIZE and (len(raw_data) % PACKET_SIZE == 0):
                    batch_records = []
                    for fields in struct.iter_unpack(DATA_FORMAT, raw_data):
                        batch_records.append({
                            "timestamp_ms": fields[0],
                            "accel_x": fields[1], "accel_y": fields[2], "accel_z": fields[3],
                            "gyro_x":  fields[4], "gyro_y":  fields[5], "gyro_z":  fields[6],
                            "mag_x":   fields[7], "mag_y":   fields[8], "mag_z":   fields[9],
                            "wand_id": 1 if fields[10] else 0
                        })
                    now = time.time()
                    with self.lock:
                        n_batch = len(batch_records)
                        self.total_packets += n_batch
                        self.last_packet_time = now
                        if batch_records:
                            self.latest_sample = batch_records[-1]

                        self._hz_counter += n_batch
                        dt_hz = now - self._last_hz_time
                        if dt_hz >= 0.5:
                            self.estimated_hz = self._hz_counter / dt_hz
                            self._hz_counter = 0
                            self._last_hz_time = now

                        self.live_stream_queue.extend(batch_records)
                        # Keep live queue bounded
                        if len(self.live_stream_queue) > 400:
                            del self.live_stream_queue[:len(self.live_stream_queue) - 200]
                        if self.is_recording:
                            self.current_buffer.extend(batch_records)
            except Exception:
                pass

    def start_streaming(self):
        """Commands ESP32 to start streaming UDP packets ('0')"""
        with self.lock:
            self._flush_udp_socket()
            self.live_stream_queue.clear()
            self.is_streaming = True
        self.send_cmd(CMD_START_STREAMING)

    def stop_streaming(self):
        """Commands ESP32 to stop streaming UDP packets ('1')"""
        self.is_streaming = False
        self.send_cmd(CMD_STOP_STREAMING)

    def get_live_samples(self):
        with self.lock:
            samples = list(self.live_stream_queue)
            self.live_stream_queue.clear()
            return samples

    def send_cmd(self, cmd: str):
        with self.lock:
            if self.tcp_client and self.is_connected:
                try:
                    self.tcp_client.sendall(cmd.encode("ascii"))
                except Exception:
                    self.is_connected = False
                    self.tcp_client = None

    def _flush_udp_socket(self):
        self.udp_sock.setblocking(False)
        try:
            while True:
                self.udp_sock.recvfrom(2048)
        except BlockingIOError:
            pass
        except Exception:
            pass
        finally:
            self.udp_sock.setblocking(True)

    def start_recording(self):
        with self.lock:
            self._flush_udp_socket()
            self.current_buffer.clear()
            self.live_stream_queue.clear()
            self.is_recording = True
        self.start_streaming()

    def stop_recording(self, stop_esp_stream=True):
        if stop_esp_stream:
            self.stop_streaming()
        with self.lock:
            self.is_recording = False
            return list(self.current_buffer)

    def prepare_signal(self):
        self.send_cmd(CMD_PREPARE)

    def wait_for_esp(self):
        while not self.is_connected:
            time.sleep(0.3)


def calculate_gyro_magnitude(sample):
    _, cal_gyro = calibration.apply(sample)
    return float(math.sqrt(cal_gyro[0]**2 + cal_gyro[1]**2 + cal_gyro[2]**2))


def trim_gesture_motion(samples, gesture_name, threshold=GYRO_ENERGY_THRESHOLD):
    """
    Trims leading and trailing silence when gyroscope energy is near stationary.
    Returns (trimmed_samples, start_idx, end_idx).
    For 'idle', returns the full unaltered window.
    """
    if not samples:
        return samples, 0, 0

    if gesture_name.lower() == "idle" or not AUTO_TRIM_ENABLED:
        return samples, 0, len(samples)

    magnitudes = [calculate_gyro_magnitude(s) for s in samples]

    # Moving window smoothing (3 samples = 30 ms) to avoid cutting during momentary zero-crossings
    smoothed = []
    window_size = 3
    for i in range(len(magnitudes)):
        start_w = max(0, i - window_size // 2)
        end_w = min(len(magnitudes), i + window_size // 2 + 1)
        smoothed.append(sum(magnitudes[start_w:end_w]) / (end_w - start_w))

    active_indices = [idx for idx, val in enumerate(smoothed) if val >= threshold]

    if not active_indices:
        return samples, 0, len(samples)

    start_active = active_indices[0]
    end_active = active_indices[-1]

    start_idx = max(0, start_active - PADDING_BEFORE_SAMPLES)
    end_idx = min(len(samples), end_active + PADDING_AFTER_SAMPLES + 1)

    trimmed = samples[start_idx:end_idx]
    return trimmed, start_idx, end_idx


def save_sample_csv(gesture_name, samples, already_trimmed=False):
    if not samples:
        return None

    if not already_trimmed:
        samples, _, _ = trim_gesture_motion(samples, gesture_name)

    target_dir = os.path.join(DATASET_DIR, gesture_name)
    os.makedirs(target_dir, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    filename = f"{gesture_name}_{timestamp_str}.csv"
    filepath = os.path.join(target_dir, filename)

    fieldnames = [
        "timestamp_ms",
        "raw_accel_x", "raw_accel_y", "raw_accel_z",
        "raw_gyro_x", "raw_gyro_y", "raw_gyro_z",
        "cal_accel_x", "cal_accel_y", "cal_accel_z",
        "cal_gyro_x", "cal_gyro_y", "cal_gyro_z",
        "mag_x", "mag_y", "mag_z",
        "wand_id"
    ]

    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            cal_acc, cal_gyro = calibration.apply(s)
            writer.writerow({
                "timestamp_ms": s["timestamp_ms"],
                "raw_accel_x": s["accel_x"], "raw_accel_y": s["accel_y"], "raw_accel_z": s["accel_z"],
                "raw_gyro_x":  s["gyro_x"],  "raw_gyro_y":  s["gyro_y"],  "raw_gyro_z":  s["gyro_z"],
                "cal_accel_x": cal_acc[0],   "cal_accel_y": cal_acc[1],   "cal_accel_z": cal_acc[2],
                "cal_gyro_x":  cal_gyro[0],  "cal_gyro_y":  cal_gyro[1],  "cal_gyro_z":  cal_gyro[2],
                "mag_x":       s["mag_x"],    "mag_y":       s["mag_y"],    "mag_z":       s["mag_z"],
                "wand_id":     s["wand_id"]
            })

    return filepath


def get_dataset_summary():
    """Returns (dict_of_counts, total_count) for all gesture classes."""
    stats = {}
    total = 0
    if os.path.exists(DATASET_DIR):
        for g_key, (name, _) in GESTURES.items():
            g_path = os.path.join(DATASET_DIR, name)
            if os.path.isdir(g_path):
                cnt = len([f for f in os.listdir(g_path) if f.endswith(".csv")])
                stats[name] = cnt
                total += cnt
            else:
                stats[name] = 0
    return stats, total


# Speed thresholds for duration categorization: (fast_max, slow_min) in seconds
NOMINAL_SPEED_THRESHOLDS = {
    "circle":   (1.15, 1.65),
    "go_up":    (0.50, 0.80),
    "go_down":  (0.60, 0.95),
    "go_left":  (0.55, 0.85),
    "go_right": (0.55, 0.80),
    "iks":      (1.80, 2.50),
    "idle":     (3.00, 5.50),
}


def analyze_dataset_durations(dataset_dir=DATASET_DIR):
    """
    Parses CSV samples in dataset_dir and computes duration distributions (min, med, max, mean),
    fast/medium/slow split counts, and actionable recommendations detecting missing speed variations.
    """
    results = {}
    if not os.path.exists(dataset_dir):
        return results

    for g_key in ["1", "2", "3", "4", "5", "6", "7"]:
        name, label = GESTURES[g_key]
        g_path = os.path.join(dataset_dir, name)
        t_fast, t_slow = NOMINAL_SPEED_THRESHOLDS.get(name, (1.0, 1.5))
        durs = []

        if os.path.isdir(g_path):
            for fname in os.listdir(g_path):
                if fname.endswith(".csv"):
                    fpath = os.path.join(g_path, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8") as fp:
                            _ = fp.readline()  # header
                            first_line = fp.readline()
                            if not first_line:
                                continue
                            last_line = first_line
                            for line in fp:
                                if line.strip():
                                    last_line = line
                            t0 = float(first_line.split(",")[0])
                            t1 = float(last_line.split(",")[0])
                            durs.append(max(0.01, (t1 - t0) / 1000.0))
                    except Exception:
                        pass

        n = len(durs)
        if n > 0:
            durs.sort()
            d_min = durs[0]
            d_max = durs[-1]
            d_med = durs[n // 2]
            d_mean = sum(durs) / n
            n_fast = sum(1 for d in durs if d <= t_fast)
            n_slow = sum(1 for d in durs if d >= t_slow)
            n_med = n - n_fast - n_slow

            if name == "idle":
                rec_tag = "BALANCED"
                rec_str = "Resting Baseline"
                advice = "Stationary background reference"
            elif n < 5:
                rec_tag = "FEW_SAMPLES"
                rec_str = f"Few samples ({n})"
                advice = f"Few samples ({n}/15 recommend) - collect more repetitions"
            else:
                min_variety = max(2, int(0.20 * n))
                if n_fast < min_variety and n_slow < min_variety:
                    if n_fast <= n_slow:
                        rec_tag = "NEED_FAST"
                        rec_str = f"Need Fast (<{t_fast:.2f}s)"
                        advice = f"Lacks fast variation (<{t_fast:.2f}s) - perform brisk, snappy strokes"
                    else:
                        rec_tag = "NEED_SLOW"
                        rec_str = f"Need Slow (>{t_slow:.2f}s)"
                        advice = f"Lacks slow variation (>{t_slow:.2f}s) - perform smooth, deliberate strokes"
                elif n_fast < min_variety:
                    rec_tag = "NEED_FAST"
                    rec_str = f"Need Fast (<{t_fast:.2f}s)"
                    advice = f"Lacks fast variation (<{t_fast:.2f}s) - perform brisk, snappy strokes"
                elif n_slow < min_variety:
                    rec_tag = "NEED_SLOW"
                    rec_str = f"Need Slow (>{t_slow:.2f}s)"
                    advice = f"Lacks slow variation (>{t_slow:.2f}s) - perform smooth, deliberate strokes"
                else:
                    rec_tag = "BALANCED"
                    rec_str = "Well balanced"
                    advice = "Well balanced across fast, normal, and slow speeds"
        else:
            d_min = d_max = d_med = d_mean = 0.0
            n_fast = n_med = n_slow = 0
            rec_tag = "NO_SAMPLES"
            rec_str = "No samples"
            advice = "No samples recorded yet - begin recording baseline"

        results[name] = {
            "name": name,
            "label": label,
            "durations": durs,
            "n": n,
            "min": d_min,
            "med": d_med,
            "median": d_med,
            "max": d_max,
            "mean": d_mean,
            "t_fast": t_fast,
            "t_slow": t_slow,
            "fast": n_fast,
            "normal": n_med,
            "med_count": n_med,
            "slow": n_slow,
            "rec_tag": rec_tag,
            "rec_str": rec_str,
            "advice": advice,
        }

    return results


# =========================================================================
# SUB-MODES: SENSOR CALIBRATION (IN-PLACE)
# =========================================================================
def mode_calibration(server: WandDataServer):
    """In-place sensor calibration flow without screen scrolling."""
    ClaudeConsole.clear()
    ClaudeConsole.hide_cursor()

    lines = [
        f"{C_CYAN}╔══════════════════════════════════════════════════════════════════════════╗{C_RESET}",
        f"{C_CYAN}║  {C_BOLD}{C_WHITE}SENSOR CALIBRATION (GYROSCOPE BIAS & ACCELEROMETER ZERO){C_RESET}{C_CYAN}                ║{C_RESET}",
        f"{C_CYAN}╚══════════════════════════════════════════════════════════════════════════╝{C_RESET}",
        "",
        f"  1. Place the wand {C_BOLD}completely still{C_RESET} on a flat, stable surface.",
        f"  2. Resting telemetry samples will be captured to eliminate drift.",
        "",
        f"  Press {C_GREEN}[SPACE / ENTER]{C_RESET} to begin calibration, or {C_RED}[ESC]{C_RESET} to cancel..."
    ]
    ClaudeConsole.render_frame(lines)

    while True:
        k = ClaudeConsole.get_key()
        if k in ("ESC", "q", "Q"):
            ClaudeConsole.show_cursor()
            ClaudeConsole.clear()
            return
        elif k in ("ENTER", " ", "space"):
            break

    server.wait_for_esp()
    server.prepare_signal()
    time.sleep(0.3)

    server.start_recording()
    beep(800, 200)

    total_sec = 3.0
    start_t = time.time()
    spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    tick = 0

    while True:
        elapsed = time.time() - start_t
        rem = max(0.0, total_sec - elapsed)
        progress = min(1.0, elapsed / total_sec)
        tick += 1
        spin = spinners[tick % len(spinners)]

        bar_len = int(progress * 30)
        bar_str = f"{C_GREEN}{'▰'*bar_len}{C_DIM}{'▱'*(30-bar_len)}{C_RESET}"

        with server.lock:
            n_samples = len(server.current_buffer)

        lines = [
            f"{C_CYAN}╔══════════════════════════════════════════════════════════════════════════╗{C_RESET}",
            f"{C_CYAN}║  {C_BOLD}{C_WHITE}COLLECTING CALIBRATION SAMPLES...{C_RESET}{C_CYAN}                                       ║{C_RESET}",
            f"{C_CYAN}╚══════════════════════════════════════════════════════════════════════════╝{C_RESET}",
            "",
            f"  {spin} Progress: [{bar_str}] {rem:3.1f}s remaining",
            f"  Samples collected: {C_BOLD}{n_samples}{C_RESET} (Keep wand motionless!)",
            "",
            f"  {C_DIM}Do not touch or move the wand...{C_RESET}"
        ]
        ClaudeConsole.render_frame(lines)

        if elapsed >= total_sec:
            break
        time.sleep(0.05)

    samples = server.stop_recording()
    beep(1200, 300)

    if len(samples) < 50:
        lines = [
            f"{C_RED}[ERROR] Insufficient samples collected ({len(samples)}). Check wand connection.{C_RESET}",
            "",
            "Press any key to return to menu..."
        ]
        ClaudeConsole.render_frame(lines)
        ClaudeConsole.get_key()
        ClaudeConsole.show_cursor()
        ClaudeConsole.clear()
        return

    success = calibration.calibrate(samples)
    if success:
        gb = calibration.gyro_bias
        lines = [
            f"{C_GREEN}╔══════════════════════════════════════════════════════════════════════════╗{C_RESET}",
            f"{C_GREEN}║  {C_BOLD}{C_WHITE}✓ SENSOR CALIBRATION COMPLETED SUCCESSFULLY{C_RESET}{C_GREEN}                             ║{C_RESET}",
            f"{C_GREEN}╚══════════════════════════════════════════════════════════════════════════╝{C_RESET}",
            "",
            f" {C_BOLD}Calibrated Biases (eliminated drift):{C_RESET}",
            f"  Gyroscope Bias:    X: {gb[0]:+7.4f} rad/s   Y: {gb[1]:+7.4f} rad/s   Z: {gb[2]:+7.4f} rad/s",
            f"  Gravity Scale:     {calibration.accel_scale:.4f}",
            "",
            f"  {C_GREEN}✓ New calibration parameters saved and active across all apps!{C_RESET}",
            "",
            f"  Press {C_CYAN}any key{C_RESET} to return to main menu..."
        ]
        ClaudeConsole.render_frame(lines)
        ClaudeConsole.get_key()

    ClaudeConsole.show_cursor()
    ClaudeConsole.clear()


# =========================================================================
# SUB-MODES: LIVE GYROSCOPE DYNAMICS & THRESHOLD TUNING (IN-PLACE)
# =========================================================================
def mode_live_monitor(server: WandDataServer):
    """
    Claude Code style in-place live telemetry HUD.
    Renders dynamic angular velocity, instantaneous sensor vectors, and threshold tuning.
    """
    global GYRO_ENERGY_THRESHOLD, AUTO_TRIM_ENABLED
    server.wait_for_esp()
    server.start_recording()
    ClaudeConsole.clear()
    ClaudeConsole.hide_cursor()

    peak_omega = 0.0
    tick = 0
    spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    try:
        while True:
            time.sleep(0.04)
            tick += 1
            spin = spinners[tick % len(spinners)]

            # Check key non-blockingly
            k = ClaudeConsole.poll_key()
            if k in ("q", "Q", "ESC", "0"):
                break
            elif k in ("+", "="):
                GYRO_ENERGY_THRESHOLD = round(min(2.0, GYRO_ENERGY_THRESHOLD + 0.05), 2)
            elif k in ("-", "_"):
                GYRO_ENERGY_THRESHOLD = round(max(0.05, GYRO_ENERGY_THRESHOLD - 0.05), 2)
            elif k in ("t", "T"):
                AUTO_TRIM_ENABLED = not AUTO_TRIM_ENABLED
            elif k in ("r", "R"):
                peak_omega = 0.0

            with server.lock:
                s = server.latest_sample
                total_pkts = server.total_packets
                hz = server.estimated_hz if (time.time() - server.last_packet_time < 1.0) else 0.0
                ip = server.tcp_addr[0] if server.tcp_addr else "192.168.0.x"

            if s:
                cal_acc, cal_gyro = calibration.apply(s)
                mag = float(math.sqrt(cal_gyro[0]**2 + cal_gyro[1]**2 + cal_gyro[2]**2))
                gx, gy, gz = cal_gyro
                ax, ay, az = cal_acc
            else:
                mag = 0.0
                gx = gy = gz = 0.0
                ax = ay = az = 0.0

            if mag > peak_omega:
                peak_omega = mag

            # Colored gauge with threshold marker
            bar_len = min(32, int((mag / 4.0) * 32))
            thresh_pos = min(32, int((GYRO_ENERGY_THRESHOLD / 4.0) * 32))

            gauge_chars = []
            for idx in range(32):
                if idx == thresh_pos:
                    gauge_chars.append(f"{C_WHITE}|{C_RESET}")
                elif idx < bar_len:
                    if mag >= GYRO_ENERGY_THRESHOLD:
                        gauge_chars.append(f"{C_RED}▰{C_RESET}")
                    elif mag >= 0.20:
                        gauge_chars.append(f"{C_AMBER}▰{C_RESET}")
                    else:
                        gauge_chars.append(f"{C_CYAN}▰{C_RESET}")
                else:
                    gauge_chars.append(f"{C_DIM}▱{C_RESET}")
            gauge_str = "".join(gauge_chars)

            if mag >= GYRO_ENERGY_THRESHOLD:
                state_badge = f"{C_RED}[● MOTION DETECTED]{C_RESET}"
            elif mag >= 0.20:
                state_badge = f"{C_AMBER}[● ARMED / APPROACH]{C_RESET}"
            else:
                state_badge = f"{C_CYAN}[○ RESTING (STATIONARY)]{C_RESET}"

            trim_badge = f"{C_GREEN}[ENABLED]{C_RESET}" if AUTO_TRIM_ENABLED else f"{C_DIM}[DISABLED]{C_RESET}"

            lines = [
                f"{C_CYAN}╔══════════════════════════════════════════════════════════════════════════╗{C_RESET}",
                f"{C_CYAN}║  {C_BOLD}{C_WHITE}WANDSENSE IMU  •  LIVE TELEMETRY & SILENCE GATE MONITOR{C_RESET}{C_CYAN}                 ║{C_RESET}",
                f"{C_CYAN}╚══════════════════════════════════════════════════════════════════════════╝{C_RESET}",
                f"  Link: {C_GREEN}[● ONLINE]{C_RESET} {spin} {ip} • {hz:4.0f} Hz • {total_pkts:,} pkts",
                "",
                f" {C_BOLD}╭── KINETIC ENERGY & ANGULAR RATE (||ω||) ────────────────────────────────╮{C_RESET}",
                f" │  Dynamics State:     {state_badge}",
                f" │  Instantaneous ||ω||: {C_BOLD}{mag:5.2f} rad/s{C_RESET}  (Peak: {C_AMBER}{peak_omega:5.2f}{C_RESET} rad/s)",
                f" │  Threshold Gauge:    [{gauge_str}]",
                f" │  Silence Gate:       {C_BOLD}{GYRO_ENERGY_THRESHOLD:.2f} rad/s{C_RESET} {trim_badge}  (White bar '|' = cut threshold)",
                f" {C_BOLD}╰────────────────────────────────────────────────────────────────────────╯{C_RESET}",
                "",
                f" {C_BOLD}╭── CALIBRATED SENSOR VECTORS ───────────────────────────────────────────╮{C_RESET}",
                f" │  Gyroscope (rad/s):   X: {gx:+6.2f}   Y: {gy:+6.2f}   Z: {gz:+6.2f}              │",
                f" │  Accelerometer (g):   X: {ax:+6.2f}   Y: {ay:+6.2f}   Z: {az:+6.2f}              │",
                f" {C_BOLD}╰────────────────────────────────────────────────────────────────────────╯{C_RESET}",
                "",
                f" {C_DIM}Shortcuts:{C_RESET} {C_CYAN}[+] / [-]{C_RESET} Threshold  •  {C_CYAN}[T]{C_RESET} Toggle Gate  •  {C_CYAN}[R]{C_RESET} Reset Peak  •  {C_CYAN}[Q / ESC]{C_RESET} Return"
            ]
            ClaudeConsole.render_frame(lines)

    except KeyboardInterrupt:
        pass
    finally:
        server.stop_recording()
        ClaudeConsole.show_cursor()
        ClaudeConsole.clear()


# =========================================================================
# SUB-MODES: ADJUST THRESHOLD
# =========================================================================
def mode_adjust_threshold():
    global GYRO_ENERGY_THRESHOLD, AUTO_TRIM_ENABLED
    ClaudeConsole.clear()
    ClaudeConsole.show_cursor()

    trim_status = f"{C_GREEN}ENABLED{C_RESET}" if AUTO_TRIM_ENABLED else f"{C_DIM}DISABLED{C_RESET}"
    lines = [
        f"{C_CYAN}╔══════════════════════════════════════════════════════════════════════════╗{C_RESET}",
        f"{C_CYAN}║  {C_BOLD}{C_WHITE}ADJUST SILENCE AUTO-TRIM THRESHOLD{C_RESET}{C_CYAN}                                      ║{C_RESET}",
        f"{C_CYAN}╚══════════════════════════════════════════════════════════════════════════╝{C_RESET}",
        f"  Current Threshold: {C_BOLD}{GYRO_ENERGY_THRESHOLD:.2f} rad/s{C_RESET}",
        f"  Current Status:    {trim_status}",
        "",
        "  Options:",
        "   - Enter a new numeric threshold (e.g. 0.35, 0.40, 0.50)",
        "   - Type 'toggle' or 't' to enable/disable auto-trim",
        "   - Press ENTER without typing to keep current",
        ""
    ]
    sys.stdout.write("".join(f"{line}\n" for line in lines))
    sys.stdout.write(f" {C_BOLD}New threshold / command: {C_RESET}")
    sys.stdout.flush()

    cmd = input().strip()
    if cmd.lower() in ("toggle", "t"):
        AUTO_TRIM_ENABLED = not AUTO_TRIM_ENABLED
    elif cmd:
        try:
            GYRO_ENERGY_THRESHOLD = float(cmd)
        except ValueError:
            pass

    ClaudeConsole.hide_cursor()
    ClaudeConsole.clear()


# =========================================================================
# SUB-MODES: ADJUST TELEMETRY UDP BATCH SIZE (DYNAMIC RUNTIME CONFIG)
# =========================================================================
def mode_adjust_batch_size(server: WandDataServer):
    """
    Interactive console HUD to change ESP32 UDP packet batching dynamically at runtime.
    Sends command B<size>\n to ESP32 without requiring firmware reflash or code reload.
    """
    ClaudeConsole.clear()
    ClaudeConsole.show_cursor()

    latency_ms = server.batch_size * 10
    rate_pkts = 100.0 / server.batch_size

    lines = [
        f"{C_CYAN}╔══════════════════════════════════════════════════════════════════════════╗{C_RESET}",
        f"{C_CYAN}║  {C_BOLD}{C_WHITE}ESP32 TELEMETRY BATCH SIZE & LATENCY TUNER{C_RESET}{C_CYAN}                              ║{C_RESET}",
        f"{C_CYAN}╚══════════════════════════════════════════════════════════════════════════╝{C_RESET}",
        f"  Current ESP32 Batch Size: {C_BOLD}{C_GREEN}{server.batch_size}{C_RESET} samples/packet",
        f"  Transmission Latency:     ~{latency_ms} ms per packet",
        f"  Packet Rate:              ~{rate_pkts:.1f} packets/second (at 100 Hz sensor clock)",
        "",
        f" {C_BOLD}Preset Latency Profiles:{C_RESET}",
        f"  [{C_GREEN}1{C_RESET}]  Ultra-Low Latency   :  1 sample/pkt  (~10 ms latency  | 100 pkts/s) - Instant twitch",
        f"  [{C_GREEN}2{C_RESET}]  Fast Response       :  2 samples/pkt (~20 ms latency  |  50 pkts/s)",
        f"  [{C_GREEN}3{C_RESET}]  Real-Time Balanced  :  3 samples/pkt (~30 ms latency  |  33 pkts/s) ★ (Optimal balance)",
        f"  [{C_GREEN}5{C_RESET}]  Standard Balanced   :  5 samples/pkt (~50 ms latency  |  20 pkts/s)",
        f"  [{C_GREEN}10{C_RESET}] Bandwidth Saver     : 10 samples/pkt (~100 ms latency |  10 pkts/s) - Default",
        f"  [{C_GREEN}20{C_RESET}] High Batching       : 20 samples/pkt (~200 ms latency |   5 pkts/s)",
        "",
        f" {C_DIM}Enter desired batch size (1-32) or press ENTER to keep {server.batch_size}:{C_RESET}"
    ]
    sys.stdout.write("".join(f"{line}\n" for line in lines))
    sys.stdout.write(f" {C_BOLD}New Batch Size [1-32]: {C_RESET}")
    sys.stdout.flush()

    raw = input().strip()
    if raw:
        try:
            val = int(raw)
            if 1 <= val <= 32:
                ok = server.set_batch_size(val)
                new_lat = val * 10
                new_rate = 100.0 / val
                if ok:
                    print(f"\n  {C_GREEN}✓ Command sent to ESP32! Batch size dynamically set to {val} ({new_lat}ms | ~{new_rate:.1f} pkts/s).{C_RESET}")
                else:
                    print(f"\n  {C_AMBER}✓ Batch size set to {val}. Will synchronize automatically as soon as ESP32 connects.{C_RESET}")
            else:
                print(f"\n  {C_RED}⚠ Invalid value. Batch size must be between 1 and 32.{C_RESET}")
        except ValueError:
            print(f"\n  {C_RED}⚠ Invalid input. Keeping {server.batch_size}.{C_RESET}")
    else:
        print(f"\n  {C_DIM}No change. Preserved {server.batch_size}.{C_RESET}")

    time.sleep(1.2)
    ClaudeConsole.hide_cursor()
    ClaudeConsole.clear()


# =========================================================================
# SUB-MODES: GESTURE DURATION & SPEED DIVERSITY ANALYSIS
# =========================================================================
def mode_duration_analysis(server: WandDataServer = None):
    """
    Claude Code style in-place duration distribution & speed diversity analyzer.
    Displays:
      - Breakdown table with min/med/max durations and fast/medium/slow split.
      - 12-char proportional speed bars.
      - Actionable gap suggestions (e.g. '⚡ Lacks FAST strokes (<1.15s)').
      - Quick jump into Infinite Mode [1] to immediately record missing variations.
    """
    ClaudeConsole.clear()
    ClaudeConsole.hide_cursor()
    block = "■"

    try:
        while True:
            analysis = analyze_dataset_durations(DATASET_DIR)
            stats_dict, total_samples = get_dataset_summary()
            box_w = 78

            lines = [
                f"{C_PURPLE}╔" + "═" * (box_w - 2) + f"╗{C_RESET}",
                f"{C_PURPLE}║  {C_BOLD}{C_WHITE}GESTURE DURATION & SPEED DIVERSITY ANALYSIS (FAST vs SLOW GAPS){C_RESET}{C_PURPLE}" + " " * (box_w - 68) + f"║{C_RESET}",
                f"{C_PURPLE}╚" + "═" * (box_w - 2) + f"╝{C_RESET}",
                f"  Dataset contains {C_BOLD}{total_samples}{C_RESET} samples across 7 classes (analyzing temporal variation)",
                "",
                f" {C_BOLD}CLASS / GESTURE    N    MIN / MED / MAX (s)    SPLIT      BAR      DIAGNOSTIC ADVICE{C_RESET}",
                " " + "─" * (box_w - 2),
            ]

            need_fast_list = []
            need_slow_list = []

            for k in ["1", "2", "3", "4", "5", "6", "7"]:
                gname, glbl = GESTURES[k]
                info = analysis.get(gname, {})
                n = info.get("n", 0)
                tf = info.get("t_fast", 1.0)
                ts = info.get("t_slow", 1.5)
                rec_tag = info.get("rec_tag", "NO_SAMPLES")

                if n > 0:
                    nf = info.get("fast", 0)
                    nm = info.get("normal", info.get("med_count", 0))
                    ns = info.get("slow", 0)
                    min_v = info.get("min", 0.0)
                    med_v = info.get("med", 0.0)
                    max_v = info.get("max", 0.0)

                    # 8-char mini bar
                    b_f = int(round((nf / n) * 8)) if n > 0 else 0
                    b_s = int(round((ns / n) * 8)) if n > 0 else 0
                    b_m = max(0, 8 - b_f - b_s)
                    bar = f"{C_AMBER}{block * b_f}{C_GREEN}{block * b_m}{C_CYAN}{block * b_s}{C_RESET}"

                    split_str = f"{C_AMBER}{nf:2d}F{C_RESET} {C_GREEN}{nm:2d}M{C_RESET} {C_CYAN}{ns:2d}S{C_RESET}"
                    stat_str = f"{min_v:4.2f}/{med_v:4.2f}/{max_v:4.2f}"

                    if gname == "idle":
                        advice_str = f"{C_GREEN}✓ Resting baseline{C_RESET}"
                    elif rec_tag == "NEED_FAST":
                        advice_str = f"{C_AMBER}⚡ Need Fast (<{tf:.2f}s){C_RESET}"
                        need_fast_list.append(f"{glbl} (<{tf:.2f}s)")
                    elif rec_tag == "NEED_SLOW":
                        advice_str = f"{C_CYAN}🐢 Need Slow (>{ts:.2f}s){C_RESET}"
                        need_slow_list.append(f"{glbl} (>{ts:.2f}s)")
                    elif rec_tag == "FEW_SAMPLES":
                        advice_str = f"{C_AMBER}⚠️ Few samples ({n}/15){C_RESET}"
                    else:
                        advice_str = f"{C_GREEN}✓ Balanced{C_RESET}"
                else:
                    bar = f"{C_DIM}--------{C_RESET}"
                    split_str = f"{C_DIM} 0F  0M  0S{C_RESET}"
                    stat_str = f"{C_DIM}  --/  --/  --{C_RESET}"
                    advice_str = f"{C_RED}No samples recorded{C_RESET}"

                lines.append(f" [{k}] {glbl:<12} {n:2d}   {stat_str}   {split_str}  [{bar}]  {advice_str}")

            lines.extend([
                " " + "─" * (box_w - 2),
                f"  {C_BOLD}Legend:{C_RESET} [{C_AMBER}■ Fast{C_RESET} | {C_GREEN}■ Normal{C_RESET} | {C_CYAN}■ Slow{C_RESET}]",
                "",
                f" {C_BOLD}╭── ACTIONABLE RECORDING RECOMMENDATIONS " + "─" * (box_w - 42) + f"{C_RESET}",
            ])

            if need_fast_list:
                lines.append(f" │  {C_AMBER}⚡ FAST MISSING:{C_RESET} {', '.join(need_fast_list)}")
            if need_slow_list:
                lines.append(f" │  {C_CYAN}🐢 SLOW MISSING:{C_RESET} {', '.join(need_slow_list)}")
            if not need_fast_list and not need_slow_list:
                lines.append(f" │  {C_GREEN}✓ All gesture classes currently exhibit balanced temporal diversity!{C_RESET}")

            lines.extend([
                f" │  {C_DIM}💡 Tip: Launch Infinite Mode [1] - live guidance prompts missing speeds.{C_RESET}",
                f" {C_BOLD}╰" + "─" * (box_w - 2) + f"{C_RESET}",
                "",
                f"  {C_PURPLE}[1 / I]{C_RESET} Launch Infinite Mode with Live Speed Guidance",
                f"  {C_CYAN}[R]{C_RESET}     Refresh Dataset Duration Analysis",
                f"  {C_DIM}[ESC/Q] Return to Main Menu{C_RESET}",
                "",
                f" {C_BOLD}Select Action:{C_RESET} "
            ])

            ClaudeConsole.render_frame(lines)

            k = ClaudeConsole.get_key()
            if k in ("1", "i", "I"):
                if server is not None:
                    mode_infinite(server)
                    ClaudeConsole.clear()
                    continue
                else:
                    break
            elif k in ("r", "R"):
                continue
            elif k in ("ESC", "q", "Q", "0"):
                break

    finally:
        ClaudeConsole.show_cursor()
        ClaudeConsole.clear()


# =========================================================================
# SUB-MODES: GESTURE SELECTION & RECORDING
# =========================================================================
def mode_infinite(server: WandDataServer):
    """
    ★ Infinite / Continuous Gesture Acquisition Mode with Live Dataset Balancer.
    Displays counts of ALL gestures, real-time distribution bars, and allows
    hotkey-switching between all classes directly in Pygame without leaving the session.
    """
    stats_dict, total_samples = get_dataset_summary()
    speed_analysis = analyze_dataset_durations(DATASET_DIR)
    ClaudeConsole.clear()
    ClaudeConsole.hide_cursor()

    # Pre-flight Dataset Balancer Table
    max_c = max(max(stats_dict.values(), default=1), 20)
    box_w = 76
    top_b = f" {C_BOLD}╭── CURRENT DATASET BALANCE & SPEED GAPS " + "─" * (box_w - 43) + f"╮{C_RESET}"
    bot_b = f" {C_BOLD}╰" + "─" * (box_w - 2) + f"╯{C_RESET}"

    lines = [
        f"{C_PURPLE}╔" + "═" * (box_w - 2) + f"╗{C_RESET}",
        f"{C_PURPLE}║  {C_BOLD}{C_WHITE}★ INFINITE GESTURE ACQUISITION MODE  •  LIVE BALANCER & SPEED ADVISOR{C_RESET}{C_PURPLE}" + " " * (box_w - 73) + f"║{C_RESET}",
        f"{C_PURPLE}╚" + "═" * (box_w - 2) + f"╝{C_RESET}",
        f"  Total recorded so far: {C_BOLD}{total_samples}{C_RESET} samples across 7 gesture classes",
        "",
        top_b,
    ]

    for k in ["1", "2", "3", "4", "5", "6", "7"]:
        name, label = GESTURES[k]
        cnt = stats_dict.get(name, 0)
        pct = (cnt / total_samples * 100.0) if total_samples > 0 else 0.0
        bar_len = min(10, int((cnt / max_c) * 10))
        bar_color = C_GREEN if cnt >= 30 else (C_AMBER if cnt >= 15 else C_RED)
        bar_str = f"{bar_color}{'▰'*bar_len}{C_DIM}{'▱'*(10-bar_len)}{C_RESET}"

        s_info = speed_analysis.get(name, {})
        rec_tag = s_info.get("rec_tag", "BALANCED")
        rec_str = s_info.get("rec_str", "Balanced")
        if rec_tag == "NEED_FAST":
            tag_str = f"{C_AMBER}⚡ {rec_str}{C_RESET}"
        elif rec_tag == "NEED_SLOW":
            tag_str = f"{C_CYAN}🐢 {rec_str}{C_RESET}"
        elif rec_tag == "NO_SAMPLES":
            tag_str = f"{C_DIM}(No samples){C_RESET}"
        elif rec_tag == "FEW_SAMPLES":
            tag_str = f"{C_AMBER}⚠️  {rec_str}{C_RESET}"
        else:
            tag_str = f"{C_GREEN}✓ {rec_str}{C_RESET}"

        row_content = f"{C_CYAN}[{k}]{C_RESET} {label:<11} : {cnt:2d} [{bar_str}] {pct:4.1f}% │ {tag_str}"
        lines.append(ClaudeConsole.box_row(row_content, box_w))

    # Find classes with lowest samples to recommend
    min_count = min(stats_dict.values()) if stats_dict else 0
    under_represented = [GESTURES[k][1] for k in ["1", "2", "3", "4", "5", "6", "7"] if stats_dict.get(GESTURES[k][0], 0) == min_count]
    rec_str = ", ".join(under_represented[:2])

    lines.extend([
        bot_b,
        f"  {C_AMBER}💡 Recommendation:{C_RESET} Prioritize {C_BOLD}{rec_str}{C_RESET} to balance dataset!",
        "",
        f" {C_BOLD}Controls in 3D Pygame Session:{C_RESET}",
        f"  - {C_CYAN}[1 - 7]{C_RESET}       : Switch active target gesture class instantly",
        f"  - {C_GREEN}[SPACE/ENTER]{C_RESET} : Accept sample & save to dataset (+1)",
        f"  - {C_RED}[D/BACKSPACE]{C_RESET} : Discard sample and retry",
        f"  - {C_DIM}[Q / ESC]{C_RESET}     : Finish infinite session & view report",
        "",
        f"  Press {C_GREEN}[SPACE / ENTER]{C_RESET} to launch 3D window, or {C_RED}[ESC]{C_RESET} to cancel..."
    ])

    ClaudeConsole.render_frame(lines)

    while True:
        k = ClaudeConsole.get_key()
        if k in ("ESC", "q", "Q"):
            ClaudeConsole.show_cursor()
            ClaudeConsole.clear()
            return
        elif k in ("ENTER", " ", "space"):
            break

    server.wait_for_esp()
    ClaudeConsole.clear()
    print(f"\n{C_PURPLE}[INFO] Launching Infinite 6-DOF Session with Live Balancer HUD...{C_RESET}")

    ui = GestureRecordingSessionUI(server, calibration, beep_fn=beep)
    ui.dataset_counts = stats_dict.copy()
    ui.speed_analysis = speed_analysis

    # Find matching initial gesture name
    matched_name = "circle"
    for _, (gname, glbl) in GESTURES.items():
        if glbl in under_represented:
            matched_name = gname
            break

    try:
        session_saved = ui.run_infinite_session(
            initial_gesture=matched_name,
            duration_sec=None,  # Dynamically adapts to active gesture (e.g. 3.2s for X, 5.0s for Idle, 2.2s for Circle)
            trim_fn=trim_gesture_motion,
            save_fn=save_sample_csv,
            speed_analysis=speed_analysis
        )
    finally:
        ui.close()

    # Post-session summary
    stats_after, total_after = get_dataset_summary()
    speed_after = analyze_dataset_durations(DATASET_DIR)
    ClaudeConsole.clear()
    ClaudeConsole.hide_cursor()

    post_top = f" {C_BOLD}╭── UPDATED DATASET BALANCE & SPEED STATUS " + "─" * (box_w - 45) + f"╮{C_RESET}"
    lines = [
        f"{C_GREEN}╔" + "═" * (box_w - 2) + f"╗{C_RESET}",
        f"{C_GREEN}║  {C_BOLD}{C_WHITE}✓ INFINITE RECORDING SESSION COMPLETED{C_RESET}{C_GREEN}" + " " * (box_w - 42) + f"║{C_RESET}",
        f"{C_GREEN}╚" + "═" * (box_w - 2) + f"╝{C_RESET}",
        f"  Recorded this session: {C_BOLD}{C_GREEN}+{session_saved} samples{C_RESET} (New Total: {total_after})",
        "",
        post_top,
    ]
    max_c_after = max(max(stats_after.values(), default=1), 20)
    for k in ["1", "2", "3", "4", "5", "6", "7"]:
        name, label = GESTURES[k]
        cnt = stats_after.get(name, 0)
        diff = cnt - stats_dict.get(name, 0)
        diff_str = f"({C_GREEN}+{diff}{C_RESET})" if diff > 0 else "    "
        pct = (cnt / total_after * 100.0) if total_after > 0 else 0.0
        bar_len = min(10, int((cnt / max_c_after) * 10))
        bar_color = C_GREEN if cnt >= 30 else (C_AMBER if cnt >= 15 else C_RED)
        bar_str = f"{bar_color}{'▰'*bar_len}{C_DIM}{'▱'*(10-bar_len)}{C_RESET}"

        s_info = speed_after.get(name, {})
        rec_tag = s_info.get("rec_tag", "BALANCED")
        rec_str = s_info.get("rec_str", "Balanced")
        if rec_tag == "NEED_FAST":
            tag_str = f"{C_AMBER}⚡ {rec_str}{C_RESET}"
        elif rec_tag == "NEED_SLOW":
            tag_str = f"{C_CYAN}🐢 {rec_str}{C_RESET}"
        elif rec_tag == "NO_SAMPLES":
            tag_str = f"{C_DIM}(No samples){C_RESET}"
        elif rec_tag == "FEW_SAMPLES":
            tag_str = f"{C_AMBER}⚠️  {rec_str}{C_RESET}"
        else:
            tag_str = f"{C_GREEN}✓ {rec_str}{C_RESET}"

        post_row = f"{C_CYAN}[{k}]{C_RESET} {label:<11} : {cnt:2d} {diff_str} [{bar_str}] {pct:4.1f}% │ {tag_str}"
        lines.append(ClaudeConsole.box_row(post_row, box_w))

    lines.extend([
        bot_b,
        "",
        f"  Press {C_CYAN}any key{C_RESET} to return to main menu..."
    ])

    ClaudeConsole.render_frame(lines)
    ClaudeConsole.get_key()
    ClaudeConsole.clear()


def mode_select_gesture(title="SELECT GESTURE TO RECORD"):
    ClaudeConsole.clear()
    ClaudeConsole.hide_cursor()

    lines = [
        f"{C_CYAN}╔══════════════════════════════════════════════════════════════════════════╗{C_RESET}",
        f"{C_CYAN}║  {C_BOLD}{C_WHITE}{title:<72}{C_RESET}{C_CYAN}║{C_RESET}",
        f"{C_CYAN}╚══════════════════════════════════════════════════════════════════════════╝{C_RESET}",
        "",
        f" {C_BOLD}╭── AVAILABLE GESTURE CLASSES ───────────────────────────────────────────╮{C_RESET}",
        f" │  {C_GREEN}[1]{C_RESET} Circle (circle)              {C_GREEN}[5]{C_RESET} Swipe Right (go_right)            │",
        f" │  {C_GREEN}[2]{C_RESET} Swipe Up (go_up)             {C_GREEN}[6]{C_RESET} Cross / X-Mark (iks)              │",
        f" │  {C_GREEN}[3]{C_RESET} Swipe Down (go_down)         {C_GREEN}[7]{C_RESET} Rest / Idle Noise (idle)          │",
        f" │  {C_GREEN}[4]{C_RESET} Swipe Left (go_left)         {C_RED}[ESC]{C_RESET} Cancel / Return to Menu         │",
        f" {C_BOLD}╰────────────────────────────────────────────────────────────────────────╯{C_RESET}",
        "",
        f"  Select gesture {C_BOLD}[1-7]{C_RESET}: "
    ]
    ClaudeConsole.render_frame(lines)

    while True:
        k = ClaudeConsole.get_key()
        if k in ("ESC", "q", "Q", "0"):
            ClaudeConsole.show_cursor()
            ClaudeConsole.clear()
            return None, None
        if k in GESTURES:
            name, label = GESTURES[k]
            ClaudeConsole.show_cursor()
            ClaudeConsole.clear()
            return name, label


def mode_paced_series(server: WandDataServer):
    """Paced recording series with continuous 3D preview and verification in Pygame."""
    gesture_name, gesture_label = mode_select_gesture("01. PACED RECORDING SERIES")
    if not gesture_name:
        return

    ClaudeConsole.show_cursor()
    print(f"\n{C_BOLD}Configuring series for [{gesture_label}] ({gesture_name}):{C_RESET}")
    default_dur = DEFAULT_GESTURE_CAPTURE_DURATIONS.get(gesture_name, 1.8)
    reps_str = input(f" Number of repetitions [default 10]: ").strip()
    dur_str = input(f" Recording window in seconds [default {default_dur}]: ").strip()

    count = int(reps_str) if reps_str.isdigit() else 10
    duration = float(dur_str) if dur_str.replace('.', '', 1).isdigit() else default_dur

    server.wait_for_esp()
    ClaudeConsole.clear()
    print(f"\n{C_CYAN}[INFO] Launching 6-DOF Pygame window for {count} trials of [{gesture_label}]...{C_RESET}")

    ui = GestureRecordingSessionUI(server, calibration, beep_fn=beep)
    ui.dataset_counts = get_dataset_summary()[0]
    ui.current_gesture_name = gesture_name
    ui.current_gesture_label = gesture_label
    saved_count = 0
    try:
        i = 0
        while i < count:
            server.wait_for_esp()
            current_num = i + 1

            # Phase 1: Countdown in Pygame with live 3D wand tracking
            ok = ui.run_countdown(gesture_label, current_num, count, countdown_sec=2.0)
            if not ok:
                break

            # Phase 2: Live recording
            raw_samples, aborted = ui.run_live_capture(
                gesture_label, current_num, count, duration_sec=duration, manual_stop=False
            )
            if aborted:
                break

            if len(raw_samples) < 10:
                time.sleep(0.5)
                continue

            # Phase 3: Auto-trim silence & review
            trimmed_samples, _, _ = trim_gesture_motion(raw_samples, gesture_name)
            decision = ui.review_and_confirm_gesture(
                trimmed_samples, len(raw_samples), gesture_label, current_num, count
            )

            if decision == "quit":
                break
            elif decision == "reject":
                pass
            elif decision == "accept":
                save_sample_csv(gesture_name, trimmed_samples, already_trimmed=True)
                ui.dataset_counts[gesture_name] = ui.dataset_counts.get(gesture_name, 0) + 1
                saved_count += 1
                i += 1
    finally:
        ui.close()

    ClaudeConsole.clear()
    ClaudeConsole.hide_cursor()
    lines = [
        f"{C_GREEN}╔══════════════════════════════════════════════════════════════════════════╗{C_RESET}",
        f"{C_GREEN}║  {C_BOLD}{C_WHITE}✓ PACED SERIES SESSION FINISHED{C_RESET}{C_GREEN}                                         ║{C_RESET}",
        f"{C_GREEN}╚══════════════════════════════════════════════════════════════════════════╝{C_RESET}",
        "",
        f"  Gesture:       {gesture_label} ({gesture_name})",
        f"  Saved Samples: {saved_count} / {count}",
        "",
        f"  Press {C_CYAN}any key{C_RESET} to return to main menu..."
    ]
    ClaudeConsole.render_frame(lines)
    ClaudeConsole.get_key()
    ClaudeConsole.clear()


def mode_manual_toggle(server: WandDataServer):
    """Manual trigger mode with 6-DOF Pygame preview."""
    gesture_name, gesture_label = mode_select_gesture("02. MANUAL RECORDING TRIGGER")
    if not gesture_name:
        return

    server.wait_for_esp()
    ClaudeConsole.clear()
    print(f"\n{C_CYAN}[INFO] Launching 6-DOF Pygame window for manual [{gesture_label}] capture...{C_RESET}")

    ui = GestureRecordingSessionUI(server, calibration, beep_fn=beep)
    ui.dataset_counts = get_dataset_summary()[0]
    ui.current_gesture_name = gesture_name
    ui.current_gesture_label = gesture_label
    counter = 1
    saved_count = 0
    try:
        while True:
            server.wait_for_esp()
            raw_samples, aborted = ui.run_live_capture(
                gesture_label, counter, total_trials="INF", duration_sec=0, manual_stop=True
            )
            if aborted:
                break

            trimmed_samples, _, _ = trim_gesture_motion(raw_samples, gesture_name)
            decision = ui.review_and_confirm_gesture(
                trimmed_samples, len(raw_samples), gesture_label, counter, total_trials="INF"
            )

            if decision == "quit":
                break
            elif decision == "reject":
                pass
            elif decision == "accept":
                save_sample_csv(gesture_name, trimmed_samples, already_trimmed=True)
                ui.dataset_counts[gesture_name] = ui.dataset_counts.get(gesture_name, 0) + 1
                saved_count += 1
                counter += 1
    finally:
        ui.close()

    ClaudeConsole.clear()
    ClaudeConsole.hide_cursor()
    lines = [
        f"{C_GREEN}╔══════════════════════════════════════════════════════════════════════════╗{C_RESET}",
        f"{C_GREEN}║  {C_BOLD}{C_WHITE}✓ MANUAL CAPTURE SESSION COMPLETED{C_RESET}{C_GREEN}                                      ║{C_RESET}",
        f"{C_GREEN}╚══════════════════════════════════════════════════════════════════════════╝{C_RESET}",
        "",
        f"  Gesture:       {gesture_label} ({gesture_name})",
        f"  Saved Samples: {saved_count}",
        "",
        f"  Press {C_CYAN}any key{C_RESET} to return to main menu..."
    ]
    ClaudeConsole.render_frame(lines)
    ClaudeConsole.get_key()
    ClaudeConsole.clear()


# =========================================================================
# SUB-MODES: 3D ORIENTATION VISUALIZER
# =========================================================================
def mode_run_3d_visualizer(server: WandDataServer):
    """Launches 3D orientation visualizer (AHRS EKF / Madgwick)."""
    server.wait_for_esp()
    ClaudeConsole.clear()
    print(f"\n{C_CYAN}[INFO] Opening 3D Orientation Visualizer in Pygame...{C_RESET}")
    server.start_recording()
    try:
        vis = Wand3DVisualizer(existing_server=server)
        vis.calib = calibration
        vis.run()
    finally:
        server.stop_recording()
    ClaudeConsole.clear()


# =========================================================================
# SUB-MODES: LIVE DEMO RECOGNITION (SHARED TELEMETRY STREAM)
# =========================================================================
def mode_live_recognition(server: WandDataServer, initial_model: str = None):
    """
    Launches Live Gesture Recognition Dashboard (Pygame) with shared WandDataServer telemetry.
    No socket closing or child process needed - ESP32 streaming continues seamlessly.
    """
    if not server.is_connected:
        ClaudeConsole.clear()
        print(f"\n{C_AMBER}[WAIT] Wand is offline. Turn on your ESP32 wand to start Live Demo...{C_RESET}")
        server.wait_for_esp()

    ClaudeConsole.clear()
    print(f"\n{C_PURPLE}[AI ENGINE] Launching Live Gesture Recognition Dashboard (Pygame)...{C_RESET}")
    server.start_streaming()
    try:
        from live_recognition import MultiModelClassifier, LiveDemoVisualizer
        clf = MultiModelClassifier()
        if initial_model:
            clf.active_model_name = initial_model
        vis = LiveDemoVisualizer(classifier=clf, existing_server=server)
        vis.calib = calibration
        vis.run()
        return server, clf.active_model_name
    except Exception as e:
        print(f"\n{C_RED}[ERROR] Live Demo failed: {e}{C_RESET}")
        time.sleep(2.0)
    finally:
        server.stop_streaming()

    ClaudeConsole.clear()
    return server, initial_model


# =========================================================================
# SUB-MODES: WAND SPACE INVADERS (ARCADE GESTURE DEFENSE)
# =========================================================================
def mode_wand_invaders(server: WandDataServer, initial_model: str = None):
    """Launches Space Invaders arcade defense game controlled by wand gestures."""
    if not server.is_connected:
        ClaudeConsole.clear()
        print(f"\n{C_AMBER}[WAIT] Wand is offline. Turn on your ESP32 wand to start Space Invaders...{C_RESET}")
        server.wait_for_esp()

    ClaudeConsole.clear()
    print(f"\n{C_PURPLE}[GAME ENGINE] Launching Wand Space Invaders (Gesture Arcade Defense)...{C_RESET}")
    server.start_streaming()
    try:
        from wand_invaders import SpaceInvadersGame
        game = SpaceInvadersGame(existing_server=server)
        game.calib = calibration
        if initial_model:
            game.clf.active_model_name = initial_model
        game.run()
        return server, game.clf.active_model_name
    except Exception as e:
        print(f"\n{C_RED}[ERROR] Wand Space Invaders failed: {e}{C_RESET}")
        time.sleep(2.0)
    finally:
        server.stop_streaming()

    ClaudeConsole.clear()
    return server, initial_model


# =========================================================================
# SUB-MODES: RETRAIN MODELS
# =========================================================================
def mode_retrain_models():
    ClaudeConsole.clear()
    ClaudeConsole.show_cursor()
    print(f"\n{C_PURPLE}╔══════════════════════════════════════════════════════════════════════════╗{C_RESET}")
    print(f"{C_PURPLE}║  RETRAINING AI MODELS (RANDOM FOREST + 1D-CNN)                           ║{C_RESET}")
    print(f"{C_PURPLE}╚══════════════════════════════════════════════════════════════════════════╝{C_RESET}")
    print(f"{C_DIM}Scanning dataset and optimizing neural weights...{C_RESET}\n")

    import subprocess
    subprocess.run([sys.executable, "train_models.py"])

    print(f"\n{C_GREEN}✓ Training completed!{C_RESET}")
    ClaudeConsole.get_key(f"\nPress {C_CYAN}any key{C_RESET} to return to main menu...")
    ClaudeConsole.hide_cursor()
    ClaudeConsole.clear()


# =========================================================================
# MAIN EVENT LOOP (CLAUDE CODE STYLE IN-PLACE DASHBOARD)
# =========================================================================
def main():
    global GYRO_ENERGY_THRESHOLD, AUTO_TRIM_ENABLED

    ClaudeConsole.setup()
    ClaudeConsole.clear()
    ClaudeConsole.hide_cursor()

    server = WandDataServer(HOST, PORT)
    server.start()

    has_lstm = os.path.exists(os.path.join("models", "gesture_conv_lstm.pth"))
    active_model = "Conv-LSTM (BiLSTM)" if has_lstm else "1D-CNN"

    spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    tick = 0
    last_dataset_check = 0.0
    cached_stats = ({}, 0)

    try:
        while True:
            time.sleep(0.05)
            tick += 1
            now = time.time()

            # Refresh dataset sample count cache every 2.0s
            if now - last_dataset_check > 2.0:
                cached_stats = get_dataset_summary()
                last_dataset_check = now

            spin = spinners[tick % len(spinners)]

            with server.lock:
                connected = server.is_connected
                ip = server.tcp_addr[0] if server.tcp_addr else None
                pkts = server.total_packets
                hz = server.estimated_hz if (now - server.last_packet_time < 1.2) else 0.0

            if connected:
                link_badge = f"{C_GREEN}[● ONLINE]{C_RESET} {spin} {ip} • {hz:4.0f} Hz ({pkts:,} pkts)"
            else:
                link_badge = f"{C_RED}[○ OFFLINE (turn on wand)]{C_RESET} {spin} Listening on :{PORT}"

            calib_badge = f"{C_GREEN}[✓ CALIBRATED]{C_RESET} 6-DOF bias compensation active" if calibration.is_calibrated else f"{C_AMBER}[⚠ UNCALIBRATED]{C_RESET} Run option 5"
            trim_badge = f"{C_GREEN}[ENABLED]{C_RESET} ({GYRO_ENERGY_THRESHOLD:.2f} rad/s)" if AUTO_TRIM_ENABLED else f"{C_DIM}[DISABLED]{C_RESET}"

            stats_dict, total_samples = cached_stats
            if total_samples > 0:
                dset_str = f"{C_BOLD}{total_samples}{C_RESET} samples across 7 classes ({stats_dict.get('circle',0)} circ, {stats_dict.get('go_up',0)} up, {stats_dict.get('go_down',0)} dn, {stats_dict.get('go_left',0)} lf, {stats_dict.get('go_right',0)} rt, {stats_dict.get('iks',0)} iks, {stats_dict.get('idle',0)} idle)"
            else:
                dset_str = f"{C_DIM}0 samples recorded{C_RESET}"

            # Check for non-blocking keypress
            k = ClaudeConsole.poll_key()
            if k:
                if k in ("1", "i", "I"):
                    mode_infinite(server)
                elif k == "2":
                    mode_paced_series(server)
                elif k == "3":
                    mode_manual_toggle(server)
                elif k in ("4", "d", "D"):
                    mode_duration_analysis(server)
                elif k == "5":
                    mode_live_monitor(server)
                elif k == "6":
                    mode_adjust_threshold()
                elif k in ("7", "c", "C"):
                    mode_calibration(server)
                elif k in ("8", "v", "V"):
                    mode_run_3d_visualizer(server)
                elif k in ("b", "B"):
                    mode_adjust_batch_size(server)
                elif k in ("9", "l", "L"):
                    server, active_model = mode_live_recognition(server, initial_model=active_model)
                elif k in ("g", "G"):
                    server, active_model = mode_wand_invaders(server, initial_model=active_model)
                elif k in ("m", "M"):
                    models_order = ["Conv-LSTM (BiLSTM)", "1D-CNN", "Random Forest"]
                    if not has_lstm and "Conv-LSTM (BiLSTM)" in models_order:
                        models_order.remove("Conv-LSTM (BiLSTM)")
                    curr_idx = models_order.index(active_model) if active_model in models_order else 0
                    active_model = models_order[(curr_idx + 1) % len(models_order)]
                elif k in ("r", "R", "t", "T"):
                    mode_retrain_models()
                elif k in ("0", "q", "Q", "ESC"):
                    break
                ClaudeConsole.clear()
                continue

            # Build in-place Claude Code dashboard frame with exact-width boxes
            box_w = 74
            top_b1 = f" {C_BOLD}╭── 01. DATA ACQUISITION & GESTURE RECORDING " + "─" * (box_w - 46) + f"╮{C_RESET}"
            bot_b  = f" {C_BOLD}╰" + "─" * (box_w - 2) + f"╯{C_RESET}"
            top_b2 = f" {C_BOLD}╭── 02. SENSOR DIAGNOSTICS & TELEMETRY " + "─" * (box_w - 40) + f"╮{C_RESET}"
            top_b3 = f" {C_BOLD}╭── 03. EDGE AI INFERENCE & MODEL RE-TRAINING " + "─" * (box_w - 47) + f"╮{C_RESET}"

            lines = [
                f"{C_CYAN}╔" + "═" * (box_w - 2) + f"╗{C_RESET}",
                f"{C_CYAN}║  {C_BOLD}{C_WHITE}WANDSENSE IMU  •  TELEMETRY & GESTURE RECOGNITION PLATFORM{C_RESET}{C_CYAN}" + " " * (box_w - 63) + f"║{C_RESET}",
                f"{C_CYAN}╚" + "═" * (box_w - 2) + f"╝{C_RESET}",
                f"  ESP32 Link:   {link_badge} {C_DIM}• Batch {server.batch_size} ({server.batch_size*10}ms){C_RESET}",
                f"  Sensor Bias:  {calib_badge}",
                f"  Silence Gate: {trim_badge}",
                f"  Dataset:      {dset_str}",
                "",
                top_b1,
                ClaudeConsole.box_row(f"{C_PURPLE}[1]{C_RESET} ★ INFINITE MODE: Continuous multi-class acquisition & balancer", box_w),
                ClaudeConsole.box_row(f"{C_GREEN}[2]{C_RESET}   Paced series recording with countdown & 6-DOF review", box_w),
                ClaudeConsole.box_row(f"{C_GREEN}[3]{C_RESET}   Manual recording trigger with ENTER toggle", box_w),
                ClaudeConsole.box_row(f"{C_CYAN}[4]{C_RESET}   Duration & speed distribution report (detect fast/slow gaps)", box_w),
                bot_b,
                top_b2,
                ClaudeConsole.box_row(f"{C_CYAN}[5]{C_RESET} Live gyroscope dynamics monitor (in-place telemetry HUD)", box_w),
                ClaudeConsole.box_row(f"{C_CYAN}[6]{C_RESET} Adjust silence auto-trim threshold (or toggle on/off)", box_w),
                ClaudeConsole.box_row(f"{C_CYAN}[7]{C_RESET} Calibrate sensors (gyroscope bias & accelerometer gravity)", box_w),
                ClaudeConsole.box_row(f"{C_CYAN}[8]{C_RESET} Interactive 3D orientation visualizer (AHRS EKF / Madgwick)", box_w),
                ClaudeConsole.box_row(f"{C_CYAN}[B]{C_RESET} Adjust telemetry UDP batching: {server.batch_size} ({server.batch_size*10}ms | ~{100/server.batch_size:.0f} pkts/s)", box_w),
                bot_b,
                top_b3,
                ClaudeConsole.box_row(f"{C_PURPLE}[9]{C_RESET} ★ LIVE DEMO: Real-time gesture recognition (Sliding Window)", box_w),
                ClaudeConsole.box_row(f"{C_EMERALD}[G]{C_RESET} ★ WAND SPACE INVADERS: Gesture Arcade Defense", box_w),
                ClaudeConsole.box_row(f"{C_VIOLET}[M]{C_RESET}   Active AI Model: {C_BOLD}{active_model}{C_RESET} (Press [M] to toggle)", box_w),
                ClaudeConsole.box_row(f"{C_PURPLE}[R]{C_RESET}   Retrain models (Random Forest + 1D-CNN)", box_w),
                bot_b,
                f"  {C_DIM}[0] Exit Platform{C_RESET}",
                "",
                f" {C_BOLD}Select Action [0-9, B, G, M, R]:{C_RESET} "
            ]
            ClaudeConsole.render_frame(lines)

    except KeyboardInterrupt:
        pass
    finally:
        ClaudeConsole.show_cursor()
        ClaudeConsole.clear()
        print(f"{C_CYAN}[INFO] WandSense IMU platform closed cleanly.{C_RESET}")
        server.running = False
        try:
            server.tcp_server.close()
            server.udp_sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

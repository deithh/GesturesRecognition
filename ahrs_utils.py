import os
import json
import math
import numpy as np

try:
    from ahrs.filters import Madgwick as _OfficialMadgwick
    from ahrs.filters import EKF as _OfficialEKF
    from ahrs.common.quaternion import Quaternion as _OfficialQuaternion
    HAS_AHRS_LIB = True
except ImportError:
    HAS_AHRS_LIB = False

CALIBRATION_FILE = "calibration.json"
G_REF = 9.80665


class SensorCalibration:
    def __init__(self):
        self.gyro_bias = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.accel_bias = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.accel_scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        self.is_calibrated = False
        self.load()

    def load(self, filepath=CALIBRATION_FILE):
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.gyro_bias = np.array(data.get("gyro_bias", [0.0, 0.0, 0.0]), dtype=np.float64)
                    self.accel_bias = np.array(data.get("accel_bias", [0.0, 0.0, 0.0]), dtype=np.float64)
                    self.accel_scale = np.array(data.get("accel_scale", [1.0, 1.0, 1.0]), dtype=np.float64)
                    self.is_calibrated = True
                    return True
            except Exception as e:
                print(f"[!] Błąd wczytywania kalibracji: {e}")
        return False

    def save(self, filepath=CALIBRATION_FILE):
        data = {
            "gyro_bias": self.gyro_bias.tolist(),
            "accel_bias": self.accel_bias.tolist(),
            "accel_scale": self.accel_scale.tolist()
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        self.is_calibrated = True
        print(f"[OK] Zapisano parametry kalibracji do: {filepath}")

    def calibrate(self, raw_samples, flat_surface=True):
        """
        Wyznacza bias spoczynkowy żyroskopu oraz kalibrację akcelerometru.
        Sprawdza również wariancję (czy pałeczka faktycznie leżała nieruchomo)
        oraz czy kąt nachylenia pozwala na bezpieczne wyzerowanie osi poziomych.
        """
        if len(raw_samples) < 50:
            print("[!] Za mało próbek do stabilnej kalibracji!")
            return False

        gyros = np.array([[s["gyro_x"], s["gyro_y"], s["gyro_z"]] for s in raw_samples], dtype=np.float64)
        accels = np.array([[s["accel_x"], s["accel_y"], s["accel_z"]] for s in raw_samples], dtype=np.float64)

        # Weryfikacja, czy pałeczka nie była poruszana podczas kalibracji
        gyro_std = np.std(gyros, axis=0)
        if np.max(gyro_std) > 0.08:
            print(f"[!] OSTRZEŻENIE: Wykryto ruch podczas kalibracji (std gyro = {np.max(gyro_std):.3f} rad/s). Powtórz kalibrację!")
            return False

        self.gyro_bias = np.mean(gyros, axis=0)

        mean_acc = np.mean(accels, axis=0)
        norm_acc = float(np.linalg.norm(mean_acc))

        if norm_acc < 1.0:
            print("[!] Nieprawidłowy odczyt akcelerometru (norma ~0)!")
            return False

        # Sprawdź, czy jedna z osi jest blisko pionu (kąt odchylenia < 12 stopni)
        dominant_axis = int(np.argmax(np.abs(mean_acc)))
        cos_tilt = abs(mean_acc[dominant_axis]) / norm_acc
        tilt_deg = math.degrees(math.acos(min(1.0, cos_tilt)))

        if flat_surface and tilt_deg <= 12.0:
            expected_acc = np.zeros(3, dtype=np.float64)
            expected_acc[dominant_axis] = math.copysign(G_REF, mean_acc[dominant_axis])
            self.accel_bias = mean_acc - expected_acc
            self.accel_scale = np.ones(3, dtype=np.float64)
            print(f"-> Wykryto płaskie ułożenie (odchylenie {tilt_deg:.1f}° od osi {'XYZ'[dominant_axis]}).")
        else:
            # Jeśli pałeczka leży pod kątem (np. ze względu na obudowę), nie psujemy osi X/Y składową grawitacji,
            # lecz normujemy skalę wektora przyspieszenia do 1g (9.80665 m/s^2).
            self.accel_bias = np.zeros(3, dtype=np.float64)
            scale_factor = G_REF / norm_acc
            self.accel_scale = np.array([scale_factor, scale_factor, scale_factor], dtype=np.float64)
            print(f"-> Pałeczka ułożona pod kątem ({tilt_deg:.1f}°). Skalibrowano skalę normy 1g (wsp. {scale_factor:.4f}) bez sztucznego zerowania osi X/Y.")

        self.save()
        print(f"-> Kalibracja zakończona sukcesem ({len(raw_samples)} próbek):")
        print(f"   Gyro bias   [rad/s]: X={self.gyro_bias[0]:+.4f}, Y={self.gyro_bias[1]:+.4f}, Z={self.gyro_bias[2]:+.4f}")
        print(f"   Accel bias  [m/s^2]: X={self.accel_bias[0]:+.4f}, Y={self.accel_bias[1]:+.4f}, Z={self.accel_bias[2]:+.4f}")
        print(f"   Accel scale [-]:     X={self.accel_scale[0]:.4f}, Y={self.accel_scale[1]:.4f}, Z={self.accel_scale[2]:.4f}")
        return True

    def apply(self, sample, adaptive_gyro=True):
        """Zwraca skorygowane wektory (cal_acc, cal_gyro) jako tablice numpy."""
        raw_acc = np.array([sample["accel_x"], sample["accel_y"], sample["accel_z"]], dtype=np.float64)
        raw_gyro = np.array([sample["gyro_x"], sample["gyro_y"], sample["gyro_z"]], dtype=np.float64)

        cal_acc = (raw_acc - self.accel_bias) * self.accel_scale
        cal_gyro = raw_gyro - self.gyro_bias

        # Ciągła mikro-korekta dryftu temperaturowego żyroskopu w spoczynku
        if adaptive_gyro and np.linalg.norm(cal_gyro) < 0.05 and abs(np.linalg.norm(cal_acc) - G_REF) < 0.6:
            self.gyro_bias = 0.995 * self.gyro_bias + 0.005 * raw_gyro
            cal_gyro = raw_gyro - self.gyro_bias

        return cal_acc, cal_gyro

    @staticmethod
    def align_mag_to_lsm6ds3(sample):
        """
        Na płytce PCB układ MMC56x3 jest obrócony o 90° względem LSM6DS3:
        - LSM6DS3: +X w prawo (wzdłuż płytki), +Y w górę, +Z do nas (z płytki)
        - MMC56x3: +X w górę (+Y_LSM), +Y w lewo (-X_LSM), +Z do nas (+Z_LSM)
        Transformacja sprowadza wektor pola magnetycznego do układu osi LSM6DS3.
        """
        mx = float(sample["mag_x"])
        my = float(sample["mag_y"])
        mz = float(sample["mag_z"])
        return np.array([-my, mx, mz], dtype=np.float64)


def quat_slerp(q1, q2, t):
    """
    Sferyczna interpolacja liniowa (SLERP) między kwaternionami q1 i q2.
    Gwarantuje idealnie gładką animację 3D nawet przy rzadkim batchowaniu telemetrii.
    """
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    dot = float(np.dot(q1, q2))
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        res = q1 + t * (q2 - q1)
        norm = float(np.linalg.norm(res))
        return res / norm if norm > 1e-12 else q1
    theta_0 = math.acos(min(1.0, dot))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    res = (s0 * q1) + (s1 * q2)
    norm = float(np.linalg.norm(res))
    return res / norm if norm > 1e-12 else q1


def quat_to_dcm(q):
    """Konwertuje znormalizowany kwaternion (w, x, y, z) na macierz rotacji 3x3 (DCM)."""
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0*(y*y + z*z),  2.0*(x*y - z*w),        2.0*(x*z + y*w)],
        [2.0*(x*y + z*w),        1.0 - 2.0*(x*x + z*z),  2.0*(y*z - x*w)],
        [2.0*(x*z - y*w),        2.0*(y*z + x*w),        1.0 - 2.0*(x*x + y*y)]
    ], dtype=np.float64)


class MadgwickAHRS:
    """
    Estymator orientacji AHRS odporny na zakłócenia magnetyczne (np. śrubki na PCB):
    - Pion (Roll i Pitch) oraz dynamika obliczane są z 6-DOF (EKF / Madgwick na Accel + Gyro),
      dzięki czemu magnetometr NIGDY nie wywraca pałeczki do góry nogami przy obrocie o 180°.
    - W trybie 9-DOF (use_mag=True) magnetometr (z automatyczną kalibracją Hard-Iron min/max)
      koryguje wyłącznie kąt poziomy Yaw wokół osi pionowej.
    """
    def __init__(self, sample_rate=100.0, gain=0.1, use_mag=True, filter_type="ekf"):
        self.default_dt = 1.0 / sample_rate
        self.gain = gain
        self.use_mag = use_mag
        self.filter_type = filter_type.lower()  # "ekf" lub "madgwick"
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.q_render = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.r_tare = np.eye(3, dtype=np.float64)
        self.last_timestamp_ms = None

        # Automatyczna kalibracja Hard-Iron magnetometru (kompensacja śrubki na PCB)
        self.mag_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
        self.mag_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
        self.yaw_mag_offset = None

        if HAS_AHRS_LIB:
            self._madgwick = _OfficialMadgwick(frequency=sample_rate, gain=gain)
            self._ekf = _OfficialEKF(frequency=sample_rate, frame="NED")
            self._ekf.mag = None
        else:
            self._madgwick = None
            self._ekf = None

    def toggle_filter(self):
        """Przełącza między Rozszerzonym Filtrem Kalmana (EKF) a filtrem Madgwicka."""
        self.filter_type = "madgwick" if self.filter_type == "ekf" else "ekf"
        return self.filter_type

    @property
    def filter_name(self):
        if not HAS_AHRS_LIB:
            return "Fallback"
        return "ahrs.EKF (Kalman)" if self.filter_type == "ekf" else "ahrs.Madgwick"

    def reset(self):
        """Ustawia bieżącą orientację w przestrzeni jako bazową (0, 0, 0)."""
        raw_r = self._get_raw_rotation_matrix()
        self.r_tare = raw_r.T.copy()
        self.q_render = self.q.copy()
        self.yaw_mag_offset = None

    def _get_raw_rotation_matrix(self):
        if HAS_AHRS_LIB:
            return _OfficialQuaternion(self.q).to_DCM()
        return quat_to_dcm(self.q)

    def get_rotation_matrix(self):
        return self._get_raw_rotation_matrix() @ self.r_tare

    def get_smooth_rotation_matrix(self, dt=0.016, smoothing_speed=24.0):
        """
        Zwraca płynnie interpolowaną macierz rotacji 3x3 (SLERP w przestrzeni kwaternionów).
        Usuwa skokowość wynikającą z batchowania próbek (np. paczek po 10 próbek co 100 ms).
        """
        alpha = 1.0 - math.exp(-smoothing_speed * max(0.001, dt))
        self.q_render = quat_slerp(self.q_render, self.q, alpha)
        return quat_to_dcm(self.q_render) @ self.r_tare

    def _apply_yaw_only_mag_correction(self, mag):
        """
        Kompensacja dryftu wyłącznie w osi poziomej Yaw (wokół pionu świata),
        po uprzednim odjęciu offsetu Hard-Iron (śrubka na PCB).
        Nie ma prawa wpłynąć na Roll ani Pitch!
        """
        mag = np.asarray(mag, dtype=np.float64)
        if not np.all(np.isfinite(mag)) or np.linalg.norm(mag) < 1e-6:
            return

        # Aktualizuj obwiednię min/max (Hard-Iron Calibration w locie)
        prev_min = self.mag_min.copy()
        prev_max = self.mag_max.copy()
        self.mag_min = np.minimum(self.mag_min, mag)
        self.mag_max = np.maximum(self.mag_max, mag)
        bounds_expanded = not (np.allclose(prev_min, self.mag_min) and np.allclose(prev_max, self.mag_max))
        span = self.mag_max - self.mag_min

        # Dopóki użytkownik nie obrócił pałeczki na tyle, by poznać środek sfery magnetycznej (span > 30 uT),
        # nie pozwalamy, by namagnesowana śrubka na PCB ściągała kąt!
        if np.min(span[:2]) < 30.0:
            return

        hard_iron_bias = 0.5 * (self.mag_max + self.mag_min)
        m_cal = mag - hard_iron_bias

        # Rzutujemy skorygowany wektor magnetyczny na płaszczyznę poziomą świata (Tilt Compensation)
        R_bw = self._get_raw_rotation_matrix()
        m_world = R_bw @ m_cal
        mag_yaw = math.atan2(m_world[1], m_world[0])

        # Jeśli obwiednia min/max właśnie się rozszerzyła (poznajemy pełniejszy zakres pola),
        # aktualizujemy kąt referencyjny zamiast ciągnąć do niepełnego środka!
        if self.yaw_mag_offset is None or bounds_expanded:
            self.yaw_mag_offset = mag_yaw
            return

        # Błąd dryftu poziomego Yaw
        yaw_err = mag_yaw - self.yaw_mag_offset
        yaw_err = (yaw_err + math.pi) % (2.0 * math.pi) - math.pi

        # Łagodna korekta wyłącznie wokół pionowej osi Z świata: q_corr * q
        half_angle = -0.008 * yaw_err * 0.5
        dq = np.array([math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)], dtype=np.float64)
        w1, x1, y1, z1 = dq
        w2, x2, y2, z2 = self.q
        self.q = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dtype=np.float64)
        self.q /= np.linalg.norm(self.q)

    def update(self, gyr, acc, mag=None, timestamp_ms=None):
        """Aktualizuje kwaternion orientacji na podstawie pojedynczej próbki (EKF lub Madgwick)."""
        dt = self.default_dt
        if timestamp_ms is not None:
            if self.last_timestamp_ms is not None:
                diff_s = (timestamp_ms - self.last_timestamp_ms) / 1000.0
                if 0.001 <= diff_s <= 0.2:
                    dt = diff_s
            self.last_timestamp_ms = timestamp_ms

        gyr = np.asarray(gyr, dtype=np.float64).copy()
        acc = np.asarray(acc, dtype=np.float64)

        norm_a = np.linalg.norm(acc)
        if norm_a < 1e-6:
            return self.q

        # Martwa strefa spoczynku (Gyro Deadzone):
        gyro_norm = np.linalg.norm(gyr)
        if gyro_norm < 0.06 and abs(norm_a - G_REF) < 0.8:
            gyr = np.array([1e-9, 0.0, 0.0], dtype=np.float64)

        if HAS_AHRS_LIB:
            if self.filter_type == "ekf" and self._ekf is not None:
                self._ekf.Dt = dt
                self._ekf.mag = None
                q_new = self._ekf.update(self.q, gyr=gyr, acc=acc, mag=None, dt=dt)
                self.q = np.asarray(q_new, dtype=np.float64)
            else:
                self._madgwick.Dt = dt
                q_new = self._madgwick.updateIMU(self.q, gyr=gyr, acc=acc, dt=dt)
                self.q = np.asarray(q_new, dtype=np.float64)
        else:
            self._update_fallback_imu(gyr, acc, dt)

        # W trybie 9-DOF używamy magnetometru WYŁĄCZNIE do stabilizacji poziomego Yaw (z odjęciem Hard-Iron),
        # dzięki czemu magnetometr nigdy nie wywraca Roll/Pitch przy obrocie o 180°!
        if self.use_mag and mag is not None:
            self._apply_yaw_only_mag_correction(mag)

        return self.q

    def update_imu(self, gyr, acc, timestamp_ms=None):
        return self.update(gyr, acc, mag=None, timestamp_ms=timestamp_ms)

    def _update_fallback_imu(self, gyr, acc, dt):
        q1, q2, q3, q4 = self.q
        ax, ay, az = acc / np.linalg.norm(acc)
        gx, gy, gz = gyr

        _2q1, _2q2, _2q3, _2q4 = 2.0 * q1, 2.0 * q2, 2.0 * q3, 2.0 * q4
        _4q1, _4q2, _4q3 = 4.0 * q1, 4.0 * q2, 4.0 * q3
        _8q2, _8q3 = 8.0 * q2, 8.0 * q3
        q1q1, q2q2, q3q3, q4q4 = q1 * q1, q2 * q2, q3 * q3, q4 * q4

        s1 = _4q1 * q3q3 + _2q3 * ax + _4q1 * q2q2 - _2q2 * ay
        s2 = _4q2 * q4q4 - _2q4 * ax + 4.0 * q1q1 * q2 - _2q1 * ay - _4q2 + _8q2 * q2q2 + _8q2 * q3q3 + _4q2 * az
        s3 = 4.0 * q1q1 * q3 + _2q1 * ax + _4q3 * q4q4 - _2q4 * ay - _4q3 + _8q3 * q2q2 + _8q3 * q3q3 + _4q3 * az
        s4 = 4.0 * q2q2 * q4 - _2q2 * ax + 4.0 * q3q3 * q4 - _2q3 * ay

        norm_s = math.sqrt(s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4)
        if norm_s > 0.0:
            s1, s2, s3, s4 = s1 / norm_s, s2 / norm_s, s3 / norm_s, s4 / norm_s

        q_dot1 = 0.5 * (-q2 * gx - q3 * gy - q4 * gz) - self.gain * s1
        q_dot2 = 0.5 * ( q1 * gx + q3 * gz - q4 * gy) - self.gain * s2
        q_dot3 = 0.5 * ( q1 * gy - q2 * gz + q4 * gx) - self.gain * s3
        q_dot4 = 0.5 * ( q1 * gz + q2 * gy - q3 * gx) - self.gain * s4

        q1 += q_dot1 * dt
        q2 += q_dot2 * dt
        q3 += q_dot3 * dt
        q4 += q_dot4 * dt

        norm_q = math.sqrt(q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4)
        self.q = np.array([q1 / norm_q, q2 / norm_q, q3 / norm_q, q4 / norm_q], dtype=np.float64)

    @property
    def roll(self):
        r_deg, _, _ = self.get_euler_angles_deg()
        return math.radians(r_deg)

    @property
    def pitch(self):
        _, p_deg, _ = self.get_euler_angles_deg()
        return math.radians(p_deg)

    @property
    def yaw(self):
        _, _, y_deg = self.get_euler_angles_deg()
        return math.radians(y_deg)

    def get_euler_angles_deg(self, smooth=True):
        if smooth:
            R = quat_to_dcm(self.q_render) @ self.r_tare
        else:
            R = self.get_rotation_matrix()
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        if sy > 1e-6:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0.0
        return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))

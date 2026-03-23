"""
Intel RealSense D455f – IMU (Gyro + Accel) ile Gerçek Zamanlı 3B Görselleştirme

Bağımlılıklar (Python 3.9+ önerilir):
    pip install numpy vpython
    # pyrealsense2 kurulumu için (işletim sistemine göre değişir):
    #   pip install pyrealsense2
    # veya Linux'ta librealsense paketleri + Python bindingleri.

Çalıştırma:
    python d455f_imu_viz.py

Notlar:
- Kamerayı başlattıktan sonraki ilk 1–2 saniye sabit tutun (filtre stabilitesine yardım eder).
- Duruş (orientation) hesaplamak için Madgwick sensör füzyonu kullanılır.
- D455f IMU birimlerinde gyro: rad/s, accel: m/s^2 verilir.
"""

import time
import math
from collections import deque

import numpy as np

try:
    import pyrealsense2 as rs
except Exception as e:
    raise SystemExit("pyrealsense2 kurulamadı / bulunamadı. Lütfen 'pip install pyrealsense2' veya librealsense kurulumunu tamamlayın.")

try:
    from vpython import canvas, box, vector, rate, color
except Exception:
    raise SystemExit("vpython gereklidir: pip install vpython")


# ------------------------- Madgwick Filtre -------------------------
class MadgwickAHRS:
    def __init__(self, sample_period=1/200.0, beta=0.05):
        self.sample_period = sample_period
        self.beta = beta
        # Birim quaternion (w, x, y, z)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    @staticmethod
    def _normalize(v):
        n = np.linalg.norm(v)
        if n == 0:
            return v
        return v / n

    def update_imu(self, gyro, accel):
        """
        gyro  : np.array([gx, gy, gz])  # rad/s
        accel : np.array([ax, ay, az])  # m/s^2
        """
        q1, q2, q3, q4 = self.q
        gx, gy, gz = gyro
        ax, ay, az = accel

        # Normalize accel (yerçekimi yönünü kullanabilmek için)
        a = np.array([ax, ay, az], dtype=float)
        a = self._normalize(a)
        ax, ay, az = a

        # Yardımcı ara terimler
        _2q1 = 2.0 * q1
        _2q2 = 2.0 * q2
        _2q3 = 2.0 * q3
        _2q4 = 2.0 * q4
        _4q1 = 4.0 * q1
        _4q2 = 4.0 * q2
        _4q3 = 4.0 * q3
        _8q2 = 8.0 * q2
        _8q3 = 8.0 * q3
        q1q1 = q1 * q1
        q2q2 = q2 * q2
        q3q3 = q3 * q3
        q4q4 = q4 * q4

        # Gradyan inme (gradient descent) – hızlandırılmış form
        s1 = _4q1 * q3q3 + _2q3 * ax + _4q1 * q2q2 - _2q2 * ay
        s2 = _4q2 * q4q4 - _2q4 * ax + 4.0 * q1q1 * q2 - _2q1 * ay - _4q2 + _8q2 * q2q2 + _8q2 * q3q3 + _4q2 * az
        s3 = 4.0 * q1q1 * q3 + _2q1 * ax + _4q3 * q4q4 - _2q4 * ay - _4q3 + _8q3 * q2q2 + _8q3 * q3q3 + _4q3 * az
        s4 = 4.0 * q2q2 * q4 - _2q2 * ax + 4.0 * q3q3 * q4 - _2q3 * ay
        s = np.array([s1, s2, s3, s4], dtype=float)
        s = self._normalize(s)

        # q türevleri (gyro ile)
        q_dot = 0.5 * self._quat_multiply(self.q, np.array([0.0, gx, gy, gz])) - self.beta * s

        # Entegrasyon
        self.q = self.q + q_dot * self.sample_period
        self.q = self._normalize(self.q)

    @staticmethod
    def _quat_multiply(q, r):
        w1, x1, y1, z1 = q
        w2, x2, y2, z2 = r
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dtype=float)

    def as_rotation_matrix(self):
        w, x, y, z = self.q
        # Dönüşüm matrisi (quaternion -> 3x3 rotasyon)
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
        ], dtype=float)
        return R


# ------------------------- RealSense IMU Akışı -------------------------
class RealSenseIMU:
    def __init__(self, gyro_fps=200, accel_fps=100):
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        # IMU akışlarını etkinleştir
        self.cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, gyro_fps)
        self.cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, accel_fps)
        self.profile = None

    def start(self):
        self.profile = self.pipe.start(self.cfg)
        # Gerçek frekans için zaman damgasından dt hesaplanacak

    def stop(self):
        try:
            self.pipe.stop()
        except Exception:
            pass

    def read_motion(self):
        """Bir frameset bekler ve varsa gyro/accel döndürür.
        Returns: (gyro_np or None, accel_np or None, timestamp_s)
        """
        fs = self.pipe.wait_for_frames()
        gyro = None
        accel = None
        ts = None
        for f in fs:
            if f.is_motion_frame():
                md = f.as_motion_frame().get_motion_data()
                ts = f.get_timestamp() * 1e-3  # ms -> s
                if f.get_profile().stream_type() == rs.stream.gyro:
                    gyro = np.array([md.x, md.y, md.z], dtype=float)
                elif f.get_profile().stream_type() == rs.stream.accel:
                    accel = np.array([md.x, md.y, md.z], dtype=float)
        return gyro, accel, ts


# ------------------------- 3B Görselleştirme -------------------------
class Visualizer3D:
    def __init__(self):
        self.scene = canvas(title='Intel RealSense D455f IMU – 3B Duruş', width=900, height=600)
        self.scene.background = color.white
        self.scene.foreground = color.black
        self.body = box(length=1, height=0.1, width=0.6, color=color.blue, opacity=0.8)
        # Referans eksenleri
        self.x_axis = box(pos=vector(1.2, 0, 0), length=2.4, height=0.01, width=0.01, color=color.red)
        self.y_axis = box(pos=vector(0, 1.2, 0), length=0.01, height=2.4, width=0.01, color=color.green)
        self.z_axis = box(pos=vector(0, 0, 1.2), length=0.01, height=0.01, width=2.4, color=color.cyan)

    def apply_rotation(self, R):
        # VPython objesini yönlendirmek için axis (x yönü) ve up (y yönü) vektörlerini kullanıyoruz.
        x_axis = vector(R[0, 0], R[1, 0], R[2, 0])
        y_axis = vector(R[0, 1], R[1, 1], R[2, 1])
        self.body.axis = x_axis
        self.body.up = y_axis


# ------------------------- Ana Döngü -------------------------

def main():
    imu = RealSenseIMU(gyro_fps=200, accel_fps=100)
    imu.start()

    vis = Visualizer3D()

    # Zaman adımı tahmini için son zaman damgası
    last_ts = None
    # Başlangıç periyodu için daha küçük beta (daha hızlı yakınsama)
    filt = MadgwickAHRS(sample_period=1/200.0, beta=0.08)

    # Basit bir kaydırmalı ortalama ile accel gürültüsünü azaltalım (opsiyonel)
    accel_window = deque(maxlen=5)

    try:
        while True:
            rate(200)  # VPython döngü hızı sınırı
            gyro, accel, ts = imu.read_motion()
            if ts is None:
                continue

            if last_ts is None:
                last_ts = ts
                continue

            dt = ts - last_ts
            # Makul dt kontrolü (örn. beklenmedik büyük atlamaları sınırla)
            if dt <= 0 or dt > 0.1:
                dt = 1/200.0
            last_ts = ts

            # Filtre örnek periyodunu güncelle
            filt.sample_period = dt

            if accel is not None:
                accel_window.append(accel)
                accel_use = np.mean(accel_window, axis=0)
            else:
                # Çok nadir, accel gelmezse son değeri kullanma
                continue

            if gyro is None:
                # Gyro yoksa yalnızca yerçekimine hizalanma olur – bir sonraki döngüyü bekle
                continue

            # Madgwick güncellemesi
            filt.update_imu(gyro=gyro, accel=accel_use)

            # 3B görselleştirmeye uygula
            R = filt.as_rotation_matrix()
            vis.apply_rotation(R)

    except KeyboardInterrupt:
        pass
    finally:
        imu.stop()


if __name__ == "__main__":
    main()

import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# YOLOv5s modelini yükle
model = YOLO('yolov5l.pt')  # ultralytics kütüphanesiyle

# RealSense pipeline başlat
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Renk ve derinlik hizalayıcı
align = rs.align(rs.stream.color)

# "cell phone" sınıf ID'si
target_class = 'cell phone'

try:
    while True:
        # Frame'leri al ve hizala
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Görüntüleri numpy array'e çevir
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # YOLO ile nesne tespiti
        results = model(color_image, verbose=False)[0]

        # Her tespit edilen nesne için
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[class_id]

            # "cell phone" tespiti ve güven eşiği
            if label == target_class and conf >= 0.8:
                # Nesnenin merkez noktası
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Uzaklık ölçümü (metre cinsinden)
                distance = depth_frame.get_distance(cx, cy)

                # Bilgileri görselleştir
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f'{label} {distance:.2f} m | Conf: {conf:.2f}'
                cv2.putText(color_image, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(color_image, (cx, cy), 5, (255, 0, 0), -1)

        # Görüntüyü göster
        cv2.imshow("YOLOv5 + RealSense", color_image)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC ile çık
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

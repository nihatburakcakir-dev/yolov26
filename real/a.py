import sys
import torch
import cv2
import os
import numpy as np
import time

# Yolov5 klasörünü path'e ekle
yolov5_path = '/home/techno/Desktop/egitim/yolov5'
if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)

from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.augmentations import letterbox

# Cihaz ayarı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# Model yükleme
model_path = 'yolov5su.pt'
model = DetectMultiBackend(model_path, device=device)
model.eval()

conf_threshold = 0.7
allowed_classes = None # None = tüm sınıflar

cap = cv2.VideoCapture("/home/techno/Downloads/tika_test.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

def gen_frames():
    prev_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # ---- Preprocess frame ----
        img = letterbox(frame, new_shape=320)[0] # letterbox resize
        img = img[:, :, ::-1].transpose(2, 0, 1) # BGR->RGB, HWC->CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        img = img.to(device)

        # ---- Inference ----
        pred = model(img)
        pred = non_max_suppression(pred, conf_threshold, 0.45)

        # ---- Sonuçları çiz ----
        for det in pred:
            if det is not None and len(det):
                det = det.cpu().numpy()
                for *box, conf, cls in det:
                    if allowed_classes and int(cls) not in allowed_classes:
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    # label = f"{model.names[int(cls)]} {conf:.2f}"
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    # cv2.putText(frame, label, (x1, y1-10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # ---- FPS Hesapla ----
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # ---- Göster ----
        cv2.imshow("YOLOv5 Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    gen_frames()
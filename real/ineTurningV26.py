from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO("/home/techno/Desktop/best.pt")

    model.train(
        data="/home/techno/Desktop/data.yaml",

        epochs=50,
        imgsz=768,          # 832 → 768 (daha hızlı)
        batch=16,           # 10 düşüktü, 4090 için artırdım
        device=0,

        workers=8,          # 4 → 8 hız artışı
        cache=True,
        amp=True,           # 🔥 çok önemli hız

        lr0=0.003,
        lrf=0.05,
        momentum=0.937,
        weight_decay=0.0005,

        cos_lr=True,

        # AUGMENTATION (seninki iyi, hafif optimize)
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.1,     # biraz ekledik (faydalı)

        scale=0.2,
        degrees=2,
        translate=0.05,

        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,

        fliplr=0.5,
        flipud=0.0,

        # SEGMENTATION
        retina_masks=True,
        overlap_mask=False,
        mask_ratio=1,

        patience=20
    )
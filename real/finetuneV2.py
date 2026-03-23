from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO("yolo26m-seg.pt")

    model.train(
        data="/home/techno/Desktop/V26Veri/data.yaml",

        epochs=200,

        imgsz=736,
        multi_scale=False,

        batch=11,
        device=0,
        workers=6,

        optimizer="MuSGD",

        lr0=0.01,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,

        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        cos_lr=True,

        mosaic=0.3,
        mixup=0.05,
        copy_paste=0.2,
        close_mosaic=30,

        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.35,

        degrees=5,
        translate=0.1,
        scale=0.3,
        shear=2,

        flipud=0.0,
        fliplr=0.5,
        perspective=0.0001,

        overlap_mask=False,
        mask_ratio=1,

        nbs=64,
        patience=40,

        cache=True   # ← Buraya ) eklendi
    )
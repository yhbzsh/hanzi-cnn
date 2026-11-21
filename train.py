"""手写汉字 CNN 训练脚本（TensorFlow/Keras）。

数据目录结构：
    data/train/<class_id>/*.png
    data/test/<class_id>/*.png

每类抽固定数量图片，等比补白到正方形后缩放，避免拉伸字形。
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def prepare(path, size):
    im = Image.open(path).convert("L")
    w, h = im.size
    if min(w, h) < 16:          # 过滤碎图
        return None
    side = max(w, h)
    canvas = Image.new("L", (side, side), 255)
    canvas.paste(im, ((side - w) // 2, (side - h) // 2))
    canvas = canvas.resize((size, size), Image.LANCZOS)
    return np.asarray(canvas, dtype="float32")[..., None] / 255.0


def load(split, per_class, class_ids, size):
    root = Path(split)
    xs, ys = [], []
    for cid in class_ids:
        for f in sorted((root / cid).glob("*.png"))[:per_class]:
            arr = prepare(f, size)
            if arr is not None:
                xs.append(arr)
                ys.append(int(cid))
    return np.array(xs), np.array(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="包含 train/ 与 test/ 的目录")
    ap.add_argument("--out", default="results")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--per-test", type=int, default=12)
    args = ap.parse_args()

    data_dir = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    class_ids = sorted([p.name for p in (data_dir / "train").iterdir()
                        if p.is_dir() and p.name.isdigit()], key=int)

    x_train_all, y_train_all = load(data_dir / "train", args.per_class, class_ids, args.size)
    x_test, y_test = load(data_dir / "test", args.per_test, class_ids, args.size)

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(x_train_all))
    x_train_all, y_train_all = x_train_all[idx], y_train_all[idx]
    val_n = max(1, int(len(x_train_all) * 0.1))
    x_val, y_val = x_train_all[:val_n], y_train_all[:val_n]
    x_train, y_train = x_train_all[val_n:], y_train_all[val_n:]

    classes = sorted(set(y_train.tolist()))
    index = {cid: i for i, cid in enumerate(classes)}
    y_train = np.array([index[int(v)] for v in y_train])
    y_val = np.array([index[int(v)] for v in y_val])
    y_test_idx = np.array([index[int(v)] for v in y_test])

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.04),
        tf.keras.layers.RandomTranslation(0.06, 0.06),
        tf.keras.layers.RandomZoom(0.08),
    ])

    model = tf.keras.Sequential([
        tf.keras.layers.Input((args.size, args.size, 1)),
        augment,
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(len(classes), activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=args.epochs, batch_size=128, verbose=2)
    test_loss, test_acc = model.evaluate(x_test, y_test_idx, verbose=0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("accuracy"); axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("loss"); axes[1].legend(); axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out / "training_curve.png"); plt.close(fig)

    metrics = {
        "classes": len(classes), "train": int(len(x_train)), "val": int(len(x_val)),
        "test": int(len(x_test)), "epochs": args.epochs, "size": args.size,
        "val_acc": round(float(history.history["val_accuracy"][-1]), 4),
        "test_acc": round(float(test_acc), 4),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()

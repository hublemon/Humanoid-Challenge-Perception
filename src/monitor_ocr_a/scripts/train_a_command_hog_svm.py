#!/usr/bin/env python3
"""Train OpenCV HOG + linear SVM models for A_command digit/icon crops.

Examples:
  python3 scripts/train_a_command_hog_svm.py --dataset /ws/datasets/a_command_hog_svm_dataset --task digit --out /ws/models/digit_hog_svm.yml
  python3 scripts/train_a_command_hog_svm.py --dataset /ws/datasets/a_command_hog_svm_dataset --task icon --out /ws/models/icon_hog_svm.yml
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


DIGIT_LABELS = [str(i) for i in range(6)]
ICON_LABELS = ["flange_nut", "gear_ring", "spacer_ring", "hex_nut", "dome_nut"]


def hog_descriptor(task):
    if task == "digit":
        return cv2.HOGDescriptor((48, 48), (16, 16), (8, 8), (8, 8), 9)
    return cv2.HOGDescriptor((96, 96), (16, 16), (8, 8), (8, 8), 9)


def load_rows(dataset, task, split):
    rows = []
    with open(Path(dataset) / "labels.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["task"] == task and row["split"] == split:
                rows.append(row)
    return rows


def features(dataset, task, split, label_to_id):
    hog = hog_descriptor(task)
    size = (48, 48) if task == "digit" else (96, 96)
    x_values = []
    y_values = []
    paths = []
    for row in load_rows(dataset, task, split):
        path = Path(dataset) / row["path"]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if (img.shape[1], img.shape[0]) != size:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        x_values.append(hog.compute(img).reshape(-1).astype(np.float32))
        y_values.append(label_to_id[row["label"]])
        paths.append(str(path))
    return np.asarray(x_values, np.float32), np.asarray(y_values, np.int32), paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", choices=["digit", "icon"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--C", type=float, default=2.5)
    args = parser.parse_args()

    labels = DIGIT_LABELS if args.task == "digit" else ICON_LABELS
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    x_train, y_train, _ = features(args.dataset, args.task, "train", label_to_id)
    x_val, y_val, _ = features(args.dataset, args.task, "val", label_to_id)
    if len(x_train) == 0:
        raise SystemExit("No training samples found")

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(args.C)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))
    svm.train(x_train, cv2.ml.ROW_SAMPLE, y_train)

    pred_train = svm.predict(x_train)[1].reshape(-1).astype(np.int32)
    train_accuracy = float((pred_train == y_train).mean())
    val_accuracy = None
    confusion = None
    if len(x_val):
        pred_val = svm.predict(x_val)[1].reshape(-1).astype(np.int32)
        val_accuracy = float((pred_val == y_val).mean())
        confusion = np.zeros((len(labels), len(labels)), dtype=int)
        for gt, pred in zip(y_val, pred_val):
            confusion[int(gt), int(pred)] += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    svm.save(str(out))
    meta = {
        "task": args.task,
        "labels": labels,
        "label_to_id": label_to_id,
        "id_to_label": {str(key): value for key, value in id_to_label.items()},
        "hog": {
            "win_size": [48, 48] if args.task == "digit" else [96, 96],
            "block_size": [16, 16],
            "block_stride": [8, 8],
            "cell_size": [8, 8],
            "nbins": 9,
        },
        "C": args.C,
        "train_samples": int(len(x_train)),
        "val_samples": int(len(x_val)),
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "confusion_matrix": confusion.tolist() if confusion is not None else None,
    }
    with open(str(out) + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

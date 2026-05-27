#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2


def stamp_float(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def ensure_dirs(out_dir):
    out = Path(out_dir)
    for name in ["pose", "rgb", "depth", "mask_cloud"]:
        (out / name).mkdir(parents=True, exist_ok=True)
    return out


def image_to_numpy(msg: Image):
    enc = msg.encoding.lower()
    h, w = msg.height, msg.width

    if enc in ["rgb8", "bgr8"]:
        dtype, channels = np.uint8, 3
    elif enc in ["rgba8", "bgra8"]:
        dtype, channels = np.uint8, 4
    elif enc in ["mono8", "8uc1"]:
        dtype, channels = np.uint8, 1
    elif enc in ["16uc1", "mono16"]:
        dtype, channels = np.uint16, 1
    elif enc == "32fc1":
        dtype, channels = np.float32, 1
    else:
        return None, enc

    itemsize = np.dtype(dtype).itemsize
    row_bytes = w * channels * itemsize
    raw = bytes(msg.data)

    if msg.step == row_bytes:
        arr = np.frombuffer(raw, dtype=dtype)
    else:
        compact = b"".join(
            raw[i * msg.step : i * msg.step + row_bytes]
            for i in range(h)
        )
        arr = np.frombuffer(compact, dtype=dtype)

    if channels == 1:
        arr = arr.reshape(h, w)
    else:
        arr = arr.reshape(h, w, channels)

    return arr.copy(), enc


def save_rgb_image(msg: Image, path: Path):
    arr, enc = image_to_numpy(msg)

    if arr is None:
        raw_path = path.with_suffix(".raw")
        raw_path.write_bytes(bytes(msg.data))
        return {"saved_as": str(raw_path), "encoding": msg.encoding, "note": "unsupported encoding, saved raw bytes"}

    if enc == "bgr8":
        arr = arr[:, :, ::-1]
    elif enc == "bgra8":
        arr = arr[:, :, [2, 1, 0]]
    elif enc == "rgba8":
        arr = arr[:, :, :3]

    if arr.ndim == 2:
        out_path = path.with_suffix(".pgm")
        with open(out_path, "wb") as f:
            f.write(f"P5\n{msg.width} {msg.height}\n255\n".encode())
            f.write(arr.astype(np.uint8).tobytes())
    else:
        out_path = path.with_suffix(".ppm")
        with open(out_path, "wb") as f:
            f.write(f"P6\n{msg.width} {msg.height}\n255\n".encode())
            f.write(arr.astype(np.uint8).tobytes())

    return {"saved_as": str(out_path), "encoding": msg.encoding}


def save_depth_image(msg: Image, path: Path):
    arr, enc = image_to_numpy(msg)

    meta = {
        "encoding": msg.encoding,
        "height": msg.height,
        "width": msg.width,
        "stamp": stamp_float(msg.header.stamp),
        "frame_id": msg.header.frame_id,
    }

    if arr is None:
        raw_path = path.with_suffix(".raw")
        raw_path.write_bytes(bytes(msg.data))
        meta["saved_raw"] = str(raw_path)
        return meta

    npy_path = path.with_suffix(".npy")
    np.save(npy_path, arr)
    meta["saved_npy"] = str(npy_path)

    if enc in ["16uc1", "mono16"]:
        pgm_path = path.with_suffix(".pgm")
        with open(pgm_path, "wb") as f:
            f.write(f"P5\n{msg.width} {msg.height}\n65535\n".encode())
            f.write(arr.astype(np.uint16).byteswap().tobytes())
        meta["saved_pgm"] = str(pgm_path)

    return meta


def save_cloud_pcd(msg: PointCloud2, path: Path):
    pts = point_cloud2.read_points(msg, skip_nans=True)
    arr = np.asarray(pts)

    if arr.size == 0:
        names = []
        xyz = np.zeros((0, 3), dtype=np.float32)
        rgb = None
    elif arr.dtype.names:
        names = list(arr.dtype.names)
        xyz = np.vstack([arr["x"], arr["y"], arr["z"]]).T.astype(np.float32)

        rgb = None
        rgb_name = "rgb" if "rgb" in names else ("rgba" if "rgba" in names else None)
        if rgb_name is not None:
            rgb_raw = arr[rgb_name]
            if rgb_raw.dtype.kind == "f":
                rgb = rgb_raw.astype(np.float32).view(np.uint32)
            else:
                rgb = rgb_raw.astype(np.uint32)
    else:
        names = []
        arr = np.asarray(list(pts), dtype=np.float32)
        if arr.size == 0:
            xyz = np.zeros((0, 3), dtype=np.float32)
            rgb = None
        else:
            xyz = arr[:, :3].astype(np.float32)
            rgb = None

    n = xyz.shape[0]

    with open(path, "w") as f:
        if rgb is not None:
            f.write("# .PCD v0.7 - Point Cloud Data file\n")
            f.write("VERSION 0.7\n")
            f.write("FIELDS x y z rgb\n")
            f.write("SIZE 4 4 4 4\n")
            f.write("TYPE F F F U\n")
            f.write("COUNT 1 1 1 1\n")
            f.write(f"WIDTH {n}\n")
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {n}\n")
            f.write("DATA ascii\n")
            for p, c in zip(xyz, rgb):
                if np.all(np.isfinite(p)):
                    f.write(f"{p[0]} {p[1]} {p[2]} {int(c)}\n")
        else:
            f.write("# .PCD v0.7 - Point Cloud Data file\n")
            f.write("VERSION 0.7\n")
            f.write("FIELDS x y z\n")
            f.write("SIZE 4 4 4\n")
            f.write("TYPE F F F\n")
            f.write("COUNT 1 1 1\n")
            f.write(f"WIDTH {n}\n")
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {n}\n")
            f.write("DATA ascii\n")
            for p in xyz:
                if np.all(np.isfinite(p)):
                    f.write(f"{p[0]} {p[1]} {p[2]}\n")

    return {
        "saved_as": str(path),
        "points": int(n),
        "frame_id": msg.header.frame_id,
        "stamp": stamp_float(msg.header.stamp),
        "fields": names,
    }


class WristOutputSaver(Node):
    def __init__(self, out_dir, target):
        super().__init__("wrist_output_saver_150")

        self.out = ensure_dirs(out_dir)
        self.target = target

        self.counts = {
            "pose": 0,
            "rgb": 0,
            "depth": 0,
            "mask_cloud": 0,
        }

        self.pose_csv_path = self.out / "pose" / "target_pose.csv"
        self.pose_jsonl_path = self.out / "pose" / "target_pose.jsonl"
        self.rgb_meta_path = self.out / "rgb" / "metadata.jsonl"
        self.depth_meta_path = self.out / "depth" / "metadata.jsonl"
        self.cloud_meta_path = self.out / "mask_cloud" / "metadata.jsonl"

        with open(self.pose_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "idx", "stamp", "frame_id",
                "x", "y", "z",
                "qx", "qy", "qz", "qw"
            ])

        self.create_subscription(
            PoseStamped,
            "/perception/wrist/target_pose",
            self.pose_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            "/perception/wrist/rgb",
            self.rgb_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            "/perception/wrist/depth",
            self.depth_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2,
            "/perception/wrist/mask_cloud",
            self.cloud_cb,
            qos_profile_sensor_data,
        )

        self.get_logger().info(f"Saving wrist outputs to: {self.out}")
        self.get_logger().info(f"Target count per topic: {self.target}")

    def done(self, key):
        return self.counts[key] >= self.target

    def check_finish(self):
        if all(v >= self.target for v in self.counts.values()):
            self.get_logger().info(f"Done. Final counts: {self.counts}")
            rclpy.shutdown()

    def log_count(self, key):
        c = self.counts[key]
        if c == 1 or c % 10 == 0 or c == self.target:
            self.get_logger().info(f"{key}: {c}/{self.target}")

    def pose_cb(self, msg):
        key = "pose"
        if self.done(key):
            return

        idx = self.counts[key]
        p = msg.pose.position
        q = msg.pose.orientation

        row = {
            "idx": idx,
            "stamp": stamp_float(msg.header.stamp),
            "frame_id": msg.header.frame_id,
            "position": {"x": p.x, "y": p.y, "z": p.z},
            "orientation": {"x": q.x, "y": q.y, "z": q.z, "w": q.w},
        }

        with open(self.pose_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                idx, row["stamp"], row["frame_id"],
                p.x, p.y, p.z,
                q.x, q.y, q.z, q.w
            ])

        with open(self.pose_jsonl_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        self.counts[key] += 1
        self.log_count(key)
        self.check_finish()

    def rgb_cb(self, msg):
        key = "rgb"
        if self.done(key):
            return

        idx = self.counts[key]
        base = self.out / "rgb" / f"frame_{idx:06d}"
        meta = save_rgb_image(msg, base)
        meta.update({
            "idx": idx,
            "stamp": stamp_float(msg.header.stamp),
            "frame_id": msg.header.frame_id,
            "height": msg.height,
            "width": msg.width,
        })

        with open(self.rgb_meta_path, "a") as f:
            f.write(json.dumps(meta) + "\n")

        self.counts[key] += 1
        self.log_count(key)
        self.check_finish()

    def depth_cb(self, msg):
        key = "depth"
        if self.done(key):
            return

        idx = self.counts[key]
        base = self.out / "depth" / f"frame_{idx:06d}"
        meta = save_depth_image(msg, base)
        meta["idx"] = idx

        with open(self.depth_meta_path, "a") as f:
            f.write(json.dumps(meta) + "\n")

        self.counts[key] += 1
        self.log_count(key)
        self.check_finish()

    def cloud_cb(self, msg):
        key = "mask_cloud"
        if self.done(key):
            return

        idx = self.counts[key]
        path = self.out / "mask_cloud" / f"frame_{idx:06d}.pcd"
        meta = save_cloud_pcd(msg, path)
        meta["idx"] = idx

        with open(self.cloud_meta_path, "a") as f:
            f.write(json.dumps(meta) + "\n")

        self.counts[key] += 1
        self.log_count(key)
        self.check_finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--target", type=int, default=150)
    args = parser.parse_args()

    rclpy.init()
    node = WristOutputSaver(args.out, args.target)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(f"Interrupted. Current counts: {node.counts}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

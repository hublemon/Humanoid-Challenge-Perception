#!/usr/bin/env python3
"""Record wrist grasp target poses and PointCloud2 frames for manipulation.

Subscribes to:
  - /perception/wrist/target_one_pose : geometry_msgs/PoseStamped
  - /perception/wrist/mask_cloud      : sensor_msgs/PointCloud2

Writes one PCD file plus pose metadata per saved cloud frame.
"""

from __future__ import annotations

import csv
import json
import math
import struct
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField


DATATYPES = {
    PointField.INT8: ("b", 1),
    PointField.UINT8: ("B", 1),
    PointField.INT16: ("h", 2),
    PointField.UINT16: ("H", 2),
    PointField.INT32: ("i", 4),
    PointField.UINT32: ("I", 4),
    PointField.FLOAT32: ("f", 4),
    PointField.FLOAT64: ("d", 8),
}


class WristGraspDatasetRecorderNode(Node):
    def __init__(self) -> None:
        super().__init__("wrist_grasp_dataset_recorder")

        self.declare_parameter("pose_topic", "/perception/wrist/target_one_pose")
        self.declare_parameter("cloud_topic", "/perception/wrist/mask_cloud")
        self.declare_parameter("output_dir", "/tmp/wrist_grasp_dataset")
        self.declare_parameter("append_timestamp", True)
        self.declare_parameter("sample_count", 30)
        self.declare_parameter("min_interval_sec", 0.0)
        self.declare_parameter("require_pose", True)
        self.declare_parameter("skip_empty_cloud", True)

        gp = self.get_parameter
        self.pose_topic = str(gp("pose_topic").value)
        self.cloud_topic = str(gp("cloud_topic").value)
        self.sample_count = max(1, int(gp("sample_count").value))
        self.min_interval_sec = max(0.0, float(gp("min_interval_sec").value))
        self.require_pose = bool(gp("require_pose").value)
        self.skip_empty_cloud = bool(gp("skip_empty_cloud").value)

        base_output_dir = Path(str(gp("output_dir").value)).expanduser()
        if bool(gp("append_timestamp").value):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_output_dir = base_output_dir / f"run_{stamp}"
        self.output_dir = base_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.latest_pose: Optional[PoseStamped] = None
        self.saved_count = 0
        self.last_save_time_sec: Optional[float] = None
        self.done = False

        self.samples_csv_path = self.output_dir / "samples.csv"
        self.samples_jsonl_path = self.output_dir / "samples.jsonl"
        self.csv_file = self.samples_csv_path.open("w", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "index",
                "pcd_file",
                "pose_file",
                "point_count",
                "pose_frame_id",
                "pose_stamp_sec",
                "pose_stamp_nanosec",
                "cloud_frame_id",
                "cloud_stamp_sec",
                "cloud_stamp_nanosec",
                "position_x",
                "position_y",
                "position_z",
                "orientation_x",
                "orientation_y",
                "orientation_z",
                "orientation_w",
            ],
        )
        self.csv_writer.writeheader()
        self.jsonl_file = self.samples_jsonl_path.open("w", encoding="utf-8")

        self.pose_sub = self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self.pose_cb,
            10,
        )
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            self.cloud_topic,
            self.cloud_cb,
            10,
        )

        self._write_run_metadata()

        self.get_logger().info(
            "WristGraspDatasetRecorder ready.\n"
            f"  pose_topic={self.pose_topic}\n"
            f"  cloud_topic={self.cloud_topic}\n"
            f"  output_dir={self.output_dir}\n"
            f"  sample_count={self.sample_count}, min_interval_sec={self.min_interval_sec}"
        )

    def pose_cb(self, msg: PoseStamped) -> None:
        self.latest_pose = msg

    def cloud_cb(self, msg: PointCloud2) -> None:
        if self.done:
            return

        if self.require_pose and self.latest_pose is None:
            self.get_logger().warn(
                f"Waiting for pose on {self.pose_topic}; cloud not saved.",
                throttle_duration_sec=5.0,
            )
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if (
            self.last_save_time_sec is not None
            and now_sec - self.last_save_time_sec < self.min_interval_sec
        ):
            return

        points = self._cloud_to_points(msg)
        if self.skip_empty_cloud and not points:
            self.get_logger().warn(
                "PointCloud2 had no valid XYZ points; cloud not saved.",
                throttle_duration_sec=5.0,
            )
            return

        idx = self.saved_count
        pcd_name = f"sample_{idx:03d}.pcd"
        pose_name = f"sample_{idx:03d}_pose.json"
        pcd_path = self.output_dir / pcd_name
        pose_path = self.output_dir / pose_name

        self._write_pcd(pcd_path, points)
        pose_payload = self._pose_payload(self.latest_pose)
        sample_payload = {
            "index": idx,
            "pcd_file": pcd_name,
            "pose_file": pose_name,
            "point_count": len(points),
            "cloud": {
                "frame_id": msg.header.frame_id,
                "stamp": {
                    "sec": int(msg.header.stamp.sec),
                    "nanosec": int(msg.header.stamp.nanosec),
                },
            },
            "pose": pose_payload,
        }
        pose_path.write_text(
            json.dumps(sample_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_sample_index(sample_payload)

        self.saved_count += 1
        self.last_save_time_sec = now_sec
        self.get_logger().info(
            f"Saved sample {self.saved_count}/{self.sample_count}: "
            f"{pcd_path} ({len(points)} points)"
        )

        if self.saved_count >= self.sample_count:
            self.done = True
            self.get_logger().info(
                f"Finished recording {self.sample_count} samples in {self.output_dir}"
            )

    def _write_run_metadata(self) -> None:
        metadata = {
            "pose_topic": self.pose_topic,
            "cloud_topic": self.cloud_topic,
            "sample_count": self.sample_count,
            "min_interval_sec": self.min_interval_sec,
            "require_pose": self.require_pose,
            "skip_empty_cloud": self.skip_empty_cloud,
            "output_dir": str(self.output_dir),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        (self.output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_sample_index(self, sample: Dict) -> None:
        pose = sample.get("pose") or {}
        position = pose.get("position") or {}
        orientation = pose.get("orientation") or {}
        pose_stamp = pose.get("stamp") or {}
        cloud = sample.get("cloud") or {}
        cloud_stamp = cloud.get("stamp") or {}

        row = {
            "index": sample["index"],
            "pcd_file": sample["pcd_file"],
            "pose_file": sample["pose_file"],
            "point_count": sample["point_count"],
            "pose_frame_id": pose.get("frame_id", ""),
            "pose_stamp_sec": pose_stamp.get("sec", ""),
            "pose_stamp_nanosec": pose_stamp.get("nanosec", ""),
            "cloud_frame_id": cloud.get("frame_id", ""),
            "cloud_stamp_sec": cloud_stamp.get("sec", ""),
            "cloud_stamp_nanosec": cloud_stamp.get("nanosec", ""),
            "position_x": position.get("x", ""),
            "position_y": position.get("y", ""),
            "position_z": position.get("z", ""),
            "orientation_x": orientation.get("x", ""),
            "orientation_y": orientation.get("y", ""),
            "orientation_z": orientation.get("z", ""),
            "orientation_w": orientation.get("w", ""),
        }
        self.csv_writer.writerow(row)
        self.csv_file.flush()
        self.jsonl_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
        self.jsonl_file.flush()

    @staticmethod
    def _pose_payload(msg: Optional[PoseStamped]) -> Optional[Dict]:
        if msg is None:
            return None

        p = msg.pose.position
        q = msg.pose.orientation
        return {
            "frame_id": msg.header.frame_id,
            "stamp": {
                "sec": int(msg.header.stamp.sec),
                "nanosec": int(msg.header.stamp.nanosec),
            },
            "position": {
                "x": float(p.x),
                "y": float(p.y),
                "z": float(p.z),
            },
            "orientation": {
                "x": float(q.x),
                "y": float(q.y),
                "z": float(q.z),
                "w": float(q.w),
            },
        }

    @staticmethod
    def _field_map(cloud: PointCloud2) -> Dict[str, PointField]:
        return {field.name: field for field in cloud.fields}

    def _cloud_to_points(self, cloud: PointCloud2) -> List[Tuple[float, float, float, float]]:
        fields = self._field_map(cloud)
        missing = [name for name in ("x", "y", "z") if name not in fields]
        if missing:
            self.get_logger().warn(f"PointCloud2 missing fields {missing}; skipping cloud.")
            return []

        endian = ">" if cloud.is_bigendian else "<"
        has_rgb = "rgb" in fields
        points = []

        for row in range(max(1, cloud.height)):
            row_offset = row * cloud.row_step
            for col in range(cloud.width):
                base = row_offset + col * cloud.point_step
                x = self._read_field(cloud.data, base, fields["x"], endian)
                y = self._read_field(cloud.data, base, fields["y"], endian)
                z = self._read_field(cloud.data, base, fields["z"], endian)
                if not all(math.isfinite(float(v)) for v in (x, y, z)):
                    continue

                rgb = 0.0
                if has_rgb:
                    rgb = self._read_rgb_as_float(cloud.data, base, fields["rgb"], endian)

                points.append((float(x), float(y), float(z), float(rgb)))

        return points

    @staticmethod
    def _read_field(data: bytes, base: int, field: PointField, endian: str):
        if field.datatype not in DATATYPES:
            raise ValueError(f"Unsupported PointField datatype: {field.datatype}")
        fmt, _ = DATATYPES[field.datatype]
        return struct.unpack_from(endian + fmt, data, base + field.offset)[0]

    def _read_rgb_as_float(self, data: bytes, base: int, field: PointField, endian: str) -> float:
        value = self._read_field(data, base, field, endian)

        if field.datatype == PointField.FLOAT32:
            return float(value)

        if field.datatype == PointField.UINT32:
            packed = struct.pack(endian + "I", int(value))
            return struct.unpack(endian + "f", packed)[0]

        return float(value)

    @staticmethod
    def _write_pcd(path: Path, points: Iterable[Tuple[float, float, float, float]]) -> None:
        pts = list(points)
        with path.open("w", encoding="utf-8") as f:
            f.write("# .PCD v0.7 - Point Cloud Data file\n")
            f.write("VERSION 0.7\n")
            f.write("FIELDS x y z rgb\n")
            f.write("SIZE 4 4 4 4\n")
            f.write("TYPE F F F F\n")
            f.write("COUNT 1 1 1 1\n")
            f.write(f"WIDTH {len(pts)}\n")
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {len(pts)}\n")
            f.write("DATA ascii\n")
            for x, y, z, rgb in pts:
                f.write(f"{x:.8f} {y:.8f} {z:.8f} {rgb:.8f}\n")

    def close(self) -> None:
        self.csv_file.close()
        self.jsonl_file.close()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WristGraspDatasetRecorderNode()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import tf2_ros


DEFAULT_BASE_FRAME = "base_link"
DEFAULT_TARGET_FRAME = "camera_right_color_optical_frame"
DEFAULT_JOINT_TOPIC = "/joint_states"


def quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm


def quaternion_to_rpy(q: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_normalize(q)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return np.asarray([roll, pitch, yaw], dtype=np.float64)


class RightWristBasePoseSaver(Node):
    def __init__(
        self,
        base_frame: str,
        target_frame: str,
        joint_topic: str,
        timeout_sec: float,
        out_path: Path,
    ) -> None:
        super().__init__("right_wrist_base_pose_saver")

        self.base_frame = base_frame
        self.target_frame = target_frame
        self.joint_topic = joint_topic
        self.timeout_sec = max(0.01, timeout_sec)
        self.out_path = out_path
        self.latest_joint_state: Optional[JointState] = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.joint_sub = self.create_subscription(
            JointState,
            self.joint_topic,
            self.joint_callback,
            10,
        )

    def joint_callback(self, msg: JointState) -> None:
        self.latest_joint_state = msg

    def wait_for_tf(self) -> bool:
        deadline = time.monotonic() + self.timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.tf_buffer.can_transform(
                self.base_frame,
                self.target_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            ):
                return True
        return False

    def wait_for_joint_state(self) -> bool:
        deadline = time.monotonic() + self.timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.latest_joint_state is not None:
                return True
        return False

    def lookup_pose(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        tf = self.tf_buffer.lookup_transform(
            self.base_frame,
            self.target_frame,
            rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=self.timeout_sec),
        )
        tr = tf.transform.translation
        rot = tf.transform.rotation
        position = np.asarray([tr.x, tr.y, tr.z], dtype=np.float64)
        quaternion = quat_normalize(
            np.asarray([rot.x, rot.y, rot.z, rot.w], dtype=np.float64)
        )
        stamp = {
            "sec": int(tf.header.stamp.sec),
            "nanosec": int(tf.header.stamp.nanosec),
        }
        return position, quaternion, stamp

    def joint_state_to_dict(self, msg: JointState) -> dict:
        joints = {}
        for idx, name in enumerate(msg.name):
            item = {}
            if idx < len(msg.position):
                item["position"] = float(msg.position[idx])
            if idx < len(msg.velocity):
                item["velocity"] = float(msg.velocity[idx])
            if idx < len(msg.effort):
                item["effort"] = float(msg.effort[idx])
            joints[name] = item

        return {
            "topic": self.joint_topic,
            "stamp": {
                "sec": int(msg.header.stamp.sec),
                "nanosec": int(msg.header.stamp.nanosec),
            },
            "name": list(msg.name),
            "position": [float(v) for v in msg.position],
            "velocity": [float(v) for v in msg.velocity],
            "effort": [float(v) for v in msg.effort],
            "by_name": joints,
        }

    def capture(self) -> dict:
        self.get_logger().info(
            f"Capturing one TF sample: {self.base_frame} <- {self.target_frame}"
        )

        if not self.wait_for_tf():
            raise RuntimeError(
                f"TF not available: {self.base_frame} <- {self.target_frame}. "
                "Check frame names or run robot_state_publisher/static TF."
            )

        if not self.wait_for_joint_state():
            raise RuntimeError(
                f"JointState not available on {self.joint_topic}. "
                "Check the joint state publisher or pass --joint-topic."
            )

        position, quaternion, stamp = self.lookup_pose()
        rpy = quaternion_to_rpy(quaternion)
        joint_state = self.joint_state_to_dict(self.latest_joint_state)

        return {
            "base_frame": self.base_frame,
            "target_frame": self.target_frame,
            "samples": 1,
            "saved_wall_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "tf_stamp": stamp,
            "pose": {
                "position": {
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "z": float(position[2]),
                },
                "orientation": {
                    "x": float(quaternion[0]),
                    "y": float(quaternion[1]),
                    "z": float(quaternion[2]),
                    "w": float(quaternion[3]),
                },
                "orientation_rpy_rad": {
                    "roll": float(rpy[0]),
                    "pitch": float(rpy[1]),
                    "yaw": float(rpy[2]),
                },
                "orientation_rpy_deg": {
                    "roll": float(math.degrees(rpy[0])),
                    "pitch": float(math.degrees(rpy[1])),
                    "yaw": float(math.degrees(rpy[2])),
                },
                "flat_xyz_quat_xyzw": [
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                    float(quaternion[0]),
                    float(quaternion[1]),
                    float(quaternion[2]),
                    float(quaternion[3]),
                ],
            },
            "joint_state": joint_state,
        }

    def save(self, payload: dict) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        csv_path = self.out_path.with_suffix(".csv")
        pose = payload["pose"]
        joint_state = payload["joint_state"]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = [
                "base_frame",
                "target_frame",
                "samples",
                "tf_stamp_sec",
                "tf_stamp_nanosec",
                "x",
                "y",
                "z",
                "qx",
                "qy",
                "qz",
                "qw",
                "roll_deg",
                "pitch_deg",
                "yaw_deg",
            ]
            header.extend([f"joint_position.{name}" for name in joint_state["name"]])
            writer.writerow(header)

            row = [
                payload["base_frame"],
                payload["target_frame"],
                payload["samples"],
                payload["tf_stamp"]["sec"],
                payload["tf_stamp"]["nanosec"],
                pose["position"]["x"],
                pose["position"]["y"],
                pose["position"]["z"],
                pose["orientation"]["x"],
                pose["orientation"]["y"],
                pose["orientation"]["z"],
                pose["orientation"]["w"],
                pose["orientation_rpy_deg"]["roll"],
                pose["orientation_rpy_deg"]["pitch"],
                pose["orientation_rpy_deg"]["yaw"],
            ]
            row.extend(joint_state["by_name"][name].get("position") for name in joint_state["name"])
            writer.writerow(row)

        self.get_logger().info(f"Saved JSON: {self.out_path}")
        self.get_logger().info(f"Saved CSV : {csv_path}")


def default_output_path() -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"captures/right_wrist_base_pose_{stamp}.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture one TF pose of the right wrist in base_link with joint states."
    )
    parser.add_argument("--base-frame", default=DEFAULT_BASE_FRAME)
    parser.add_argument("--target-frame", default=DEFAULT_TARGET_FRAME)
    parser.add_argument("--joint-topic", default=DEFAULT_JOINT_TOPIC)
    parser.add_argument("--timeout-sec", type=float, default=3.0)
    parser.add_argument("--out", default=default_output_path())
    args = parser.parse_args()

    rclpy.init()
    node = RightWristBasePoseSaver(
        base_frame=args.base_frame,
        target_frame=args.target_frame,
        joint_topic=args.joint_topic,
        timeout_sec=args.timeout_sec,
        out_path=Path(args.out),
    )

    try:
        payload = node.capture()
        node.save(payload)
        p = payload["pose"]["position"]
        q = payload["pose"]["orientation"]
        rpy = payload["pose"]["orientation_rpy_deg"]
        joint_count = len(payload["joint_state"]["name"])
        print(
            "Captured pose "
            f"{payload['base_frame']} <- {payload['target_frame']}: "
            f"x={p['x']:.6f}, y={p['y']:.6f}, z={p['z']:.6f}, "
            f"qx={q['x']:.6f}, qy={q['y']:.6f}, qz={q['z']:.6f}, qw={q['w']:.6f}, "
            f"rpy_deg=({rpy['roll']:.3f}, {rpy['pitch']:.3f}, {rpy['yaw']:.3f}), "
            f"joints={joint_count}"
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
#
# Copyright 2026 perception
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Wrist bolt-hole 3D center node."""

import rclpy

from perception_nodes.wrist_targets.wrist_target_common import TargetPreset, WristTargetCenterNode


PRESET = TargetPreset(
    node_name='bolt_hole_center',
    target_class='bolt_hole',
    default_detections_topic='/detections/wrist/scenario_d/bolt_hole',
    default_out_pose_topic='/perception/wrist/bolt_hole_center',
    target_mode='hole',
    default_debug_topic='/perception/wrist/debug/bolt_hole_center_image',
)


class BoltHoleCenterNode(WristTargetCenterNode):
    """Publish the 3D center of the bolt insertion hole from wrist camera."""

    def __init__(self) -> None:
        super().__init__(PRESET)


def main(args=None) -> None:
    """Run the bolt-hole center node."""
    rclpy.init(args=args)
    node = BoltHoleCenterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

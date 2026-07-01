#!/usr/bin/env python3
#
# Copyright 2026 perception
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Wrist bolt-top 3D center node."""

import rclpy

from perception_nodes.wrist_targets.wrist_target_common import TargetPreset, WristTargetCenterNode


PRESET = TargetPreset(
    node_name='bolt_top_center',
    target_class='bolt_top',
    default_detections_topic='/detections/wrist/scenario_d/bolt_top',
    default_out_pose_topic='/perception/wrist/bolt_top_center',
    target_mode='top_surface',
    default_debug_topic='/perception/wrist/debug/bolt_top_center_image',
)


class BoltTopCenterNode(WristTargetCenterNode):
    """Publish the 3D center of the bolt top from wrist camera."""

    def __init__(self) -> None:
        super().__init__(PRESET)


def main(args=None) -> None:
    """Run the bolt-top center node."""
    rclpy.init(args=args)
    node = BoltTopCenterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

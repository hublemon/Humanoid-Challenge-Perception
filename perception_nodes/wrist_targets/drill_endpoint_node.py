#!/usr/bin/env python3
#
# Copyright 2026 perception
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Wrist drill endpoint 3D node."""

import rclpy

from perception_nodes.wrist_targets.wrist_target_common import TargetPreset, WristTargetCenterNode


PRESET = TargetPreset(
    node_name='drill_endpoint',
    target_class='drill',
    default_detections_topic='/detections/wrist/scenario_d/drill',
    default_out_pose_topic='/perception/wrist/drill_endpoint',
    target_mode='endpoint',
    default_debug_topic='/perception/wrist/debug/drill_endpoint_image',
)


class DrillEndpointNode(WristTargetCenterNode):
    """Publish the 3D endpoint of the detected full drill mask/bbox from wrist camera."""

    def __init__(self) -> None:
        super().__init__(PRESET)


def main(args=None) -> None:
    """Run the drill endpoint node."""
    rclpy.init(args=args)
    node = DrillEndpointNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

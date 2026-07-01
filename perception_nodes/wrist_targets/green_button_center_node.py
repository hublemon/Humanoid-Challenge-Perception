#!/usr/bin/env python3
#
# Copyright 2026 perception
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Wrist green-button 3D center node."""

import rclpy

from perception_nodes.wrist_targets.wrist_target_common import TargetPreset, WristTargetCenterNode


PRESET = TargetPreset(
    node_name='green_button_center',
    target_class='green_button',
    default_detections_topic='/detections/wrist/scenario_c/green_button',
    default_out_pose_topic='/perception/wrist/green_button_center',
    target_mode='surface',
    default_debug_topic='/perception/wrist/debug/green_button_center_image',
)


class GreenButtonCenterNode(WristTargetCenterNode):
    """Publish the 3D center of the green button from wrist camera."""

    def __init__(self) -> None:
        super().__init__(PRESET)


def main(args=None) -> None:
    """Run the green-button center node."""
    rclpy.init(args=args)
    node = GreenButtonCenterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

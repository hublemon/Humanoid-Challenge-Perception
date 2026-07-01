#!/usr/bin/env python3
#
# Copyright 2026 perception
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""ZED wheel-hole 3D center node."""

import rclpy

from perception_nodes.zed_targets.zed_target_common import TargetPreset, ZedTargetCenterNode


PRESET = TargetPreset(
    node_name='wheel_hole_center',
    target_class='wheel_hole',
    default_detections_topic='/detections/scenario_d/wheel_hole',
    default_out_pose_topic='/perception/zed/wheel_hole_center',
    target_mode='hole',
    default_debug_topic='/perception/zed/debug/wheel_hole_center_image',
    default_detections_msg_type='single',
)


class WheelHoleCenterNode(ZedTargetCenterNode):
    """Publish the 3D center of the wheel center hole."""

    def __init__(self) -> None:
        super().__init__(PRESET)


def main(args=None) -> None:
    """Run the wheel-hole center node."""
    rclpy.init(args=args)
    node = WheelHoleCenterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

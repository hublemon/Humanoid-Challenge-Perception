#!/usr/bin/env python3
#
# Copyright 2026 perception
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""ZED bolt-top 3D center node."""

import rclpy

from perception_nodes.zed_targets.zed_target_common import TargetPreset, ZedTargetCenterNode


PRESET = TargetPreset(
    node_name='bolt_top_center',
    target_class='bolt_top',
    default_detections_topic='/detections/scenario_d/bolt_top',
    default_out_pose_topic='/perception/zed/bolt_top_center',
    target_mode='top_surface',
    default_debug_topic='/perception/zed/debug/bolt_top_center_image',
    default_detections_msg_type='single',
)


class BoltTopCenterNode(ZedTargetCenterNode):
    """Publish the 3D center of the visible bolt top surface."""

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

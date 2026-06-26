# perception_wrist_targets

Wrist camera target-to-3D package. This package mirrors `perception_zed_targets`
but uses the right wrist RealSense camera, where RGB and depth are **not**
registered. Each node subscribes to one target-specific `PartDetectionArray`
topic and publishes one `geometry_msgs/PoseStamped` in `base_link`.

## Nodes

| executable | detection topic | class_name | output topic | mode |
| --- | --- | --- | --- | --- |
| `green_button_center_node` | `/detections/wrist/scenario_c/green_button` | `green_button` | `/perception/wrist/green_button_center` | surface |
| `bolt_top_center_node` | `/detections/wrist/scenario_d/bolt_top` | `bolt_top` | `/perception/wrist/bolt_top_center` | top_surface |
| `wheel_hole_center_node` | `/detections/wrist/scenario_d/wheel_hole` | `wheel_hole` | `/perception/wrist/wheel_hole_center` | hole |
| `bolt_hole_center_node` | `/detections/wrist/scenario_d/bolt_hole` | `bolt_hole` | `/perception/wrist/bolt_hole_center` | hole |
| `drill_endpoint_node` | `/detections/wrist/scenario_d/drill` | `drill` | `/perception/wrist/drill_endpoint` | endpoint |

All output messages are `geometry_msgs/msg/PoseStamped`. Orientation is identity;
manipulation is expected to apply grasp offsets and approach orientation.

## Wrist RGB-D handling

The wrist RGB and depth images are not aligned. The common node therefore:

1. back-projects the full depth image using depth CameraInfo;
2. transforms depth-frame points into the color optical frame using TF
   `depth_frame -> rgb_frame` or the fallback extrinsics in `params.yaml`;
3. projects color-frame points into the RGB image plane;
4. selects projected 3D points that fall inside the target mask/ROI;
5. estimates the target center according to target mode;
6. transforms the result into `base_link` and publishes `PoseStamped`.

## Target modes

- `surface`: use mask/bbox inner region depth and 2D target center.
- `top_surface`: use near-depth points and optional plane fitting for top faces.
- `hole`: use ellipse annulus/ring depth; never use the hole center pixel depth directly.
- `endpoint`: select a drill endpoint/corner in 2D, sample depth at an inset point,
  and publish the endpoint ray with inset depth.

## Build and run

```bash
cd /ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-up-to perception_wrist_targets
source install/setup.bash

ros2 launch perception_wrist_targets wrist_targets_all.launch.py
```

Run one node:

```bash
ros2 launch perception_wrist_targets drill_endpoint.launch.py
ros2 topic echo /perception/wrist/drill_endpoint
```

## Hole-mode stabilization parameters

These parameters apply to any node whose `target_mode` is `hole`
(`wheel_hole_center`, `bolt_hole_center`).  
`wheel_hole_center` ships with stabilization enabled by default; all other nodes
use the disabled/plane defaults.

| parameter | type | default | wheel_hole default | description |
| --- | --- | --- | --- | --- |
| `hole_center_stabilization_enable` | bool | `false` | `true` | Enable EMA smoothing on the 2-D hole center pixel. |
| `hole_center_smoothing_alpha` | float | `0.30` | `0.30` | EMA weight for the new sample (higher = faster response). |
| `hole_center_reset_gate_px` | float | `35.0` | `35.0` | Distance (px) at which the EMA resets instead of smoothing. |
| `hole_depth_policy` | str | `plane` | `median` | Depth estimation policy: `plane` (existing plane-fit + fallback), `median`, or `percentile`. |
| `hole_depth_percentile` | float | `50.0` | `50.0` | Ring-depth percentile used when `hole_depth_policy: percentile`. |
| `hole_depth_smoothing_enable` | bool | `false` | `true` | Enable EMA smoothing on the estimated depth Z. |
| `hole_depth_smoothing_alpha` | float | `0.25` | `0.25` | EMA weight for the new depth sample. |
| `hole_depth_jump_gate_m` | float | `0.08` | `0.08` | Depth jump (m) at which the EMA resets instead of smoothing. |

**`hole_depth_policy` values:**

- `plane` — existing behavior: rim percentile filtering → robust plane fit →
  ray–plane intersection; falls back to ring median if plane fit fails.
- `median` — skip plane fitting entirely; use `np.median` of ring-rim depths.
  Stable across noisy/partial rings. Recommended for `wheel_hole`.
- `percentile` — skip plane fitting; use `hole_depth_percentile`-th percentile
  of ring-rim depths.

## Dependencies

`perception_part_detector` must be built in the same workspace. The package also
requires `cv_bridge`, `message_filters`, `tf2_ros`, and `tf2_geometry_msgs`.

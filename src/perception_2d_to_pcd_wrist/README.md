# perception_2d_to_pcd_wrist

Wrist right RGB-D camera detections are converted into one task-aware grasp
target pose.

## Node

| Node | Output | Role |
| --- | --- | --- |
| `wrist_task_grasp_planner_node` | `/perception/wrist/target_one_pose` (`geometry_msgs/PoseStamped`), `/perception/wrist/target_one_detection` (`std_msgs/String` JSON) | Pick one 3D target from wrist detections |

The planner subscribes to `/perception/task_list` and only considers parts with
a positive remaining count. Among valid wrist detections, the score combines
detection confidence and proximity to the configured arm reference point.

Depth is reprojected into the wrist RGB image plane before each detection mask is
converted to a 3D point set.

With temporal smoothing enabled, the planner waits until the same class and
nearby 3D location are observed repeatedly before publishing. The latest
selected pose is republished at `republish_last_pose_hz` while it remains within
`hold_last_pose_sec`.

## Key Parameters

| Parameter | Default | Purpose |
| --- | --- | --- |
| `min_score_to_publish` | `0.20` | Minimum weighted score to publish |
| `weight_confidence` | `0.45` | Confidence contribution to score |
| `weight_arm_proximity` | `0.55` | Arm-reference proximity contribution to score |
| `arm_reference_frame` | `""` | Optional frame for `arm_reference_xyz`; empty means `base_frame` |
| `arm_reference_xyz` | `[0.0, 0.0, 0.0]` | Arm reference point |
| `temporal_smoothing_enable` | `true` | Require repeated observations before lock |
| `temporal_window_sec` | `0.8` | Temporal history window |
| `temporal_min_observations` | `2` | Minimum observations inside the window |
| `republish_last_pose_hz` | `2.0` | Last pose republish rate |
| `hold_last_pose_sec` | `2.0` | Last pose lifetime; `<=0` means indefinite |

## Run

```bash
ros2 launch perception_2d_to_pcd_wrist wrist_task_grasp_planner.launch.py
```

Common overrides are forwarded by the launch file:

```bash
ros2 launch perception_2d_to_pcd_wrist wrist_task_grasp_planner.launch.py \
  temporal_smoothing_enable:=false \
  republish_last_pose_hz:=5.0 \
  allow_all_without_task:=true
```

`wrist_all.launch.py` is kept as an alias that launches the same planner.

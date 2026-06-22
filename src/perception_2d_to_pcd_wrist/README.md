# perception_2d_to_pcd_wrist

Wrist right RGB-D camera detections are converted into one task-aware grasp
target pose.

## Node

| Node | Output | Role |
| --- | --- | --- |
| `wrist_task_grasp_planner_node` | `/perception/wrist/target_one_pose` (`geometry_msgs/PoseStamped`), `/perception/wrist/target_one_detection` (`std_msgs/String` JSON) | Pick one 3D target from wrist detections |

The planner subscribes to `/perception/task_list` and only considers parts with
a positive remaining count. Among valid wrist detections, the score is based on:

1. detection confidence
2. distance to the configured arm reference point

Depth is reprojected into the wrist RGB image plane before each detection mask is
converted to a 3D point set.

## Run

```bash
ros2 launch perception_2d_to_pcd_wrist wrist_task_grasp_planner.launch.py
```

`wrist_all.launch.py` is kept as an alias that launches the same planner.

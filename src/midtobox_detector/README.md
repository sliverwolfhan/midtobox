# `midtobox_detector`

C++17 ROS 2 Humble perception package for detecting isolated `35 × 35 × 35 cm` cubes (configurable) from **Livox Mid-360** point clouds using **PCL + Eigen only (zero OpenCV)**.

## Features

- **Dual Livox input formats**: `sensor_msgs/msg/PointCloud2` (`xfer_format=0`, default) and `livox_ros_driver2/msg/CustomMsg` (`xfer_format=1`), parsing Mid-360 `tag` flags and rejecting `(0,0,0)` / near-range returns before PCL conversion.
- **Inverted or upright mounting**: One-click `mounting.inverted: true/false` toggle (`roll=+180°` when inverted) and configurable height (`mounting.z`, default `0.70 m`), with optional built-in static TF `base_footprint -> livox_frame`.
- **Automatic motion-compensated sliding window**: Accumulates `200–300 ms` of point clouds using `odom -> base_footprint` TF when available, and gracefully falls back to static accumulation when no `odom` TF is published.
- **Known-size geometric fitting**:
  - Ego-vehicle 3D CropBox + workspace ROI crop in leveled `base_footprint`.
  - Tilt-constrained RANSAC ground plane removal & leveling (`SACMODEL_PERPENDICULAR_PLANE`).
  - ROR + per-cluster SOR suppression of Mid-360 edge-dragging ("veil") outliers.
  - PCL Euclidean clustering separating cubes `>= 10 cm` apart, with explicit oversized touching/stacked rejection (`XY > 0.50 m` or `Z > 0.45 m`) flagged via debug markers.
  - Zhang et al. (IV 2017) L-shape closeness search + explicit 2-face L-shape (coupled orthogonal Huber IRLS) vs 1-face front-only (RANSAC+Huber normal push + symmetric `p5/p95` midpoint, eliminating single-face corner-flip bias) + optional pure 2D SE(2) Huber refinement.
- **Tracking & locking**: Hungarian assignment + 6D constant-velocity Kalman filter + 4-fold `[-45°, +45°)` square yaw canonicalization with temporal unwrapping + nearest-target primary selection with hysteresis + quality lock gating.
- **Configurable output frame**: Defaults to `base_footprint`; switchable via YAML or launch arg to `livox_frame` or `odom`.

## Build

```bash
source /opt/ros/humble/setup.bash
cd /home/xjh/Desktop/midtobox
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

## Launch

```bash
# Detector only (e.g. bag replay or external driver)
ros2 launch midtobox_detector midtobox.launch.py

# Detector + livox_ros_driver2 (PointCloud2, xfer_format=0) + RViz2
ros2 launch midtobox_detector midtobox.launch.py launch_driver:=true launch_rviz:=true

# Detector + livox_ros_driver2 (CustomMsg, xfer_format=1)
ros2 launch midtobox_detector midtobox.launch.py launch_driver:=true xfer_format:=1

# Switch output frame or mounting mode on the fly
ros2 launch midtobox_detector midtobox.launch.py output_frame:=livox_frame inverted:=true mounting_z:=0.70
```

## Published Topics

| Topic | Type | Description |
|---|---|---|
| `/midtobox/target_pose` | `geometry_msgs/msg/PoseStamped` | Primary tracked cube centroid & orientation (published when valid target exists) |
| `/midtobox/detections` | `vision_msgs/msg/Detection3DArray` | All confirmed valid cubes simultaneously |
| `/midtobox/locked` | `std_msgs/msg/Bool` | `true` iff primary cube meets point count, inlier ratio, RMSE, and stability gates |
| `/midtobox/markers` | `visualization_msgs/msg/MarkerArray` | Valid cubes, centroids, heading arrows, labels, and red rejected oversized cluster markers |
| `/midtobox/debug/accumulated_cloud` | `sensor_msgs/msg/PointCloud2` | Optional ROI-cropped accumulated cloud (`debug.publish_clouds:=true`) |
| `/midtobox/debug/non_ground_cloud` | `sensor_msgs/msg/PointCloud2` | Optional filtered non-ground cloud (`debug.publish_clouds:=true`) |

## Attribution

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for MIT / Apache-2.0 attributions (`SS47816/lidar_obstacle_detector`, `autoware_shape_estimation`, `livox_ros_driver2`).

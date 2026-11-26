# Third-Party Open-Source Algorithm Attributions & Licenses

`midtobox_detector` is a clean C++17 (PCL + Eigen, zero OpenCV) implementation
that adapts and fixes core algorithmic ideas from the following open-source
projects:

## 1. SS47816/lidar_obstacle_detector (MIT License)

- **Repository**: https://github.com/SS47816/lidar_obstacle_detector
- **License**: MIT License
- **Adapted components**:
  - Two-stage coarse-to-fine L-shape closeness fitting (`box_fitter.cpp`),
    originally based on Xiao Zhang, Wenda Xu, Chiyu Dong, and John M. Dolan,
    *"Efficient L-Shape Fitting for Vehicle Detection Using Laser Scanners"*,
    IEEE Intelligent Vehicles Symposium (IV), 2017.
  - Tilt-constrained RANSAC ground plane segmentation (`ground_segmenter.cpp`),
    simplified to leveled Z-band + `pcl::SACMODEL_PERPENDICULAR_PLANE` without
    Patchwork++ or automotive-scale parameter clamps.
  - PCL KD-tree Euclidean cluster extraction (`detector_pipeline.cpp`).
  - Constant-velocity Kalman filter + Hungarian linear assignment tracking
    architecture (`tracker.cpp`), extended with 4-fold square yaw unwrapping
    and nearest-target hysteresis.

## 2. autowarefoundation/autoware.universe — autoware_shape_estimation (Apache-2.0 License)

- **Repository**: https://github.com/autowarefoundation/autoware.universe
  (`perception/autoware_shape_estimation`)
- **License**: Apache License 2.0
- **Adapted components & bug fixes**:
  - Known-size partial-view centroid correction (`box_fitter.cpp`), extending
    observed face geometry by known box dimensions `(L, W, H)`.
  - **Bug fix over stock `autoware_shape_estimation`**: Explicitly distinguishes
    2-face L-shape views from 1-face front-only views. On 1-face views, instead
    of anchoring to an arbitrary near corner (which causes corner-flip bias on
    partial single-face observations), `midtobox_detector` fits the front plane
    via RANSAC + Huber IRLS, computes the symmetric `p5/p95` lateral midpoint
    along the face tangent, and pushes inward by `L/2 + range_bias_offset` along
    the sensor-facing normal.

## 3. Livox-SDK/livox_ros_driver2 (MIT License)

- **Location in workspace**: `src/livox_ros_driver2`
- **License**: MIT License (see `src/livox_ros_driver2/LICENSE.txt`)
- Used unchanged as a workspace ROS 2 dependency for `CustomMsg` / `CustomPoint`
  definitions and optional hardware driver launch.

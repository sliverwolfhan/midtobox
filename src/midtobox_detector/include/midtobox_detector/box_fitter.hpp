// Copyright 2026 midtobox contributors
// SPDX-License-Identifier: Apache-2.0
//
// Known-size 3D cube geometric fitting from partial-view LiDAR point clusters.
// Adapts:
//   - Zhang et al. (IV 2017) L-shape closeness fitting from
//     SS47816/lidar_obstacle_detector (MIT License)
//   - Known-size partial-view centroid correction from
//     autoware_shape_estimation (Apache-2.0 License), with explicit 2-face vs
//     1-face branching to eliminate single-face corner-flip bias.

#pragma once

#include <Eigen/Core>

#include <string>
#include <vector>

#include "midtobox_detector/types.hpp"

namespace midtobox
{

struct Obb2D
{
  double major_span{0.0};
  double minor_span{0.0};
  double yaw{0.0};
  Eigen::Vector2d center{Eigen::Vector2d::Zero()};
};

class BoxFitter
{
public:
  /// Compute robust 2D oriented bounding box (minimal-area scan using p1/p99 quantiles).
  static Obb2D computeRobustObb2D(const std::vector<Eigen::Vector2d> & pts);

  /// Compute linear-interpolated quantile q in [0, 1].
  static double computeQuantile(std::vector<double> values, double q);

  /// Zhang IV 2017 closeness criterion L-shape orientation search in [0, pi/2).
  static double searchLShapeOrientation(
    const std::vector<Eigen::Vector2d> & pts,
    const BoxFittingConfig & cfg);

  /// Fit known-size box to a cluster in the ground-leveled frame G (z = 0 on ground).
  /// `sensor_origin_ground` is the LiDAR origin expressed in frame G (used to orient outward normals).
  /// Outputs `detection_ground` expressed in frame G.
  /// Returns true on valid fit; on failure returns false and populates `reject_reason`.
  static bool fitBoxInGroundFrame(
    const std::vector<Eigen::Vector3d> & cluster_pts,
    const Eigen::Vector3d & sensor_origin_ground,
    const BoxFittingConfig & cfg,
    const GateConfig & gate_cfg,
    BoxDetection & detection_ground,
    std::string & reject_reason);
};

}  // namespace midtobox

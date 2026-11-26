// Copyright 2026 midtobox contributors
// SPDX-License-Identifier: Apache-2.0
//
// Core geometry models, detection/tracking result structures, and yaw utilities.
// Tunable YAML parameter structs live in config_types.hpp (loaded from config/midtobox.yaml).

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "midtobox_detector/config_types.hpp"

namespace midtobox
{

/// Normalized plane equation a*x + b*y + c*z + d = 0 with ||(a,b,c)|| = 1 and c > 0.
struct PlaneModel
{
  double a{0.0};
  double b{0.0};
  double c{1.0};
  double d{0.0};
  bool from_ransac{false};
  int inlier_count{0};

  Eigen::Vector3d normal() const { return Eigen::Vector3d(a, b, c); }

  double signedDistance(const Eigen::Vector3d & p) const
  {
    return a * p.x() + b * p.y() + c * p.z() + d;
  }

  /// Minimal rotation R_level that aligns the plane normal to +Z (0,0,1).
  Eigen::Matrix3d levelRotation() const
  {
    return Eigen::Quaterniond::FromTwoVectors(normal().normalized(), Eigen::Vector3d::UnitZ())
           .toRotationMatrix();
  }

  /// Map a point from base_footprint into the ground-leveled frame G (where z_G == signedDistance).
  Eigen::Vector3d toGroundFrame(const Eigen::Vector3d & p_base) const
  {
    const Eigen::Matrix3d R = levelRotation();
    return R * p_base + Eigen::Vector3d(0.0, 0.0, d);
  }

  /// Inverse map from ground-leveled frame G back to base_footprint.
  Eigen::Vector3d fromGroundFrame(const Eigen::Vector3d & p_ground) const
  {
    const Eigen::Matrix3d R = levelRotation();
    return R.transpose() * p_ground - d * normal();
  }
};

enum class FitMode
{
  kSingleFace = 1,
  kTwoFaceLShape = 2
};

struct BoxDetection
{
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
  double yaw{0.0};  // canonical yaw in base_footprint
  Eigen::Vector3d size{0.35, 0.35, 0.35};

  FitMode fit_mode{FitMode::kSingleFace};
  bool has_top_face{false};
  int point_count{0};
  int side_point_count{0};
  int top_point_count{0};
  double inlier_ratio{0.0};
  double rmse{0.0};
  double score{0.0};
};

struct RejectedCluster
{
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};
  Eigen::Vector3d size{Eigen::Vector3d::Zero()};  // AABB/OBB span in base_footprint
  double yaw{0.0};
  int point_count{0};
  bool is_oversized{false};  // true for touching/stacked clusters flagged as oversized
  std::string reason;
};

struct TrackedBox
{
  uint64_t track_id{0};
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};
  Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
  double yaw{0.0};  // temporally unwrapped yaw
  Eigen::Vector3d size{0.35, 0.35, 0.35};

  FitMode fit_mode{FitMode::kSingleFace};
  bool has_top_face{false};
  int point_count{0};
  double inlier_ratio{0.0};
  double rmse{0.0};
  double score{0.0};

  int hits{0};
  int missed{0};
  bool confirmed{false};
  bool locked{false};
  bool is_primary{false};
  double pos_jitter{0.0};
  double yaw_jitter_deg{0.0};
};

struct PipelineResult
{
  PlaneModel ground_plane;
  std::vector<BoxDetection> raw_detections;
  std::vector<TrackedBox> tracks;
  std::vector<RejectedCluster> rejected_clusters;

  bool has_primary{false};
  TrackedBox primary_target;
  bool locked{false};

  int input_points{0};
  int roi_points{0};
  int non_ground_points{0};
  double compute_time_ms{0.0};
};

/// Canonicalize angle into [-period/2, +period/2).
/// For square boxes (period = pi/2), returns angle in [-pi/4, +pi/4).
inline double canonicalizeSymmetricYaw(double yaw, double period = kHalfPi)
{
  const double half = 0.5 * period;
  double a = std::fmod(yaw + half, period);
  if (a < 0.0) {
    a += period;
  }
  // Guard against floating-point rounding hitting exactly period
  if (a >= period) {
    a -= period;
  }
  return a - half;
}

/// Shortest signed difference (target - ref) modulo `period`, in [-period/2, +period/2).
inline double shortestSymmetricAngleDiff(double target, double ref, double period = kHalfPi)
{
  return canonicalizeSymmetricYaw(target - ref, period);
}

}  // namespace midtobox

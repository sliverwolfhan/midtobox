// Copyright 2026 midtobox contributors
// SPDX-License-Identifier: Apache-2.0
//
// Core configuration and result data structures for midtobox_detector.

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace midtobox
{

constexpr double kPi = 3.14159265358979323846;
constexpr double kHalfPi = kPi * 0.5;
constexpr double kQuarterPi = kPi * 0.25;
constexpr double kDegToRad = kPi / 180.0;
constexpr double kRadToDeg = 180.0 / kPi;

struct PreFilterConfig
{
  double min_range{0.25};
  double max_range{6.0};
  bool use_tag_filter{true};
  uint8_t tag_reject_mask{0x3F};  // rejects bits 0..5 (stray / edge-dragging / low-refl)
};

struct MountingConfig
{
  bool inverted{true};  // true => adds 180 deg roll before fine trim
  double x{0.0};
  double y{0.0};
  double z{0.70};
  double roll_deg{0.0};
  double pitch_deg{0.0};
  double yaw_deg{0.0};

  /// Compute T_base_lidar (transforms points from lidar_frame into base_footprint).
  Eigen::Isometry3d computeTransform() const
  {
    const double eff_roll = (inverted ? kPi : 0.0) + roll_deg * kDegToRad;
    const double pitch = pitch_deg * kDegToRad;
    const double yaw = yaw_deg * kDegToRad;

    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    const Eigen::AngleAxisd rz(yaw, Eigen::Vector3d::UnitZ());
    const Eigen::AngleAxisd ry(pitch, Eigen::Vector3d::UnitY());
    const Eigen::AngleAxisd rx(eff_roll, Eigen::Vector3d::UnitX());
    T.linear() = (rz * ry * rx).toRotationMatrix();
    T.translation() = Eigen::Vector3d(x, y, z);
    return T;
  }
};

struct CropConfig
{
  bool ego_enabled{true};
  double ego_min_x{-0.50};
  double ego_max_x{0.40};
  double ego_min_y{-0.45};
  double ego_max_y{0.45};
  double ego_min_z{-0.10};
  double ego_max_z{1.10};

  double roi_min_x{0.45};
  double roi_max_x{3.20};
  double roi_min_y{-1.20};
  double roi_max_y{1.20};
  double roi_min_z{-0.25};
  double roi_max_z{0.90};

  double voxel_leaf_size{0.0};  // <= 0 disables voxel downsampling
};

struct GroundConfig
{
  bool use_ransac{true};
  double band_min_z{-0.25};
  double band_max_z{0.25};
  double max_tilt_deg{8.0};
  double ransac_dist_thresh{0.025};
  int ransac_max_iter{200};
  int min_inliers{30};
  double max_plane_offset{0.20};
  double nominal_z{0.0};
  double clearance_height{0.045};
  bool level_to_ground{true};
};

struct OutlierFilterConfig
{
  bool ror_enabled{true};
  double ror_radius{0.06};
  int ror_min_neighbors{3};

  bool sor_enabled{true};
  int sor_mean_k{12};
  double sor_stddev_mul{2.5};
};

struct ClusterConfig
{
  double tolerance{0.06};
  int min_size{25};
  int max_size{25000};
};

struct GateConfig
{
  int min_points{35};
  double min_xy_span{0.18};
  double max_xy_span{0.50};
  double min_z{0.20};
  double max_z{0.45};
  double max_bottom_z{0.25};
  double height_consistency_tol{0.07};
  double min_inlier_ratio{0.50};
  double max_rmse{0.040};
};

struct BoxFittingConfig
{
  double length{0.35};
  double width{0.35};
  double height{0.35};
  double range_bias_offset{0.0};

  double closeness_d0{0.01};
  double coarse_step_deg{1.0};
  double fine_step_deg{0.1};

  double top_band_thickness{0.030};
  double side_min_z{0.05};
  double side_top_margin{0.045};
  int top_min_points{10};
  double top_max_z_mad{0.025};

  double face_band{0.035};
  double two_face_min_span{0.12};
  int two_face_min_pts{12};
  double two_face_min_excl_span{0.08};

  double huber_delta{0.015};
  int ransac_line_iters{100};
  double ransac_line_thresh{0.020};

  bool enable_se2_refine{true};
  int se2_max_iters{10};
  double se2_max_shift{0.04};
  double se2_max_angle_deg{6.0};

  double inlier_dist_thresh{0.030};
};

struct TrackerConfig
{
  double max_assoc_dist{0.25};
  int min_hits{2};
  int max_missed_frames{3};

  double kf_process_noise_pos{0.020};
  double kf_process_noise_vel{0.80};
  double kf_meas_noise_pos{0.015};
  double kf_vel_damping{0.95};
  double yaw_alpha{0.45};

  double switch_hysteresis_m{0.15};

  int lock_min_points{80};
  double lock_min_inlier_ratio{0.70};
  double lock_max_rmse{0.025};
  int lock_window_frames{3};
  double lock_max_pos_jitter{0.015};
  double lock_max_yaw_jitter_deg{3.0};
};

struct PipelineConfig
{
  CropConfig crop;
  GroundConfig ground;
  OutlierFilterConfig outlier;
  ClusterConfig cluster;
  GateConfig gate;
  BoxFittingConfig fitting;
  TrackerConfig tracker;
};

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

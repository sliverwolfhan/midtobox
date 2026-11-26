// Copyright 2026 midtobox contributors
// SPDX-License-Identifier: Apache-2.0
//
// Multi-object box tracker: Hungarian optimal assignment + 6D constant-velocity
// Kalman filter + 4-fold symmetric temporal yaw unwrapping + lock quality gating
// + nearest-target primary selector with hysteresis.
// Tracking architecture adapted from SS47816/lidar_obstacle_detector (MIT License).

#pragma once

#include <Eigen/Core>

#include <cstdint>
#include <deque>
#include <utility>
#include <vector>

#include "midtobox_detector/types.hpp"

namespace midtobox
{

/// Solve rectangular linear sum assignment (minimum total cost) using the
/// Kuhn-Munkres (Hungarian) algorithm in O(N^3).
/// Returns pairs of (row_idx, col_idx) whose cost <= max_cost.
std::vector<std::pair<size_t, size_t>> solveHungarianAssignment(
  const std::vector<std::vector<double>> & cost_matrix,
  double max_cost);

class BoxTracker
{
public:
  BoxTracker() = default;

  void reset();

  /// Transform existing track states, covariances, orientations, and stability
  /// histories from base_footprint(t_{k-1}) into base_footprint(t_k) using
  /// T_curr_prev = T_odom_base(t_k)^{-1} * T_odom_base(t_{k-1}).
  void applyEgoMotion(const Eigen::Isometry3d & T_curr_prev);

  /// Update tracker with current-frame detections (expressed in base_footprint).
  /// `stamp_sec` is monotonic timestamp in seconds.
  /// `is_square` selects 90-degree (pi/2) vs 180-degree (pi) yaw symmetry period.
  /// Optional `T_curr_prev` compensates robot ego-motion when odom TF is active.
  void update(
    const std::vector<BoxDetection> & detections,
    double stamp_sec,
    const TrackerConfig & cfg,
    bool is_square,
    std::vector<TrackedBox> & tracks_out,
    bool & has_primary_out,
    TrackedBox & primary_out,
    bool & locked_out,
    const Eigen::Isometry3d * T_curr_prev = nullptr);

private:
  struct TrackState
  {
    uint64_t id{0};

    // 6D Kalman state: [x, y, z, vx, vy, vz]
    Eigen::Matrix<double, 6, 1> x{Eigen::Matrix<double, 6, 1>::Zero()};
    Eigen::Matrix<double, 6, 6> P{Eigen::Matrix<double, 6, 6>::Identity()};

    // Temporally unwrapped yaw (initialized canonical in [-period/2, +period/2))
    double yaw{0.0};
    // Tilt component from latest measurement (pitch/roll relative to base_footprint)
    Eigen::Quaterniond latest_tilt{Eigen::Quaterniond::Identity()};

    Eigen::Vector3d size{0.35, 0.35, 0.35};
    FitMode fit_mode{FitMode::kSingleFace};
    bool has_top_face{false};
    int point_count{0};
    double inlier_ratio{0.0};
    double rmse{0.0};
    double score{0.0};

    int hits{0};
    int missed{0};

    // Sliding history of recent matched measurements (unwrapped) for lock jitter gating
    std::deque<Eigen::Vector3d> pos_history;
    std::deque<double> yaw_history;
  };

  void predictTrack(TrackState & trk, double dt, const TrackerConfig & cfg) const;
  void updateTrack(
    TrackState & trk,
    const BoxDetection & det,
    const TrackerConfig & cfg,
    double sym_period) const;
  void evaluateLock(
    const TrackState & trk,
    const TrackerConfig & cfg,
    bool & locked_out,
    double & pos_jitter_out,
    double & yaw_jitter_deg_out) const;

  std::vector<TrackState> tracks_;
  uint64_t next_track_id_{1};
  bool has_last_stamp_{false};
  double last_stamp_sec_{0.0};
  uint64_t primary_track_id_{0};
};

}  // namespace midtobox

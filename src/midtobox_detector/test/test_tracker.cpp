// Copyright 2026 midtobox contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "midtobox_detector/tracker.hpp"
#include "midtobox_detector/types.hpp"

namespace midtobox
{
namespace
{

BoxDetection makeDet(
  const Eigen::Vector3d & center,
  double yaw_deg,
  int pts = 180,
  double inlier_ratio = 0.90,
  double rmse = 0.010)
{
  BoxDetection d;
  d.center = center;
  d.yaw = canonicalizeSymmetricYaw(yaw_deg * kDegToRad, kHalfPi);
  d.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(d.yaw, Eigen::Vector3d::UnitZ()));
  d.size = Eigen::Vector3d(0.35, 0.35, 0.35);
  d.fit_mode = FitMode::kTwoFaceLShape;
  d.point_count = pts;
  d.inlier_ratio = inlier_ratio;
  d.rmse = rmse;
  d.score = 0.92;
  return d;
}

TEST(TrackerTest, HungarianAssignmentOptimalAndGated)
{
  // Greedy picks (0,0)=1.0 then (1,1)=10.0 (total 11.0).
  // Optimal Hungarian picks (0,1)=2.0 and (1,0)=2.0 (total 4.0).
  const std::vector<std::vector<double>> cost = {
    {1.0, 2.0},
    {2.0, 10.0}
  };
  const auto matches = solveHungarianAssignment(cost, 5.0);
  ASSERT_EQ(matches.size(), 2u);
  int matched_0 = -1;
  int matched_1 = -1;
  for (const auto & p : matches) {
    if (p.first == 0) matched_0 = static_cast<int>(p.second);
    if (p.first == 1) matched_1 = static_cast<int>(p.second);
  }
  EXPECT_EQ(matched_0, 1);
  EXPECT_EQ(matched_1, 0);

  // Rectangular matrix with max_cost gating
  const std::vector<std::vector<double>> rect = {
    {0.05, 0.80, 0.90},
    {0.90, 0.70, 0.12}
  };
  const auto rect_matches = solveHungarianAssignment(rect, 0.25);
  ASSERT_EQ(rect_matches.size(), 2u);
}

TEST(TrackerTest, TemporalYawUnwrappingAcrossFortyFiveBoundary)
{
  BoxTracker tracker;
  TrackerConfig cfg;
  cfg.min_hits = 2;
  cfg.lock_window_frames = 3;

  // Measurements oscillate across +45 / -45 deg (raw canonical values jump ~89 deg!)
  const std::vector<double> raw_yaws_deg = {44.4, -44.9, 44.7, -44.6, 44.8, -44.8};
  double prev_yaw_deg = 0.0;
  bool has_prev = false;
  bool locked_final = false;

  for (size_t k = 0; k < raw_yaws_deg.size(); ++k) {
    std::vector<TrackedBox> tracks;
    bool has_primary = false;
    TrackedBox primary;
    bool locked = false;

    tracker.update(
      {makeDet(Eigen::Vector3d(1.80, 0.0, 0.175), raw_yaws_deg[k])},
      0.1 * static_cast<double>(k),
      cfg, true, tracks, has_primary, primary, locked);

    if (has_primary) {
      const double cur_yaw_deg = primary.yaw * kRadToDeg;
      if (has_prev) {
        // Must NEVER experience a ~90 deg boundary jump!
        EXPECT_LT(std::abs(cur_yaw_deg - prev_yaw_deg), 2.0)
          << "Boundary jump at frame " << k << ": " << prev_yaw_deg << " -> " << cur_yaw_deg;
      }
      prev_yaw_deg = cur_yaw_deg;
      has_prev = true;
      locked_final = locked;
    }
  }
  EXPECT_TRUE(locked_final);
}

TEST(TrackerTest, MultiTargetHysteresisPrimarySelection)
{
  BoxTracker tracker;
  TrackerConfig cfg;
  cfg.min_hits = 2;
  cfg.switch_hysteresis_m = 0.15;

  std::vector<TrackedBox> tracks;
  bool has_primary = false;
  TrackedBox primary;
  bool locked = false;

  // Frame 0 & 1: Cube A at 1.50 m, Cube B at 1.65 m -> A becomes primary
  for (int k = 0; k < 2; ++k) {
    tracker.update(
      {makeDet({1.50, -0.35, 0.175}, 10.0), makeDet({1.65, 0.35, 0.175}, -15.0)},
      0.1 * k, cfg, true, tracks, has_primary, primary, locked);
  }
  ASSERT_TRUE(has_primary);
  ASSERT_EQ(tracks.size(), 2u);
  const uint64_t id_a = primary.track_id;
  EXPECT_NEAR(primary.center.x(), 1.50, 0.05);

  // Frame 2: Cube B moves slightly closer (1.42 m vs 1.50 m -> diff 0.08 m < 0.15 m hysteresis)
  // Primary MUST stay Cube A!
  tracker.update(
    {makeDet({1.50, -0.35, 0.175}, 10.0), makeDet({1.42, 0.35, 0.175}, -15.0)},
    0.2, cfg, true, tracks, has_primary, primary, locked);
  ASSERT_TRUE(has_primary);
  EXPECT_EQ(primary.track_id, id_a);

  // Frame 3 & 4: Cube B moves much closer (1.25 m vs 1.50 m -> diff 0.25 m > 0.15 m hysteresis)
  // Primary MUST switch to Cube B!
  tracker.update(
    {makeDet({1.50, -0.35, 0.175}, 10.0), makeDet({1.25, 0.35, 0.175}, -15.0)},
    0.3, cfg, true, tracks, has_primary, primary, locked);
  tracker.update(
    {makeDet({1.50, -0.35, 0.175}, 10.0), makeDet({1.20, 0.35, 0.175}, -15.0)},
    0.4, cfg, true, tracks, has_primary, primary, locked);
  ASSERT_TRUE(has_primary);
  EXPECT_NE(primary.track_id, id_a);
}

TEST(TrackerTest, LockGatingEnforcesQualityCriteria)
{
  BoxTracker tracker;
  TrackerConfig cfg;
  cfg.min_hits = 2;
  cfg.lock_window_frames = 3;
  cfg.lock_min_points = 80;
  cfg.lock_min_inlier_ratio = 0.70;
  cfg.lock_max_rmse = 0.025;
  cfg.lock_max_pos_jitter = 0.015;
  cfg.lock_max_yaw_jitter_deg = 3.0;

  std::vector<TrackedBox> tracks;
  bool has_primary = false;
  TrackedBox primary;
  bool locked = false;

  // Low point count (60 < 80) -> must NOT lock even after 4 stable frames
  for (int k = 0; k < 4; ++k) {
    tracker.update(
      {makeDet({1.80, 0.0, 0.175}, 5.0, 60, 0.90, 0.010)},
      0.1 * k, cfg, true, tracks, has_primary, primary, locked);
  }
  EXPECT_TRUE(has_primary);
  EXPECT_FALSE(locked);

  // High RMSE (0.035 > 0.025) -> must NOT lock
  tracker.update(
    {makeDet({1.80, 0.0, 0.175}, 5.0, 150, 0.90, 0.035)},
    0.4, cfg, true, tracks, has_primary, primary, locked);
  EXPECT_FALSE(locked);

  // Clean high-quality measurement -> locks immediately once quality gates pass
  tracker.update(
    {makeDet({1.80, 0.0, 0.175}, 5.0, 150, 0.90, 0.010)},
    0.5, cfg, true, tracks, has_primary, primary, locked);
  EXPECT_TRUE(locked);

  // Timestamp jump backwards -> cleanly resets tracker
  tracker.update(
    {}, -5.0, cfg, true, tracks, has_primary, primary, locked);
  EXPECT_FALSE(has_primary);
  EXPECT_FALSE(locked);
  EXPECT_TRUE(tracks.empty());
}

TEST(TrackerTest, EgoMotionCompensationTracksThroughFastTurns)
{
  BoxTracker tracker;
  TrackerConfig cfg;
  cfg.min_hits = 2;
  cfg.lock_window_frames = 3;
  cfg.max_assoc_dist = 0.25;

  // Stationary box in odom frame while robot drives at 0.4 m/s and turns at 1.4 rad/s
  // (apparent box displacement in base_footprint is ~0.28 m/frame > max_assoc_dist without comp!)
  const Eigen::Vector3d box_odom(2.00, 0.0, 0.175);
  const double box_yaw_odom = 20.0 * kDegToRad;

  Eigen::Isometry3d prev_T_odom_base = Eigen::Isometry3d::Identity();
  bool has_prev = false;
  uint64_t first_id = 0;
  bool last_locked = false;

  for (int k = 0; k < 8; ++k) {
    const double t = 0.1 * k;
    const double rx = 0.4 * t;
    const double ryaw = -0.45 + 1.4 * t;

    Eigen::Isometry3d T_odom_base = Eigen::Isometry3d::Identity();
    T_odom_base.linear() = Eigen::AngleAxisd(ryaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    T_odom_base.translation() = Eigen::Vector3d(rx, 0.0, 0.0);

    const Eigen::Vector3d c_base = T_odom_base.inverse() * box_odom;
    const double yaw_base_deg = (box_yaw_odom - ryaw) * kRadToDeg;

    Eigen::Isometry3d T_curr_prev = Eigen::Isometry3d::Identity();
    const Eigen::Isometry3d * ego_ptr = nullptr;
    if (has_prev) {
      T_curr_prev = T_odom_base.inverse() * prev_T_odom_base;
      ego_ptr = &T_curr_prev;
    }

    std::vector<TrackedBox> tracks;
    bool has_primary = false;
    TrackedBox primary;
    bool locked = false;

    tracker.update(
      {makeDet(c_base, yaw_base_deg)}, t, cfg, true,
      tracks, has_primary, primary, locked, ego_ptr);

    prev_T_odom_base = T_odom_base;
    has_prev = true;

    if (k >= 1) {
      ASSERT_TRUE(has_primary) << "Lost track at frame " << k;
      if (first_id == 0) {
        first_id = primary.track_id;
      } else {
        EXPECT_EQ(primary.track_id, first_id) << "Track ID broke at frame " << k;
      }
      EXPECT_LT((primary.center - c_base).norm(), 0.005) << "frame " << k;
      const double yaw_err_deg =
        std::abs(shortestSymmetricAngleDiff(primary.yaw, box_yaw_odom - ryaw, kHalfPi)) * kRadToDeg;
      EXPECT_LT(yaw_err_deg, 0.5) << "frame " << k;
      last_locked = locked;
    }
  }
  EXPECT_TRUE(last_locked);
}

}  // namespace
}  // namespace midtobox

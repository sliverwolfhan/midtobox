// Copyright 2026 midtobox contributors
// SPDX-License-Identifier: Apache-2.0

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "midtobox_detector/box_fitter.hpp"
#include "midtobox_detector/types.hpp"

namespace midtobox
{
namespace
{

// Generate partial-view surface points of a cube visible from `sensor_origin`
// (faces whose outward normal satisfies n · (sensor - face_center) > 0.05).
std::vector<Eigen::Vector3d> sampleVisibleCubePoints(
  const Eigen::Vector3d & center,
  double yaw,
  const Eigen::Vector3d & size,
  const Eigen::Vector3d & sensor_origin,
  double noise_std,
  int pts_per_face,
  unsigned int seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uni(-0.5, 0.5);
  std::normal_distribution<double> gauss(0.0, noise_std);

  const Eigen::Matrix3d R = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  std::vector<Eigen::Vector3d> pts;

  // 4 side faces + top face (+Z)
  struct FaceSpec { Eigen::Vector3d normal_local; int u_axis; int v_axis; };
  const std::vector<FaceSpec> faces = {
    {{1, 0, 0}, 1, 2},
    {{-1, 0, 0}, 1, 2},
    {{0, 1, 0}, 0, 2},
    {{0, -1, 0}, 0, 2},
    {{0, 0, 1}, 0, 1},
  };

  for (const auto & f : faces) {
    const Eigen::Vector3d n_world = R * f.normal_local;
    Eigen::Vector3d face_offset_local = f.normal_local.cwiseProduct(0.5 * size);
    const Eigen::Vector3d face_center = center + R * face_offset_local;
    const Eigen::Vector3d to_sensor = (sensor_origin - face_center).normalized();
    const double cos_inc = n_world.dot(to_sensor);
    if (cos_inc <= 0.12) {
      continue;  // back-facing or extreme grazing
    }

    const int n_pts = std::max(15, static_cast<int>(std::round(pts_per_face * cos_inc)));
    for (int i = 0; i < n_pts; ++i) {
      Eigen::Vector3d p_local = face_offset_local;
      p_local[f.u_axis] = uni(rng) * size[f.u_axis];
      p_local[f.v_axis] = uni(rng) * size[f.v_axis];
      Eigen::Vector3d p_world = center + R * p_local;
      // Add range noise along sensor ray
      const Eigen::Vector3d ray = (p_world - sensor_origin).normalized();
      p_world += ray * gauss(rng);
      if (p_world.z() >= 0.045) {
        pts.push_back(p_world);
      }
    }
  }
  return pts;
}

TEST(BoxFitterTest, YawCanonicalizationAndShortestDiff)
{
  EXPECT_NEAR(canonicalizeSymmetricYaw(0.0), 0.0, 1e-9);
  EXPECT_NEAR(canonicalizeSymmetricYaw(44.0 * kDegToRad), 44.0 * kDegToRad, 1e-9);
  EXPECT_NEAR(canonicalizeSymmetricYaw(45.0 * kDegToRad), -45.0 * kDegToRad, 1e-9);
  EXPECT_NEAR(canonicalizeSymmetricYaw(-45.0 * kDegToRad), -45.0 * kDegToRad, 1e-9);
  EXPECT_NEAR(canonicalizeSymmetricYaw(-46.0 * kDegToRad), 44.0 * kDegToRad, 1e-9);
  EXPECT_NEAR(canonicalizeSymmetricYaw(90.0 * kDegToRad), 0.0, 1e-9);
  EXPECT_NEAR(canonicalizeSymmetricYaw(180.0 * kDegToRad), 0.0, 1e-9);

  // Crossing +45/-45 boundary
  const double diff = shortestSymmetricAngleDiff(-44.7 * kDegToRad, 44.5 * kDegToRad, kHalfPi);
  EXPECT_NEAR(diff * kRadToDeg, 0.8, 1e-6);
}

TEST(BoxFitterTest, SingleFaceFrontOnlyZeroBias)
{
  const Eigen::Vector3d sensor(0.0, 0.0, 0.70);
  const Eigen::Vector3d gt_center(1.80, 0.05, 0.175);
  const Eigen::Vector3d size(0.35, 0.35, 0.35);

  for (double gt_yaw_deg : {0.0, 4.0, -4.0}) {
    const double gt_yaw = gt_yaw_deg * kDegToRad;
    auto pts = sampleVisibleCubePoints(gt_center, gt_yaw, size, sensor, 0.012, 220, 101);
    BoxFittingConfig cfg;
    GateConfig gate;
    BoxDetection det;
    std::string reason;

    ASSERT_TRUE(BoxFitter::fitBoxInGroundFrame(pts, sensor, cfg, gate, det, reason))
      << "yaw=" << gt_yaw_deg << " reason=" << reason;
    EXPECT_EQ(det.fit_mode, FitMode::kSingleFace);
    EXPECT_TRUE(det.has_top_face);
    EXPECT_LT((det.center - gt_center).norm(), 0.015)
      << "est=" << det.center.transpose() << " gt=" << gt_center.transpose();
    const double yaw_err_deg =
      std::abs(shortestSymmetricAngleDiff(det.yaw, gt_yaw, kHalfPi)) * kRadToDeg;
    EXPECT_LT(yaw_err_deg, 2.5);
  }
}

TEST(BoxFitterTest, SingleFaceRangeBiasOffsetPreservedWithSE2)
{
  const Eigen::Vector3d sensor(0.0, 0.0, 0.70);
  const Eigen::Vector3d gt_center(1.80, 0.0, 0.175);
  const Eigen::Vector3d size(0.35, 0.35, 0.35);
  auto pts = sampleVisibleCubePoints(gt_center, 0.0, size, sensor, 0.005, 220, 202);

  BoxFittingConfig cfg0;
  cfg0.enable_se2_refine = true;
  cfg0.range_bias_offset = 0.0;

  BoxFittingConfig cfg_bias = cfg0;
  cfg_bias.range_bias_offset = 0.015;  // +1.5 cm inward material offset

  GateConfig gate;
  BoxDetection det0, det_bias;
  std::string reason;
  ASSERT_TRUE(BoxFitter::fitBoxInGroundFrame(pts, sensor, cfg0, gate, det0, reason));
  ASSERT_TRUE(BoxFitter::fitBoxInGroundFrame(pts, sensor, cfg_bias, gate, det_bias, reason));

  // Front normal points along -X toward sensor, so +bias pushes center along +X by +0.015 m
  EXPECT_NEAR(det_bias.center.x() - det0.center.x(), 0.015, 0.002);
  EXPECT_NEAR(det_bias.center.y() - det0.center.y(), 0.0, 0.002);
}

TEST(BoxFitterTest, TwoFaceLShapeObliqueAngles)
{
  const Eigen::Vector3d sensor(0.0, 0.0, 0.70);
  const Eigen::Vector3d gt_center(1.65, -0.20, 0.175);
  const Eigen::Vector3d size(0.35, 0.35, 0.35);

  for (double gt_yaw_deg : {22.0, 35.0, -32.0, 44.5, -44.5}) {
    const double gt_yaw = gt_yaw_deg * kDegToRad;
    auto pts = sampleVisibleCubePoints(gt_center, gt_yaw, size, sensor, 0.012, 240, 303);
    BoxFittingConfig cfg;
    GateConfig gate;
    BoxDetection det;
    std::string reason;

    ASSERT_TRUE(BoxFitter::fitBoxInGroundFrame(pts, sensor, cfg, gate, det, reason))
      << "yaw=" << gt_yaw_deg << " reason=" << reason;
    EXPECT_EQ(det.fit_mode, FitMode::kTwoFaceLShape) << "yaw=" << gt_yaw_deg;
    EXPECT_LT((det.center - gt_center).norm(), 0.015)
      << "yaw=" << gt_yaw_deg << " est=" << det.center.transpose();
    const double yaw_err_deg =
      std::abs(shortestSymmetricAngleDiff(det.yaw, gt_yaw, kHalfPi)) * kRadToDeg;
    EXPECT_LT(yaw_err_deg, 1.8) << "yaw=" << gt_yaw_deg;
    EXPECT_GE(det.yaw, -kQuarterPi - 1e-9);
    EXPECT_LT(det.yaw, kQuarterPi + 1e-9);
  }
}

TEST(BoxFitterTest, EdgeDraggingVeilOutliersRobustness)
{
  const Eigen::Vector3d sensor(0.0, 0.0, 0.70);
  const Eigen::Vector3d gt_center(1.80, 0.0, 0.175);
  const Eigen::Vector3d size(0.35, 0.35, 0.35);
  auto pts = sampleVisibleCubePoints(gt_center, 0.0, size, sensor, 0.010, 220, 404);

  // Inject 25 edge-dragging "veil" outliers stretching behind the right edge (+y = +0.175)
  std::mt19937 rng(99);
  std::uniform_real_distribution<double> depth_trail(0.03, 0.14);
  std::uniform_real_distribution<double> z_dist(0.06, 0.32);
  for (int i = 0; i < 25; ++i) {
    const double dx = depth_trail(rng);
    const double x = gt_center.x() - 0.175 + dx;
    const double y = gt_center.y() + 0.175 + 0.015 + 0.08 * dx;
    pts.emplace_back(x, y, z_dist(rng));
  }

  BoxFittingConfig cfg;
  GateConfig gate;
  BoxDetection det;
  std::string reason;
  ASSERT_TRUE(BoxFitter::fitBoxInGroundFrame(pts, sensor, cfg, gate, det, reason))
    << "reason=" << reason;
  EXPECT_LT((det.center - gt_center).norm(), 0.020);
  const double yaw_err_deg =
    std::abs(shortestSymmetricAngleDiff(det.yaw, 0.0, kHalfPi)) * kRadToDeg;
  EXPECT_LT(yaw_err_deg, 3.0);
}

TEST(BoxFitterTest, HeightMismatchRejected)
{
  const Eigen::Vector3d sensor(0.0, 0.0, 0.70);
  // Box elevated on a 15 cm pedestal -> top at 0.50 m vs expected 0.35 m
  const Eigen::Vector3d gt_center(1.80, 0.0, 0.325);
  const Eigen::Vector3d size(0.35, 0.35, 0.35);
  auto pts = sampleVisibleCubePoints(gt_center, 0.2, size, sensor, 0.008, 200, 505);

  BoxFittingConfig cfg;
  GateConfig gate;
  BoxDetection det;
  std::string reason;
  EXPECT_FALSE(BoxFitter::fitBoxInGroundFrame(pts, sensor, cfg, gate, det, reason));
  EXPECT_EQ(reason, "height_mismatch");
}

TEST(BoxFitterTest, EmptyAndTinyClusterRejected)
{
  BoxFittingConfig cfg;
  GateConfig gate;
  BoxDetection det;
  std::string reason;
  EXPECT_FALSE(BoxFitter::fitBoxInGroundFrame({}, Eigen::Vector3d(0, 0, 0.7), cfg, gate, det, reason));
  EXPECT_EQ(reason, "too_few_points");
}

TEST(BoxFitterTest, TwoFaceRangeBiasOffsetRadialUnitPush)
{
  const Eigen::Vector3d sensor(0.0, 0.0, 0.70);
  const Eigen::Vector3d gt_center(1.80, 0.0, 0.175);
  const Eigen::Vector3d size(0.35, 0.35, 0.35);

  for (double yaw_deg : {20.0, 45.0}) {
    auto pts = sampleVisibleCubePoints(gt_center, yaw_deg * kDegToRad, size, sensor, 0.004, 240, 606);

    BoxFittingConfig cfg0;
    cfg0.range_bias_offset = 0.0;
    BoxFittingConfig cfg_bias = cfg0;
    cfg_bias.range_bias_offset = 0.020;  // +2.0 cm radial range offset

    GateConfig gate;
    BoxDetection det0, det_bias;
    std::string reason;
    ASSERT_TRUE(BoxFitter::fitBoxInGroundFrame(pts, sensor, cfg0, gate, det0, reason));
    ASSERT_TRUE(BoxFitter::fitBoxInGroundFrame(pts, sensor, cfg_bias, gate, det_bias, reason));
    ASSERT_EQ(det0.fit_mode, FitMode::kTwoFaceLShape);

    // Sensor is at (0,0), box at (1.80,0) -> radial push must be (+0.020, 0.0), NOT sqrt(2) diagonal
    EXPECT_NEAR(det_bias.center.x() - det0.center.x(), 0.020, 0.002) << "yaw=" << yaw_deg;
    EXPECT_NEAR(det_bias.center.y() - det0.center.y(), 0.0, 0.002) << "yaw=" << yaw_deg;
  }
}

TEST(BoxFitterTest, UprightNoTopFaceLShapeZeroZBias)
{
  // Sensor at z = 0.25 m (below 0.35 m cube top) -> zero top-face points
  const Eigen::Vector3d sensor(0.0, 0.0, 0.25);
  const Eigen::Vector3d gt_center(1.75, -0.15, 0.175);
  const Eigen::Vector3d size(0.35, 0.35, 0.35);
  auto pts = sampleVisibleCubePoints(gt_center, 25.0 * kDegToRad, size, sensor, 0.012, 240, 707);

  BoxFittingConfig cfg;
  GateConfig gate;
  BoxDetection det;
  std::string reason;
  ASSERT_TRUE(BoxFitter::fitBoxInGroundFrame(pts, sensor, cfg, gate, det, reason)) << reason;
  EXPECT_EQ(det.fit_mode, FitMode::kTwoFaceLShape);
  EXPECT_FALSE(det.has_top_face);
  EXPECT_NEAR(det.center.z(), 0.175, 0.002);
  EXPECT_LT((det.center - gt_center).norm(), 0.012);
}

TEST(BoxFitterTest, ShortBoxWithoutTopFaceRejected)
{
  // 23 cm short crate viewed from upright sensor at z = 0.20 m (below top face)
  const Eigen::Vector3d sensor(0.0, 0.0, 0.20);
  const Eigen::Vector3d short_center(1.60, 0.0, 0.115);
  const Eigen::Vector3d short_size(0.35, 0.35, 0.23);
  auto pts = sampleVisibleCubePoints(short_center, 0.0, short_size, sensor, 0.008, 220, 808);

  BoxFittingConfig cfg;  // expects 0.35 m cube height
  GateConfig gate;
  BoxDetection det;
  std::string reason;
  EXPECT_FALSE(BoxFitter::fitBoxInGroundFrame(pts, sensor, cfg, gate, det, reason));
  EXPECT_EQ(reason, "height_mismatch");
}

TEST(BoxFitterTest, NonSquareRectangularBoxFit)
{
  const Eigen::Vector3d sensor(0.0, 0.0, 0.70);
  const Eigen::Vector3d gt_center(1.75, 0.0, 0.15);
  const Eigen::Vector3d size(0.45, 0.25, 0.30);

  BoxFittingConfig cfg;
  cfg.length = 0.45;
  cfg.width = 0.25;
  cfg.height = 0.30;
  GateConfig gate;
  gate.max_xy_span = 0.60;

  for (double yaw_deg : {0.0, 90.0, 25.0, -30.0}) {
    auto pts = sampleVisibleCubePoints(gt_center, yaw_deg * kDegToRad, size, sensor, 0.010, 240, 909);
    BoxDetection det;
    std::string reason;
    ASSERT_TRUE(BoxFitter::fitBoxInGroundFrame(pts, sensor, cfg, gate, det, reason))
      << "yaw=" << yaw_deg << " reason=" << reason;
    EXPECT_LT((det.center - gt_center).norm(), 0.015) << "yaw=" << yaw_deg;
    const double yaw_err_deg =
      std::abs(shortestSymmetricAngleDiff(det.yaw, yaw_deg * kDegToRad, kPi)) * kRadToDeg;
    EXPECT_LT(yaw_err_deg, 2.5) << "yaw=" << yaw_deg;
  }
}

TEST(BoxFitterTest, TopMinPointsParameterRespected)
{
  const Eigen::Vector3d sensor(0.0, 0.0, 0.70);
  // 0.38 m tall box on ground (within height_consistency_tol of 0.35 m):
  // has_top_face=true yields cz ~ 0.205 m, while has_top_face=false falls back to cz_ground=0.175 m.
  const Eigen::Vector3d gt_center(1.80, 0.0, 0.190);
  const Eigen::Vector3d tall_size(0.35, 0.35, 0.38);
  auto pts = sampleVisibleCubePoints(gt_center, 20.0 * kDegToRad, tall_size, sensor, 0.006, 220, 910);

  BoxFittingConfig cfg_default;
  GateConfig gate;
  BoxDetection det_default;
  std::string reason;
  ASSERT_TRUE(BoxFitter::fitBoxInGroundFrame(pts, sensor, cfg_default, gate, det_default, reason));
  EXPECT_TRUE(det_default.has_top_face);
  EXPECT_NEAR(det_default.center.z(), 0.205, 0.008);

  BoxFittingConfig cfg_strict = cfg_default;
  cfg_strict.top_min_points = 200;
  BoxDetection det_strict;
  ASSERT_TRUE(BoxFitter::fitBoxInGroundFrame(pts, sensor, cfg_strict, gate, det_strict, reason));
  EXPECT_FALSE(det_strict.has_top_face);
  EXPECT_NEAR(det_strict.center.z(), 0.175, 0.002);
}

}  // namespace
}  // namespace midtobox

// Copyright 2026 midtobox contributors
// SPDX-License-Identifier: Apache-2.0

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "midtobox_detector/detector_pipeline.hpp"
#include "midtobox_detector/types.hpp"

namespace midtobox
{
namespace
{

void appendGroundPlane(
  DetectorPipeline::CloudPtr & cloud,
  const Eigen::Matrix3d & R_ground_to_base,
  double z_offset,
  double noise_std,
  unsigned int seed)
{
  std::mt19937 rng(seed);
  std::normal_distribution<double> gauss(0.0, noise_std);
  for (double x = 0.55; x <= 3.0; x += 0.07) {
    for (double y = -1.1; y <= 1.1; y += 0.07) {
      Eigen::Vector3d pg(x, y, gauss(rng));
      Eigen::Vector3d pb = R_ground_to_base * pg + Eigen::Vector3d(0.0, 0.0, z_offset);
      cloud->points.emplace_back(
        static_cast<float>(pb.x()), static_cast<float>(pb.y()), static_cast<float>(pb.z()));
    }
  }
}

void appendVisibleBoxOnGround(
  DetectorPipeline::CloudPtr & cloud,
  const Eigen::Vector3d & center_ground,
  double yaw_ground,
  const Eigen::Vector3d & size,
  const Eigen::Matrix3d & R_ground_to_base,
  double z_offset,
  const Eigen::Vector3d & sensor_base,
  double noise_std,
  unsigned int seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uni(-0.5, 0.5);
  std::normal_distribution<double> gauss(0.0, noise_std);

  const Eigen::Vector3d sensor_ground =
    R_ground_to_base.transpose() * (sensor_base - Eigen::Vector3d(0.0, 0.0, z_offset));
  const Eigen::Matrix3d R_box =
    Eigen::AngleAxisd(yaw_ground, Eigen::Vector3d::UnitZ()).toRotationMatrix();

  struct FaceSpec { Eigen::Vector3d normal_local; int u_axis; int v_axis; };
  const std::vector<FaceSpec> faces = {
    {{1, 0, 0}, 1, 2},
    {{-1, 0, 0}, 1, 2},
    {{0, 1, 0}, 0, 2},
    {{0, -1, 0}, 0, 2},
    {{0, 0, 1}, 0, 1},
  };

  for (const auto & f : faces) {
    const Eigen::Vector3d n_g = R_box * f.normal_local;
    const Eigen::Vector3d face_off = f.normal_local.cwiseProduct(0.5 * size);
    const Eigen::Vector3d face_center = center_ground + R_box * face_off;
    const double cos_inc = n_g.dot((sensor_ground - face_center).normalized());
    if (cos_inc <= 0.12) {
      continue;
    }
    const int n_pts = std::max(25, static_cast<int>(std::round(260 * cos_inc)));
    for (int i = 0; i < n_pts; ++i) {
      Eigen::Vector3d p_loc = face_off;
      p_loc[f.u_axis] = uni(rng) * size[f.u_axis];
      p_loc[f.v_axis] = uni(rng) * size[f.v_axis];
      Eigen::Vector3d pg = center_ground + R_box * p_loc;
      const Eigen::Vector3d ray = (pg - sensor_ground).normalized();
      pg += ray * gauss(rng);
      const Eigen::Vector3d pb = R_ground_to_base * pg + Eigen::Vector3d(0.0, 0.0, z_offset);
      cloud->points.emplace_back(
        static_cast<float>(pb.x()), static_cast<float>(pb.y()), static_cast<float>(pb.z()));
    }
  }
}

TEST(PipelineTest, MultiCubeSeparatedBy12cmAndEgoCropped)
{
  const Eigen::Vector3d sensor(0.0, 0.0, 0.70);
  const Eigen::Vector3d sz(0.35, 0.35, 0.35);

  // Case A: Two rotated cubes with >= 13 cm corner-to-corner clearance
  {
    DetectorPipeline pipeline;
    const Eigen::Vector3d c1(1.50, -0.29, 0.175);  // +Y corner at -0.066 m
    const Eigen::Vector3d c2(1.55, +0.29, 0.175);  // -Y corner at +0.087 m -> 15.3 cm gap

    PipelineResult res;
    for (int frame = 0; frame < 4; ++frame) {
      DetectorPipeline::CloudPtr cloud(new DetectorPipeline::CloudT);
      appendGroundPlane(cloud, Eigen::Matrix3d::Identity(), 0.0, 0.008, 10 + frame);
      appendVisibleBoxOnGround(
        cloud, c1, 20.0 * kDegToRad, sz, Eigen::Matrix3d::Identity(), 0.0, sensor, 0.010, 100 + frame);
      appendVisibleBoxOnGround(
        cloud, c2, -10.0 * kDegToRad, sz, Eigen::Matrix3d::Identity(), 0.0, sensor, 0.010, 200 + frame);
      // Inject ego chassis points inside ego cropbox (must be removed)
      for (int i = 0; i < 60; ++i) {
        cloud->points.emplace_back(0.15f, -0.1f + 0.003f * i, 0.25f);
      }
      cloud->width = static_cast<uint32_t>(cloud->points.size());
      cloud->height = 1;
      res = pipeline.process(cloud, sensor, 0.1 * frame);
    }

    ASSERT_EQ(res.tracks.size(), 2u);
    ASSERT_TRUE(res.has_primary);
    EXPECT_TRUE(res.locked);
    EXPECT_LT((res.primary_target.center - c1).norm(), 0.018);
    const double yaw_err_deg =
      std::abs(shortestSymmetricAngleDiff(res.primary_target.yaw, 20.0 * kDegToRad, kHalfPi)) *
      kRadToDeg;
    EXPECT_LT(yaw_err_deg, 2.0);
  }

  // Case B: Two axis-aligned cubes separated by exact 12 cm face-to-face gap (0.47 m c-to-c)
  {
    DetectorPipeline pipeline;
    const Eigen::Vector3d c1(1.60, -0.235, 0.175);
    const Eigen::Vector3d c2(1.60, +0.235, 0.175);

    PipelineResult res;
    for (int frame = 0; frame < 4; ++frame) {
      DetectorPipeline::CloudPtr cloud(new DetectorPipeline::CloudT);
      appendGroundPlane(cloud, Eigen::Matrix3d::Identity(), 0.0, 0.008, 50 + frame);
      appendVisibleBoxOnGround(
        cloud, c1, 0.0, sz, Eigen::Matrix3d::Identity(), 0.0, sensor, 0.008, 600 + frame);
      appendVisibleBoxOnGround(
        cloud, c2, 0.0, sz, Eigen::Matrix3d::Identity(), 0.0, sensor, 0.008, 700 + frame);
      cloud->width = static_cast<uint32_t>(cloud->points.size());
      cloud->height = 1;
      res = pipeline.process(cloud, sensor, 0.1 * frame);
    }
    ASSERT_EQ(res.tracks.size(), 2u);
    EXPECT_TRUE(res.locked);
  }
}

TEST(PipelineTest, CubeAt45DegreesPassesObbGate)
{
  DetectorPipeline pipeline;
  const Eigen::Vector3d sensor(0.0, 0.0, 0.70);
  const Eigen::Vector3d gt(1.80, 0.0, 0.175);
  const Eigen::Vector3d sz(0.35, 0.35, 0.35);

  PipelineResult res;
  for (int frame = 0; frame < 4; ++frame) {
    DetectorPipeline::CloudPtr cloud(new DetectorPipeline::CloudT);
    appendGroundPlane(cloud, Eigen::Matrix3d::Identity(), 0.0, 0.008, 20 + frame);
    appendVisibleBoxOnGround(
      cloud, gt, 45.0 * kDegToRad, sz, Eigen::Matrix3d::Identity(), 0.0, sensor, 0.012, 300 + frame);
    cloud->width = static_cast<uint32_t>(cloud->points.size());
    cloud->height = 1;
    res = pipeline.process(cloud, sensor, 0.1 * frame);
  }

  ASSERT_TRUE(res.has_primary);
  EXPECT_TRUE(res.locked);
  EXPECT_LT((res.primary_target.center - gt).norm(), 0.018);
}

TEST(PipelineTest, TiltedGroundFourDegreesCompensated)
{
  DetectorPipeline pipeline;
  const Eigen::Vector3d sensor(0.0, 0.0, 0.70);
  // 4-degree pitch slope around Y
  const double tilt = 4.0 * kDegToRad;
  const Eigen::Matrix3d R_tilt = Eigen::AngleAxisd(-tilt, Eigen::Vector3d::UnitY()).toRotationMatrix();
  const Eigen::Vector3d c_ground(1.80, 0.10, 0.175);
  const Eigen::Vector3d c_base_gt = R_tilt * c_ground;
  const Eigen::Vector3d sz(0.35, 0.35, 0.35);

  PipelineResult res;
  for (int frame = 0; frame < 4; ++frame) {
    DetectorPipeline::CloudPtr cloud(new DetectorPipeline::CloudT);
    appendGroundPlane(cloud, R_tilt, 0.0, 0.008, 30 + frame);
    appendVisibleBoxOnGround(
      cloud, c_ground, 25.0 * kDegToRad, sz, R_tilt, 0.0, sensor, 0.010, 400 + frame);
    cloud->width = static_cast<uint32_t>(cloud->points.size());
    cloud->height = 1;
    res = pipeline.process(cloud, sensor, 0.1 * frame);
  }

  EXPECT_TRUE(res.ground_plane.from_ransac);
  const Eigen::Vector3d expected_normal = R_tilt * Eigen::Vector3d::UnitZ();
  const double normal_err_deg =
    std::acos(std::clamp(res.ground_plane.normal().dot(expected_normal), -1.0, 1.0)) * kRadToDeg;
  EXPECT_LT(normal_err_deg, 0.8);

  ASSERT_TRUE(res.has_primary);
  EXPECT_TRUE(res.locked);
  EXPECT_LT((res.primary_target.center - c_base_gt).norm(), 0.018)
    << "est=" << res.primary_target.center.transpose()
    << " gt=" << c_base_gt.transpose();
}

TEST(PipelineTest, OversizedTouchingAndStackedClustersRejected)
{
  const Eigen::Vector3d sensor(0.0, 0.0, 0.70);
  const Eigen::Vector3d sz(0.35, 0.35, 0.35);

  // 1. Touching side-by-side cubes (total Y span = 0.70 m > 0.50 m)
  {
    DetectorPipeline pipeline;
    DetectorPipeline::CloudPtr cloud(new DetectorPipeline::CloudT);
    appendGroundPlane(cloud, Eigen::Matrix3d::Identity(), 0.0, 0.006, 40);
    appendVisibleBoxOnGround(
      cloud, {1.80, -0.175, 0.175}, 0.0, sz, Eigen::Matrix3d::Identity(), 0.0, sensor, 0.008, 501);
    appendVisibleBoxOnGround(
      cloud, {1.80, +0.175, 0.175}, 0.0, sz, Eigen::Matrix3d::Identity(), 0.0, sensor, 0.008, 502);
    cloud->width = static_cast<uint32_t>(cloud->points.size());
    cloud->height = 1;

    const auto res = pipeline.process(cloud, sensor, 0.1);
    EXPECT_TRUE(res.raw_detections.empty());
    EXPECT_FALSE(res.has_primary);
    EXPECT_FALSE(res.locked);
    bool found_oversized = false;
    for (const auto & rej : res.rejected_clusters) {
      if (rej.is_oversized && rej.reason.find("oversized_xy") != std::string::npos) {
        found_oversized = true;
      }
    }
    EXPECT_TRUE(found_oversized);
  }

  // 2. Stacked cubes (total Z height = 0.70 m > 0.45 m)
  {
    DetectorPipeline pipeline;
    DetectorPipeline::CloudPtr cloud(new DetectorPipeline::CloudT);
    appendGroundPlane(cloud, Eigen::Matrix3d::Identity(), 0.0, 0.006, 41);
    appendVisibleBoxOnGround(
      cloud, {1.80, 0.0, 0.175}, 0.0, sz, Eigen::Matrix3d::Identity(), 0.0, sensor, 0.008, 503);
    appendVisibleBoxOnGround(
      cloud, {1.80, 0.0, 0.525}, 0.0, sz, Eigen::Matrix3d::Identity(), 0.0, sensor, 0.008, 504);
    cloud->width = static_cast<uint32_t>(cloud->points.size());
    cloud->height = 1;

    const auto res = pipeline.process(cloud, sensor, 0.1);
    EXPECT_TRUE(res.raw_detections.empty());
    EXPECT_FALSE(res.has_primary);
    EXPECT_FALSE(res.locked);
    bool found_oversized_z = false;
    for (const auto & rej : res.rejected_clusters) {
      if (rej.is_oversized && rej.reason.find("oversized_z") != std::string::npos) {
        found_oversized_z = true;
      }
    }
    EXPECT_TRUE(found_oversized_z);
  }
}

TEST(PipelineTest, EmptyAndPureGroundCloudSafe)
{
  DetectorPipeline pipeline;
  const Eigen::Vector3d sensor(0.0, 0.0, 0.70);

  DetectorPipeline::CloudPtr empty(new DetectorPipeline::CloudT);
  auto res1 = pipeline.process(empty, sensor, 0.0);
  EXPECT_FALSE(res1.has_primary);
  EXPECT_FALSE(res1.locked);
  EXPECT_TRUE(res1.tracks.empty());

  DetectorPipeline::CloudPtr ground(new DetectorPipeline::CloudT);
  appendGroundPlane(ground, Eigen::Matrix3d::Identity(), 0.0, 0.008, 77);
  ground->width = static_cast<uint32_t>(ground->points.size());
  ground->height = 1;
  auto res2 = pipeline.process(ground, sensor, 0.1);
  EXPECT_TRUE(res2.ground_plane.from_ransac);
  EXPECT_FALSE(res2.has_primary);
  EXPECT_FALSE(res2.locked);
}

TEST(PipelineTest, ShortCrateNotFlaggedOversized)
{
  DetectorPipeline pipeline;
  const Eigen::Vector3d sensor(0.0, 0.0, 0.70);
  const Eigen::Vector3d short_sz(0.35, 0.35, 0.23);

  DetectorPipeline::CloudPtr cloud(new DetectorPipeline::CloudT);
  appendGroundPlane(cloud, Eigen::Matrix3d::Identity(), 0.0, 0.006, 81);
  appendVisibleBoxOnGround(
    cloud, {1.75, 0.0, 0.115}, 0.0, short_sz, Eigen::Matrix3d::Identity(), 0.0, sensor, 0.008, 82);
  cloud->width = static_cast<uint32_t>(cloud->points.size());
  cloud->height = 1;

  const auto res = pipeline.process(cloud, sensor, 0.1);
  EXPECT_TRUE(res.raw_detections.empty());
  EXPECT_FALSE(res.has_primary);
  ASSERT_FALSE(res.rejected_clusters.empty());
  for (const auto & rej : res.rejected_clusters) {
    EXPECT_FALSE(rej.is_oversized) << "Short crate falsely marked oversized: " << rej.reason;
  }
}

TEST(PipelineTest, LevelToGroundFalsePreservesRansacVerticalOffset)
{
  PipelineConfig cfg;
  cfg.ground.use_ransac = true;
  cfg.ground.level_to_ground = false;
  DetectorPipeline pipeline(cfg);

  const Eigen::Vector3d sensor(0.0, 0.0, 0.70);
  const double z_off = -0.09;
  const Eigen::Vector3d c_ground(1.80, 0.0, 0.175);
  const Eigen::Vector3d c_base_gt = c_ground + Eigen::Vector3d(0.0, 0.0, z_off);
  const Eigen::Vector3d sz(0.35, 0.35, 0.35);

  PipelineResult res;
  for (int frame = 0; frame < 4; ++frame) {
    DetectorPipeline::CloudPtr cloud(new DetectorPipeline::CloudT);
    appendGroundPlane(cloud, Eigen::Matrix3d::Identity(), z_off, 0.008, 90 + frame);
    appendVisibleBoxOnGround(
      cloud, c_ground, 20.0 * kDegToRad, sz, Eigen::Matrix3d::Identity(), z_off,
      sensor, 0.010, 900 + frame);
    cloud->width = static_cast<uint32_t>(cloud->points.size());
    cloud->height = 1;
    res = pipeline.process(cloud, sensor, 0.1 * frame);
  }

  EXPECT_TRUE(res.ground_plane.from_ransac);
  ASSERT_TRUE(res.has_primary);
  EXPECT_TRUE(res.locked);
  EXPECT_LT((res.primary_target.center - c_base_gt).norm(), 0.018)
    << "est=" << res.primary_target.center.transpose() << " gt=" << c_base_gt.transpose();
}

}  // namespace
}  // namespace midtobox

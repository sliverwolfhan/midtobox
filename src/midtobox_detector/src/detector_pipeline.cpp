// Copyright 2026 midtobox contributors
// SPDX-License-Identifier: Apache-2.0
//
// 纯 C++（PCL + Eigen）方块质心感知流水线实现。
// 欧氏聚类流程借鉴自 SS47816/lidar_obstacle_detector (MIT License)。

#include "midtobox_detector/detector_pipeline.hpp"

#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace midtobox
{

DetectorPipeline::DetectorPipeline(const PipelineConfig & config)
: config_(config)
{
}

void DetectorPipeline::setConfig(const PipelineConfig & config)
{
  config_ = config;
}

void DetectorPipeline::resetTracker()
{
  tracker_.reset();
}

PipelineResult DetectorPipeline::process(
  const CloudConstPtr & base_cloud,
  const Eigen::Vector3d & sensor_origin_base,
  double stamp_sec,
  CloudPtr roi_cloud_out,
  CloudPtr non_ground_cloud_out,
  const Eigen::Isometry3d * T_curr_prev_base)
{
  const auto t_start = std::chrono::steady_clock::now();
  PipelineResult result;

  if (!base_cloud) {
    return result;
  }
  result.input_points = static_cast<int>(base_cloud->size());

  // 1. 自车 3D 包围盒剔除 + 作业区 ROI 裁剪（均在整平后的 base_footprint 下执行）
  CloudPtr roi_cloud(new CloudT);
  roi_cloud->points.reserve(base_cloud->size());
  const auto & cc = config_.crop;
  for (const auto & pt : base_cloud->points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
      continue;
    }
    if (cc.ego_enabled &&
        pt.x >= cc.ego_min_x && pt.x <= cc.ego_max_x &&
        pt.y >= cc.ego_min_y && pt.y <= cc.ego_max_y &&
        pt.z >= cc.ego_min_z && pt.z <= cc.ego_max_z)
    {
      continue;
    }
    if (pt.x < cc.roi_min_x || pt.x > cc.roi_max_x ||
        pt.y < cc.roi_min_y || pt.y > cc.roi_max_y ||
        pt.z < cc.roi_min_z || pt.z > cc.roi_max_z)
    {
      continue;
    }
    roi_cloud->points.push_back(pt);
  }
  roi_cloud->width = static_cast<uint32_t>(roi_cloud->points.size());
  roi_cloud->height = 1;
  roi_cloud->is_dense = true;

  // 可选体素降采样（默认关闭以保持方块边缘精度）
  if (cc.voxel_leaf_size > 1e-4 && !roi_cloud->empty()) {
    CloudPtr down(new CloudT);
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(roi_cloud);
    const float leaf = static_cast<float>(cc.voxel_leaf_size);
    vg.setLeafSize(leaf, leaf, leaf);
    vg.filter(*down);
    roi_cloud = down;
  }

  result.roi_points = static_cast<int>(roi_cloud->size());
  if (roi_cloud_out) {
    *roi_cloud_out = *roi_cloud;
  }

  // 2. 倾角约束 RANSAC 地面分割
  CloudPtr raw_non_ground(new CloudT);
  result.ground_plane = ground_segmenter_.segment(roi_cloud, config_.ground, raw_non_ground);

  // 3. 聚类前先执行 ROR 半径离群滤波，打断 Mid-360 边缘“拉丝”飞点形成的块间桥接
  CloudPtr filtered_non_ground =
    ground_segmenter_.applyRadiusOutlierRemoval(raw_non_ground, config_.outlier);
  result.non_ground_points = static_cast<int>(filtered_non_ground->size());

  if (non_ground_cloud_out) {
    *non_ground_cloud_out = *filtered_non_ground;
  }

  // 判断目标是否为正方形（90° 四折对称）还是长方形（180° 二折对称）
  const bool is_square = std::abs(config_.fitting.length - config_.fitting.width) < 0.02;
  const double sym_period = is_square ? kHalfPi : kPi;

  if (static_cast<int>(filtered_non_ground->size()) >= config_.cluster.min_size) {
    // 4. 将非地面点云变换到地面整平系 G（z_G 表示真实离地高度，z_G = 0 为地面）
    const Eigen::Matrix3d R_level =
      config_.ground.level_to_ground ? result.ground_plane.levelRotation()
                                     : Eigen::Matrix3d::Identity();
    // 关闭旋转整平时，若 RANSAC 拟合成功仍保留垂直高度偏移 d/c，避免地面标高偏差误杀高度门控
    const double d_shift = config_.ground.level_to_ground
      ? result.ground_plane.d
      : ((result.ground_plane.from_ransac && result.ground_plane.c > 1e-6)
        ? (result.ground_plane.d / result.ground_plane.c)
        : -config_.ground.nominal_z);
    const Eigen::Vector3d n_ground =
      config_.ground.level_to_ground ? result.ground_plane.normal()
                                     : Eigen::Vector3d::UnitZ();

    auto toGround = [&](const Eigen::Vector3d & p_base) -> Eigen::Vector3d {
        return R_level * p_base + Eigen::Vector3d(0.0, 0.0, d_shift);
      };
    auto fromGround = [&](const Eigen::Vector3d & p_ground) -> Eigen::Vector3d {
        return R_level.transpose() * p_ground - d_shift * n_ground;
      };

    const Eigen::Vector3d sensor_origin_ground = toGround(sensor_origin_base);

    CloudPtr leveled_cloud(new CloudT);
    leveled_cloud->points.resize(filtered_non_ground->size());
    for (size_t i = 0; i < filtered_non_ground->size(); ++i) {
      const auto & pt = filtered_non_ground->points[i];
      const Eigen::Vector3d pg = toGround(Eigen::Vector3d(pt.x, pt.y, pt.z));
      leveled_cloud->points[i].x = static_cast<float>(pg.x());
      leveled_cloud->points[i].y = static_cast<float>(pg.y());
      leveled_cloud->points[i].z = static_cast<float>(pg.z());
    }
    leveled_cloud->width = static_cast<uint32_t>(leveled_cloud->points.size());
    leveled_cloud->height = 1;
    leveled_cloud->is_dense = true;

    // 5. PCL KD-Tree 欧氏聚类，分离间距 >= 10 cm 的多个独立方块
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(leveled_cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(config_.cluster.tolerance);
    ec.setMinClusterSize(config_.cluster.min_size);
    ec.setMaxClusterSize(config_.cluster.max_size);
    ec.setSearchMethod(tree);
    ec.setInputCloud(leveled_cloud);
    ec.extract(cluster_indices);

    // 6. 逐簇处理：SOR 修剪边缘飞点 -> 尺寸/高度门控（显式拦截超大粘连/堆叠）-> 已知尺寸几何拟合
    for (const auto & indices : cluster_indices) {
      CloudPtr cluster_cloud(new CloudT);
      cluster_cloud->points.reserve(indices.indices.size());
      for (int idx : indices.indices) {
        cluster_cloud->points.push_back(leveled_cloud->points[idx]);
      }
      cluster_cloud->width = static_cast<uint32_t>(cluster_cloud->points.size());
      cluster_cloud->height = 1;
      cluster_cloud->is_dense = true;

      CloudPtr clean_cluster =
        ground_segmenter_.applyStatisticalOutlierRemoval(cluster_cloud, config_.outlier);

      std::vector<Eigen::Vector3d> pts_3d;
      std::vector<Eigen::Vector2d> pts_2d;
      std::vector<double> z_vals;
      pts_3d.reserve(clean_cluster->size());
      pts_2d.reserve(clean_cluster->size());
      z_vals.reserve(clean_cluster->size());

      for (const auto & pt : clean_cluster->points) {
        pts_3d.emplace_back(pt.x, pt.y, pt.z);
        pts_2d.emplace_back(pt.x, pt.y);
        z_vals.push_back(pt.z);
      }

      if (pts_3d.empty()) {
        continue;
      }

      // 计算簇的鲁棒 2D OBB 跨度与 p03/p97 高度范围，用于尺寸门控
      const Obb2D obb = BoxFitter::computeRobustObb2D(pts_2d);
      const double z_p97 = BoxFitter::computeQuantile(z_vals, 0.97);
      const double z_p03 = BoxFitter::computeQuantile(z_vals, 0.03);
      const Eigen::Vector3d cluster_center_base =
        fromGround(Eigen::Vector3d(obb.center.x(), obb.center.y(), 0.5 * z_p97));

      // 超大簇拦截：水平超限（并排粘连）或高度超限（上下堆叠）直接拒判并标记供 RViz 标红
      const bool oversized_xy =
        (obb.major_span > config_.gate.max_xy_span) ||
        (obb.minor_span > config_.gate.max_xy_span);
      const bool oversized_z = (z_p97 > config_.gate.max_z);

      if (oversized_xy || oversized_z) {
        RejectedCluster rej;
        rej.center = cluster_center_base;
        rej.size = Eigen::Vector3d(
          std::max(obb.major_span, 0.05),
          std::max(obb.minor_span, 0.05),
          std::max(z_p97, 0.05));
        rej.yaw = obb.yaw;
        rej.point_count = static_cast<int>(pts_3d.size());
        rej.is_oversized = true;
        if (oversized_xy && oversized_z) {
          rej.reason = "oversized_xy_z";
        } else if (oversized_xy) {
          rej.reason = "oversized_xy";
        } else {
          rej.reason = "oversized_z";
        }
        result.rejected_clusters.push_back(std::move(rej));
        continue;
      }

      // 最小尺寸、点数与落地校验（剔除过小杂物和悬空干扰物）
      if (static_cast<int>(pts_3d.size()) < config_.gate.min_points ||
          obb.major_span < config_.gate.min_xy_span ||
          z_p97 < config_.gate.min_z ||
          z_p03 > config_.gate.max_bottom_z)
      {
        RejectedCluster rej;
        rej.center = cluster_center_base;
        rej.size = Eigen::Vector3d(
          std::max(obb.major_span, 0.05),
          std::max(obb.minor_span, 0.05),
          std::max(z_p97, 0.05));
        rej.yaw = obb.yaw;
        rej.point_count = static_cast<int>(pts_3d.size());
        rej.is_oversized = false;
        if (static_cast<int>(pts_3d.size()) < config_.gate.min_points) {
          rej.reason = "too_few_points";
        } else if (z_p03 > config_.gate.max_bottom_z) {
          rej.reason = "floating_object";
        } else {
          rej.reason = "undersized";
        }
        result.rejected_clusters.push_back(std::move(rej));
        continue;
      }

      // 在地面整平系 G 下执行已知尺寸方块几何拟合
      BoxDetection det_ground;
      std::string reject_reason;
      if (!BoxFitter::fitBoxInGroundFrame(
          pts_3d, sensor_origin_ground, config_.fitting, config_.gate,
          det_ground, reject_reason))
      {
        RejectedCluster rej;
        rej.center = cluster_center_base;
        rej.size = Eigen::Vector3d(
          std::max(obb.major_span, 0.05),
          std::max(obb.minor_span, 0.05),
          std::max(z_p97, 0.05));
        rej.yaw = obb.yaw;
        rej.point_count = static_cast<int>(pts_3d.size());
        // 仅当顶面纵深超长或高度高于目标块时才标记为超大拒判（矮箱不误标超大）
        rej.is_oversized =
          (reject_reason == "top_depth_oversized") ||
          (reject_reason == "height_mismatch" && z_p97 > config_.fitting.height);
        rej.reason = reject_reason;
        result.rejected_clusters.push_back(std::move(rej));
        continue;
      }

      // 将 G 系拟合结果变换回 base_footprint，并保留真实地面坡度倾角
      BoxDetection det_base = det_ground;
      det_base.center = fromGround(det_ground.center);

      const Eigen::Vector3d x_axis_ground(std::cos(det_ground.yaw), std::sin(det_ground.yaw), 0.0);
      const Eigen::Vector3d x_axis_base = R_level.transpose() * x_axis_ground;
      const double yaw_base =
        canonicalizeSymmetricYaw(std::atan2(x_axis_base.y(), x_axis_base.x()), sym_period);

      // 将 G 系偏航角对齐到与 yaw_base 一致的对称分支，再左乘地面逆旋转得到 3D 姿态
      const double sym_shift =
        std::round((yaw_base - det_ground.yaw) / sym_period) * sym_period;
      const double yaw_g_aligned = det_ground.yaw + sym_shift;
      const Eigen::Matrix3d R_box_base =
        R_level.transpose() *
        Eigen::AngleAxisd(yaw_g_aligned, Eigen::Vector3d::UnitZ()).toRotationMatrix();

      det_base.yaw = yaw_base;
      det_base.orientation = Eigen::Quaterniond(R_box_base).normalized();
      result.raw_detections.push_back(std::move(det_base));
    }
  }

  // 7. 多目标跟踪更新（匈牙利匹配 + 6D 卡尔曼滤波 + 连续 yaw 展开 + 主目标滞回与锁定门控）
  tracker_.update(
    result.raw_detections,
    stamp_sec,
    config_.tracker,
    is_square,
    result.tracks,
    result.has_primary,
    result.primary_target,
    result.locked,
    T_curr_prev_base);

  const auto t_end = std::chrono::steady_clock::now();
  result.compute_time_ms =
    std::chrono::duration<double, std::milli>(t_end - t_start).count();
  return result;
}

}  // namespace midtobox

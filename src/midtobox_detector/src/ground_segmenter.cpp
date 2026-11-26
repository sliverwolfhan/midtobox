// Copyright 2026 midtobox contributors
// SPDX-License-Identifier: Apache-2.0
//
// 倾角约束 RANSAC 地面分割与边缘拉丝离群滤波（ROR / SOR）实现。
// 地面分割思路参考 SS47816/lidar_obstacle_detector (MIT License)。

#include "midtobox_detector/ground_segmenter.hpp"

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <algorithm>
#include <cmath>

namespace midtobox
{

PlaneModel GroundSegmenter::segment(
  const CloudConstPtr & roi_cloud,
  const GroundConfig & cfg,
  CloudPtr & non_ground_out) const
{
  PlaneModel plane;
  plane.a = 0.0;
  plane.b = 0.0;
  plane.c = 1.0;
  plane.d = -cfg.nominal_z;
  plane.from_ransac = false;
  plane.inlier_count = 0;

  if (!non_ground_out) {
    non_ground_out.reset(new CloudT);
  }
  non_ground_out->clear();

  if (!roi_cloud || roi_cloud->empty()) {
    return plane;
  }

  if (cfg.use_ransac) {
    // 1. 先截取整平坐标系下接近地面的 Z 带点云作为候选地面点，加速 RANSAC 并防止拟合到墙面/顶面
    CloudPtr band_cloud(new CloudT);
    band_cloud->points.reserve(roi_cloud->size());
    for (const auto & pt : roi_cloud->points) {
      if (pt.z >= static_cast<float>(cfg.band_min_z) &&
          pt.z <= static_cast<float>(cfg.band_max_z))
      {
        band_cloud->points.push_back(pt);
      }
    }
    band_cloud->width = static_cast<uint32_t>(band_cloud->points.size());
    band_cloud->height = 1;
    band_cloud->is_dense = true;

    if (static_cast<int>(band_cloud->size()) >= std::max(cfg.min_inliers, 3)) {
      // 使用垂直于 +Z 轴且带最大倾角约束的 RANSAC 拟合地面平面
      pcl::SACSegmentation<PointT> seg;
      seg.setOptimizeCoefficients(true);
      seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
      seg.setMethodType(pcl::SAC_RANSAC);
      seg.setAxis(Eigen::Vector3f(0.0f, 0.0f, 1.0f));
      const double max_tilt_rad = cfg.max_tilt_deg * kDegToRad;
      seg.setEpsAngle(max_tilt_rad);
      seg.setDistanceThreshold(cfg.ransac_dist_thresh);
      seg.setMaxIterations(cfg.ransac_max_iter);
      seg.setInputCloud(band_cloud);

      pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
      pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients);
      seg.segment(*inliers, *coeffs);

      if (coeffs->values.size() == 4 &&
          static_cast<int>(inliers->indices.size()) >= cfg.min_inliers)
      {
        double a = coeffs->values[0];
        double b = coeffs->values[1];
        double c = coeffs->values[2];
        double d = coeffs->values[3];
        const double norm = std::sqrt(a * a + b * b + c * c);
        if (norm > 1e-9) {
          // 归一化平面系数，并强制法向量朝上 (c > 0)
          a /= norm;
          b /= norm;
          c /= norm;
          d /= norm;
          if (c < 0.0) {
            a = -a;
            b = -b;
            c = -c;
            d = -d;
          }
          const double clamped_c = std::clamp(c, -1.0, 1.0);
          const double tilt_rad = std::acos(clamped_c);
          // 校验原点 (x=0, y=0) 处地面高度 z0 = -d / c 是否落在允许偏差范围内
          const double z_origin = (c > 1e-6) ? (-d / c) : 1e9;
          if (tilt_rad <= max_tilt_rad + 1e-3 &&
              std::abs(z_origin - cfg.nominal_z) <= cfg.max_plane_offset)
          {
            plane.a = a;
            plane.b = b;
            plane.c = c;
            plane.d = d;
            plane.from_ransac = true;
            plane.inlier_count = static_cast<int>(inliers->indices.size());
          }
        }
      }
    }
  }

  // 2. 按点到拟合地面的精确有符号高度分离非地面目标点（h >= clearance_height）
  non_ground_out->points.reserve(roi_cloud->size());
  for (const auto & pt : roi_cloud->points) {
    const double h = plane.a * pt.x + plane.b * pt.y + plane.c * pt.z + plane.d;
    if (h >= cfg.clearance_height) {
      non_ground_out->points.push_back(pt);
    }
  }
  non_ground_out->width = static_cast<uint32_t>(non_ground_out->points.size());
  non_ground_out->height = 1;
  non_ground_out->is_dense = true;

  return plane;
}

GroundSegmenter::CloudPtr GroundSegmenter::applyRadiusOutlierRemoval(
  const CloudConstPtr & input,
  const OutlierFilterConfig & cfg) const
{
  CloudPtr output(new CloudT);
  if (!input || input->empty()) {
    return output;
  }
  if (!cfg.ror_enabled || cfg.ror_min_neighbors <= 0 || cfg.ror_radius <= 0.0) {
    *output = *input;
    return output;
  }

  pcl::RadiusOutlierRemoval<PointT> ror;
  ror.setInputCloud(input);
  ror.setRadiusSearch(cfg.ror_radius);
  ror.setMinNeighborsInRadius(cfg.ror_min_neighbors);
  ror.filter(*output);
  return output;
}

GroundSegmenter::CloudPtr GroundSegmenter::applyStatisticalOutlierRemoval(
  const CloudConstPtr & input,
  const OutlierFilterConfig & cfg) const
{
  CloudPtr output(new CloudT);
  if (!input || input->empty()) {
    return output;
  }
  if (!cfg.sor_enabled || static_cast<int>(input->size()) <= cfg.sor_mean_k) {
    *output = *input;
    return output;
  }

  pcl::StatisticalOutlierRemoval<PointT> sor;
  sor.setInputCloud(input);
  sor.setMeanK(cfg.sor_mean_k);
  sor.setStddevMulThresh(cfg.sor_stddev_mul);
  sor.filter(*output);
  return output;
}

}  // namespace midtobox

// Copyright 2026 midtobox contributors
// SPDX-License-Identifier: Apache-2.0
//
// Tilt-constrained RANSAC ground segmentation and outlier removal (ROR/SOR).
// Ground plane RANSAC concept adapted from SS47816/lidar_obstacle_detector (MIT),
// redesigned for leveled Z-band + bounded-tilt SACMODEL_PERPENDICULAR_PLANE.

#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "midtobox_detector/types.hpp"

namespace midtobox
{

class GroundSegmenter
{
public:
  using PointT = pcl::PointXYZ;
  using CloudT = pcl::PointCloud<PointT>;
  using CloudPtr = CloudT::Ptr;
  using CloudConstPtr = CloudT::ConstPtr;

  /// Estimate ground plane in leveled base_footprint and separate non-ground points.
  /// Returns PlaneModel with unit normal (c > 0) and populates non_ground_out (in base_footprint).
  PlaneModel segment(
    const CloudConstPtr & roi_cloud,
    const GroundConfig & cfg,
    CloudPtr & non_ground_out) const;

  /// Apply RadiusOutlierRemoval (if enabled) to strip sparse Mid-360 edge-dragging veil points.
  CloudPtr applyRadiusOutlierRemoval(
    const CloudConstPtr & input,
    const OutlierFilterConfig & cfg) const;

  /// Apply StatisticalOutlierRemoval (if enabled) to trim fringe outliers on a cluster.
  CloudPtr applyStatisticalOutlierRemoval(
    const CloudConstPtr & input,
    const OutlierFilterConfig & cfg) const;
};

}  // namespace midtobox

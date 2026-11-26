// Copyright 2026 midtobox contributors
// SPDX-License-Identifier: Apache-2.0
//
// Pure C++ (PCL + Eigen) perception pipeline:
//   Ego/ROI crop -> tilt-constrained RANSAC ground removal & leveling ->
//   ROR outlier stripping -> Euclidean clustering -> per-cluster SOR ->
//   oversized touching/stacked rejection -> known-size 2-face/1-face box fitting ->
//   Hungarian + Kalman tracking with yaw unwrapping & lock gating.

#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "midtobox_detector/box_fitter.hpp"
#include "midtobox_detector/ground_segmenter.hpp"
#include "midtobox_detector/tracker.hpp"
#include "midtobox_detector/types.hpp"

namespace midtobox
{

class DetectorPipeline
{
public:
  using PointT = pcl::PointXYZ;
  using CloudT = pcl::PointCloud<PointT>;
  using CloudPtr = CloudT::Ptr;
  using CloudConstPtr = CloudT::ConstPtr;

  DetectorPipeline() = default;
  explicit DetectorPipeline(const PipelineConfig & config);

  void setConfig(const PipelineConfig & config);
  const PipelineConfig & config() const { return config_; }

  void resetTracker();

  /// Run the complete perception pipeline on a cloud expressed in leveled base_footprint.
  /// `sensor_origin_base` is the LiDAR origin in base_footprint (used for outward normal direction).
  /// Optionally populates `roi_cloud_out` and `non_ground_cloud_out` (both in base_footprint)
  /// when non-null pointers are passed for debug publishing.
  /// Optional `T_curr_prev_base` (T_odom_base(t_k)^{-1} * T_odom_base(t_{k-1})) compensates
  /// tracker states for robot translation & yaw rotation when odom TF is active.
  PipelineResult process(
    const CloudConstPtr & base_cloud,
    const Eigen::Vector3d & sensor_origin_base,
    double stamp_sec,
    CloudPtr roi_cloud_out = nullptr,
    CloudPtr non_ground_cloud_out = nullptr,
    const Eigen::Isometry3d * T_curr_prev_base = nullptr);

private:
  PipelineConfig config_;
  GroundSegmenter ground_segmenter_;
  BoxTracker tracker_;
};

}  // namespace midtobox

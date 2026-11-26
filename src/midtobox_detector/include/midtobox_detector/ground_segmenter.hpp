// Copyright 2026 midtobox contributors
// SPDX-License-Identifier: Apache-2.0
//
// 倾角约束 RANSAC 地面分割与边缘拉丝离群滤波（ROR / SOR）。
// 地面 RANSAC 思路借鉴自 SS47816/lidar_obstacle_detector (MIT License)，
// 针对室内近距平地重构为「整平 Z 带预筛 + SACMODEL_PERPENDICULAR_PLANE 倾角约束」。

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

  /// 在整平后的 base_footprint 坐标系中估计地面平面并分离非地面点云。
  /// 返回法向朝上 (c > 0) 的单位平面模型 PlaneModel，并将非地面点写入 non_ground_out。
  PlaneModel segment(
    const CloudConstPtr & roi_cloud,
    const GroundConfig & cfg,
    CloudPtr & non_ground_out) const;

  /// 执行半径离群滤波（ROR），剔除 Mid-360 边缘稀疏的“拉丝”拖尾点，防止相邻块粘连
  CloudPtr applyRadiusOutlierRemoval(
    const CloudConstPtr & input,
    const OutlierFilterConfig & cfg) const;

  /// 对单个候选簇执行统计离群滤波（SOR），进一步修剪方块边缘毛刺飞点
  CloudPtr applyStatisticalOutlierRemoval(
    const CloudConstPtr & input,
    const OutlierFilterConfig & cfg) const;
};

}  // namespace midtobox

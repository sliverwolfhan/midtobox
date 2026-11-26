// Copyright 2026 midtobox contributors
// SPDX-License-Identifier: Apache-2.0
//
// 纯 C++（PCL + Eigen，零 OpenCV）感知流水线总控：
//   自车/ROI 裁剪 -> 倾角约束 RANSAC 去地与整平 -> ROR 去拉丝 -> 欧氏聚类 ->
//   每簇 SOR 去毛刺 -> 超大粘连/堆叠拒判 -> 已知尺寸双面/单面几何拟合 ->
//   匈牙利 + 卡尔曼多目标跟踪与锁定门控。

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

  /// 更新流水线参数配置（支持运行期热更新）
  void setConfig(const PipelineConfig & config);

  /// 获取当前只读参数配置
  const PipelineConfig & config() const { return config_; }

  /// 重置内部多目标跟踪器
  void resetTracker();

  /// 对整平后 base_footprint 坐标系下的点云执行完整识块与跟踪流水线。
  /// - `base_cloud`：base_footprint 下的单帧或多帧累积点云
  /// - `sensor_origin_base`：雷达原点在 base_footprint 中的坐标（用于确定朝向雷达的外法向）
  /// - `stamp_sec`：当前帧时间戳 (s)
  /// - `roi_cloud_out` / `non_ground_cloud_out`：可选，传入非空指针时输出调试点云（base_footprint 系）
  /// - `T_curr_prev_base`：可选，odom 有效时传入 T_odom_base(t_k)^{-1} * T_odom_base(t_{k-1}) 补偿跟踪状态
  PipelineResult process(
    const CloudConstPtr & base_cloud,
    const Eigen::Vector3d & sensor_origin_base,
    double stamp_sec,
    CloudPtr roi_cloud_out = nullptr,
    CloudPtr non_ground_cloud_out = nullptr,
    const Eigen::Isometry3d * T_curr_prev_base = nullptr);

private:
  PipelineConfig config_;             // 全阶段参数配置
  GroundSegmenter ground_segmenter_;  // 地面分割与离群滤波模块
  BoxTracker tracker_;                // 多目标卡尔曼跟踪器
};

}  // namespace midtobox

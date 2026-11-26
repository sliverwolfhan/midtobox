// Copyright 2026 midtobox contributors
// SPDX-License-Identifier: Apache-2.0
//
// 已知尺寸 3D 方块单视角几何拟合模块。
// 开源算法借鉴与修正：
//   - Zhang et al. (IV 2017) L-shape Closeness 朝向搜索（参考 SS47816/lidar_obstacle_detector, MIT）
//   - 已知尺寸部分视角质心修正（参考 autoware_shape_estimation, Apache-2.0），
//     并显式区分「双立面 L 形」与「单立面正面」分支，彻底消除单正面角点跳变偏置。

#pragma once

#include <Eigen/Core>

#include <string>
#include <vector>

#include "midtobox_detector/types.hpp"

namespace midtobox
{

/// 2D 鲁棒有向包围盒（基于 p1/p99 分位数面积最小化搜索，抗边缘飞点）
struct Obb2D
{
  double major_span{0.0};                           // 主轴（较长边）跨度 (m)
  double minor_span{0.0};                           // 次轴（较短边）跨度 (m)
  double yaw{0.0};                                  // 主轴偏航角 (rad)
  Eigen::Vector2d center{Eigen::Vector2d::Zero()};  // 包围盒 2D 中心 (m)
};

class BoxFitter
{
public:
  /// 计算 2D 点集的鲁棒最小面积有向包围盒（以 2° 步长扫描，用 p1/p99 抗飞点）
  static Obb2D computeRobustObb2D(const std::vector<Eigen::Vector2d> & pts);

  /// 计算线性插值分位数 q ∈ [0, 1]（内部采用 O(N) nth_element 实现）
  static double computeQuantile(std::vector<double> values, double q);

  /// 基于 Zhang IV 2017 Closeness 准则的两阶段（粗搜+精搜）L-shape 朝向搜索，返回角度 ∈ [0, pi/2)
  static double searchLShapeOrientation(
    const std::vector<Eigen::Vector2d> & pts,
    const BoxFittingConfig & cfg);

  /// 在地面整平系 G（地面位于 z = 0）下对候选点簇执行已知尺寸方块拟合。
  /// `sensor_origin_ground` 为雷达原点在 G 系下的坐标（用于确定朝向雷达侧的外法向）。
  /// 拟合成功返回 true 并将结果写入 `detection_ground`（G 系）；失败返回 false 并填写 `reject_reason`。
  static bool fitBoxInGroundFrame(
    const std::vector<Eigen::Vector3d> & cluster_pts,
    const Eigen::Vector3d & sensor_origin_ground,
    const BoxFittingConfig & cfg,
    const GateConfig & gate_cfg,
    BoxDetection & detection_ground,
    std::string & reject_reason);
};

}  // namespace midtobox

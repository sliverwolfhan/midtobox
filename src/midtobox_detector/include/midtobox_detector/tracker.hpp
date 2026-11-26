// Copyright 2026 midtobox contributors
// SPDX-License-Identifier: Apache-2.0
//
// 多目标方块跟踪模块：匈牙利最优匹配 + 6D 常速卡尔曼滤波 + 4 折对称偏航角连续展开
// + 锁定质量/抖动门控 + 带滞回防抖的最近主目标选择。
// 跟踪整体架构借鉴自 SS47816/lidar_obstacle_detector (MIT License)。

#pragma once

#include <Eigen/Core>

#include <cstdint>
#include <deque>
#include <utility>
#include <vector>

#include "midtobox_detector/types.hpp"

namespace midtobox
{

/// 使用经典 Kuhn-Munkres（匈牙利）算法在 O(N^3) 时间内求解矩形最小总代价指派。
/// 仅返回代价 <= max_cost 的有效匹配对 (行索引, 列索引)。
std::vector<std::pair<size_t, size_t>> solveHungarianAssignment(
  const std::vector<std::vector<double>> & cost_matrix,
  double max_cost);

class BoxTracker
{
public:
  BoxTracker() = default;

  /// 清空所有历史轨迹并重置跟踪器状态（用于时间戳跳变或外参切换）
  void reset();

  /// 自车运动补偿：利用 T_curr_prev = T_odom_base(t_k)^{-1} * T_odom_base(t_{k-1})
  /// 将上一帧 base_footprint(t_{k-1}) 下的轨迹状态、协方差、姿态与抖动滑窗
  /// 统一变换到当前帧 base_footprint(t_k)，避免机器人平移/转向时产生滤波滞后。
  void applyEgoMotion(const Eigen::Isometry3d & T_curr_prev);

  /// 用当前帧（base_footprint 下）的原始检测列表更新所有跟踪轨迹。
  /// - `stamp_sec`：单调递增时间戳 (s)
  /// - `is_square`：正方形选 90° (pi/2) 对称周期，长方形选 180° (pi) 对称周期
  /// - `T_curr_prev`：可选，有 odom TF 时传入帧间自车运动变换以消除转向/平移滞后
  void update(
    const std::vector<BoxDetection> & detections,
    double stamp_sec,
    const TrackerConfig & cfg,
    bool is_square,
    std::vector<TrackedBox> & tracks_out,
    bool & has_primary_out,
    TrackedBox & primary_out,
    bool & locked_out,
    const Eigen::Isometry3d * T_curr_prev = nullptr);

private:
  /// 单条跟踪轨迹的内部滤波与历史状态
  struct TrackState
  {
    uint64_t id{0};  // 唯一轨迹编号

    // 6D 卡尔曼状态向量：[x, y, z, vx, vy, vz] 与状态协方差矩阵 P
    Eigen::Matrix<double, 6, 1> x{Eigen::Matrix<double, 6, 1>::Zero()};
    Eigen::Matrix<double, 6, 6> P{Eigen::Matrix<double, 6, 6>::Identity()};

    // 帧间连续展开后的偏航角（初始化归约在 [-period/2, +period/2)）
    double yaw{0.0};
    // 最新观测的纯倾斜四元数分量（相对 base_footprint 的地面坡度 pitch/roll）
    Eigen::Quaterniond latest_tilt{Eigen::Quaterniond::Identity()};

    Eigen::Vector3d size{0.35, 0.35, 0.35};
    FitMode fit_mode{FitMode::kSingleFace};
    bool has_top_face{false};
    int point_count{0};
    double inlier_ratio{0.0};
    double rmse{0.0};
    double score{0.0};

    int hits{0};    // 累计命中帧数
    int missed{0};  // 连续丢帧计数

    // 最近若干帧匹配观测的滑动历史（yaw 已做连续展开），用于计算锁定抖动
    std::deque<Eigen::Vector3d> pos_history;
    std::deque<double> yaw_history;
  };

  /// 卡尔曼预测步（带轻微速度阻尼防止静止方块过冲）
  void predictTrack(TrackState & trk, double dt, const TrackerConfig & cfg) const;

  /// 卡尔曼观测更新步（Joseph 形式协方差更新 + 对称 yaw 连续展开平滑）
  void updateTrack(
    TrackState & trk,
    const BoxDetection & det,
    const TrackerConfig & cfg,
    double sym_period) const;

  /// 评估单条轨迹是否满足点数、内点率、RMSE 及连续帧位置/偏航抖动锁定判据
  void evaluateLock(
    const TrackState & trk,
    const TrackerConfig & cfg,
    bool & locked_out,
    double & pos_jitter_out,
    double & yaw_jitter_deg_out) const;

  std::vector<TrackState> tracks_;  // 当前活跃轨迹集合
  uint64_t next_track_id_{1};       // 下一个分配的轨迹 ID
  bool has_last_stamp_{false};      // 是否已有上一帧时间戳
  double last_stamp_sec_{0.0};      // 上一帧时间戳 (s)
  uint64_t primary_track_id_{0};    // 当前锁定的主目标轨迹 ID（用于滞回防抖）
};

}  // namespace midtobox

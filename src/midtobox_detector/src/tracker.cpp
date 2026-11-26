// Copyright 2026 midtobox contributors
// SPDX-License-Identifier: Apache-2.0
//
// 多目标方块跟踪实现：匈牙利最优匹配 + 6D 常速卡尔曼滤波 + 4 折对称偏航角连续展开
// + 锁定质量/抖动门控 + 带滞回防抖的最近主目标选择。
// 跟踪架构参考 SS47816/lidar_obstacle_detector (MIT License)。

#include "midtobox_detector/tracker.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>

namespace midtobox
{

std::vector<std::pair<size_t, size_t>> solveHungarianAssignment(
  const std::vector<std::vector<double>> & cost_matrix,
  double max_cost)
{
  std::vector<std::pair<size_t, size_t>> matches;
  const size_t n_rows = cost_matrix.size();
  if (n_rows == 0) {
    return matches;
  }
  const size_t n_cols = cost_matrix[0].size();
  if (n_cols == 0) {
    return matches;
  }

  // 补齐为 N x N 方阵（无效/超距项填充大代价值），以支持矩形代价矩阵求解
  const size_t N = std::max(n_rows, n_cols);
  const double kLargeCost = std::max(max_cost * 100.0, 1e6);

  // 经典 1-indexed Jonker / Kuhn-Munkres 势能法实现，时间复杂度 O(N^3)
  std::vector<std::vector<double>> a(N + 1, std::vector<double>(N + 1, kLargeCost));
  for (size_t i = 0; i < n_rows; ++i) {
    for (size_t j = 0; j < n_cols; ++j) {
      const double c = cost_matrix[i][j];
      a[i + 1][j + 1] = (std::isfinite(c) && c <= max_cost) ? c : kLargeCost;
    }
  }

  std::vector<double> u(N + 1, 0.0);
  std::vector<double> v(N + 1, 0.0);
  std::vector<size_t> p(N + 1, 0);
  std::vector<size_t> way(N + 1, 0);

  for (size_t i = 1; i <= N; ++i) {
    p[0] = i;
    size_t j0 = 0;
    std::vector<double> minv(N + 1, std::numeric_limits<double>::infinity());
    std::vector<bool> used(N + 1, false);

    do {
      used[j0] = true;
      const size_t i0 = p[j0];
      double delta = std::numeric_limits<double>::infinity();
      size_t j1 = 0;

      for (size_t j = 1; j <= N; ++j) {
        if (!used[j]) {
          const double cur = a[i0][j] - u[i0] - v[j];
          if (cur < minv[j]) {
            minv[j] = cur;
            way[j] = j0;
          }
          if (minv[j] < delta) {
            delta = minv[j];
            j1 = j;
          }
        }
      }

      for (size_t j = 0; j <= N; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);

    do {
      const size_t j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 != 0);
  }

  for (size_t j = 1; j <= n_cols; ++j) {
    if (p[j] >= 1 && p[j] <= n_rows) {
      const size_t r = p[j] - 1;
      const size_t c = j - 1;
      if (cost_matrix[r][c] <= max_cost) {
        matches.emplace_back(r, c);
      }
    }
  }
  return matches;
}

void BoxTracker::reset()
{
  tracks_.clear();
  next_track_id_ = 1;
  has_last_stamp_ = false;
  last_stamp_sec_ = 0.0;
  primary_track_id_ = 0;
}

void BoxTracker::predictTrack(TrackState & trk, double dt, const TrackerConfig & cfg) const
{
  // 常速运动转移矩阵（带轻微速度阻尼，抑制静止方块因残余速度引起的位置过冲）
  Eigen::Matrix<double, 6, 6> F = Eigen::Matrix<double, 6, 6>::Identity();
  F(0, 3) = dt;
  F(1, 4) = dt;
  F(2, 5) = dt;
  F(3, 3) = cfg.kf_vel_damping;
  F(4, 4) = cfg.kf_vel_damping;
  F(5, 5) = cfg.kf_vel_damping;

  Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Zero();
  const double q_pos = cfg.kf_process_noise_pos * cfg.kf_process_noise_pos * std::max(dt, 0.02);
  const double q_vel = cfg.kf_process_noise_vel * cfg.kf_process_noise_vel * std::max(dt, 0.02);
  Q(0, 0) = q_pos;
  Q(1, 1) = q_pos;
  Q(2, 2) = q_pos * 0.5;
  Q(3, 3) = q_vel;
  Q(4, 4) = q_vel;
  Q(5, 5) = q_vel * 0.5;

  trk.x = F * trk.x;
  trk.P = F * trk.P * F.transpose() + Q;
}

void BoxTracker::updateTrack(
  TrackState & trk,
  const BoxDetection & det,
  const TrackerConfig & cfg,
  double sym_period) const
{
  Eigen::Matrix<double, 3, 6> H = Eigen::Matrix<double, 3, 6>::Zero();
  H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

  const double meas_std = std::max(cfg.kf_meas_noise_pos, 1e-4);
  const double r_var = meas_std * meas_std;
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity() * r_var;

  const Eigen::Vector3d y = det.center - H * trk.x;
  const Eigen::Matrix3d S = H * trk.P * H.transpose() + R;
  const Eigen::Matrix<double, 6, 3> K = trk.P * H.transpose() * S.inverse();

  trk.x = trk.x + K * y;
  const Eigen::Matrix<double, 6, 6> I = Eigen::Matrix<double, 6, 6>::Identity();
  // 采用 Joseph 形式更新协方差，确保数值对称正定
  const Eigen::Matrix<double, 6, 6> IKH = I - K * H;
  trk.P = IKH * trk.P * IKH.transpose() + K * R * K.transpose();

  // 帧间连续展开偏航角并做 EMA 平滑，彻底防止正方形在 ±45° 边界处发生 90° 跳变
  const double yaw_diff = shortestSymmetricAngleDiff(det.yaw, trk.yaw, sym_period);
  const double meas_yaw_unwrapped = trk.yaw + yaw_diff;
  trk.yaw = trk.yaw + std::clamp(cfg.yaw_alpha, 0.05, 1.0) * yaw_diff;

  // 从观测姿态中分离出纯地面坡度倾斜分量（剔除绕 Z 轴偏航）
  const Eigen::Quaterniond q_yaw_det(Eigen::AngleAxisd(det.yaw, Eigen::Vector3d::UnitZ()));
  trk.latest_tilt = (det.orientation * q_yaw_det.inverse()).normalized();

  trk.size = det.size;
  trk.fit_mode = det.fit_mode;
  trk.has_top_face = det.has_top_face;
  trk.point_count = det.point_count;
  trk.inlier_ratio = det.inlier_ratio;
  trk.rmse = det.rmse;
  trk.score = det.score;

  trk.hits += 1;
  trk.missed = 0;

  // 将当前帧匹配观测（连续展开后的 yaw）压入锁定抖动滑窗历史
  const size_t max_hist = static_cast<size_t>(std::max(cfg.lock_window_frames, 2));
  trk.pos_history.push_back(det.center);
  trk.yaw_history.push_back(meas_yaw_unwrapped);
  while (trk.pos_history.size() > max_hist) {
    trk.pos_history.pop_front();
  }
  while (trk.yaw_history.size() > max_hist) {
    trk.yaw_history.pop_front();
  }
}

void BoxTracker::evaluateLock(
  const TrackState & trk,
  const TrackerConfig & cfg,
  bool & locked_out,
  double & pos_jitter_out,
  double & yaw_jitter_deg_out) const
{
  locked_out = false;
  pos_jitter_out = 0.0;
  yaw_jitter_deg_out = 0.0;

  const size_t req_frames = static_cast<size_t>(std::max(cfg.lock_window_frames, 1));
  if (trk.pos_history.size() < req_frames || trk.yaw_history.size() < req_frames) {
    return;
  }

  Eigen::Vector3d mean_pos = Eigen::Vector3d::Zero();
  for (const auto & p : trk.pos_history) {
    mean_pos += p;
  }
  mean_pos /= static_cast<double>(trk.pos_history.size());

  double max_pos_dev = 0.0;
  for (const auto & p : trk.pos_history) {
    max_pos_dev = std::max(max_pos_dev, (p - mean_pos).norm());
  }

  double mean_yaw = 0.0;
  for (double y : trk.yaw_history) {
    mean_yaw += y;
  }
  mean_yaw /= static_cast<double>(trk.yaw_history.size());

  double max_yaw_dev = 0.0;
  for (double y : trk.yaw_history) {
    max_yaw_dev = std::max(max_yaw_dev, std::abs(y - mean_yaw));
  }

  pos_jitter_out = max_pos_dev;
  yaw_jitter_deg_out = max_yaw_dev * kRadToDeg;

  locked_out =
    (trk.missed == 0) &&
    (trk.hits >= std::max(cfg.min_hits, cfg.lock_window_frames)) &&
    (trk.point_count >= cfg.lock_min_points) &&
    (trk.inlier_ratio >= cfg.lock_min_inlier_ratio) &&
    (trk.rmse <= cfg.lock_max_rmse) &&
    (pos_jitter_out < cfg.lock_max_pos_jitter) &&
    (yaw_jitter_deg_out < cfg.lock_max_yaw_jitter_deg);
}

void BoxTracker::applyEgoMotion(const Eigen::Isometry3d & T_curr_prev)
{
  if (tracks_.empty()) {
    return;
  }
  const Eigen::Matrix3d R = T_curr_prev.linear();
  const Eigen::Vector3d t = T_curr_prev.translation();
  const double dyaw = std::atan2(R(1, 0), R(0, 0));

  Eigen::Matrix<double, 6, 6> M = Eigen::Matrix<double, 6, 6>::Zero();
  M.block<3, 3>(0, 0) = R;
  M.block<3, 3>(3, 3) = R;

  for (auto & trk : tracks_) {
    trk.x.head<3>() = R * trk.x.head<3>() + t;
    trk.x.tail<3>() = R * trk.x.tail<3>();
    trk.P = M * trk.P * M.transpose();

    const Eigen::Matrix3d R_prev_box =
      trk.latest_tilt.toRotationMatrix() *
      Eigen::AngleAxisd(trk.yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const Eigen::Matrix3d R_curr_box = R * R_prev_box;
    trk.yaw += dyaw;
    const Eigen::Matrix3d R_yaw_inv =
      Eigen::AngleAxisd(-trk.yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    trk.latest_tilt = Eigen::Quaterniond(R_curr_box * R_yaw_inv).normalized();

    for (auto & p_hist : trk.pos_history) {
      p_hist = R * p_hist + t;
    }
    for (double & y_hist : trk.yaw_history) {
      y_hist += dyaw;
    }
  }
}

void BoxTracker::update(
  const std::vector<BoxDetection> & detections,
  double stamp_sec,
  const TrackerConfig & cfg,
  bool is_square,
  std::vector<TrackedBox> & tracks_out,
  bool & has_primary_out,
  TrackedBox & primary_out,
  bool & locked_out,
  const Eigen::Isometry3d * T_curr_prev)
{
  const double sym_period = is_square ? kHalfPi : kPi;

  double dt = 0.1;
  bool did_reset = false;
  if (has_last_stamp_) {
    const double raw_dt = stamp_sec - last_stamp_sec_;
    if (raw_dt < -0.5 || raw_dt > 2.0) {
      // 时间戳大幅跳变（如 rosbag 循环回放或数据流中断）-> 清空旧轨迹重新跟踪
      reset();
      did_reset = true;
      dt = 0.1;
    } else {
      dt = std::clamp(raw_dt, 0.01, 0.5);
    }
  }
  has_last_stamp_ = true;
  last_stamp_sec_ = stamp_sec;

  // 0. 当 odom TF 有效时，先将历史轨迹状态补偿到当前帧 base_footprint 消除自车运动滞后
  if (!did_reset && T_curr_prev != nullptr) {
    applyEgoMotion(*T_curr_prev);
  }

  // 1. 对所有现有轨迹执行卡尔曼预测
  for (auto & trk : tracks_) {
    predictTrack(trk, dt, cfg);
  }

  // 2. 构建 3D 欧氏距离代价矩阵并求解匈牙利最优关联
  std::vector<bool> track_matched(tracks_.size(), false);
  std::vector<bool> det_matched(detections.size(), false);

  if (!tracks_.empty() && !detections.empty()) {
    std::vector<std::vector<double>> cost(
      tracks_.size(), std::vector<double>(detections.size(), 1e9));
    for (size_t i = 0; i < tracks_.size(); ++i) {
      const Eigen::Vector3d pred_pos = tracks_[i].x.head<3>();
      for (size_t j = 0; j < detections.size(); ++j) {
        cost[i][j] = (pred_pos - detections[j].center).norm();
      }
    }

    const auto assignments = solveHungarianAssignment(cost, cfg.max_assoc_dist);
    for (const auto & pair : assignments) {
      const size_t ti = pair.first;
      const size_t di = pair.second;
      updateTrack(tracks_[ti], detections[di], cfg, sym_period);
      track_matched[ti] = true;
      det_matched[di] = true;
    }
  }

  // 3. 对未匹配轨迹递增丢帧计数并清理超时轨迹
  for (size_t i = 0; i < tracks_.size(); ++i) {
    if (!track_matched[i]) {
      tracks_[i].missed += 1;
    }
  }
  tracks_.erase(
    std::remove_if(
      tracks_.begin(), tracks_.end(),
      [&](const TrackState & trk) { return trk.missed > cfg.max_missed_frames; }),
    tracks_.end());

  // 4. 为未关联的新检测初始化新轨迹
  for (size_t j = 0; j < detections.size(); ++j) {
    if (det_matched[j]) {
      continue;
    }
    const auto & det = detections[j];
    TrackState trk;
    trk.id = next_track_id_++;
    trk.x.head<3>() = det.center;
    trk.x.tail<3>().setZero();
    trk.P = Eigen::Matrix<double, 6, 6>::Identity() * 0.05;
    trk.P(3, 3) = 0.25;
    trk.P(4, 4) = 0.25;
    trk.P(5, 5) = 0.10;
    // 初始偏航角严格归约到 [-period/2, +period/2)
    trk.yaw = canonicalizeSymmetricYaw(det.yaw, sym_period);
    const Eigen::Quaterniond q_yaw_det(Eigen::AngleAxisd(det.yaw, Eigen::Vector3d::UnitZ()));
    trk.latest_tilt = (det.orientation * q_yaw_det.inverse()).normalized();
    trk.size = det.size;
    trk.fit_mode = det.fit_mode;
    trk.has_top_face = det.has_top_face;
    trk.point_count = det.point_count;
    trk.inlier_ratio = det.inlier_ratio;
    trk.rmse = det.rmse;
    trk.score = det.score;
    trk.hits = 1;
    trk.missed = 0;
    trk.pos_history.push_back(det.center);
    trk.yaw_history.push_back(trk.yaw);
    tracks_.push_back(std::move(trk));
  }

  // 5. 生成已确认目标输出列表，并基于最近距离 + 滞回防抖策略选出主目标
  tracks_out.clear();
  tracks_out.reserve(tracks_.size());

  int prev_primary_idx = -1;
  int nearest_idx = -1;
  double nearest_dist = std::numeric_limits<double>::infinity();

  for (const auto & trk : tracks_) {
    const bool confirmed = (trk.hits >= cfg.min_hits);
    if (!confirmed) {
      continue;
    }
    TrackedBox out;
    out.track_id = trk.id;
    out.center = trk.x.head<3>();
    out.velocity = trk.x.tail<3>();
    out.yaw = trk.yaw;
    out.orientation =
      (trk.latest_tilt * Eigen::Quaterniond(Eigen::AngleAxisd(trk.yaw, Eigen::Vector3d::UnitZ())))
      .normalized();
    out.size = trk.size;
    out.fit_mode = trk.fit_mode;
    out.has_top_face = trk.has_top_face;
    out.point_count = trk.point_count;
    out.inlier_ratio = trk.inlier_ratio;
    out.rmse = trk.rmse;
    out.score = trk.score;
    out.hits = trk.hits;
    out.missed = trk.missed;
    out.confirmed = true;
    evaluateLock(trk, cfg, out.locked, out.pos_jitter, out.yaw_jitter_deg);

    const int cur_idx = static_cast<int>(tracks_out.size());
    tracks_out.push_back(out);

    // 仅允许当前观测到或最多丢失 1 帧的目标参与主目标竞争
    if (out.missed <= 1) {
      const double dist_xy = std::hypot(out.center.x(), out.center.y());
      if (out.track_id == primary_track_id_) {
        prev_primary_idx = cur_idx;
      }
      if (dist_xy < nearest_dist) {
        nearest_dist = dist_xy;
        nearest_idx = cur_idx;
      }
    }
  }

  has_primary_out = false;
  locked_out = false;

  if (nearest_idx >= 0) {
    int chosen_idx = nearest_idx;
    if (prev_primary_idx >= 0 && prev_primary_idx != nearest_idx) {
      const double prev_dist = std::hypot(
        tracks_out[prev_primary_idx].center.x(),
        tracks_out[prev_primary_idx].center.y());
      // 保持原主目标，除非新目标比原主目标至少近 switch_hysteresis_m 才切换
      if (nearest_dist + cfg.switch_hysteresis_m >= prev_dist) {
        chosen_idx = prev_primary_idx;
      }
    }
    tracks_out[chosen_idx].is_primary = true;
    primary_track_id_ = tracks_out[chosen_idx].track_id;
    has_primary_out = true;
    primary_out = tracks_out[chosen_idx];
    locked_out = primary_out.locked;
  } else {
    primary_track_id_ = 0;
  }
}

}  // namespace midtobox

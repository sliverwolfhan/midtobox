// Copyright 2026 midtobox contributors
// SPDX-License-Identifier: Apache-2.0
//
// 核心几何模型、检测/跟踪输出数据结构及对称偏航角计算工具。
// 所有可调 YAML 参数结构体定义在 config_types.hpp（运行时从 config/midtobox.yaml 加载）。

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "midtobox_detector/config_types.hpp"

namespace midtobox
{

/// 归一化地面平面模型：a*x + b*y + c*z + d = 0，满足 ||(a,b,c)|| = 1 且 c > 0（法向朝上）
struct PlaneModel
{
  double a{0.0};
  double b{0.0};
  double c{1.0};
  double d{0.0};
  bool from_ransac{false};  // true 表示由 RANSAC 拟合得到，false 表示回退名义水平面
  int inlier_count{0};      // 地面 RANSAC 内点数

  /// 返回单位法向量（朝向 +Z 半空间）
  Eigen::Vector3d normal() const { return Eigen::Vector3d(a, b, c); }

  /// 计算点 p 到地面的有符号高度（平面上方为正）
  double signedDistance(const Eigen::Vector3d & p) const
  {
    return a * p.x() + b * p.y() + c * p.z() + d;
  }

  /// 计算将地面法向对齐到 +Z (0,0,1) 的最小旋转矩阵 R_level
  Eigen::Matrix3d levelRotation() const
  {
    return Eigen::Quaterniond::FromTwoVectors(normal().normalized(), Eigen::Vector3d::UnitZ())
           .toRotationMatrix();
  }

  /// 将 base_footprint 中的点映射到地面整平系 G（满足 z_G 等于离地高度 signedDistance）
  Eigen::Vector3d toGroundFrame(const Eigen::Vector3d & p_base) const
  {
    const Eigen::Matrix3d R = levelRotation();
    return R * p_base + Eigen::Vector3d(0.0, 0.0, d);
  }

  /// 将地面整平系 G 中的点逆映射回 base_footprint
  Eigen::Vector3d fromGroundFrame(const Eigen::Vector3d & p_ground) const
  {
    const Eigen::Matrix3d R = levelRotation();
    return R.transpose() * p_ground - d * normal();
  }
};

/// 几何拟合分支模式（用于区分单正面外推与 L 形双立面角点外推）
enum class FitMode
{
  kSingleFace = 1,     // 单立面模式：仅可见正面（可能附带顶面）
  kTwoFaceLShape = 2   // L 形双立面模式：同时清晰可见两个相邻正交立面
};

/// 单帧几何拟合输出的方块检测结果
struct BoxDetection
{
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};                 // 3D 质心 (m)
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};  // 3D 姿态四元数（含地面倾角）
  double yaw{0.0};                                                 // 归约后的偏航角 (rad)
  Eigen::Vector3d size{0.35, 0.35, 0.35};                          // 方块尺寸 (L, W, H) (m)

  FitMode fit_mode{FitMode::kSingleFace};  // 实际采用的拟合模式
  bool has_top_face{false};                // 是否检测到有效内缩顶面点
  int point_count{0};                      // 参与拟合的总点数
  int side_point_count{0};                 // 侧面点数
  int top_point_count{0};                  // 内缩顶面点数
  double inlier_ratio{0.0};                // 3D 表面内点比例 [0, 1]
  double rmse{0.0};                        // 表面内点均方根误差 (m)
  double score{0.0};                       // 综合置信度评分 [0, 1]
};

/// 被尺寸/高度/质量门控拒绝的聚类簇（用于 RViz 标红可视化排查）
struct RejectedCluster
{
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};  // 簇在 base_footprint 下的近似中心 (m)
  Eigen::Vector3d size{Eigen::Vector3d::Zero()};    // 簇的 OBB 水平跨度与高度 (m)
  double yaw{0.0};                                  // OBB 主轴偏航角 (rad)
  int point_count{0};                               // 簇点数
  bool is_oversized{false};                         // 是否因粘连/堆叠超尺寸被拒（RViz 标红显示）
  std::string reason;                               // 拒绝原因标识（如 oversized_xy / height_mismatch）
};

/// 经多帧卡尔曼滤波与跟踪后的方块目标状态
struct TrackedBox
{
  uint64_t track_id{0};                                            // 唯一跟踪轨迹 ID
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};                 // 滤波后的 3D 质心 (m)
  Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};               // 估计速度 (m/s)
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};  // 滤波后的 3D 姿态
  double yaw{0.0};                                                 // 帧间连续展开后的偏航角 (rad)
  Eigen::Vector3d size{0.35, 0.35, 0.35};                          // 方块尺寸 (m)

  FitMode fit_mode{FitMode::kSingleFace};  // 最新一帧拟合模式
  bool has_top_face{false};                // 最新一帧是否可见顶面
  int point_count{0};                      // 最新一帧点数
  double inlier_ratio{0.0};                // 最新一帧内点率
  double rmse{0.0};                        // 最新一帧拟合 RMSE (m)
  double score{0.0};                       // 综合评分 [0, 1]

  int hits{0};                             // 累计命中帧数
  int missed{0};                           // 当前连续丢帧数
  bool confirmed{false};                   // 是否已达到 min_hits 成为确认目标
  bool locked{false};                      // 是否满足全部锁定质量与稳定性判据
  bool is_primary{false};                  // 是否被选为当前主目标（输出至 target_pose）
  double pos_jitter{0.0};                  // 锁定滑窗内质心最大抖动 (m)
  double yaw_jitter_deg{0.0};              // 锁定滑窗内偏航角最大抖动 (deg)
};

/// 整条感知流水线单帧执行结果总包
struct PipelineResult
{
  PlaneModel ground_plane;                         // 当前帧地面平面模型
  std::vector<BoxDetection> raw_detections;        // 单帧原始拟合结果列表
  std::vector<TrackedBox> tracks;                  // 所有已确认的跟踪目标列表
  std::vector<RejectedCluster> rejected_clusters;  // 被门控拒判的簇列表

  bool has_primary{false};    // 是否存在有效主目标
  TrackedBox primary_target;  // 主目标状态
  bool locked{false};         // 主目标是否已锁定

  int input_points{0};        // 输入点云总点数
  int roi_points{0};          // ROI 裁剪后点数
  int non_ground_points{0};   // 去地与 ROR 滤波后非地面点数
  double compute_time_ms{0.0};  // 单帧纯计算耗时 (ms)
};

/// 将对称偏航角归约到 [-period/2, +period/2) 区间。
/// 对于正方形（period = pi/2），输出范围固定为 [-pi/4, +pi/4)，消除 90° 对称歧义。
inline double canonicalizeSymmetricYaw(double yaw, double period = kHalfPi)
{
  const double half = 0.5 * period;
  double a = std::fmod(yaw + half, period);
  if (a < 0.0) {
    a += period;
  }
  // 防止浮点舍入导致恰好等于 period
  if (a >= period) {
    a -= period;
  }
  return a - half;
}

/// 计算两个对称角度之间模 `period` 意义下的最短有符号差值 (target - ref)，范围 [-period/2, +period/2)
inline double shortestSymmetricAngleDiff(double target, double ref, double period = kHalfPi)
{
  return canonicalizeSymmetricYaw(target - ref, period);
}

}  // namespace midtobox

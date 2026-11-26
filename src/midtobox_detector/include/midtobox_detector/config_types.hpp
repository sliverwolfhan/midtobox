// Copyright 2026 midtobox contributors
// SPDX-License-Identifier: Apache-2.0
//
// 参数配置结构体定义（与 config/midtobox.yaml 1:1 对应）。
// 日常调参请直接修改 config/midtobox.yaml（或运行期用 ros2 param set 热修改）；
// 此处的成员初始值仅作为 C++ 单元测试不经 ROS 启动时的安全兜底默认值。

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <cstdint>

namespace midtobox
{

// 常用数学常量
constexpr double kPi = 3.14159265358979323846;
constexpr double kHalfPi = kPi * 0.5;
constexpr double kQuarterPi = kPi * 0.25;
constexpr double kDegToRad = kPi / 180.0;
constexpr double kRadToDeg = 180.0 / kPi;

/// Mid-360 原始点云预过滤参数（在转换为 PCL 点云前执行）
struct PreFilterConfig
{
  double min_range{0.25};         // 最小有效测距 (m)，剔除 (0,0,0) 无效点及近距杂散点
  double max_range{6.0};          // 最大有效测距 (m)
  bool use_tag_filter{true};      // 是否启用 Livox Mid-360 硬件 tag 噪点过滤
  uint8_t tag_reject_mask{0x3F};  // 拒绝位掩码：0x3F 过滤 bit 0..5（近距杂散/拉丝/低反噪点）
};

/// 雷达安装外参（描述 base_footprint -> livox_frame 的刚体变换）
struct MountingConfig
{
  bool inverted{true};  // true 表示倒装（自动叠加 +180° roll，将视场变为 -52°~+7°）
  double x{0.0};        // 雷达相对车体基准系 X 偏移 (m)
  double y{0.0};        // 雷达相对车体基准系 Y 偏移 (m)
  double z{0.70};       // 雷达离地安装高度 Z (m)
  double roll_deg{0.0};   // 滚转角标定微调 (deg)
  double pitch_deg{0.0};  // 俯仰角标定微调 (deg)
  double yaw_deg{0.0};    // 偏航角标定微调 (deg)

  /// 计算刚体变换 T_base_lidar（将 lidar_frame 中的点转换到整平后的 base_footprint）
  Eigen::Isometry3d computeTransform() const
  {
    const double eff_roll = (inverted ? kPi : 0.0) + roll_deg * kDegToRad;
    const double pitch = pitch_deg * kDegToRad;
    const double yaw = yaw_deg * kDegToRad;

    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    const Eigen::AngleAxisd rz(yaw, Eigen::Vector3d::UnitZ());
    const Eigen::AngleAxisd ry(pitch, Eigen::Vector3d::UnitY());
    const Eigen::AngleAxisd rx(eff_roll, Eigen::Vector3d::UnitX());
    T.linear() = (rz * ry * rx).toRotationMatrix();
    T.translation() = Eigen::Vector3d(x, y, z);
    return T;
  }
};

/// 自车 3D 剔除盒与作业区 ROI 裁剪参数（在 base_footprint 下生效）
struct CropConfig
{
  bool ego_enabled{true};    // 是否剔除自车车体/机械臂包络点云
  double ego_min_x{-0.50};   // 自车剔除盒 X 下界 (m)
  double ego_max_x{0.40};    // 自车剔除盒 X 上界 (m)
  double ego_min_y{-0.45};   // 自车剔除盒 Y 下界 (m)
  double ego_max_y{0.45};    // 自车剔除盒 Y 上界 (m)
  double ego_min_z{-0.10};   // 自车剔除盒 Z 下界 (m)
  double ego_max_z{1.10};    // 自车剔除盒 Z 上界 (m)

  double roi_min_x{0.45};    // 作业检测区 ROI X 下界 (m)
  double roi_max_x{3.20};    // 作业检测区 ROI X 上界 (m)
  double roi_min_y{-1.20};   // 作业检测区 ROI Y 下界 (m)
  double roi_max_y{1.20};    // 作业检测区 ROI Y 上界 (m)
  double roi_min_z{-0.25};   // 作业检测区 ROI Z 下界 (m)
  double roi_max_z{0.90};    // 作业检测区 ROI Z 上界 (m)

  double voxel_leaf_size{0.0};  // 体素降采样边长 (m)，<= 0 表示关闭以保持边缘锐度
};

/// 倾角约束 RANSAC 地面分割与整平参数
struct GroundConfig
{
  bool use_ransac{true};           // 是否使用 RANSAC 拟合地面平面（false 则用名义水平面）
  double band_min_z{-0.25};        // 地面候选点 Z 搜索带下界 (m)
  double band_max_z{0.25};         // 地面候选点 Z 搜索带上界 (m)
  double max_tilt_deg{8.0};        // 允许的地面最大倾角 (deg)
  double ransac_dist_thresh{0.025};  // RANSAC 地面内点距离阈值 (m)
  int ransac_max_iter{200};        // RANSAC 最大迭代次数
  int min_inliers{30};             // 接受拟合地面所需最少内点数
  double max_plane_offset{0.20};   // 拟合地面相对 nominal_z 允许的最大高度偏差 (m)
  double nominal_z{0.0};           // 名义地面高度 (m)
  double clearance_height{0.045};  // 离地净空阈值 (m)，高于此值判定为非地面点
  bool level_to_ground{true};      // 是否将非地面点云旋转整平到拟合地面坐标系 G
};

/// 边缘“拉丝”飞点去噪参数（ROR + 每簇 SOR）
struct OutlierFilterConfig
{
  bool ror_enabled{true};      // 是否在聚类前启用半径离群滤波（ROR）
  double ror_radius{0.06};     // ROR 搜索半径 (m)
  int ror_min_neighbors{3};    // ROR 半径内最少邻居点数

  bool sor_enabled{true};      // 是否对每个候选簇启用统计离群滤波（SOR）
  int sor_mean_k{12};          // SOR 统计邻域点数 K
  double sor_stddev_mul{2.5};  // SOR 标准差倍率（建议 >=2.2 以保留斜视稀疏顶面）
};

/// PCL 欧氏聚类参数
struct ClusterConfig
{
  double tolerance{0.06};  // 欧氏聚类连通半径 (m)
  int min_size{25};        // 有效簇最少点数
  int max_size{25000};     // 有效簇最多点数
};

/// 候选簇尺寸与拟合质量门控参数（用于拦截杂物与超大粘连/堆叠簇）
struct GateConfig
{
  int min_points{35};                // 进入拟合所需最少点数
  double min_xy_span{0.18};          // 水平主跨度下限 (m)
  double max_xy_span{0.50};          // 水平跨度上限 (m)，超限判定为多块粘连并拒判
  double min_z{0.20};                // 高度下限 (m)
  double max_z{0.45};                // 高度上限 (m)，超限判定为上下堆叠并拒判
  double max_bottom_z{0.25};         // 底部离地最大高度 (m)，过滤悬空物体
  double height_consistency_tol{0.07};  // 实测高度与已知 box.height 的允许偏差 (m)
  double min_inlier_ratio{0.50};     // 3D 箱面内点率下限
  double max_rmse{0.040};            // 3D 箱面内点 RMSE 上限 (m)
};

/// 已知尺寸方块几何拟合参数（L-shape 搜索 + 双面/单面解算 + SE(2) 精修）
struct BoxFittingConfig
{
  double length{0.35};             // 目标方块长度 L (m)
  double width{0.35};              // 目标方块宽度 W (m)
  double height{0.35};             // 目标方块高度 H (m)
  double range_bias_offset{0.0};   // 实测材质测距系统偏置补偿 (m)

  double closeness_d0{0.01};       // Zhang IV 2017 Closeness 准则最小距离下限 d0 (m)
  double coarse_step_deg{1.0};     // 粗搜索角度步长 (deg)
  double fine_step_deg{0.1};       // 精搜索角度步长 (deg)

  double top_band_thickness{0.030};  // 顶部候选点带厚度 (m)
  double side_min_z{0.05};         // 侧面点最小离地高度 (m)
  double side_top_margin{0.045};   // 侧面点距顶面留白 (m)
  int top_min_points{10};          // 判定存在真实顶面所需的最少内缩顶面点数
  double top_max_z_mad{0.025};     // 顶面高度中位绝对偏差 (MAD) 上限 (m)

  double face_band{0.035};         // 单立面归属距离带宽度 (m)
  double two_face_min_span{0.12};  // 双立面判定：两正交轴最小跨度 (m)
  int two_face_min_pts{12};        // 双立面判定：每面独有点最少个数
  double two_face_min_excl_span{0.08};  // 双立面判定：每面独有点最小延伸跨度 (m)

  double huber_delta{0.015};       // Huber 鲁棒损失拐点 (m)
  int ransac_line_iters{100};      // 单正面 RANSAC 直线拟合迭代次数
  double ransac_line_thresh{0.020};  // 单正面 RANSAC 直线内点阈值 (m)

  bool enable_se2_refine{true};    // 是否启用纯 2D SE(2) Huber 点到边高斯牛顿精修
  int se2_max_iters{10};           // SE(2) 精修最大迭代次数
  double se2_max_shift{0.04};      // SE(2) 精修最大允许平移偏移 (m)
  double se2_max_angle_deg{6.0};   // SE(2) 精修最大允许角度修正 (deg)

  double inlier_dist_thresh{0.030};  // 计算 3D 表面内点比例的距离阈值 (m)
};

/// 多目标跟踪（匈牙利 + 6D 卡尔曼）与锁定质量门控参数
struct TrackerConfig
{
  double max_assoc_dist{0.25};       // 帧间最大匹配距离 (m)
  int min_hits{2};                   // 确认有效轨迹所需最少命中帧数
  int max_missed_frames{3};          // 轨迹销毁前允许的最大连续丢帧数

  double kf_process_noise_pos{0.020};  // 卡尔曼位置过程噪声标准差 (m)
  double kf_process_noise_vel{0.80};   // 卡尔曼速度过程噪声标准差 (m/s)
  double kf_meas_noise_pos{0.015};     // 卡尔曼观测位置噪声标准差 (m)
  double kf_vel_damping{0.95};       // 速度阻尼系数（抑制静止目标过冲）
  double yaw_alpha{0.45};            // 偏航角指数滑动平均滤波系数

  double switch_hysteresis_m{0.15};  // 主目标切换滞回距离 (m)

  int lock_min_points{80};           // 锁定门控：最少点数
  double lock_min_inlier_ratio{0.70};  // 锁定门控：最小内点比例
  double lock_max_rmse{0.025};       // 锁定门控：最大表面 RMSE (m)
  int lock_window_frames{3};         // 锁定门控：抖动评估滑窗帧数
  double lock_max_pos_jitter{0.015};  // 锁定门控：滑窗内最大质心抖动 (m)
  double lock_max_yaw_jitter_deg{3.0};  // 锁定门控：滑窗内最大偏航角抖动 (deg)
};

/// 整条纯 C++ 感知流水线的汇总配置
struct PipelineConfig
{
  CropConfig crop;
  GroundConfig ground;
  OutlierFilterConfig outlier;
  ClusterConfig cluster;
  GateConfig gate;
  BoxFittingConfig fitting;
  TrackerConfig tracker;
};

}  // namespace midtobox

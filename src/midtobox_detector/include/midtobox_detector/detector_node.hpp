// Copyright 2026 midtobox contributors
// SPDX-License-Identifier: Apache-2.0
//
// ROS 2 Humble 识块感知节点，封装 DetectorPipeline 并提供：
//   - 双输入格式支持：sensor_msgs/msg/PointCloud2 与 livox_ros_driver2/msg/CustomMsg
//   - 转 PCL 前执行 Mid-360 tag 位掩码与近距/(0,0,0) 预过滤
//   - 可开关内置静态 TF 发布（base_footprint -> livox_frame）
//   - 200~300 ms 滑动窗口累积（有 odom TF 自动运动补偿，无 odom 平滑退化为静止累积）
//   - 可配置输出坐标系（base_footprint / livox_frame / odom）与运行期热调参
//   - 发布主目标位姿、全量检测数组、锁定状态、RViz 可视化标记及可选调试点云

#pragma once

#include <tf2_ros/buffer.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <deque>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/bool.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "midtobox_detector/detector_pipeline.hpp"
#include "midtobox_detector/types.hpp"

namespace midtobox
{

class MidtoboxDetectorNode : public rclcpp::Node
{
public:
  explicit MidtoboxDetectorNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  /// 滑动窗口内缓存的单帧点云与位姿快照（点云已变换到该帧时刻的 base_footprint 且剔除自车）
  struct WindowFrame
  {
    double stamp_sec{0.0};                                         // 帧时间戳 (s)
    bool has_odom{false};                                          // 该帧时刻是否获取到新鲜 odom TF
    Eigen::Isometry3d T_odom_base{Eigen::Isometry3d::Identity()};  // odom -> base_footprint 变换
    DetectorPipeline::CloudPtr cloud;                              // 帧内裁剪后的点云
  };

  /// 声明并从 YAML 加载全部 ROS 2 参数
  void declareAndLoadParameters();

  /// 运行期动态参数回调（支持 ros2 param set 原子热更新，自动拒绝启动项与非法值）
  rcl_interfaces::msg::SetParametersResult onParametersSet(
    const std::vector<rclcpp::Parameter> & params);

  /// 发布静态 TF：base_footprint -> livox_frame
  void publishStaticLidarTf();

  /// PointCloud2 话题回调（xfer_format=0）
  void pointCloud2Callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);

  /// Livox CustomMsg 话题回调（xfer_format=1）
  void customMsgCallback(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr msg);

  /// 解析 PointCloud2（支持 2D row_step 填充，同步过滤 Mid-360 tag 与距离）
  DetectorPipeline::CloudPtr parsePointCloud2(const sensor_msgs::msg::PointCloud2 & msg);

  /// 解析 Livox CustomMsg（同步过滤 Mid-360 tag 与距离）
  DetectorPipeline::CloudPtr parseCustomMsg(
    const livox_ros_driver2::msg::CustomMsg & msg) const;

  /// 统一处理传感器点云：整平 -> 每帧去自车 -> 滑窗累积 -> 识块跟踪 -> 坐标变换发布
  void processSensorCloud(
    const DetectorPipeline::CloudConstPtr & sensor_cloud,
    const std_msgs::msg::Header & header);

  /// 查询传感器坐标系到 base_footprint 的变换（外部 TF 不可用且为默认雷达系时回退 YAML 外参）
  bool lookupSensorToBase(
    const std::string & sensor_frame,
    const rclcpp::Time & stamp,
    Eigen::Isometry3d & T_base_sensor_out);

  /// 查询新鲜的 odom -> base_footprint 变换（超过 odom_tf_max_age_sec_ 视为无效）
  bool lookupOdomToBase(
    const rclcpp::Time & stamp,
    Eigen::Isometry3d & T_odom_base_out);

  /// 查询 base_footprint 到目标输出坐标系 output_frame 的变换（超时/缺失时安全回退 base_footprint）
  bool lookupBaseToOutput(
    const rclcpp::Time & stamp,
    bool has_odom,
    const Eigen::Isometry3d & T_odom_base,
    Eigen::Isometry3d & T_out_base_out,
    std::string & resolved_frame_out);

  /// 将流水线结果转换到输出坐标系并发布全部 ROS 2 话题与 RViz 标记
  void publishResults(
    const PipelineResult & res,
    const rclcpp::Time & stamp,
    const Eigen::Isometry3d & T_out_base,
    const std::string & out_frame,
    const DetectorPipeline::CloudConstPtr & roi_cloud,
    const DetectorPipeline::CloudConstPtr & non_ground_cloud);

  // 节点级 IO / 话题 / 坐标系配置（默认值与 config/midtobox.yaml 一致）
  std::string input_topic_{"/livox/lidar"};
  std::string input_format_{"pointcloud2"};
  std::string qos_reliability_{"best_effort"};
  std::string target_pose_topic_{"/midtobox/target_pose"};
  std::string detections_topic_{"/midtobox/detections"};
  std::string locked_topic_{"/midtobox/locked"};
  std::string markers_topic_{"/midtobox/markers"};
  std::string debug_accum_topic_{"/midtobox/debug/accumulated_cloud"};
  std::string debug_nonground_topic_{"/midtobox/debug/non_ground_cloud"};
  std::string base_frame_{"base_footprint"};
  std::string lidar_frame_{"livox_frame"};
  std::string odom_frame_{"odom"};
  std::string output_frame_{"base_footprint"};
  bool publish_static_tf_{true};

  PreFilterConfig prefilter_cfg_;
  MountingConfig mounting_cfg_;
  Eigen::Isometry3d T_base_lidar_{Eigen::Isometry3d::Identity()};

  // 滑动窗口配置
  double window_duration_sec_{0.25};
  int window_max_frames_{4};
  double odom_tf_max_age_sec_{0.15};
  double storage_roi_padding_{1.0};

  // 调试可视化开关
  bool debug_publish_clouds_{false};
  bool debug_publish_roi_marker_{true};
  bool debug_publish_rejected_markers_{true};

  // 动态参数回调句柄
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;

  // 感知流水线与滑动窗口运行状态
  DetectorPipeline pipeline_;
  std::deque<WindowFrame> window_buffer_;
  bool odom_mode_logged_{false};
  bool last_odom_available_{false};
  bool has_prev_odom_{false};
  Eigen::Isometry3d prev_T_odom_base_{Eigen::Isometry3d::Identity()};

  // TF2 缓存与静态广播器
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;

  // ROS 2 订阅器与发布器
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc2_sub_;
  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr custom_sub_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_pub_;
  rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr detections_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr locked_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_accum_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_nonground_pub_;
};

}  // namespace midtobox

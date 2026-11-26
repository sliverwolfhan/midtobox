// Copyright 2026 midtobox contributors
// SPDX-License-Identifier: Apache-2.0
//
// ROS 2 Humble node wrapping DetectorPipeline with:
//   - Dual input support: sensor_msgs/msg/PointCloud2 & livox_ros_driver2/msg/CustomMsg
//   - Mid-360 tag & min-range pre-filtering before PCL conversion
//   - Built-in toggleable static TF broadcaster (base_footprint -> livox_frame)
//   - 200-300 ms sliding-window accumulation with automatic odom->base_footprint
//     motion compensation and graceful static fallback
//   - Configurable output_frame (base_footprint, livox_frame, odom)
//   - Outputs: /midtobox/target_pose, /midtobox/detections, /midtobox/locked,
//     /midtobox/markers, and optional debug point clouds.

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
  struct WindowFrame
  {
    double stamp_sec{0.0};
    bool has_odom{false};
    Eigen::Isometry3d T_odom_base{Eigen::Isometry3d::Identity()};
    DetectorPipeline::CloudPtr cloud;
  };

  void declareAndLoadParameters();
  rcl_interfaces::msg::SetParametersResult onParametersSet(
    const std::vector<rclcpp::Parameter> & params);
  void publishStaticLidarTf();

  void pointCloud2Callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
  void customMsgCallback(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr msg);

  DetectorPipeline::CloudPtr parsePointCloud2(const sensor_msgs::msg::PointCloud2 & msg);
  DetectorPipeline::CloudPtr parseCustomMsg(
    const livox_ros_driver2::msg::CustomMsg & msg) const;

  void processSensorCloud(
    const DetectorPipeline::CloudConstPtr & sensor_cloud,
    const std_msgs::msg::Header & header);

  bool lookupSensorToBase(
    const std::string & sensor_frame,
    const rclcpp::Time & stamp,
    Eigen::Isometry3d & T_base_sensor_out);

  bool lookupOdomToBase(
    const rclcpp::Time & stamp,
    Eigen::Isometry3d & T_odom_base_out);

  bool lookupBaseToOutput(
    const rclcpp::Time & stamp,
    bool has_odom,
    const Eigen::Isometry3d & T_odom_base,
    Eigen::Isometry3d & T_out_base_out,
    std::string & resolved_frame_out);

  void publishResults(
    const PipelineResult & res,
    const rclcpp::Time & stamp,
    const Eigen::Isometry3d & T_out_base,
    const std::string & out_frame,
    const DetectorPipeline::CloudConstPtr & roi_cloud,
    const DetectorPipeline::CloudConstPtr & non_ground_cloud);

  // Configuration
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

  double window_duration_sec_{0.25};
  int window_max_frames_{4};
  double odom_tf_max_age_sec_{0.15};
  double storage_roi_padding_{1.0};

  bool debug_publish_clouds_{false};
  bool debug_publish_roi_marker_{true};
  bool debug_publish_rejected_markers_{true};

  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;

  // Pipeline & sliding-window state
  DetectorPipeline pipeline_;
  std::deque<WindowFrame> window_buffer_;
  bool odom_mode_logged_{false};
  bool last_odom_available_{false};
  bool has_prev_odom_{false};
  Eigen::Isometry3d prev_T_odom_base_{Eigen::Isometry3d::Identity()};

  // TF2
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;

  // ROS 2 Subscriptions & Publishers
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

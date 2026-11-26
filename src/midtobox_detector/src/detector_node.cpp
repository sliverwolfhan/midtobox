// Copyright 2026 midtobox contributors
// SPDX-License-Identifier: Apache-2.0
//
// ROS 2 Humble node implementation for midtobox_detector.

#include "midtobox_detector/detector_node.hpp"

#include <pcl_conversions/pcl_conversions.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace midtobox
{

namespace
{

geometry_msgs::msg::Pose eigenPoseToMsg(
  const Eigen::Vector3d & pos,
  const Eigen::Quaterniond & quat)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = pos.x();
  pose.position.y = pos.y();
  pose.position.z = pos.z();
  const Eigen::Quaterniond q = quat.normalized();
  pose.orientation.x = q.x();
  pose.orientation.y = q.y();
  pose.orientation.z = q.z();
  pose.orientation.w = q.w();
  return pose;
}

double quaternionToYaw(const Eigen::Quaterniond & q)
{
  const Eigen::Quaterniond qn = q.normalized();
  const double siny_cosp = 2.0 * (qn.w() * qn.z() + qn.x() * qn.y());
  const double cosy_cosp = 1.0 - 2.0 * (qn.y() * qn.y() + qn.z() * qn.z());
  return std::atan2(siny_cosp, cosy_cosp);
}

std::string stripLeadingSlash(std::string frame)
{
  while (!frame.empty() && frame.front() == '/') {
    frame.erase(frame.begin());
  }
  return frame;
}

}  // namespace

MidtoboxDetectorNode::MidtoboxDetectorNode(const rclcpp::NodeOptions & options)
: Node("midtobox_detector", options)
{
  declareAndLoadParameters();

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  if (publish_static_tf_) {
    static_tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    publishStaticLidarTf();
  }

  // Output publishers (RELIABLE for downstream consumers and RViz)
  const auto reliable_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
  target_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
    target_pose_topic_, reliable_qos);
  detections_pub_ = this->create_publisher<vision_msgs::msg::Detection3DArray>(
    detections_topic_, reliable_qos);
  locked_pub_ = this->create_publisher<std_msgs::msg::Bool>(
    locked_topic_, reliable_qos);
  markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    markers_topic_, reliable_qos);

  if (debug_publish_clouds_) {
    const auto debug_qos = rclcpp::SensorDataQoS().keep_last(2);
    debug_accum_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      debug_accum_topic_, debug_qos);
    debug_nonground_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      debug_nonground_topic_, debug_qos);
  }

  // Sensor subscription QoS: BEST_EFFORT subscribes cleanly to both RELIABLE (livox_ros_driver2)
  // and BEST_EFFORT (SensorDataQoS / sim / bag) publishers under DDS RxO rules.
  rclcpp::QoS sub_qos = rclcpp::QoS(rclcpp::KeepLast(20)).durability_volatile();
  if (qos_reliability_ == "reliable") {
    sub_qos.reliable();
  } else {
    if (qos_reliability_ != "best_effort") {
      RCLCPP_WARN(
        this->get_logger(),
        "Unrecognized qos_reliability='%s'; expected 'best_effort' or 'reliable'. "
        "Falling back to 'best_effort'.",
        qos_reliability_.c_str());
    }
    sub_qos.best_effort();
  }

  if (input_format_ == "custom_msg") {
    custom_sub_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
      input_topic_, sub_qos,
      std::bind(&MidtoboxDetectorNode::customMsgCallback, this, std::placeholders::_1));
    RCLCPP_INFO(
      this->get_logger(),
      "midtobox_detector subscribed to %s (format=custom_msg, qos=%s)",
      input_topic_.c_str(), qos_reliability_.c_str());
  } else {
    if (input_format_ != "pointcloud2") {
      RCLCPP_WARN(
        this->get_logger(),
        "Unrecognized input_format='%s'; expected 'pointcloud2' or 'custom_msg'. "
        "Falling back to 'pointcloud2'.",
        input_format_.c_str());
    }
    pc2_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, sub_qos,
      std::bind(&MidtoboxDetectorNode::pointCloud2Callback, this, std::placeholders::_1));
    RCLCPP_INFO(
      this->get_logger(),
      "midtobox_detector subscribed to %s (format=pointcloud2, qos=%s)",
      input_topic_.c_str(), qos_reliability_.c_str());
  }

  // Enable live parameter tuning via `ros2 param set` after startup declaration completes
  param_cb_handle_ = this->add_on_set_parameters_callback(
    std::bind(&MidtoboxDetectorNode::onParametersSet, this, std::placeholders::_1));

  RCLCPP_INFO(
    this->get_logger(),
    "Mounting: inverted=%s, xyz=(%.3f, %.3f, %.3f), trim_rpy_deg=(%.2f, %.2f, %.2f) | "
    "frames: base=%s, lidar=%s, odom=%s, output=%s | static_tf=%s | box=%.3fx%.3fx%.3fm",
    mounting_cfg_.inverted ? "true" : "false",
    mounting_cfg_.x, mounting_cfg_.y, mounting_cfg_.z,
    mounting_cfg_.roll_deg, mounting_cfg_.pitch_deg, mounting_cfg_.yaw_deg,
    base_frame_.c_str(), lidar_frame_.c_str(), odom_frame_.c_str(), output_frame_.c_str(),
    publish_static_tf_ ? "true" : "false",
    pipeline_.config().fitting.length,
    pipeline_.config().fitting.width,
    pipeline_.config().fitting.height);
}

void MidtoboxDetectorNode::declareAndLoadParameters()
{
  // IO, topics & frames (defaults sourced from member initializers / config_types.hpp)
  input_topic_ = this->declare_parameter<std::string>("input_topic", input_topic_);
  input_format_ = this->declare_parameter<std::string>("input_format", input_format_);
  qos_reliability_ = this->declare_parameter<std::string>("qos_reliability", qos_reliability_);

  target_pose_topic_ =
    this->declare_parameter<std::string>("topics.target_pose", target_pose_topic_);
  detections_topic_ =
    this->declare_parameter<std::string>("topics.detections", detections_topic_);
  locked_topic_ = this->declare_parameter<std::string>("topics.locked", locked_topic_);
  markers_topic_ = this->declare_parameter<std::string>("topics.markers", markers_topic_);
  debug_accum_topic_ = this->declare_parameter<std::string>(
    "topics.debug_accumulated_cloud", debug_accum_topic_);
  debug_nonground_topic_ = this->declare_parameter<std::string>(
    "topics.debug_non_ground_cloud", debug_nonground_topic_);

  base_frame_ = stripLeadingSlash(
    this->declare_parameter<std::string>("base_frame", base_frame_));
  lidar_frame_ = stripLeadingSlash(
    this->declare_parameter<std::string>("lidar_frame", lidar_frame_));
  odom_frame_ = stripLeadingSlash(
    this->declare_parameter<std::string>("odom_frame", odom_frame_));
  output_frame_ = stripLeadingSlash(
    this->declare_parameter<std::string>("output_frame", output_frame_));
  publish_static_tf_ = this->declare_parameter<bool>("publish_static_tf", publish_static_tf_);

  // Mounting
  mounting_cfg_.inverted =
    this->declare_parameter<bool>("mounting.inverted", mounting_cfg_.inverted);
  mounting_cfg_.x = this->declare_parameter<double>("mounting.x", mounting_cfg_.x);
  mounting_cfg_.y = this->declare_parameter<double>("mounting.y", mounting_cfg_.y);
  mounting_cfg_.z = this->declare_parameter<double>("mounting.z", mounting_cfg_.z);
  mounting_cfg_.roll_deg =
    this->declare_parameter<double>("mounting.roll_deg", mounting_cfg_.roll_deg);
  mounting_cfg_.pitch_deg =
    this->declare_parameter<double>("mounting.pitch_deg", mounting_cfg_.pitch_deg);
  mounting_cfg_.yaw_deg =
    this->declare_parameter<double>("mounting.yaw_deg", mounting_cfg_.yaw_deg);
  T_base_lidar_ = mounting_cfg_.computeTransform();

  // Sliding window
  window_duration_sec_ =
    this->declare_parameter<double>("window.duration_sec", window_duration_sec_);
  window_max_frames_ = this->declare_parameter<int>("window.max_frames", window_max_frames_);
  odom_tf_max_age_sec_ =
    this->declare_parameter<double>("window.odom_tf_max_age_sec", odom_tf_max_age_sec_);
  storage_roi_padding_ =
    this->declare_parameter<double>("window.storage_roi_padding", storage_roi_padding_);

  // Pre-filter (Mid-360 tag & range)
  prefilter_cfg_.min_range =
    this->declare_parameter<double>("prefilter.min_range", prefilter_cfg_.min_range);
  prefilter_cfg_.max_range =
    this->declare_parameter<double>("prefilter.max_range", prefilter_cfg_.max_range);
  prefilter_cfg_.use_tag_filter =
    this->declare_parameter<bool>("prefilter.use_tag_filter", prefilter_cfg_.use_tag_filter);
  const int mask_int = this->declare_parameter<int>(
    "prefilter.tag_reject_mask", static_cast<int>(prefilter_cfg_.tag_reject_mask));
  prefilter_cfg_.tag_reject_mask = static_cast<uint8_t>(std::clamp(mask_int, 0, 255));

  PipelineConfig cfg;

  // Ego crop & workspace ROI
  cfg.crop.ego_enabled =
    this->declare_parameter<bool>("crop.ego_enabled", cfg.crop.ego_enabled);
  cfg.crop.ego_min_x = this->declare_parameter<double>("crop.ego_min_x", cfg.crop.ego_min_x);
  cfg.crop.ego_max_x = this->declare_parameter<double>("crop.ego_max_x", cfg.crop.ego_max_x);
  cfg.crop.ego_min_y = this->declare_parameter<double>("crop.ego_min_y", cfg.crop.ego_min_y);
  cfg.crop.ego_max_y = this->declare_parameter<double>("crop.ego_max_y", cfg.crop.ego_max_y);
  cfg.crop.ego_min_z = this->declare_parameter<double>("crop.ego_min_z", cfg.crop.ego_min_z);
  cfg.crop.ego_max_z = this->declare_parameter<double>("crop.ego_max_z", cfg.crop.ego_max_z);
  cfg.crop.roi_min_x = this->declare_parameter<double>("crop.roi_min_x", cfg.crop.roi_min_x);
  cfg.crop.roi_max_x = this->declare_parameter<double>("crop.roi_max_x", cfg.crop.roi_max_x);
  cfg.crop.roi_min_y = this->declare_parameter<double>("crop.roi_min_y", cfg.crop.roi_min_y);
  cfg.crop.roi_max_y = this->declare_parameter<double>("crop.roi_max_y", cfg.crop.roi_max_y);
  cfg.crop.roi_min_z = this->declare_parameter<double>("crop.roi_min_z", cfg.crop.roi_min_z);
  cfg.crop.roi_max_z = this->declare_parameter<double>("crop.roi_max_z", cfg.crop.roi_max_z);
  cfg.crop.voxel_leaf_size =
    this->declare_parameter<double>("crop.voxel_leaf_size", cfg.crop.voxel_leaf_size);

  // Ground segmentation
  cfg.ground.use_ransac =
    this->declare_parameter<bool>("ground.use_ransac", cfg.ground.use_ransac);
  cfg.ground.band_min_z =
    this->declare_parameter<double>("ground.band_min_z", cfg.ground.band_min_z);
  cfg.ground.band_max_z =
    this->declare_parameter<double>("ground.band_max_z", cfg.ground.band_max_z);
  cfg.ground.max_tilt_deg =
    this->declare_parameter<double>("ground.max_tilt_deg", cfg.ground.max_tilt_deg);
  cfg.ground.ransac_dist_thresh = this->declare_parameter<double>(
    "ground.ransac_dist_thresh", cfg.ground.ransac_dist_thresh);
  cfg.ground.ransac_max_iter =
    this->declare_parameter<int>("ground.ransac_max_iter", cfg.ground.ransac_max_iter);
  cfg.ground.min_inliers =
    this->declare_parameter<int>("ground.min_inliers", cfg.ground.min_inliers);
  cfg.ground.max_plane_offset =
    this->declare_parameter<double>("ground.max_plane_offset", cfg.ground.max_plane_offset);
  cfg.ground.nominal_z =
    this->declare_parameter<double>("ground.nominal_z", cfg.ground.nominal_z);
  cfg.ground.clearance_height =
    this->declare_parameter<double>("ground.clearance_height", cfg.ground.clearance_height);
  cfg.ground.level_to_ground =
    this->declare_parameter<bool>("ground.level_to_ground", cfg.ground.level_to_ground);

  // Outlier removal
  cfg.outlier.ror_enabled =
    this->declare_parameter<bool>("outlier.ror_enabled", cfg.outlier.ror_enabled);
  cfg.outlier.ror_radius =
    this->declare_parameter<double>("outlier.ror_radius", cfg.outlier.ror_radius);
  cfg.outlier.ror_min_neighbors =
    this->declare_parameter<int>("outlier.ror_min_neighbors", cfg.outlier.ror_min_neighbors);
  cfg.outlier.sor_enabled =
    this->declare_parameter<bool>("outlier.sor_enabled", cfg.outlier.sor_enabled);
  cfg.outlier.sor_mean_k =
    this->declare_parameter<int>("outlier.sor_mean_k", cfg.outlier.sor_mean_k);
  cfg.outlier.sor_stddev_mul =
    this->declare_parameter<double>("outlier.sor_stddev_mul", cfg.outlier.sor_stddev_mul);

  // Clustering
  cfg.cluster.tolerance =
    this->declare_parameter<double>("cluster.tolerance", cfg.cluster.tolerance);
  cfg.cluster.min_size = this->declare_parameter<int>("cluster.min_size", cfg.cluster.min_size);
  cfg.cluster.max_size = this->declare_parameter<int>("cluster.max_size", cfg.cluster.max_size);

  // Size & quality gates
  cfg.gate.min_points = this->declare_parameter<int>("gate.min_points", cfg.gate.min_points);
  cfg.gate.min_xy_span = this->declare_parameter<double>("gate.min_xy_span", cfg.gate.min_xy_span);
  cfg.gate.max_xy_span = this->declare_parameter<double>("gate.max_xy_span", cfg.gate.max_xy_span);
  cfg.gate.min_z = this->declare_parameter<double>("gate.min_z", cfg.gate.min_z);
  cfg.gate.max_z = this->declare_parameter<double>("gate.max_z", cfg.gate.max_z);
  cfg.gate.max_bottom_z =
    this->declare_parameter<double>("gate.max_bottom_z", cfg.gate.max_bottom_z);
  cfg.gate.height_consistency_tol = this->declare_parameter<double>(
    "gate.height_consistency_tol", cfg.gate.height_consistency_tol);
  cfg.gate.min_inlier_ratio =
    this->declare_parameter<double>("gate.min_inlier_ratio", cfg.gate.min_inlier_ratio);
  cfg.gate.max_rmse = this->declare_parameter<double>("gate.max_rmse", cfg.gate.max_rmse);

  // Box dimensions & fitting
  cfg.fitting.length = this->declare_parameter<double>("box.length", cfg.fitting.length);
  cfg.fitting.width = this->declare_parameter<double>("box.width", cfg.fitting.width);
  cfg.fitting.height = this->declare_parameter<double>("box.height", cfg.fitting.height);
  cfg.fitting.range_bias_offset =
    this->declare_parameter<double>("box.range_bias_offset", cfg.fitting.range_bias_offset);

  cfg.fitting.closeness_d0 =
    this->declare_parameter<double>("fitting.closeness_d0", cfg.fitting.closeness_d0);
  cfg.fitting.coarse_step_deg =
    this->declare_parameter<double>("fitting.coarse_step_deg", cfg.fitting.coarse_step_deg);
  cfg.fitting.fine_step_deg =
    this->declare_parameter<double>("fitting.fine_step_deg", cfg.fitting.fine_step_deg);
  cfg.fitting.top_band_thickness = this->declare_parameter<double>(
    "fitting.top_band_thickness", cfg.fitting.top_band_thickness);
  cfg.fitting.side_min_z =
    this->declare_parameter<double>("fitting.side_min_z", cfg.fitting.side_min_z);
  cfg.fitting.side_top_margin =
    this->declare_parameter<double>("fitting.side_top_margin", cfg.fitting.side_top_margin);
  cfg.fitting.top_min_points =
    this->declare_parameter<int>("fitting.top_min_points", cfg.fitting.top_min_points);
  cfg.fitting.top_max_z_mad =
    this->declare_parameter<double>("fitting.top_max_z_mad", cfg.fitting.top_max_z_mad);
  cfg.fitting.face_band =
    this->declare_parameter<double>("fitting.face_band", cfg.fitting.face_band);
  cfg.fitting.two_face_min_span = this->declare_parameter<double>(
    "fitting.two_face_min_span", cfg.fitting.two_face_min_span);
  cfg.fitting.two_face_min_pts =
    this->declare_parameter<int>("fitting.two_face_min_pts", cfg.fitting.two_face_min_pts);
  cfg.fitting.two_face_min_excl_span = this->declare_parameter<double>(
    "fitting.two_face_min_excl_span", cfg.fitting.two_face_min_excl_span);
  cfg.fitting.huber_delta =
    this->declare_parameter<double>("fitting.huber_delta", cfg.fitting.huber_delta);
  cfg.fitting.ransac_line_iters =
    this->declare_parameter<int>("fitting.ransac_line_iters", cfg.fitting.ransac_line_iters);
  cfg.fitting.ransac_line_thresh = this->declare_parameter<double>(
    "fitting.ransac_line_thresh", cfg.fitting.ransac_line_thresh);
  cfg.fitting.enable_se2_refine =
    this->declare_parameter<bool>("fitting.enable_se2_refine", cfg.fitting.enable_se2_refine);
  cfg.fitting.se2_max_iters =
    this->declare_parameter<int>("fitting.se2_max_iters", cfg.fitting.se2_max_iters);
  cfg.fitting.se2_max_shift =
    this->declare_parameter<double>("fitting.se2_max_shift", cfg.fitting.se2_max_shift);
  cfg.fitting.se2_max_angle_deg = this->declare_parameter<double>(
    "fitting.se2_max_angle_deg", cfg.fitting.se2_max_angle_deg);
  cfg.fitting.inlier_dist_thresh = this->declare_parameter<double>(
    "fitting.inlier_dist_thresh", cfg.fitting.inlier_dist_thresh);

  // Tracker & lock gating
  cfg.tracker.max_assoc_dist =
    this->declare_parameter<double>("tracker.max_assoc_dist", cfg.tracker.max_assoc_dist);
  cfg.tracker.min_hits = this->declare_parameter<int>("tracker.min_hits", cfg.tracker.min_hits);
  cfg.tracker.max_missed_frames =
    this->declare_parameter<int>("tracker.max_missed_frames", cfg.tracker.max_missed_frames);
  cfg.tracker.kf_process_noise_pos = this->declare_parameter<double>(
    "tracker.kf_process_noise_pos", cfg.tracker.kf_process_noise_pos);
  cfg.tracker.kf_process_noise_vel = this->declare_parameter<double>(
    "tracker.kf_process_noise_vel", cfg.tracker.kf_process_noise_vel);
  cfg.tracker.kf_meas_noise_pos = this->declare_parameter<double>(
    "tracker.kf_meas_noise_pos", cfg.tracker.kf_meas_noise_pos);
  cfg.tracker.kf_vel_damping =
    this->declare_parameter<double>("tracker.kf_vel_damping", cfg.tracker.kf_vel_damping);
  cfg.tracker.yaw_alpha =
    this->declare_parameter<double>("tracker.yaw_alpha", cfg.tracker.yaw_alpha);
  cfg.tracker.switch_hysteresis_m = this->declare_parameter<double>(
    "tracker.switch_hysteresis_m", cfg.tracker.switch_hysteresis_m);

  cfg.tracker.lock_min_points =
    this->declare_parameter<int>("lock.min_points", cfg.tracker.lock_min_points);
  cfg.tracker.lock_min_inlier_ratio = this->declare_parameter<double>(
    "lock.min_inlier_ratio", cfg.tracker.lock_min_inlier_ratio);
  cfg.tracker.lock_max_rmse =
    this->declare_parameter<double>("lock.max_rmse", cfg.tracker.lock_max_rmse);
  cfg.tracker.lock_window_frames =
    this->declare_parameter<int>("lock.window_frames", cfg.tracker.lock_window_frames);
  cfg.tracker.lock_max_pos_jitter = this->declare_parameter<double>(
    "lock.max_pos_jitter", cfg.tracker.lock_max_pos_jitter);
  cfg.tracker.lock_max_yaw_jitter_deg = this->declare_parameter<double>(
    "lock.max_yaw_jitter_deg", cfg.tracker.lock_max_yaw_jitter_deg);

  // Debug
  debug_publish_clouds_ =
    this->declare_parameter<bool>("debug.publish_clouds", debug_publish_clouds_);
  debug_publish_roi_marker_ =
    this->declare_parameter<bool>("debug.publish_roi_marker", debug_publish_roi_marker_);
  debug_publish_rejected_markers_ = this->declare_parameter<bool>(
    "debug.publish_rejected_markers", debug_publish_rejected_markers_);

  pipeline_.setConfig(cfg);
}

rcl_interfaces::msg::SetParametersResult MidtoboxDetectorNode::onParametersSet(
  const std::vector<rclcpp::Parameter> & params)
{
  rcl_interfaces::msg::SetParametersResult res;
  res.successful = true;

  PipelineConfig next_cfg = pipeline_.config();
  MountingConfig next_mounting = mounting_cfg_;
  PreFilterConfig next_prefilter = prefilter_cfg_;
  double next_window_dur = window_duration_sec_;
  int next_window_max = window_max_frames_;
  double next_odom_age = odom_tf_max_age_sec_;
  double next_pad = storage_roi_padding_;
  std::string next_output_frame = output_frame_;
  bool next_pub_roi = debug_publish_roi_marker_;
  bool next_pub_rej = debug_publish_rejected_markers_;
  bool mounting_changed = false;

  for (const auto & p : params) {
    const std::string & name = p.get_name();

    // Startup-only parameters (subscriptions, publishers, static frame names)
    if (name == "input_topic" || name == "input_format" || name == "qos_reliability" ||
        name == "base_frame" || name == "lidar_frame" || name == "odom_frame" ||
        name == "publish_static_tf" || name == "debug.publish_clouds" ||
        name.rfind("topics.", 0) == 0)
    {
      res.successful = false;
      res.reason = "Parameter '" + name + "' requires node restart";
      return res;
    }

    if (name == "output_frame") {
      next_output_frame = stripLeadingSlash(p.as_string());
    } else if (name == "mounting.inverted") {
      next_mounting.inverted = p.as_bool();
      mounting_changed = true;
    } else if (name == "mounting.x") {
      next_mounting.x = p.as_double();
      mounting_changed = true;
    } else if (name == "mounting.y") {
      next_mounting.y = p.as_double();
      mounting_changed = true;
    } else if (name == "mounting.z") {
      next_mounting.z = p.as_double();
      mounting_changed = true;
    } else if (name == "mounting.roll_deg") {
      next_mounting.roll_deg = p.as_double();
      mounting_changed = true;
    } else if (name == "mounting.pitch_deg") {
      next_mounting.pitch_deg = p.as_double();
      mounting_changed = true;
    } else if (name == "mounting.yaw_deg") {
      next_mounting.yaw_deg = p.as_double();
      mounting_changed = true;
    } else if (name == "window.duration_sec") {
      if (p.as_double() < 0.0) {
        res.successful = false;
        res.reason = "window.duration_sec must be >= 0";
        return res;
      }
      next_window_dur = p.as_double();
    } else if (name == "window.max_frames") {
      if (p.as_int() < 1) {
        res.successful = false;
        res.reason = "window.max_frames must be >= 1";
        return res;
      }
      next_window_max = static_cast<int>(p.as_int());
    } else if (name == "window.odom_tf_max_age_sec") {
      next_odom_age = p.as_double();
    } else if (name == "window.storage_roi_padding") {
      next_pad = p.as_double();
    } else if (name == "prefilter.min_range") {
      next_prefilter.min_range = p.as_double();
    } else if (name == "prefilter.max_range") {
      next_prefilter.max_range = p.as_double();
    } else if (name == "prefilter.use_tag_filter") {
      next_prefilter.use_tag_filter = p.as_bool();
    } else if (name == "prefilter.tag_reject_mask") {
      const int m = static_cast<int>(p.as_int());
      if (m < 0 || m > 255) {
        res.successful = false;
        res.reason = "prefilter.tag_reject_mask must be in [0, 255]";
        return res;
      }
      next_prefilter.tag_reject_mask = static_cast<uint8_t>(m);
    } else if (name == "crop.ego_enabled") {
      next_cfg.crop.ego_enabled = p.as_bool();
    } else if (name == "crop.ego_min_x") {
      next_cfg.crop.ego_min_x = p.as_double();
    } else if (name == "crop.ego_max_x") {
      next_cfg.crop.ego_max_x = p.as_double();
    } else if (name == "crop.ego_min_y") {
      next_cfg.crop.ego_min_y = p.as_double();
    } else if (name == "crop.ego_max_y") {
      next_cfg.crop.ego_max_y = p.as_double();
    } else if (name == "crop.ego_min_z") {
      next_cfg.crop.ego_min_z = p.as_double();
    } else if (name == "crop.ego_max_z") {
      next_cfg.crop.ego_max_z = p.as_double();
    } else if (name == "crop.roi_min_x") {
      next_cfg.crop.roi_min_x = p.as_double();
    } else if (name == "crop.roi_max_x") {
      next_cfg.crop.roi_max_x = p.as_double();
    } else if (name == "crop.roi_min_y") {
      next_cfg.crop.roi_min_y = p.as_double();
    } else if (name == "crop.roi_max_y") {
      next_cfg.crop.roi_max_y = p.as_double();
    } else if (name == "crop.roi_min_z") {
      next_cfg.crop.roi_min_z = p.as_double();
    } else if (name == "crop.roi_max_z") {
      next_cfg.crop.roi_max_z = p.as_double();
    } else if (name == "crop.voxel_leaf_size") {
      next_cfg.crop.voxel_leaf_size = p.as_double();
    } else if (name == "ground.use_ransac") {
      next_cfg.ground.use_ransac = p.as_bool();
    } else if (name == "ground.band_min_z") {
      next_cfg.ground.band_min_z = p.as_double();
    } else if (name == "ground.band_max_z") {
      next_cfg.ground.band_max_z = p.as_double();
    } else if (name == "ground.max_tilt_deg") {
      next_cfg.ground.max_tilt_deg = p.as_double();
    } else if (name == "ground.ransac_dist_thresh") {
      next_cfg.ground.ransac_dist_thresh = p.as_double();
    } else if (name == "ground.ransac_max_iter") {
      next_cfg.ground.ransac_max_iter = static_cast<int>(p.as_int());
    } else if (name == "ground.min_inliers") {
      next_cfg.ground.min_inliers = static_cast<int>(p.as_int());
    } else if (name == "ground.max_plane_offset") {
      next_cfg.ground.max_plane_offset = p.as_double();
    } else if (name == "ground.nominal_z") {
      next_cfg.ground.nominal_z = p.as_double();
    } else if (name == "ground.clearance_height") {
      next_cfg.ground.clearance_height = p.as_double();
    } else if (name == "ground.level_to_ground") {
      next_cfg.ground.level_to_ground = p.as_bool();
    } else if (name == "outlier.ror_enabled") {
      next_cfg.outlier.ror_enabled = p.as_bool();
    } else if (name == "outlier.ror_radius") {
      next_cfg.outlier.ror_radius = p.as_double();
    } else if (name == "outlier.ror_min_neighbors") {
      next_cfg.outlier.ror_min_neighbors = static_cast<int>(p.as_int());
    } else if (name == "outlier.sor_enabled") {
      next_cfg.outlier.sor_enabled = p.as_bool();
    } else if (name == "outlier.sor_mean_k") {
      next_cfg.outlier.sor_mean_k = static_cast<int>(p.as_int());
    } else if (name == "outlier.sor_stddev_mul") {
      next_cfg.outlier.sor_stddev_mul = p.as_double();
    } else if (name == "cluster.tolerance") {
      if (p.as_double() <= 0.0) {
        res.successful = false;
        res.reason = "cluster.tolerance must be > 0";
        return res;
      }
      next_cfg.cluster.tolerance = p.as_double();
    } else if (name == "cluster.min_size") {
      next_cfg.cluster.min_size = static_cast<int>(p.as_int());
    } else if (name == "cluster.max_size") {
      next_cfg.cluster.max_size = static_cast<int>(p.as_int());
    } else if (name == "gate.min_points") {
      next_cfg.gate.min_points = static_cast<int>(p.as_int());
    } else if (name == "gate.min_xy_span") {
      next_cfg.gate.min_xy_span = p.as_double();
    } else if (name == "gate.max_xy_span") {
      next_cfg.gate.max_xy_span = p.as_double();
    } else if (name == "gate.min_z") {
      next_cfg.gate.min_z = p.as_double();
    } else if (name == "gate.max_z") {
      next_cfg.gate.max_z = p.as_double();
    } else if (name == "gate.max_bottom_z") {
      next_cfg.gate.max_bottom_z = p.as_double();
    } else if (name == "gate.height_consistency_tol") {
      next_cfg.gate.height_consistency_tol = p.as_double();
    } else if (name == "gate.min_inlier_ratio") {
      next_cfg.gate.min_inlier_ratio = p.as_double();
    } else if (name == "gate.max_rmse") {
      next_cfg.gate.max_rmse = p.as_double();
    } else if (name == "box.length" || name == "box.width" || name == "box.height") {
      if (p.as_double() <= 0.02) {
        res.successful = false;
        res.reason = name + " must be > 0.02 m";
        return res;
      }
      if (name == "box.length") {
        next_cfg.fitting.length = p.as_double();
      } else if (name == "box.width") {
        next_cfg.fitting.width = p.as_double();
      } else {
        next_cfg.fitting.height = p.as_double();
      }
    } else if (name == "box.range_bias_offset") {
      next_cfg.fitting.range_bias_offset = p.as_double();
    } else if (name == "fitting.closeness_d0") {
      next_cfg.fitting.closeness_d0 = p.as_double();
    } else if (name == "fitting.coarse_step_deg") {
      next_cfg.fitting.coarse_step_deg = p.as_double();
    } else if (name == "fitting.fine_step_deg") {
      next_cfg.fitting.fine_step_deg = p.as_double();
    } else if (name == "fitting.top_band_thickness") {
      next_cfg.fitting.top_band_thickness = p.as_double();
    } else if (name == "fitting.side_min_z") {
      next_cfg.fitting.side_min_z = p.as_double();
    } else if (name == "fitting.side_top_margin") {
      next_cfg.fitting.side_top_margin = p.as_double();
    } else if (name == "fitting.top_min_points") {
      next_cfg.fitting.top_min_points = static_cast<int>(p.as_int());
    } else if (name == "fitting.top_max_z_mad") {
      next_cfg.fitting.top_max_z_mad = p.as_double();
    } else if (name == "fitting.face_band") {
      next_cfg.fitting.face_band = p.as_double();
    } else if (name == "fitting.two_face_min_span") {
      next_cfg.fitting.two_face_min_span = p.as_double();
    } else if (name == "fitting.two_face_min_pts") {
      next_cfg.fitting.two_face_min_pts = static_cast<int>(p.as_int());
    } else if (name == "fitting.two_face_min_excl_span") {
      next_cfg.fitting.two_face_min_excl_span = p.as_double();
    } else if (name == "fitting.huber_delta") {
      next_cfg.fitting.huber_delta = p.as_double();
    } else if (name == "fitting.ransac_line_iters") {
      next_cfg.fitting.ransac_line_iters = static_cast<int>(p.as_int());
    } else if (name == "fitting.ransac_line_thresh") {
      next_cfg.fitting.ransac_line_thresh = p.as_double();
    } else if (name == "fitting.enable_se2_refine") {
      next_cfg.fitting.enable_se2_refine = p.as_bool();
    } else if (name == "fitting.se2_max_iters") {
      next_cfg.fitting.se2_max_iters = static_cast<int>(p.as_int());
    } else if (name == "fitting.se2_max_shift") {
      next_cfg.fitting.se2_max_shift = p.as_double();
    } else if (name == "fitting.se2_max_angle_deg") {
      next_cfg.fitting.se2_max_angle_deg = p.as_double();
    } else if (name == "fitting.inlier_dist_thresh") {
      next_cfg.fitting.inlier_dist_thresh = p.as_double();
    } else if (name == "tracker.max_assoc_dist") {
      next_cfg.tracker.max_assoc_dist = p.as_double();
    } else if (name == "tracker.min_hits") {
      next_cfg.tracker.min_hits = static_cast<int>(p.as_int());
    } else if (name == "tracker.max_missed_frames") {
      next_cfg.tracker.max_missed_frames = static_cast<int>(p.as_int());
    } else if (name == "tracker.kf_process_noise_pos") {
      next_cfg.tracker.kf_process_noise_pos = p.as_double();
    } else if (name == "tracker.kf_process_noise_vel") {
      next_cfg.tracker.kf_process_noise_vel = p.as_double();
    } else if (name == "tracker.kf_meas_noise_pos") {
      next_cfg.tracker.kf_meas_noise_pos = p.as_double();
    } else if (name == "tracker.kf_vel_damping") {
      next_cfg.tracker.kf_vel_damping = p.as_double();
    } else if (name == "tracker.yaw_alpha") {
      next_cfg.tracker.yaw_alpha = p.as_double();
    } else if (name == "tracker.switch_hysteresis_m") {
      next_cfg.tracker.switch_hysteresis_m = p.as_double();
    } else if (name == "lock.min_points") {
      next_cfg.tracker.lock_min_points = static_cast<int>(p.as_int());
    } else if (name == "lock.min_inlier_ratio") {
      next_cfg.tracker.lock_min_inlier_ratio = p.as_double();
    } else if (name == "lock.max_rmse") {
      next_cfg.tracker.lock_max_rmse = p.as_double();
    } else if (name == "lock.window_frames") {
      next_cfg.tracker.lock_window_frames = static_cast<int>(p.as_int());
    } else if (name == "lock.max_pos_jitter") {
      next_cfg.tracker.lock_max_pos_jitter = p.as_double();
    } else if (name == "lock.max_yaw_jitter_deg") {
      next_cfg.tracker.lock_max_yaw_jitter_deg = p.as_double();
    } else if (name == "debug.publish_roi_marker") {
      next_pub_roi = p.as_bool();
    } else if (name == "debug.publish_rejected_markers") {
      next_pub_rej = p.as_bool();
    }
  }

  // Commit atomically
  pipeline_.setConfig(next_cfg);
  mounting_cfg_ = next_mounting;
  prefilter_cfg_ = next_prefilter;
  window_duration_sec_ = next_window_dur;
  window_max_frames_ = next_window_max;
  odom_tf_max_age_sec_ = next_odom_age;
  storage_roi_padding_ = next_pad;
  output_frame_ = next_output_frame;
  debug_publish_roi_marker_ = next_pub_roi;
  debug_publish_rejected_markers_ = next_pub_rej;

  if (mounting_changed) {
    T_base_lidar_ = mounting_cfg_.computeTransform();
    window_buffer_.clear();
    pipeline_.resetTracker();
    has_prev_odom_ = false;
    if (publish_static_tf_ && static_tf_broadcaster_) {
      publishStaticLidarTf();
    }
  }

  return res;
}

void MidtoboxDetectorNode::publishStaticLidarTf()
{
  geometry_msgs::msg::TransformStamped tf_msg = tf2::eigenToTransform(T_base_lidar_);
  tf_msg.header.stamp = this->get_clock()->now();
  tf_msg.header.frame_id = base_frame_;
  tf_msg.child_frame_id = lidar_frame_;
  static_tf_broadcaster_->sendTransform(tf_msg);
}

DetectorPipeline::CloudPtr MidtoboxDetectorNode::parsePointCloud2(
  const sensor_msgs::msg::PointCloud2 & msg)
{
  DetectorPipeline::CloudPtr cloud(new DetectorPipeline::CloudT);
  const uint32_t step = msg.point_step;
  if (msg.width == 0 || msg.height == 0 || step == 0) {
    return cloud;
  }
  if (msg.is_bigendian) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "Big-endian PointCloud2 is not supported; ignoring frame");
    return cloud;
  }

  const size_t min_row_bytes = static_cast<size_t>(msg.width) * static_cast<size_t>(step);
  const size_t row_step =
    (msg.row_step >= min_row_bytes) ? static_cast<size_t>(msg.row_step) : min_row_bytes;
  if (msg.data.size() < static_cast<size_t>(msg.height) * row_step) {
    return cloud;
  }

  int x_off = -1;
  int y_off = -1;
  int z_off = -1;
  int tag_off = -1;
  uint8_t tag_dtype = 0;

  for (const auto & f : msg.fields) {
    if (f.name == "x" && f.datatype == sensor_msgs::msg::PointField::FLOAT32) {
      x_off = static_cast<int>(f.offset);
    } else if (f.name == "y" && f.datatype == sensor_msgs::msg::PointField::FLOAT32) {
      y_off = static_cast<int>(f.offset);
    } else if (f.name == "z" && f.datatype == sensor_msgs::msg::PointField::FLOAT32) {
      z_off = static_cast<int>(f.offset);
    } else if (f.name == "tag") {
      tag_off = static_cast<int>(f.offset);
      tag_dtype = f.datatype;
    }
  }

  if (x_off < 0 || y_off < 0 || z_off < 0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "PointCloud2 missing FLOAT32 x/y/z fields; ignoring frame");
    return cloud;
  }

  if (static_cast<uint32_t>(x_off) + 4 > step ||
      static_cast<uint32_t>(y_off) + 4 > step ||
      static_cast<uint32_t>(z_off) + 4 > step)
  {
    return cloud;
  }

  const bool check_tag =
    prefilter_cfg_.use_tag_filter &&
    (tag_off >= 0) &&
    (tag_dtype == sensor_msgs::msg::PointField::UINT8) &&
    (static_cast<uint32_t>(tag_off) + 1 <= step);

  const float min_r2 = static_cast<float>(prefilter_cfg_.min_range * prefilter_cfg_.min_range);
  const float max_r2 = (prefilter_cfg_.max_range > 0.0)
    ? static_cast<float>(prefilter_cfg_.max_range * prefilter_cfg_.max_range)
    : std::numeric_limits<float>::infinity();

  const size_t num_pts = static_cast<size_t>(msg.width) * static_cast<size_t>(msg.height);
  cloud->points.reserve(num_pts);
  const uint8_t * base_ptr = msg.data.data();

  for (uint32_t r = 0; r < msg.height; ++r) {
    const uint8_t * row_ptr = base_ptr + static_cast<size_t>(r) * row_step;
    for (uint32_t c = 0; c < msg.width; ++c) {
      const uint8_t * ptr = row_ptr + static_cast<size_t>(c) * step;

      if (check_tag) {
        const uint8_t tag = ptr[tag_off];
        if ((tag & prefilter_cfg_.tag_reject_mask) != 0) {
          continue;
        }
      }

      float x = 0.0f;
      float y = 0.0f;
      float z = 0.0f;
      std::memcpy(&x, ptr + x_off, sizeof(float));
      std::memcpy(&y, ptr + y_off, sizeof(float));
      std::memcpy(&z, ptr + z_off, sizeof(float));

      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        continue;
      }

      const float r2 = x * x + y * y + z * z;
      if (r2 < min_r2 || r2 > max_r2) {
        continue;
      }

      cloud->points.emplace_back(x, y, z);
    }
  }

  cloud->width = static_cast<uint32_t>(cloud->points.size());
  cloud->height = 1;
  cloud->is_dense = true;
  return cloud;
}

DetectorPipeline::CloudPtr MidtoboxDetectorNode::parseCustomMsg(
  const livox_ros_driver2::msg::CustomMsg & msg) const
{
  DetectorPipeline::CloudPtr cloud(new DetectorPipeline::CloudT);
  cloud->points.reserve(msg.points.size());

  const float min_r2 = static_cast<float>(prefilter_cfg_.min_range * prefilter_cfg_.min_range);
  const float max_r2 = (prefilter_cfg_.max_range > 0.0)
    ? static_cast<float>(prefilter_cfg_.max_range * prefilter_cfg_.max_range)
    : std::numeric_limits<float>::infinity();

  for (const auto & pt : msg.points) {
    if (prefilter_cfg_.use_tag_filter && ((pt.tag & prefilter_cfg_.tag_reject_mask) != 0)) {
      continue;
    }
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
      continue;
    }
    const float r2 = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
    if (r2 < min_r2 || r2 > max_r2) {
      continue;
    }
    cloud->points.emplace_back(pt.x, pt.y, pt.z);
  }

  cloud->width = static_cast<uint32_t>(cloud->points.size());
  cloud->height = 1;
  cloud->is_dense = true;
  return cloud;
}

void MidtoboxDetectorNode::pointCloud2Callback(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
  auto sensor_cloud = parsePointCloud2(*msg);
  processSensorCloud(sensor_cloud, msg->header);
}

void MidtoboxDetectorNode::customMsgCallback(
  const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr msg)
{
  auto sensor_cloud = parseCustomMsg(*msg);
  processSensorCloud(sensor_cloud, msg->header);
}

bool MidtoboxDetectorNode::lookupSensorToBase(
  const std::string & sensor_frame,
  const rclcpp::Time & stamp,
  Eigen::Isometry3d & T_base_sensor_out)
{
  const std::string clean_frame = stripLeadingSlash(sensor_frame);

  if (clean_frame.empty() || clean_frame == base_frame_) {
    T_base_sensor_out = Eigen::Isometry3d::Identity();
    return true;
  }

  try {
    const auto tf = tf_buffer_->lookupTransform(
      base_frame_, clean_frame, stamp, tf2::durationFromSec(0.0));
    T_base_sensor_out = tf2::transformToEigen(tf.transform);
    return true;
  } catch (const tf2::TransformException &) {
    try {
      const auto tf = tf_buffer_->lookupTransform(
        base_frame_, clean_frame, tf2::TimePointZero);
      T_base_sensor_out = tf2::transformToEigen(tf.transform);
      return true;
    } catch (const tf2::TransformException &) {
      if (clean_frame == lidar_frame_) {
        if (!publish_static_tf_) {
          RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 3000,
            "publish_static_tf=false and external TF %s -> %s not yet available; "
            "temporarily falling back to YAML mounting.* extrinsics",
            base_frame_.c_str(), lidar_frame_.c_str());
        }
        T_base_sensor_out = T_base_lidar_;
        return true;
      }
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Cannot transform point cloud from '%s' to '%s'",
        clean_frame.c_str(), base_frame_.c_str());
      return false;
    }
  }
}

bool MidtoboxDetectorNode::lookupOdomToBase(
  const rclcpp::Time & stamp,
  Eigen::Isometry3d & T_odom_base_out)
{
  if (odom_frame_.empty() || odom_frame_ == base_frame_) {
    return false;
  }

  try {
    const auto tf = tf_buffer_->lookupTransform(
      odom_frame_, base_frame_, stamp, tf2::durationFromSec(0.0));
    T_odom_base_out = tf2::transformToEigen(tf.transform);
    return true;
  } catch (const tf2::ExtrapolationException &) {
    // Allow very recent latest TF if lidar timestamp leads odom slightly
    try {
      const auto tf = tf_buffer_->lookupTransform(
        odom_frame_, base_frame_, tf2::TimePointZero);
      const rclcpp::Time tf_time(tf.header.stamp, stamp.get_clock_type());
      const double age = std::abs((stamp - tf_time).seconds());
      if (age <= odom_tf_max_age_sec_) {
        T_odom_base_out = tf2::transformToEigen(tf.transform);
        return true;
      }
    } catch (const tf2::TransformException &) {
      return false;
    }
    return false;
  } catch (const tf2::TransformException &) {
    return false;
  }
}

bool MidtoboxDetectorNode::lookupBaseToOutput(
  const rclcpp::Time & stamp,
  bool has_odom,
  const Eigen::Isometry3d & T_odom_base,
  Eigen::Isometry3d & T_out_base_out,
  std::string & resolved_frame_out)
{
  if (output_frame_.empty() || output_frame_ == base_frame_) {
    T_out_base_out = Eigen::Isometry3d::Identity();
    resolved_frame_out = base_frame_;
    return true;
  }

  if (output_frame_ == odom_frame_) {
    if (has_odom) {
      T_out_base_out = T_odom_base;
      resolved_frame_out = odom_frame_;
      return true;
    }
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Output frame '%s' requested but fresh %s->%s TF unavailable (max age %.2fs); "
      "falling back to '%s'",
      output_frame_.c_str(), odom_frame_.c_str(), base_frame_.c_str(),
      odom_tf_max_age_sec_, base_frame_.c_str());
    T_out_base_out = Eigen::Isometry3d::Identity();
    resolved_frame_out = base_frame_;
    return false;
  }

  try {
    const auto tf = tf_buffer_->lookupTransform(
      output_frame_, base_frame_, stamp, tf2::durationFromSec(0.0));
    T_out_base_out = tf2::transformToEigen(tf.transform);
    resolved_frame_out = output_frame_;
    return true;
  } catch (const tf2::TransformException &) {
    try {
      const auto tf = tf_buffer_->lookupTransform(
        output_frame_, base_frame_, tf2::TimePointZero);
      // Reject stale dynamic transforms (static transforms have stamp == 0)
      if (tf.header.stamp.sec != 0 || tf.header.stamp.nanosec != 0) {
        const rclcpp::Time tf_time(tf.header.stamp, stamp.get_clock_type());
        const double age = std::abs((stamp - tf_time).seconds());
        if (age > odom_tf_max_age_sec_) {
          throw tf2::ExtrapolationException("Output frame TF is stale");
        }
      }
      T_out_base_out = tf2::transformToEigen(tf.transform);
      resolved_frame_out = output_frame_;
      return true;
    } catch (const tf2::TransformException &) {
      if (output_frame_ == lidar_frame_) {
        if (!publish_static_tf_) {
          RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 3000,
            "publish_static_tf=false and external TF %s -> %s not yet available for output; "
            "temporarily falling back to YAML mounting.* extrinsics",
            base_frame_.c_str(), lidar_frame_.c_str());
        }
        T_out_base_out = T_base_lidar_.inverse();
        resolved_frame_out = lidar_frame_;
        return true;
      }
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Output frame '%s' unavailable from '%s'; falling back to '%s'",
        output_frame_.c_str(), base_frame_.c_str(), base_frame_.c_str());
      T_out_base_out = Eigen::Isometry3d::Identity();
      resolved_frame_out = base_frame_;
      return false;
    }
  }
}

void MidtoboxDetectorNode::processSensorCloud(
  const DetectorPipeline::CloudConstPtr & sensor_cloud,
  const std_msgs::msg::Header & header)
{
  rclcpp::Time stamp = this->get_clock()->now();
  if (header.stamp.sec != 0 || header.stamp.nanosec != 0) {
    stamp = rclcpp::Time(header.stamp, this->get_clock()->get_clock_type());
  }
  const double stamp_sec = stamp.seconds();

  // Check timestamp monotonicity before buffering
  if (!window_buffer_.empty()) {
    const double dt_last = stamp_sec - window_buffer_.back().stamp_sec;
    if (dt_last < -0.5 || dt_last > 2.0) {
      window_buffer_.clear();
      pipeline_.resetTracker();
      has_prev_odom_ = false;
    } else if (dt_last < -1e-6) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping out-of-order point cloud frame (dt=%.3fs)", dt_last);
      return;
    }
  }

  // 1. Transform sensor cloud into leveled base_footprint FIRST
  Eigen::Isometry3d T_base_sensor = Eigen::Isometry3d::Identity();
  if (!lookupSensorToBase(header.frame_id, stamp, T_base_sensor)) {
    return;
  }

  const Eigen::Matrix3f R_bs = T_base_sensor.linear().cast<float>();
  const Eigen::Vector3f t_bs = T_base_sensor.translation().cast<float>();

  // Pre-crop with generous padding before buffering in sliding window
  const auto & cc = pipeline_.config().crop;
  const float pad = static_cast<float>(std::max(storage_roi_padding_, 0.2));
  const float min_x = static_cast<float>(cc.roi_min_x) - pad;
  const float max_x = static_cast<float>(cc.roi_max_x) + pad;
  const float min_y = static_cast<float>(cc.roi_min_y) - pad;
  const float max_y = static_cast<float>(cc.roi_max_y) + pad;
  const float min_z = static_cast<float>(cc.roi_min_z) - pad;
  const float max_z = static_cast<float>(cc.roi_max_z) + pad;

  const float ego_min_x = static_cast<float>(cc.ego_min_x);
  const float ego_max_x = static_cast<float>(cc.ego_max_x);
  const float ego_min_y = static_cast<float>(cc.ego_min_y);
  const float ego_max_y = static_cast<float>(cc.ego_max_y);
  const float ego_min_z = static_cast<float>(cc.ego_min_z);
  const float ego_max_z = static_cast<float>(cc.ego_max_z);

  DetectorPipeline::CloudPtr frame_base(new DetectorPipeline::CloudT);
  frame_base->points.reserve(sensor_cloud->size() / 2);
  for (const auto & pt : sensor_cloud->points) {
    const Eigen::Vector3f pb = R_bs * Eigen::Vector3f(pt.x, pt.y, pt.z) + t_bs;
    // Strip ego-vehicle / arm envelope in each scan's own base_footprint(t_k) BEFORE
    // sliding-window odom motion compensation so backing up or turning never projects
    // past self-body points into the workspace ROI as phantom obstacles.
    if (cc.ego_enabled &&
        pb.x() >= ego_min_x && pb.x() <= ego_max_x &&
        pb.y() >= ego_min_y && pb.y() <= ego_max_y &&
        pb.z() >= ego_min_z && pb.z() <= ego_max_z)
    {
      continue;
    }
    if (pb.x() >= min_x && pb.x() <= max_x &&
        pb.y() >= min_y && pb.y() <= max_y &&
        pb.z() >= min_z && pb.z() <= max_z)
    {
      frame_base->points.emplace_back(pb.x(), pb.y(), pb.z());
    }
  }
  frame_base->width = static_cast<uint32_t>(frame_base->points.size());
  frame_base->height = 1;
  frame_base->is_dense = true;

  // 2. Sliding-window accumulation (with automatic odom->base_footprint motion compensation
  //    or graceful static fallback)
  Eigen::Isometry3d T_odom_base = Eigen::Isometry3d::Identity();
  const bool has_odom = lookupOdomToBase(stamp, T_odom_base);
  if (!odom_mode_logged_ || has_odom != last_odom_available_) {
    RCLCPP_INFO(
      this->get_logger(),
      "Sliding-window accumulation mode: %s (window=%.0f ms, max_frames=%d)",
      has_odom ? "odom->base_footprint motion-compensated" : "static fallback (no odom TF)",
      window_duration_sec_ * 1000.0, window_max_frames_);
    odom_mode_logged_ = true;
    last_odom_available_ = has_odom;
  }

  WindowFrame wf;
  wf.stamp_sec = stamp_sec;
  wf.has_odom = has_odom;
  wf.T_odom_base = T_odom_base;
  wf.cloud = frame_base;
  window_buffer_.push_back(std::move(wf));

  while (window_buffer_.size() > 1 &&
         ((stamp_sec - window_buffer_.front().stamp_sec) > window_duration_sec_ + 1e-6 ||
         static_cast<int>(window_buffer_.size()) > std::max(window_max_frames_, 1)))
  {
    window_buffer_.pop_front();
  }

  DetectorPipeline::CloudPtr accum_cloud(new DetectorPipeline::CloudT);
  size_t total_pts = 0;
  for (const auto & f : window_buffer_) {
    total_pts += f.cloud->size();
  }
  accum_cloud->points.reserve(total_pts);

  if (has_odom) {
    const Eigen::Isometry3d T_base_curr_odom = T_odom_base.inverse();
    for (const auto & f : window_buffer_) {
      if (!f.has_odom) {
        continue;
      }
      const Eigen::Isometry3d T_curr_prev = T_base_curr_odom * f.T_odom_base;
      const Eigen::Matrix3f R_rel = T_curr_prev.linear().cast<float>();
      const Eigen::Vector3f t_rel = T_curr_prev.translation().cast<float>();
      for (const auto & pt : f.cloud->points) {
        const Eigen::Vector3f p = R_rel * Eigen::Vector3f(pt.x, pt.y, pt.z) + t_rel;
        accum_cloud->points.emplace_back(p.x(), p.y(), p.z());
      }
    }
  } else {
    for (const auto & f : window_buffer_) {
      if (f.has_odom) {
        // Avoid smearing previously motion-tracked frames uncompensated after odom dropout
        continue;
      }
      accum_cloud->points.insert(
        accum_cloud->points.end(), f.cloud->points.begin(), f.cloud->points.end());
    }
  }
  accum_cloud->width = static_cast<uint32_t>(accum_cloud->points.size());
  accum_cloud->height = 1;
  accum_cloud->is_dense = true;

  // 3. Run pure-C++ detection + fitting + tracking pipeline in leveled base_footprint
  DetectorPipeline::CloudPtr debug_roi;
  DetectorPipeline::CloudPtr debug_nonground;
  if (debug_publish_clouds_) {
    debug_roi.reset(new DetectorPipeline::CloudT);
    debug_nonground.reset(new DetectorPipeline::CloudT);
  }

  Eigen::Isometry3d T_curr_prev_tracker = Eigen::Isometry3d::Identity();
  const Eigen::Isometry3d * tracker_ego_ptr = nullptr;
  if (has_odom && has_prev_odom_) {
    T_curr_prev_tracker = T_odom_base.inverse() * prev_T_odom_base_;
    tracker_ego_ptr = &T_curr_prev_tracker;
  }

  const Eigen::Vector3d sensor_origin_base = T_base_sensor.translation();
  const PipelineResult res = pipeline_.process(
    accum_cloud, sensor_origin_base, stamp_sec, debug_roi, debug_nonground, tracker_ego_ptr);

  has_prev_odom_ = has_odom;
  if (has_odom) {
    prev_T_odom_base_ = T_odom_base;
  }

  // 4. Transform outputs into requested output_frame and publish
  Eigen::Isometry3d T_out_base = Eigen::Isometry3d::Identity();
  std::string active_out_frame = base_frame_;
  lookupBaseToOutput(stamp, has_odom, T_odom_base, T_out_base, active_out_frame);

  publishResults(res, stamp, T_out_base, active_out_frame, debug_roi, debug_nonground);
}

void MidtoboxDetectorNode::publishResults(
  const PipelineResult & res,
  const rclcpp::Time & stamp,
  const Eigen::Isometry3d & T_out_base,
  const std::string & out_frame,
  const DetectorPipeline::CloudConstPtr & roi_cloud,
  const DetectorPipeline::CloudConstPtr & non_ground_cloud)
{
  const Eigen::Quaterniond q_out_base(T_out_base.linear());

  // 1. /midtobox/locked
  std_msgs::msg::Bool locked_msg;
  locked_msg.data = res.locked;
  locked_pub_->publish(locked_msg);

  // 2. /midtobox/target_pose (only when a confirmed primary target exists)
  if (res.has_primary) {
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = out_frame;
    const Eigen::Vector3d pos_out = T_out_base * res.primary_target.center;
    const Eigen::Quaterniond q_out = (q_out_base * res.primary_target.orientation).normalized();
    pose_msg.pose = eigenPoseToMsg(pos_out, q_out);
    target_pose_pub_->publish(pose_msg);
  }

  // 3. /midtobox/detections (ALL confirmed valid cubes simultaneously)
  vision_msgs::msg::Detection3DArray det_array;
  det_array.header.stamp = stamp;
  det_array.header.frame_id = out_frame;
  det_array.detections.reserve(res.tracks.size());

  for (const auto & trk : res.tracks) {
    vision_msgs::msg::Detection3D det_msg;
    det_msg.header = det_array.header;
    det_msg.id = std::to_string(trk.track_id);

    const Eigen::Vector3d pos_out = T_out_base * trk.center;
    const Eigen::Quaterniond q_out = (q_out_base * trk.orientation).normalized();
    const auto pose = eigenPoseToMsg(pos_out, q_out);

    det_msg.bbox.center = pose;
    det_msg.bbox.size.x = trk.size.x();
    det_msg.bbox.size.y = trk.size.y();
    det_msg.bbox.size.z = trk.size.z();

    vision_msgs::msg::ObjectHypothesisWithPose hyp;
    hyp.hypothesis.class_id = "box";
    hyp.hypothesis.score = trk.score;
    hyp.pose.pose = pose;
    det_msg.results.push_back(hyp);

    det_array.detections.push_back(std::move(det_msg));
  }
  detections_pub_->publish(det_array);

  // 4. /midtobox/markers (DELETEALL + valid tracks + oversized/rejected cluster debug markers + ROI)
  visualization_msgs::msg::MarkerArray marker_array;
  {
    visualization_msgs::msg::Marker clear_m;
    clear_m.header.stamp = stamp;
    clear_m.header.frame_id = out_frame;
    clear_m.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(clear_m);
  }

  int marker_id = 1;
  for (const auto & trk : res.tracks) {
    const Eigen::Vector3d pos_out = T_out_base * trk.center;
    const Eigen::Quaterniond q_out = (q_out_base * trk.orientation).normalized();

    // Cube body
    visualization_msgs::msg::Marker cube;
    cube.header.stamp = stamp;
    cube.header.frame_id = out_frame;
    cube.ns = "valid_boxes";
    cube.id = marker_id++;
    cube.type = visualization_msgs::msg::Marker::CUBE;
    cube.action = visualization_msgs::msg::Marker::ADD;
    cube.pose = eigenPoseToMsg(pos_out, q_out);
    cube.scale.x = trk.size.x();
    cube.scale.y = trk.size.y();
    cube.scale.z = trk.size.z();
    if (trk.is_primary && trk.locked) {
      cube.color.r = 0.1f; cube.color.g = 0.95f; cube.color.b = 0.2f; cube.color.a = 0.45f;
    } else if (trk.is_primary) {
      cube.color.r = 0.1f; cube.color.g = 0.85f; cube.color.b = 0.95f; cube.color.a = 0.40f;
    } else {
      cube.color.r = 0.25f; cube.color.g = 0.50f; cube.color.b = 1.0f; cube.color.a = 0.35f;
    }
    marker_array.markers.push_back(cube);

    // Centroid sphere
    visualization_msgs::msg::Marker sphere;
    sphere.header = cube.header;
    sphere.ns = "centroids";
    sphere.id = marker_id++;
    sphere.type = visualization_msgs::msg::Marker::SPHERE;
    sphere.action = visualization_msgs::msg::Marker::ADD;
    sphere.pose = cube.pose;
    sphere.scale.x = 0.05;
    sphere.scale.y = 0.05;
    sphere.scale.z = 0.05;
    sphere.color.r = 1.0f; sphere.color.g = 0.95f; sphere.color.b = 0.1f; sphere.color.a = 0.95f;
    marker_array.markers.push_back(sphere);

    // Heading arrow
    visualization_msgs::msg::Marker arrow;
    arrow.header = cube.header;
    arrow.ns = "headings";
    arrow.id = marker_id++;
    arrow.type = visualization_msgs::msg::Marker::ARROW;
    arrow.action = visualization_msgs::msg::Marker::ADD;
    arrow.pose = cube.pose;
    arrow.scale.x = 0.28;
    arrow.scale.y = 0.03;
    arrow.scale.z = 0.03;
    arrow.color.r = 1.0f; arrow.color.g = 0.8f; arrow.color.b = 0.0f; arrow.color.a = 0.95f;
    marker_array.markers.push_back(arrow);

    // Info text label (offset upward in base_footprint so it sits above cube in any output frame)
    const Eigen::Vector3d label_pos_out =
      T_out_base * (trk.center + Eigen::Vector3d(0.0, 0.0, 0.5 * trk.size.z() + 0.12));
    visualization_msgs::msg::Marker text;
    text.header = cube.header;
    text.ns = "labels";
    text.id = marker_id++;
    text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text.action = visualization_msgs::msg::Marker::ADD;
    text.pose.position.x = label_pos_out.x();
    text.pose.position.y = label_pos_out.y();
    text.pose.position.z = label_pos_out.z();
    text.pose.orientation.w = 1.0;
    text.scale.z = 0.075;
    text.color.r = 1.0f; text.color.g = 1.0f; text.color.b = 1.0f; text.color.a = 1.0f;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2)
        << "ID " << trk.track_id
        << (trk.is_primary ? " [PRI]" : "")
        << (trk.locked ? " LOCK" : "") << "\n"
        << "(" << pos_out.x() << ", " << pos_out.y() << ", " << pos_out.z() << ")m\n"
        << "yaw=" << std::setprecision(1) << (quaternionToYaw(q_out) * kRadToDeg) << "deg "
        << (trk.fit_mode == FitMode::kTwoFaceLShape ? "2F" : "1F")
        << " pts=" << trk.point_count;
    text.text = oss.str();
    marker_array.markers.push_back(text);
  }

  // Oversized touching/stacked clusters flagged visibly in red/orange
  if (debug_publish_rejected_markers_) {
    for (const auto & rej : res.rejected_clusters) {
      if (!rej.is_oversized) {
        continue;
      }
      const Eigen::Vector3d pos_out = T_out_base * rej.center;
      const Eigen::Quaterniond q_rej_base(Eigen::AngleAxisd(rej.yaw, Eigen::Vector3d::UnitZ()));
      const Eigen::Quaterniond q_out = (q_out_base * q_rej_base).normalized();

      visualization_msgs::msg::Marker box;
      box.header.stamp = stamp;
      box.header.frame_id = out_frame;
      box.ns = "rejected_oversized";
      box.id = marker_id++;
      box.type = visualization_msgs::msg::Marker::CUBE;
      box.action = visualization_msgs::msg::Marker::ADD;
      box.pose = eigenPoseToMsg(pos_out, q_out);
      box.scale.x = rej.size.x();
      box.scale.y = rej.size.y();
      box.scale.z = rej.size.z();
      box.color.r = 1.0f; box.color.g = 0.25f; box.color.b = 0.05f; box.color.a = 0.35f;
      marker_array.markers.push_back(box);

      const Eigen::Vector3d rej_label_pos_out =
        T_out_base * (rej.center + Eigen::Vector3d(0.0, 0.0, 0.5 * rej.size.z() + 0.10));
      visualization_msgs::msg::Marker text;
      text.header = box.header;
      text.ns = "rejected_labels";
      text.id = marker_id++;
      text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text.action = visualization_msgs::msg::Marker::ADD;
      text.pose.position.x = rej_label_pos_out.x();
      text.pose.position.y = rej_label_pos_out.y();
      text.pose.position.z = rej_label_pos_out.z();
      text.pose.orientation.w = 1.0;
      text.scale.z = 0.07;
      text.color.r = 1.0f; text.color.g = 0.4f; text.color.b = 0.2f; text.color.a = 1.0f;
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(2)
          << "REJECT: " << rej.reason << "\n"
          << rej.size.x() << "x" << rej.size.y() << "x" << rej.size.z() << "m";
      text.text = oss.str();
      marker_array.markers.push_back(text);
    }
  }

  if (debug_publish_roi_marker_) {
    const auto & cc = pipeline_.config().crop;
    const Eigen::Vector3d roi_center_base(
      0.5 * (cc.roi_min_x + cc.roi_max_x),
      0.5 * (cc.roi_min_y + cc.roi_max_y),
      0.5 * (cc.roi_min_z + cc.roi_max_z));
    const Eigen::Vector3d roi_center_out = T_out_base * roi_center_base;

    visualization_msgs::msg::Marker roi_m;
    roi_m.header.stamp = stamp;
    roi_m.header.frame_id = out_frame;
    roi_m.ns = "roi_box";
    roi_m.id = marker_id++;
    roi_m.type = visualization_msgs::msg::Marker::CUBE;
    roi_m.action = visualization_msgs::msg::Marker::ADD;
    roi_m.pose = eigenPoseToMsg(roi_center_out, q_out_base);
    roi_m.scale.x = cc.roi_max_x - cc.roi_min_x;
    roi_m.scale.y = cc.roi_max_y - cc.roi_min_y;
    roi_m.scale.z = cc.roi_max_z - cc.roi_min_z;
    roi_m.color.r = 0.8f; roi_m.color.g = 0.8f; roi_m.color.b = 0.8f; roi_m.color.a = 0.05f;
    marker_array.markers.push_back(roi_m);
  }

  markers_pub_->publish(marker_array);

  // 5. Optional debug point clouds (always in base_footprint for direct geometry inspection)
  if (debug_publish_clouds_) {
    if (debug_accum_pub_ && roi_cloud) {
      sensor_msgs::msg::PointCloud2 msg;
      pcl::toROSMsg(*roi_cloud, msg);
      msg.header.stamp = stamp;
      msg.header.frame_id = base_frame_;
      debug_accum_pub_->publish(msg);
    }
    if (debug_nonground_pub_ && non_ground_cloud) {
      sensor_msgs::msg::PointCloud2 msg;
      pcl::toROSMsg(*non_ground_cloud, msg);
      msg.header.stamp = stamp;
      msg.header.frame_id = base_frame_;
      debug_nonground_pub_->publish(msg);
    }
  }
}

}  // namespace midtobox

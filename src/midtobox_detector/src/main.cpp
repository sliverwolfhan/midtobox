// Copyright 2026 midtobox contributors
// SPDX-License-Identifier: Apache-2.0
//
// midtobox_detector ROS 2 节点入口文件。

#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "midtobox_detector/detector_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<midtobox::MidtoboxDetectorNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

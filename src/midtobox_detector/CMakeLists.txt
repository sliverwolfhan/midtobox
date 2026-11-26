cmake_minimum_required(VERSION 3.14)
project(midtobox_detector LANGUAGES C CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Workaround for Ubuntu 22.04 VTK 9.1 MPI::MPI_C target pulled in by pcl_conversions
find_package(MPI QUIET COMPONENTS C)

# ROS 2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(livox_ros_driver2 REQUIRED)

# Strict PCL + Eigen only (ZERO OpenCV)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters kdtree sample_consensus search segmentation)

# Pure C++ core perception library (zero ROS 2 runtime coupling; unit-testable directly)
add_library(${PROJECT_NAME}_core SHARED
  src/ground_segmenter.cpp
  src/box_fitter.cpp
  src/tracker.cpp
  src/detector_pipeline.cpp
)

target_include_directories(${PROJECT_NAME}_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)

target_compile_definitions(${PROJECT_NAME}_core PUBLIC ${PCL_DEFINITIONS})

target_link_libraries(${PROJECT_NAME}_core PUBLIC
  Eigen3::Eigen
  ${PCL_LIBRARIES}
)

# ROS 2 detector node executable
add_executable(${PROJECT_NAME}_node
  src/detector_node.cpp
  src/main.cpp
)

target_include_directories(${PROJECT_NAME}_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(${PROJECT_NAME}_node
  ${PROJECT_NAME}_core
)

ament_target_dependencies(${PROJECT_NAME}_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
  vision_msgs
  tf2
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  pcl_conversions
  livox_ros_driver2
)

# Install targets & resources
install(TARGETS
  ${PROJECT_NAME}_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS
  ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config launch rviz
  DESTINATION share/${PROJECT_NAME}
)

install(FILES THIRD_PARTY_NOTICES.md
  DESTINATION share/${PROJECT_NAME}
)

# C++ GoogleTest unit test suite
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_box_fitter test/test_box_fitter.cpp)
  if(TARGET test_box_fitter)
    target_link_libraries(test_box_fitter ${PROJECT_NAME}_core)
  endif()

  ament_add_gtest(test_tracker test/test_tracker.cpp)
  if(TARGET test_tracker)
    target_link_libraries(test_tracker ${PROJECT_NAME}_core)
  endif()

  ament_add_gtest(test_pipeline test/test_pipeline.cpp)
  if(TARGET test_pipeline)
    target_link_libraries(test_pipeline ${PROJECT_NAME}_core)
  endif()
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME}_core)
ament_export_dependencies(Eigen3 PCL)

ament_package()

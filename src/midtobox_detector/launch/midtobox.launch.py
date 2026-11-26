"""Unified launch file for midtobox_detector, optional livox_ros_driver2, and optional RViz2."""

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _infer_xfer_format_from_yaml(params_file: str) -> int:
    try:
        with open(params_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        ros_params = data.get('midtobox_detector', {}).get('ros__parameters', {})
        if str(ros_params.get('input_format', 'pointcloud2')).strip().lower() == 'custom_msg':
            return 1
    except Exception:
        pass
    return 0


def _parse_bool(name: str, value: str) -> bool:
    norm = value.strip().lower()
    if norm == 'true':
        return True
    if norm == 'false':
        return False
    raise ValueError(f"Invalid {name}='{value}'; expected 'true' or 'false'")


def _parse_optional_bool(name: str, value: str):
    if not value.strip():
        return None
    return _parse_bool(name, value)


def _build_nodes(context):
    launch_driver = _parse_bool(
        'launch_driver', LaunchConfiguration('launch_driver').perform(context)
    )
    launch_rviz = _parse_bool(
        'launch_rviz', LaunchConfiguration('launch_rviz').perform(context)
    )
    use_sim_time = _parse_bool(
        'use_sim_time', LaunchConfiguration('use_sim_time').perform(context)
    )
    xfer_format_str = LaunchConfiguration('xfer_format').perform(context).strip()

    params_file = LaunchConfiguration('params_file').perform(context)
    output_frame = LaunchConfiguration('output_frame').perform(context).strip()
    inverted = _parse_optional_bool('inverted', LaunchConfiguration('inverted').perform(context))
    mounting_z_str = LaunchConfiguration('mounting_z').perform(context).strip()
    publish_static_tf = _parse_optional_bool(
        'publish_static_tf', LaunchConfiguration('publish_static_tf').perform(context)
    )
    publish_debug_clouds = _parse_optional_bool(
        'publish_debug_clouds', LaunchConfiguration('publish_debug_clouds').perform(context)
    )

    detector_overrides = {
        'use_sim_time': use_sim_time,
    }

    if xfer_format_str:
        if xfer_format_str not in ('0', '1'):
            raise ValueError(
                f"Invalid xfer_format='{xfer_format_str}'; "
                "expected '0' (PointCloud2) or '1' (CustomMsg)"
            )
        xfer_format = int(xfer_format_str)
        detector_overrides['input_format'] = 'custom_msg' if xfer_format == 1 else 'pointcloud2'
    else:
        # Defer detector input_format to params_file YAML; sync driver xfer_format if launched
        xfer_format = _infer_xfer_format_from_yaml(params_file)

    if output_frame:
        detector_overrides['output_frame'] = output_frame
    if inverted is not None:
        detector_overrides['mounting.inverted'] = inverted
    if mounting_z_str:
        detector_overrides['mounting.z'] = float(mounting_z_str)
    if publish_static_tf is not None:
        detector_overrides['publish_static_tf'] = publish_static_tf
    if publish_debug_clouds is not None:
        detector_overrides['debug.publish_clouds'] = publish_debug_clouds

    actions = []

    if launch_driver:
        driver_config = PathJoinSubstitution([
            FindPackageShare('livox_ros_driver2'),
            'config',
            'MID360_config.json',
        ]).perform(context)

        livox_driver = Node(
            package='livox_ros_driver2',
            executable='livox_ros_driver2_node',
            name='livox_lidar_publisher',
            output='screen',
            parameters=[
                {'xfer_format': xfer_format},
                {'multi_topic': 0},
                {'data_src': 0},
                {'publish_freq': 10.0},
                {'output_data_type': 0},
                {'frame_id': 'livox_frame'},
                {'lvx_file_path': '/tmp/livox_unused.lvx'},
                {'user_config_path': driver_config},
                {'cmdline_input_bd_code': 'livox0000000001'},
            ],
        )
        actions.append(livox_driver)

    detector_node = Node(
        package='midtobox_detector',
        executable='midtobox_detector_node',
        name='midtobox_detector',
        output='screen',
        parameters=[params_file, detector_overrides],
    )
    actions.append(detector_node)

    if launch_rviz:
        rviz_config = PathJoinSubstitution([
            FindPackageShare('midtobox_detector'),
            'rviz',
            'midtobox.rviz',
        ]).perform(context)

        rviz_node = Node(
            package='rviz2',
            executable='rviz2',
            name='midtobox_rviz2',
            output='screen',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': use_sim_time}],
        )
        actions.append(rviz_node)

    return actions


def generate_launch_description():
    default_params = PathJoinSubstitution([
        FindPackageShare('midtobox_detector'),
        'config',
        'midtobox.yaml',
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'launch_driver',
            default_value='false',
            description='Launch livox_ros_driver2_node with workspace MID360_config.json',
        ),
        DeclareLaunchArgument(
            'launch_rviz',
            default_value='false',
            description='Launch RViz2 with midtobox visualization config',
        ),
        DeclareLaunchArgument(
            'xfer_format',
            default_value='',
            description='Optional override: 0 = PointCloud2, 1 = CustomMsg (empty defers to YAML)',
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Path to midtobox_detector YAML parameter file',
        ),
        DeclareLaunchArgument(
            'output_frame',
            default_value='',
            description='Optional override for output_frame (base_footprint, livox_frame, odom)',
        ),
        DeclareLaunchArgument(
            'inverted',
            default_value='',
            description='Optional override for mounting.inverted (true or false)',
        ),
        DeclareLaunchArgument(
            'mounting_z',
            default_value='',
            description='Optional override for mounting.z height in meters',
        ),
        DeclareLaunchArgument(
            'publish_static_tf',
            default_value='',
            description='Optional override for publish_static_tf (true or false)',
        ),
        DeclareLaunchArgument(
            'publish_debug_clouds',
            default_value='',
            description=(
                'Optional override for debug.publish_clouds '
                '(true or false; empty defers to YAML)'
            ),
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation / bag clock',
        ),
        OpaqueFunction(function=_build_nodes),
    ])

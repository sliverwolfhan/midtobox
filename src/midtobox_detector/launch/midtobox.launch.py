"""midtobox_detector 统一启动文件：可选拉起 livox_ros_driver2 驱动与 RViz2 可视化。"""

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _parse_bool(raw: str, name: str) -> bool:
    """严格解析必填布尔启动参数，拼写错误直接抛出异常避免静默忽略。"""
    val = raw.strip().lower()
    if val == 'true':
        return True
    if val == 'false':
        return False
    raise ValueError(f"参数 {name}='{raw}' 无效；仅接受 'true' 或 'false'")


def _parse_optional_bool(raw: str, name: str):
    """解析可选布尔启动参数（空字符串表示沿用 YAML 默认值）。"""
    val = raw.strip().lower()
    if not val:
        return None
    return _parse_bool(val, name)


def _infer_xfer_format_from_yaml(params_file: str) -> int:
    """根据 YAML 中的 input_format 自动推断驱动的 xfer_format（0=PointCloud2, 1=CustomMsg）。"""
    try:
        with open(params_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        ros_params = data.get('midtobox_detector', {}).get('ros__parameters', {})
        if str(ros_params.get('input_format', 'pointcloud2')).strip().lower() == 'custom_msg':
            return 1
    except Exception:
        pass
    return 0


def _build_nodes(context):
    # 读取并校验启动参数
    launch_driver = _parse_bool(
        LaunchConfiguration('launch_driver').perform(context), 'launch_driver'
    )
    launch_rviz = _parse_bool(
        LaunchConfiguration('launch_rviz').perform(context), 'launch_rviz'
    )
    use_sim_time = _parse_bool(
        LaunchConfiguration('use_sim_time').perform(context), 'use_sim_time'
    )
    xfer_format_str = LaunchConfiguration('xfer_format').perform(context).strip()

    params_file = LaunchConfiguration('params_file').perform(context)
    output_frame = LaunchConfiguration('output_frame').perform(context).strip()
    inverted_opt = _parse_optional_bool(
        LaunchConfiguration('inverted').perform(context), 'inverted'
    )
    mounting_z_str = LaunchConfiguration('mounting_z').perform(context).strip()
    publish_static_tf_opt = _parse_optional_bool(
        LaunchConfiguration('publish_static_tf').perform(context), 'publish_static_tf'
    )
    publish_debug_clouds_opt = _parse_optional_bool(
        LaunchConfiguration('publish_debug_clouds').perform(context), 'publish_debug_clouds'
    )

    # 仅将命令行显式指定的参数覆盖到 YAML 之上，避免误覆盖用户自定义 YAML
    detector_overrides = {
        'use_sim_time': use_sim_time,
    }

    if xfer_format_str:
        if xfer_format_str not in ('0', '1'):
            raise ValueError(
                f"参数 xfer_format='{xfer_format_str}' 无效；"
                "仅接受 '0' (PointCloud2) 或 '1' (CustomMsg)"
            )
        xfer_format = int(xfer_format_str)
        detector_overrides['input_format'] = 'custom_msg' if xfer_format == 1 else 'pointcloud2'
    else:
        # 未指定 xfer_format 时沿用 YAML 的 input_format，并同步给驱动节点
        xfer_format = _infer_xfer_format_from_yaml(params_file)

    if output_frame:
        detector_overrides['output_frame'] = output_frame
    if inverted_opt is not None:
        detector_overrides['mounting.inverted'] = inverted_opt
    if mounting_z_str:
        detector_overrides['mounting.z'] = float(mounting_z_str)
    if publish_static_tf_opt is not None:
        detector_overrides['publish_static_tf'] = publish_static_tf_opt
    if publish_debug_clouds_opt is not None:
        detector_overrides['debug.publish_clouds'] = publish_debug_clouds_opt

    actions = []

    # 1. 可选：启动工作区内的 Livox Mid-360 驱动（直接读取 MID360_config.json，不改动 IP）
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

    # 2. 核心：启动 midtobox_detector 方块质心识别节点
    detector_node = Node(
        package='midtobox_detector',
        executable='midtobox_detector_node',
        name='midtobox_detector',
        output='screen',
        parameters=[params_file, detector_overrides],
    )
    actions.append(detector_node)

    # 3. 可选：启动预配置好的 RViz2 可视化界面
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
            description='是否同时启动工作区内的 livox_ros_driver2_node 驱动节点',
        ),
        DeclareLaunchArgument(
            'launch_rviz',
            default_value='false',
            description='是否同时启动预配置的 RViz2 调试可视化窗口',
        ),
        DeclareLaunchArgument(
            'xfer_format',
            default_value='',
            description='可选覆盖：0=PointCloud2，1=CustomMsg（留空则沿用 YAML 配置）',
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='midtobox_detector 参数 YAML 文件路径',
        ),
        DeclareLaunchArgument(
            'output_frame',
            default_value='',
            description='可选覆盖输出坐标系（base_footprint / livox_frame / odom）',
        ),
        DeclareLaunchArgument(
            'inverted',
            default_value='',
            description='可选覆盖雷达正倒装开关 mounting.inverted（true 或 false）',
        ),
        DeclareLaunchArgument(
            'mounting_z',
            default_value='',
            description='可选覆盖雷达离地安装高度 mounting.z（单位 m）',
        ),
        DeclareLaunchArgument(
            'publish_static_tf',
            default_value='',
            description='可选覆盖静态 TF 发布开关 publish_static_tf（true 或 false）',
        ),
        DeclareLaunchArgument(
            'publish_debug_clouds',
            default_value='',
            description='可选覆盖调试点云开关 debug.publish_clouds（true/false，留空沿用 YAML）',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='是否使用仿真时钟 / rosbag 回放时钟',
        ),
        OpaqueFunction(function=_build_nodes),
    ])

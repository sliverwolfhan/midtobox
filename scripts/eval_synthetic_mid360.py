#!/usr/bin/env python3
"""
Synthetic Inverted Livox Mid-360 Raycast Simulator & Quantitative E2E Benchmark.

Generates physically realistic inverted Mid-360 point clouds (roll=180 deg, z=0.70 m,
vertical FOV [-52 deg, +7 deg], non-repetitive flower-petal scan pattern, 1.5 cm range
noise, ~25% (0,0,0) invalid returns, tagged & untagged edge-dragging "veil" outliers)
and feeds them directly into the real compiled C++ `midtobox_detector_node` over ROS 2
topics (both PointCloud2 and livox_ros_driver2/msg/CustomMsg).

Usage:
  source /opt/ros/humble/setup.bash
  source /home/xjh/Desktop/midtobox/install/setup.bash
  python3 /home/xjh/Desktop/midtobox/scripts/eval_synthetic_mid360.py
"""

import argparse
import math
import os
import signal
import subprocess
import sys
import threading
import time
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from builtin_interfaces.msg import Time as RosTime
from geometry_msgs.msg import PoseStamped, TransformStamped
from livox_ros_driver2.msg import CustomMsg, CustomPoint
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import MarkerArray


# ---------------------------------------------------------------------------
# Geometry & Raycasting Helpers
# ---------------------------------------------------------------------------

def rot_x(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def rot_y(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rot_z(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def shortest_square_yaw_err_deg(est_rad: float, gt_rad: float) -> float:
    half_pi = 0.5 * math.pi
    q_pi = 0.25 * math.pi
    d = math.fmod(est_rad - gt_rad + q_pi, half_pi)
    if d < 0.0:
        d += half_pi
    return abs((d - q_pi) * 180.0 / math.pi)


@dataclass
class BoxSpec:
    center_ground: np.ndarray  # (3,) in ground frame (z = height/2 on ground)
    yaw_ground: float          # radians
    size: np.ndarray = field(default_factory=lambda: np.array([0.35, 0.35, 0.35]))


def generate_mid360_rays_lidar_frame(
    frame_idx: int,
    num_rays: int = 18000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate non-repetitive flower-petal ray directions in upright livox_frame.
    Mid-360 physical vertical FOV in livox_frame is [-7 deg, +52 deg], horizontal 360 deg.
    Each frame advances phase to reproduce Mid-360 non-repetitive gap filling across 200-300 ms.
    """
    if rng is None:
        rng = np.random.default_rng(1000 + frame_idx)

    # Non-repetitive petal scan parametric pattern + slight jitter
    t = np.linspace(0.0, 1.0, num_rays, endpoint=False)
    phase_az = frame_idx * 0.4137
    phase_el = frame_idx * 0.7291

    # Azimuth wraps rapidly across 360 deg with non-integer harmonic ratio
    az = (
        2.0 * math.pi * (37.0 * t + phase_az) + rng.normal(0.0, 0.002, num_rays)
    ) % (2.0 * math.pi)
    # Elevation oscillates between -7 deg and +52 deg (slightly denser in lower half)
    el_norm = 0.5 * (1.0 - np.cos(2.0 * math.pi * (11.3 * t + phase_el)))
    # Mix uniform stratified coverage to avoid blind bands in synthetic benchmark
    el_mix = 0.5 * el_norm + 0.5 * rng.uniform(0.0, 1.0, num_rays)
    el = np.deg2rad(-7.0 + 59.0 * el_mix)

    cos_el = np.cos(el)
    dirs = np.stack([cos_el * np.cos(az), cos_el * np.sin(az), np.sin(el)], axis=1)
    return dirs


def raycast_inverted_mid360(
    boxes: List[BoxSpec],
    frame_idx: int,
    inverted: bool = True,
    mount_z: float = 0.70,
    ground_pitch_deg: float = 0.0,
    range_noise_std: float = 0.014,
    add_veil_outliers: bool = True,
    heavy_untagged_veil: bool = False,
    invalid_zero_ratio: float = 0.25,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Physically raycast Mid-360 rays against ground plane + oriented boxes.
    Returns (pts_lidar [M,3] float32, tags [M] uint8) in livox_frame.
    """
    if rng is None:
        rng = np.random.default_rng(5000 + frame_idx)

    R_base_lidar = rot_x(math.pi) if inverted else np.eye(3)
    t_base_lidar = np.array([0.0, 0.0, mount_z], dtype=np.float64)

    # Ground frame -> base_footprint rotation (negative pitch around Y lifts +X ground up)
    R_base_ground = rot_y(-math.radians(ground_pitch_deg))
    n_ground_base = R_base_ground @ np.array([0.0, 0.0, 1.0])

    # Cast rays in lidar frame -> transform to base_footprint
    dirs_lidar = generate_mid360_rays_lidar_frame(frame_idx, num_rays=18000, rng=rng)
    dirs_base = dirs_lidar @ R_base_lidar.T
    origin_base = t_base_lidar

    max_range = 5.5
    best_t = np.full(len(dirs_base), np.inf, dtype=np.float64)
    hit_box_idx = np.full(len(dirs_base), -1, dtype=np.int32)

    # 1. Intersect with ground plane (passes through origin of base_footprint)
    denom_g = dirs_base @ n_ground_base
    valid_g = denom_g < -1e-5
    t_g = np.full(len(dirs_base), np.inf, dtype=np.float64)
    t_g[valid_g] = -(origin_base @ n_ground_base) / denom_g[valid_g]
    mask_g = (t_g > 0.30) & (t_g < max_range)
    best_t[mask_g] = t_g[mask_g]

    # 2. Intersect with each oriented box (slab method in box-local frame)
    box_world_info = []
    for b_idx, box in enumerate(boxes):
        R_box_ground = rot_z(box.yaw_ground)
        R_box_base = R_base_ground @ R_box_ground
        c_base = R_base_ground @ box.center_ground
        half = 0.5 * box.size
        box_world_info.append((c_base, R_box_base, half))

        o_loc = R_box_base.T @ (origin_base - c_base)
        d_loc = dirs_base @ R_box_base

        # Avoid division by zero
        safe_d = np.where(np.abs(d_loc) > 1e-9, d_loc, 1.0)
        inv_d = np.where(np.abs(d_loc) > 1e-9, 1.0 / safe_d, 1e12)
        t1 = (-half - o_loc) * inv_d
        t2 = (half - o_loc) * inv_d
        tmin = np.max(np.minimum(t1, t2), axis=1)
        tmax = np.min(np.maximum(t1, t2), axis=1)

        hit = (tmax >= np.maximum(tmin, 0.25)) & (tmin < best_t) & (tmin < max_range)
        best_t[hit] = tmin[hit]
        hit_box_idx[hit] = b_idx

    valid_hits = np.isfinite(best_t)
    hit_dirs = dirs_base[valid_hits]
    hit_t = best_t[valid_hits]
    hit_is_box = hit_box_idx[valid_hits] >= 0

    # Add realistic Mid-360 Gaussian range noise along rays
    noisy_t = hit_t + rng.normal(0.0, range_noise_std, size=hit_t.shape)
    pts_base = origin_base[None, :] + hit_dirs * noisy_t[:, None]
    tags = np.zeros(len(pts_base), dtype=np.uint8)

    # Inject realistic Mid-360 edge-dragging "veil" (mixed-pixel) outliers near box silhouettes
    if add_veil_outliers and len(box_world_info) > 0:
        veil_pts = []
        veil_tags = []
        for c_base, R_box_base, half in box_world_info:
            # Find box hit points near lateral edges (|ly| > 0.85*hy or |lx| > 0.85*hx)
            b_pts = pts_base[hit_is_box]
            if len(b_pts) == 0:
                continue
            loc = (b_pts - c_base[None, :]) @ R_box_base
            near_edge = (
                (np.abs(loc[:, 0]) > 0.82 * half[0]) | (np.abs(loc[:, 1]) > 0.82 * half[1])
            ) & (np.abs(loc[:, 2]) < 0.85 * half[2])
            edge_pts = b_pts[near_edge]
            if len(edge_pts) == 0:
                continue

            # Tagged veil points (tag = 0x04 / 0x08 -> stripped by Mid-360 tag filter)
            n_tagged = min(len(edge_pts), 28)
            sel_t = edge_pts[rng.choice(len(edge_pts), size=n_tagged, replace=False)]
            ray_u = sel_t - origin_base[None, :]
            ray_u /= np.linalg.norm(ray_u, axis=1, keepdims=True)
            drag_t = rng.uniform(0.04, 0.25, size=(n_tagged, 1))
            veil_pts.append(sel_t + ray_u * drag_t)
            tag_choices = np.array([0x04, 0x08, 0x10], dtype=np.uint8)
            veil_tags.append(rng.choice(tag_choices, size=n_tagged))

            # Untagged mixed-pixel veil points (tag = 0x00 -> must be handled by ROR/SOR + Huber!)
            n_untagged = min(len(edge_pts), 22 if heavy_untagged_veil else 8)
            sel_u = edge_pts[rng.choice(len(edge_pts), size=n_untagged, replace=False)]
            ray_u2 = sel_u - origin_base[None, :]
            ray_u2 /= np.linalg.norm(ray_u2, axis=1, keepdims=True)
            drag_u = rng.uniform(0.05, 0.16, size=(n_untagged, 1))
            veil_pts.append(sel_u + ray_u2 * drag_u)
            veil_tags.append(np.zeros(n_untagged, dtype=np.uint8))

        if veil_pts:
            pts_base = np.vstack([pts_base] + veil_pts)
            tags = np.concatenate([tags] + veil_tags)

    # Add ego-chassis points inside CropBox (must be stripped by detector)
    ego_pts = np.column_stack([
        rng.uniform(-0.30, 0.30, 80),
        rng.uniform(-0.30, 0.30, 80),
        rng.uniform(0.05, 0.60, 80),
    ])
    pts_base = np.vstack([pts_base, ego_pts])
    tags = np.concatenate([tags, np.zeros(len(ego_pts), dtype=np.uint8)])

    # Transform back from base_footprint to livox_frame
    pts_lidar = (pts_base - t_base_lidar[None, :]) @ R_base_lidar

    # Add near-field stray points (r < 0.20 m, tag=0x10) and ~25% (0,0,0) invalid returns
    n_stray = 120
    stray = rng.uniform(-0.12, 0.12, size=(n_stray, 3))
    stray_tags = np.full(n_stray, 0x10, dtype=np.uint8)

    n_zeros = int(len(pts_lidar) * invalid_zero_ratio)
    zeros = np.zeros((n_zeros, 3), dtype=np.float64)
    zero_tags = np.zeros(n_zeros, dtype=np.uint8)

    pts_all = np.vstack([pts_lidar, stray, zeros]).astype(np.float32)
    tags_all = np.concatenate([tags, stray_tags, zero_tags]).astype(np.uint8)
    return pts_all, tags_all


def pack_livox_pointcloud2(
    pts: np.ndarray,
    tags: np.ndarray,
    stamp_sec: float,
    frame_id: str = 'livox_frame',
) -> PointCloud2:
    """Pack points into exact 26-byte Livox Mid-360 PointCloud2 schema."""
    n = len(pts)
    dt = np.dtype({
        'names': ['x', 'y', 'z', 'intensity', 'tag', 'line', 'timestamp'],
        'formats': ['<f4', '<f4', '<f4', '<f4', 'u1', 'u1', '<f8'],
        'offsets': [0, 4, 8, 12, 16, 17, 18],
        'itemsize': 26,
    })
    arr = np.zeros(n, dtype=dt)
    arr['x'] = pts[:, 0]
    arr['y'] = pts[:, 1]
    arr['z'] = pts[:, 2]
    arr['intensity'] = 100.0
    arr['tag'] = tags
    arr['line'] = 0
    arr['timestamp'] = stamp_sec * 1e9

    msg = PointCloud2()
    sec = int(math.floor(stamp_sec))
    nsec = int(round((stamp_sec - sec) * 1e9))
    msg.header.stamp = RosTime(sec=sec, nanosec=nsec)
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = n
    msg.is_bigendian = False
    msg.is_dense = True  # Intentionally mirrors real Livox driver quirk!
    msg.point_step = 26
    msg.row_step = 26 * n
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name='tag', offset=16, datatype=PointField.UINT8, count=1),
        PointField(name='line', offset=17, datatype=PointField.UINT8, count=1),
        PointField(name='timestamp', offset=18, datatype=PointField.FLOAT64, count=1),
    ]
    msg.data = arr.tobytes()
    return msg


def pack_livox_custom_msg(
    pts: np.ndarray,
    tags: np.ndarray,
    stamp_sec: float,
    frame_id: str = 'livox_frame',
) -> CustomMsg:
    """Pack points into livox_ros_driver2/msg/CustomMsg."""
    msg = CustomMsg()
    sec = int(math.floor(stamp_sec))
    nsec = int(round((stamp_sec - sec) * 1e9))
    msg.header.stamp = RosTime(sec=sec, nanosec=nsec)
    msg.header.frame_id = frame_id
    msg.timebase = int(round(stamp_sec * 1e9))
    msg.point_num = len(pts)
    msg.lidar_id = 1
    points_list = []
    for i in range(len(pts)):
        cp = CustomPoint()
        cp.offset_time = i * 5000
        cp.x = float(pts[i, 0])
        cp.y = float(pts[i, 1])
        cp.z = float(pts[i, 2])
        cp.reflectivity = 100
        cp.tag = int(tags[i])
        cp.line = 0
        points_list.append(cp)
    msg.points = points_list
    return msg


# ---------------------------------------------------------------------------
# ROS 2 Harness Driving Real C++ midtobox_detector_node
# ---------------------------------------------------------------------------

class EvalHarnessNode(Node):
    def __init__(self, input_format: str = 'pointcloud2'):
        super().__init__('midtobox_eval_harness')
        self.input_format = input_format

        pub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        if input_format == 'custom_msg':
            self.cloud_pub = self.create_publisher(CustomMsg, '/livox/lidar', pub_qos)
        else:
            self.cloud_pub = self.create_publisher(PointCloud2, '/livox/lidar', pub_qos)

        self.tf_broadcaster = TransformBroadcaster(self)

        sub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.lock = threading.Lock()
        self.det_count_rx = 0
        self.latest_pose: Optional[PoseStamped] = None
        self.latest_dets: Optional[Detection3DArray] = None
        self.latest_locked: Optional[bool] = None
        self.latest_markers: Optional[MarkerArray] = None

        self.create_subscription(PoseStamped, '/midtobox/target_pose', self._on_pose, sub_qos)
        self.create_subscription(Detection3DArray, '/midtobox/detections', self._on_dets, sub_qos)
        self.create_subscription(Bool, '/midtobox/locked', self._on_locked, sub_qos)
        self.create_subscription(MarkerArray, '/midtobox/markers', self._on_markers, sub_qos)

    def _on_pose(self, msg: PoseStamped):
        with self.lock:
            self.latest_pose = msg

    def _on_dets(self, msg: Detection3DArray):
        with self.lock:
            self.latest_dets = msg
            self.det_count_rx += 1

    def _on_locked(self, msg: Bool):
        with self.lock:
            self.latest_locked = msg.data

    def _on_markers(self, msg: MarkerArray):
        with self.lock:
            self.latest_markers = msg

    def publish_odom_tf(self, stamp_sec: float, tx: float, ty: float, yaw: float):
        tf = TransformStamped()
        sec = int(math.floor(stamp_sec))
        nsec = int(round((stamp_sec - sec) * 1e9))
        tf.header.stamp = RosTime(sec=sec, nanosec=nsec)
        tf.header.frame_id = 'odom'
        tf.child_frame_id = 'base_footprint'
        tf.transform.translation.x = tx
        tf.transform.translation.y = ty
        tf.transform.translation.z = 0.0
        tf.transform.rotation.z = math.sin(0.5 * yaw)
        tf.transform.rotation.w = math.cos(0.5 * yaw)
        self.tf_broadcaster.sendTransform(tf)

    def send_and_wait(self, msg, timeout: float = 2.0) -> Tuple[
        Optional[PoseStamped], Optional[Detection3DArray], Optional[bool], Optional[MarkerArray]
    ]:
        with self.lock:
            prev_rx = self.det_count_rx
            self.latest_pose = None
        self.cloud_pub.publish(msg)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.01)
            with self.lock:
                if self.det_count_rx > prev_rx:
                    # Spin briefly to catch PoseStamped / MarkerArray in same batch
                    break
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.005)
        with self.lock:
            return self.latest_pose, self.latest_dets, self.latest_locked, self.latest_markers


class DetectorProcess:
    def __init__(self, extra_params: Dict[str, str]):
        from ament_index_python.packages import get_package_prefix
        exe = os.path.join(
            get_package_prefix('midtobox_detector'),
            'lib', 'midtobox_detector', 'midtobox_detector_node',
        )
        cmd = [exe, '--ros-args']
        for k, v in extra_params.items():
            cmd.extend(['-p', f'{k}:={v}'])
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def stop(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=2.0)
        # Allow DDS discovery to unregister endpoints cleanly before next scenario
        time.sleep(0.15)


def wait_for_subscriber(harness: EvalHarnessNode, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rclpy.spin_once(harness, timeout_sec=0.05)
        if harness.cloud_pub.get_subscription_count() >= 1:
            time.sleep(0.15)
            return True
    return False


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    detail: str
    err_3d_cm: Optional[float] = None
    yaw_err_deg: Optional[float] = None
    locked: Optional[bool] = None


def run_standard_scenarios() -> List[ScenarioResult]:
    results: List[ScenarioResult] = []
    det_proc = DetectorProcess({
        'input_format': 'pointcloud2',
        'qos_reliability': 'reliable',
        'mounting.inverted': 'true',
        'mounting.z': '0.70',
        'output_frame': 'base_footprint',
        'publish_static_tf': 'true',
    })
    harness = EvalHarnessNode('pointcloud2')
    try:
        if not wait_for_subscriber(harness):
            raise RuntimeError('midtobox_detector_node did not subscribe within timeout')

        t_cursor = 100.0  # Jump by +10.0 s between scenarios to cleanly reset window & tracker

        # ------------------------------------------------------------------
        # Single-box accuracy scenarios (front-only, oblique L-shape, boundary,
        # tilted ground, heavy untagged veil outliers)
        # ------------------------------------------------------------------
        single_cases = [
            # name, center_g, yaw_deg, pitch_deg, heavy_veil, max_pos_cm, max_yaw_deg
            ('single_face_0deg_1.8m',     [1.80,  0.00, 0.175],   0.0, 0.0, False, 2.0, 3.5),
            ('single_face_5deg_2.2m',     [2.20,  0.15, 0.175],   5.0, 0.0, False, 2.5, 4.0),
            ('lshape_25deg_1.6m',         [1.60, -0.18, 0.175],  25.0, 0.0, False, 2.0, 2.5),
            ('lshape_neg35deg_2.0m',      [2.00,  0.22, 0.175], -35.0, 0.0, False, 1.5, 2.5),
            ('boundary_44.5deg_unwrap',   [1.75,  0.00, 0.175],  44.5, 0.0, False, 2.0, 2.5),
            ('tilted_ground_3.5deg',      [1.80,  0.05, 0.175],  22.0, 3.5, False, 2.5, 3.0),
            ('heavy_veil_outliers',       [1.75, -0.05, 0.175],   0.0, 0.0, True,  2.5, 4.0),
        ]

        for name, c_g, yaw_deg, pitch_deg, heavy_veil, max_pos_cm, max_yaw_deg in single_cases:
            t_cursor += 10.0
            box = BoxSpec(np.array(c_g, dtype=np.float64), math.radians(yaw_deg))
            R_bg = rot_y(-math.radians(pitch_deg))
            gt_base = R_bg @ box.center_ground
            gt_yaw = math.radians(yaw_deg)

            yaws_seen: List[float] = []
            last_pose = None
            last_locked = False
            name_seed = zlib.crc32(name.encode('utf-8')) % 100000
            for k in range(10):
                stamp = t_cursor + 0.1 * k
                pts, tags = raycast_inverted_mid360(
                    [box], frame_idx=k, ground_pitch_deg=pitch_deg,
                    heavy_untagged_veil=heavy_veil,
                    rng=np.random.default_rng(10000 + name_seed + k),
                )
                msg = pack_livox_pointcloud2(pts, tags, stamp)
                pose, dets, locked, _ = harness.send_and_wait(msg)
                if pose is not None:
                    last_pose = pose
                    last_locked = bool(locked)
                    q = pose.pose.orientation
                    yaws_seen.append(quat_to_yaw(q.x, q.y, q.z, q.w))

            if last_pose is None:
                results.append(ScenarioResult(name, False, 'No target_pose received'))
                continue

            est = np.array([
                last_pose.pose.position.x,
                last_pose.pose.position.y,
                last_pose.pose.position.z,
            ])
            err_vec = (est - gt_base) * 100.0
            err_3d = float(np.linalg.norm(err_vec))
            est_yaw = yaws_seen[-1]
            yaw_err = shortest_square_yaw_err_deg(est_yaw, gt_yaw)

            max_jump = 0.0
            for j in range(1, len(yaws_seen)):
                max_jump = max(max_jump, abs(yaws_seen[j] - yaws_seen[j - 1]) * 180.0 / math.pi)

            ok = (
                (err_3d <= max_pos_cm)
                and (yaw_err <= max_yaw_deg)
                and last_locked
                and (max_jump < 10.0)
            )
            detail = (
                f'dxyz=({err_vec[0]:+.2f},{err_vec[1]:+.2f},{err_vec[2]:+.2f})cm '
                f'3D={err_3d:.2f}cm yaw_err={yaw_err:.2f}deg max_jump={max_jump:.2f}deg '
                f'locked={last_locked}'
            )
            results.append(ScenarioResult(name, ok, detail, err_3d, yaw_err, last_locked))

        # ------------------------------------------------------------------
        # Multi-cube separation (>= 13 cm corner-to-corner gap) + primary selection + hysteresis
        # ------------------------------------------------------------------
        t_cursor += 10.0
        # b1 +Y corner at -0.066 m; b2 -Y corner at +0.070 m -> 13.6 cm gap
        b1 = BoxSpec(np.array([1.50, -0.29, 0.175]), math.radians(20.0))
        b2 = BoxSpec(np.array([1.62,  0.29, 0.175]), math.radians(-18.0))
        det_counts = []
        primary_y_phase1 = None
        for k in range(8):
            stamp = t_cursor + 0.1 * k
            pts, tags = raycast_inverted_mid360(
                [b1, b2], frame_idx=k, rng=np.random.default_rng(20000 + k))
            pose, dets, locked, _ = harness.send_and_wait(pack_livox_pointcloud2(pts, tags, stamp))
            if dets is not None:
                det_counts.append(len(dets.detections))
            if pose is not None:
                primary_y_phase1 = pose.pose.position.y

        # Phase 2: Move b2 slightly closer (r_b2=1.45 m vs r_b1=1.53 m, diff 0.08 < 0.15 m)
        # Primary MUST remain b1 (y < 0)!
        b2_nearer_hyst = BoxSpec(np.array([1.42, 0.29, 0.175]), math.radians(-18.0))
        primary_y_phase2 = None
        for k in range(8, 13):
            stamp = t_cursor + 0.1 * k
            pts, tags = raycast_inverted_mid360(
                [b1, b2_nearer_hyst], frame_idx=k, rng=np.random.default_rng(20000 + k))
            pose, _, _, _ = harness.send_and_wait(pack_livox_pointcloud2(pts, tags, stamp))
            if pose is not None:
                primary_y_phase2 = pose.pose.position.y

        # Phase 3: Move b2 much closer (r_b2=1.25 m vs r_b1=1.53 m, diff 0.28 > 0.15 m)
        # Primary MUST switch to b2 (y > 0)!
        b2_closest = BoxSpec(np.array([1.22, 0.29, 0.175]), math.radians(-18.0))
        primary_y_phase3 = None
        for k in range(13, 20):
            stamp = t_cursor + 0.1 * k
            pts, tags = raycast_inverted_mid360(
                [b1, b2_closest], frame_idx=k, rng=np.random.default_rng(20000 + k))
            pose, _, _, _ = harness.send_and_wait(pack_livox_pointcloud2(pts, tags, stamp))
            if pose is not None:
                primary_y_phase3 = pose.pose.position.y

        multi_ok = (
            max(det_counts, default=0) == 2
            and primary_y_phase1 is not None and primary_y_phase1 < -0.10
            and primary_y_phase2 is not None and primary_y_phase2 < -0.10
            and primary_y_phase3 is not None and primary_y_phase3 > +0.10
        )
        results.append(ScenarioResult(
            'multi_cube_13cm_gap_hysteresis',
            multi_ok,
            f'max_dets={max(det_counts, default=0)} y_p1={primary_y_phase1} '
            f'y_p2(hyst)={primary_y_phase2} y_p3(switch)={primary_y_phase3}',
        ))

        # ------------------------------------------------------------------
        # Oversized touching cubes (0.70 m XY) & stacked cubes (0.70 m Z) rejected
        # ------------------------------------------------------------------
        for rej_name, boxes_rej in [
            (
                'oversized_touching_rejected',
                [
                    BoxSpec(np.array([1.75, -0.175, 0.175]), 0.0),
                    BoxSpec(np.array([1.75, +0.175, 0.175]), 0.0),
                ],
            ),
            (
                'oversized_stacked_rejected',
                [
                    BoxSpec(np.array([1.75, 0.0, 0.175]), 0.0),
                    BoxSpec(np.array([1.75, 0.0, 0.525]), 0.0),
                ],
            ),
        ]:
            t_cursor += 10.0
            any_det = False
            any_lock = False
            saw_rej_marker = False
            for k in range(6):
                stamp = t_cursor + 0.1 * k
                pts, tags = raycast_inverted_mid360(
                    boxes_rej, frame_idx=k, rng=np.random.default_rng(30000 + k))
                _, dets, locked, markers = harness.send_and_wait(
                    pack_livox_pointcloud2(pts, tags, stamp))
                if dets is not None and len(dets.detections) > 0:
                    any_det = True
                if locked:
                    any_lock = True
                if markers is not None:
                    for m in markers.markers:
                        if m.ns == 'rejected_oversized':
                            saw_rej_marker = True
            rej_ok = (not any_det) and (not any_lock) and saw_rej_marker
            results.append(ScenarioResult(
                rej_name, rej_ok,
                f'any_det={any_det} any_lock={any_lock} rej_marker={saw_rej_marker}',
            ))

        # ------------------------------------------------------------------
        # Moving robot with odom -> base_footprint TF motion compensation
        # Robot drives forward +0.5 m/s along +X toward a stationary box at x_odom=2.60 m.
        # Without motion compensation, 250 ms window smears front face by 12.5 cm!
        # ------------------------------------------------------------------
        t_cursor += 10.0
        box_odom_x = 2.60
        box_odom_y = 0.10
        box_yaw = math.radians(28.0)
        vx = 0.50  # m/s
        last_pose = None
        gt_last_base = None
        for k in range(10):
            stamp = t_cursor + 0.1 * k
            robot_x = vx * (0.1 * k)
            # Publish odom -> base_footprint TF BEFORE publishing cloud
            harness.publish_odom_tf(stamp, robot_x, 0.0, 0.0)
            time.sleep(0.02)
            box_in_base = BoxSpec(
                np.array([box_odom_x - robot_x, box_odom_y, 0.175]), box_yaw)
            pts, tags = raycast_inverted_mid360(
                [box_in_base], frame_idx=k, rng=np.random.default_rng(40000 + k))
            pose, _, _, _ = harness.send_and_wait(pack_livox_pointcloud2(pts, tags, stamp))
            if pose is not None:
                last_pose = pose
                gt_last_base = box_in_base.center_ground

        if last_pose is not None and gt_last_base is not None:
            est = np.array([
                last_pose.pose.position.x,
                last_pose.pose.position.y,
                last_pose.pose.position.z,
            ])
            err_3d = float(np.linalg.norm(est - gt_last_base) * 100.0)
            q = last_pose.pose.orientation
            yaw_err = shortest_square_yaw_err_deg(quat_to_yaw(q.x, q.y, q.z, q.w), box_yaw)
            ok = err_3d <= 3.0 and yaw_err <= 3.0
            results.append(ScenarioResult(
                'moving_robot_odom_comp', ok,
                f'vx={vx}m/s 3D_err={err_3d:.2f}cm yaw_err={yaw_err:.2f}deg',
                err_3d, yaw_err,
            ))
        else:
            results.append(ScenarioResult('moving_robot_odom_comp', False, 'No pose received'))

        # ------------------------------------------------------------------
        # Turning + translating robot (vx=0.3 m/s, omega=1.0 rad/s) with odom TF
        # Verifies ego-motion-compensated tracker has <2cm pos err, <2deg yaw lag, and locks
        # ------------------------------------------------------------------
        t_cursor += 10.0
        box_odom_pos = np.array([2.00, 0.00, 0.175])
        box_odom_yaw = math.radians(20.0)
        turn_errs = []
        turn_yerrs = []
        turn_tids = set()
        last_locked_turn = False
        for k in range(10):
            stamp = t_cursor + 0.1 * k
            rx = 0.03 * k
            ryaw = -0.35 + 0.10 * k  # 1.0 rad/s yaw rate
            harness.publish_odom_tf(stamp, rx, 0.0, ryaw)
            time.sleep(0.02)
            R_bo = rot_z(-ryaw)
            c_base = R_bo @ (box_odom_pos - np.array([rx, 0.0, 0.0]))
            yaw_base = box_odom_yaw - ryaw
            pts, tags = raycast_inverted_mid360(
                [BoxSpec(c_base, yaw_base)], frame_idx=k, rng=np.random.default_rng(45000 + k))
            pose, dets, locked, _ = harness.send_and_wait(pack_livox_pointcloud2(pts, tags, stamp))
            if pose is not None and k >= 3:
                est = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
                turn_errs.append(float(np.linalg.norm(est - c_base) * 100.0))
                q = pose.pose.orientation
                turn_yerrs.append(
                    shortest_square_yaw_err_deg(quat_to_yaw(q.x, q.y, q.z, q.w), yaw_base)
                )
                last_locked_turn = bool(locked)
            if dets and dets.detections:
                turn_tids.add(dets.detections[0].id)

        if turn_errs:
            max_pe = max(turn_errs)
            max_ye = max(turn_yerrs)
            ok = (len(turn_tids) == 1) and (max_pe <= 2.0) and (max_ye <= 2.0) and last_locked_turn
            results.append(ScenarioResult(
                'turning_robot_odom_comp', ok,
                f'omega=1.0rad/s tids={len(turn_tids)} max_3D={max_pe:.2f}cm '
                f'max_yaw={max_ye:.2f}deg locked={last_locked_turn}',
                max_pe, max_ye, last_locked_turn,
            ))
        else:
            results.append(ScenarioResult('turning_robot_odom_comp', False, 'No pose received'))

        # ------------------------------------------------------------------
        # Reversing robot (-0.8 m/s) with dense ego-bumper points inside ego CropBox
        # on an empty floor: must NOT leave phantom ego-chassis detections in ROI!
        # ------------------------------------------------------------------
        t_cursor += 10.0
        R_bl_inv = rot_x(math.pi)
        t_bl_inv = np.array([0.0, 0.0, 0.70])
        ghost_dets = 0
        for k in range(10):
            stamp = t_cursor + 0.1 * k
            rx = -0.8 * (0.1 * k)
            harness.publish_odom_tf(stamp, rx, 0.0, 0.0)
            time.sleep(0.02)
            rng_rev = np.random.default_rng(47000 + k)
            pts_l, tags_l = raycast_inverted_mid360([], frame_idx=k, rng=rng_rev)
            ego_base = np.column_stack([
                rng_rev.uniform(0.32, 0.395, 180),
                rng_rev.uniform(-0.17, 0.17, 180),
                rng_rev.uniform(0.06, 0.35, 180),
            ])
            ego_lidar = ((ego_base - t_bl_inv[None, :]) @ R_bl_inv).astype(np.float32)
            pts_all = np.vstack([pts_l, ego_lidar])
            tags_all = np.concatenate([tags_l, np.zeros(len(ego_lidar), dtype=np.uint8)])
            _, dets, _, _ = harness.send_and_wait(
                pack_livox_pointcloud2(pts_all, tags_all, stamp)
            )
            if dets is not None and len(dets.detections) > 0:
                ghost_dets += 1
        results.append(ScenarioResult(
            'reversing_ego_no_ghost',
            ghost_dets == 0,
            f'vx=-0.8m/s false_det_frames={ghost_dets}/10',
        ))

    finally:
        harness.destroy_node()
        det_proc.stop()

    return results


def run_custom_msg_and_output_frame_scenarios() -> List[ScenarioResult]:
    results: List[ScenarioResult] = []

    # 1. CustomMsg (xfer_format=1) + output_frame:=livox_frame (inverted!)
    det_proc = DetectorProcess({
        'input_format': 'custom_msg',
        'qos_reliability': 'reliable',
        'mounting.inverted': 'true',
        'mounting.z': '0.70',
        'output_frame': 'livox_frame',
        'publish_static_tf': 'true',
    })
    harness = EvalHarnessNode('custom_msg')
    try:
        if not wait_for_subscriber(harness):
            raise RuntimeError('CustomMsg detector did not subscribe')

        box = BoxSpec(np.array([1.70, 0.25, 0.175]), math.radians(20.0))
        # Expected center & yaw in inverted livox_frame (roll=180 deg, tz=0.70):
        # x_l = x_b = 1.70, y_l = -y_b = -0.25, z_l = -(z_b - 0.70) = +0.525, yaw_l = -20 deg
        gt_livox = np.array([1.70, -0.25, 0.525])
        gt_yaw_livox = -math.radians(20.0)

        last_pose = None
        last_locked = False
        last_markers = None
        for k in range(8):
            stamp = 500.0 + 0.1 * k
            pts, tags = raycast_inverted_mid360(
                [box], frame_idx=k, rng=np.random.default_rng(50000 + k))
            pose, _, locked, markers = harness.send_and_wait(
                pack_livox_custom_msg(pts, tags, stamp)
            )
            if pose is not None:
                last_pose = pose
                last_locked = bool(locked)
            if markers is not None:
                last_markers = markers

        if last_pose is None:
            results.append(ScenarioResult('custom_msg_and_livox_frame', False, 'No pose received'))
        else:
            est = np.array([
                last_pose.pose.position.x,
                last_pose.pose.position.y,
                last_pose.pose.position.z,
            ])
            err_3d = float(np.linalg.norm(est - gt_livox) * 100.0)
            q = last_pose.pose.orientation
            yaw_err = shortest_square_yaw_err_deg(quat_to_yaw(q.x, q.y, q.z, q.w), gt_yaw_livox)
            label_z = None
            if last_markers is not None:
                for m in last_markers.markers:
                    if m.ns == 'labels':
                        label_z = m.pose.position.z
            # In inverted livox_frame, floor is at z=+0.70 and box top is at z=+0.35;
            # label above cube must have z < 0.35.
            label_above_ok = (label_z is not None) and (label_z < 0.35)
            ok = (
                last_pose.header.frame_id == 'livox_frame'
                and err_3d <= 2.0
                and yaw_err <= 2.5
                and label_above_ok
                and last_locked
            )
            results.append(ScenarioResult(
                'custom_msg_and_livox_frame', ok,
                f'frame={last_pose.header.frame_id} '
                f'3D_err={err_3d:.2f}cm yaw_err={yaw_err:.2f}deg '
                f'label_z={label_z:.3f}m locked={last_locked}',
                err_3d, yaw_err, last_locked,
            ))
    finally:
        harness.destroy_node()
        det_proc.stop()

    # 2. output_frame:=odom with live odom -> base_footprint TF + stale odom fallback
    det_proc2 = DetectorProcess({
        'input_format': 'pointcloud2',
        'qos_reliability': 'reliable',
        'mounting.inverted': 'true',
        'mounting.z': '0.70',
        'output_frame': 'odom',
        'publish_static_tf': 'true',
    })
    harness2 = EvalHarnessNode('pointcloud2')
    try:
        if not wait_for_subscriber(harness2):
            raise RuntimeError('odom-frame detector did not subscribe')

        robot_odom = (3.0, -1.0, math.radians(30.0))
        R_ob = rot_z(robot_odom[2])
        box = BoxSpec(np.array([1.70, 0.10, 0.175]), math.radians(15.0))
        gt_odom = R_ob @ box.center_ground + np.array([robot_odom[0], robot_odom[1], 0.0])
        gt_yaw_odom = robot_odom[2] + box.yaw_ground

        last_pose = None
        last_locked = False
        for k in range(8):
            stamp = 600.0 + 0.1 * k
            harness2.publish_odom_tf(stamp, robot_odom[0], robot_odom[1], robot_odom[2])
            time.sleep(0.02)
            pts, tags = raycast_inverted_mid360(
                [box], frame_idx=k, rng=np.random.default_rng(60000 + k))
            pose, _, locked, _ = harness2.send_and_wait(pack_livox_pointcloud2(pts, tags, stamp))
            if pose is not None:
                last_pose = pose
                last_locked = bool(locked)

        if last_pose is None:
            results.append(ScenarioResult('output_frame_odom', False, 'No pose received'))
        else:
            est = np.array([
                last_pose.pose.position.x,
                last_pose.pose.position.y,
                last_pose.pose.position.z,
            ])
            err_3d = float(np.linalg.norm(est - gt_odom) * 100.0)
            q = last_pose.pose.orientation
            yaw_err = shortest_square_yaw_err_deg(quat_to_yaw(q.x, q.y, q.z, q.w), gt_yaw_odom)
            ok = (
                last_pose.header.frame_id == 'odom'
                and err_3d <= 2.5
                and yaw_err <= 2.5
                and last_locked
            )
            results.append(ScenarioResult(
                'output_frame_odom', ok,
                f'frame={last_pose.header.frame_id} '
                f'3D_err={err_3d:.2f}cm yaw_err={yaw_err:.2f}deg locked={last_locked}',
                err_3d, yaw_err, last_locked,
            ))

        # Stop updating odom TF and advance timestamps by +5 s -> fall back to base_footprint
        stale_pose = None
        for k in range(6):
            stamp = 606.0 + 0.1 * k
            pts, tags = raycast_inverted_mid360(
                [box], frame_idx=k, rng=np.random.default_rng(61000 + k))
            pose, _, _, _ = harness2.send_and_wait(pack_livox_pointcloud2(pts, tags, stamp))
            if pose is not None:
                stale_pose = pose
        stale_ok = (stale_pose is not None) and (stale_pose.header.frame_id == 'base_footprint')
        stale_frame = stale_pose.header.frame_id if stale_pose else None
        results.append(ScenarioResult(
            'stale_odom_output_fallback', stale_ok,
            f'frame={stale_frame} (expected base_footprint)',
        ))
    finally:
        harness2.destroy_node()
        det_proc2.stop()

    # 3. Upright mounting toggle (mounting.inverted:=false, mounting.z:=0.30 m)
    #    + short 23 cm distractor rejection without top face
    det_proc3 = DetectorProcess({
        'input_format': 'pointcloud2',
        'qos_reliability': 'reliable',
        'mounting.inverted': 'false',
        'mounting.z': '0.30',
        'output_frame': 'base_footprint',
        'publish_static_tf': 'true',
    })
    harness3 = EvalHarnessNode('pointcloud2')
    try:
        if not wait_for_subscriber(harness3):
            raise RuntimeError('upright-mode detector did not subscribe')

        box = BoxSpec(np.array([1.75, -0.15, 0.175]), math.radians(24.0))
        last_pose = None
        last_locked = False
        for k in range(8):
            stamp = 700.0 + 0.1 * k
            pts, tags = raycast_inverted_mid360(
                [box], frame_idx=k, inverted=False, mount_z=0.30,
                rng=np.random.default_rng(70000 + k))
            pose, _, locked, _ = harness3.send_and_wait(pack_livox_pointcloud2(pts, tags, stamp))
            if pose is not None:
                last_pose = pose
                last_locked = bool(locked)

        if last_pose is None:
            results.append(ScenarioResult('upright_mount_toggle', False, 'No pose received'))
        else:
            est = np.array([
                last_pose.pose.position.x,
                last_pose.pose.position.y,
                last_pose.pose.position.z,
            ])
            err_vec = (est - box.center_ground) * 100.0
            err_3d = float(np.linalg.norm(err_vec))
            q = last_pose.pose.orientation
            yaw_err = shortest_square_yaw_err_deg(quat_to_yaw(q.x, q.y, q.z, q.w), box.yaw_ground)
            ok = err_3d <= 1.2 and abs(err_vec[2]) <= 0.5 and yaw_err <= 2.5 and last_locked
            results.append(ScenarioResult(
                'upright_mount_toggle', ok,
                f'inverted=False dxyz=({err_vec[0]:+.2f},{err_vec[1]:+.2f},{err_vec[2]:+.2f})cm '
                f'3D={err_3d:.2f}cm yaw_err={yaw_err:.2f}deg locked={last_locked}',
                err_3d, yaw_err, last_locked,
            ))
    finally:
        harness3.destroy_node()
        det_proc3.stop()

    # 4. 2-face L-shape radial range bias offset compensation (material -2.0 cm range bias)
    det_proc4 = DetectorProcess({
        'input_format': 'pointcloud2',
        'qos_reliability': 'reliable',
        'mounting.inverted': 'true',
        'mounting.z': '0.70',
        'box.range_bias_offset': '0.02',
    })
    harness4 = EvalHarnessNode('pointcloud2')
    try:
        if not wait_for_subscriber(harness4):
            raise RuntimeError('range-bias detector did not subscribe')

        box = BoxSpec(np.array([1.80, 0.0, 0.175]), math.radians(20.0))
        last_pose = None
        last_locked = False
        for k in range(8):
            stamp = 800.0 + 0.1 * k
            pts, tags = raycast_inverted_mid360(
                [box], frame_idx=k, range_noise_std=0.006, add_veil_outliers=False,
                rng=np.random.default_rng(80000 + k))
            # Shift only box-height points (material reflectivity bias on box, not floor)
            pts_base_z = 0.70 - pts[:, 2]  # inverted z=0.70
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            is_box_pt = (norms[:, 0] > 0.25) & (pts_base_z > 0.04) & (pts[:, 0] > 0.45)
            pts[is_box_pt] -= 0.02 * (pts[is_box_pt] / norms[is_box_pt])
            pose, _, locked, _ = harness4.send_and_wait(pack_livox_pointcloud2(pts, tags, stamp))
            if pose is not None:
                last_pose = pose
                last_locked = bool(locked)

        if last_pose is None:
            results.append(ScenarioResult('two_face_range_bias_20deg', False, 'No pose received'))
        else:
            est = np.array([
                last_pose.pose.position.x,
                last_pose.pose.position.y,
                last_pose.pose.position.z,
            ])
            err_vec = (est - box.center_ground) * 100.0
            err_2d = float(math.hypot(err_vec[0], err_vec[1]))
            ok = err_2d <= 0.8 and abs(err_vec[1]) <= 0.5 and last_locked
            results.append(ScenarioResult(
                'two_face_range_bias_20deg', ok,
                f'dxyz=({err_vec[0]:+.2f},{err_vec[1]:+.2f},{err_vec[2]:+.2f})cm '
                f'2D={err_2d:.2f}cm locked={last_locked}',
                float(np.linalg.norm(err_vec)), None, last_locked,
            ))
    finally:
        harness4.destroy_node()
        det_proc4.stop()

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Synthetic inverted Mid-360 raycast E2E benchmark'
    )
    parser.parse_args()

    rclpy.init()
    try:
        all_results = run_standard_scenarios() + run_custom_msg_and_output_frame_scenarios()
    finally:
        rclpy.shutdown()

    print('\n' + '=' * 92)
    print(f'{"SCENARIO":<34} | {"STATUS":<6} | DETAILS')
    print('-' * 92)
    n_pass = 0
    for r in all_results:
        status = 'PASS' if r.passed else 'FAIL'
        if r.passed:
            n_pass += 1
        print(f'{r.name:<34} | {status:<6} | {r.detail}')
    print('=' * 92)
    print(f'Summary: {n_pass}/{len(all_results)} scenarios passed.')

    sys.exit(0 if n_pass == len(all_results) else 1)


if __name__ == '__main__':
    main()

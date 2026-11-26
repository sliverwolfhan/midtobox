#!/usr/bin/env python3
"""
Read-only Real Livox Mid-360 Rosbag Smoke & Throughput Test.

Reads a ROS 2 Humble sqlite3 bag directly via sqlite3 (read-only URI) +
rclpy.serialization.deserialize_message (works even when metadata.yaml is 0 bytes,
and NEVER modifies any file outside /home/xjh/Desktop/midtobox).

Feeds real Mid-360 PointCloud2 frames into the compiled C++ `midtobox_detector_node`
and verifies:
  - Raw / zero-point / Mid-360 tag filtering statistics
  - Zero crashes / errors over real Mid-360 traffic
  - Per-frame end-to-end processing latency & sustained throughput (Hz)
  - Valid publication on /midtobox/detections, /midtobox/locked, /midtobox/markers.

Usage:
  source /opt/ros/humble/setup.bash
  source /home/xjh/Desktop/midtobox/install/setup.bash
  python3 /home/xjh/Desktop/midtobox/scripts/smoke_test_real_bag.py
"""

import argparse
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import MarkerArray

DEFAULT_BAG_DB3 = '/home/xjh/Desktop/learn/pointlib/ros_bag2(点云示例)/ros_bag2_0.db3'


class BagSmokeNode(Node):
    def __init__(self, topic_name: str):
        super().__init__('midtobox_bag_smoke_tester')
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pub = self.create_publisher(PointCloud2, topic_name, qos)

        self.lock = threading.Lock()
        self.det_rx = 0
        self.locked_rx = 0
        self.marker_rx = 0
        self.total_dets = 0
        self.total_rejected_markers = 0

        self.create_subscription(Detection3DArray, '/midtobox/detections', self._on_dets, qos)
        self.create_subscription(Bool, '/midtobox/locked', self._on_locked, qos)
        self.create_subscription(MarkerArray, '/midtobox/markers', self._on_markers, qos)

    def _on_dets(self, msg: Detection3DArray):
        with self.lock:
            self.det_rx += 1
            self.total_dets += len(msg.detections)

    def _on_locked(self, msg: Bool):
        with self.lock:
            self.locked_rx += 1

    def _on_markers(self, msg: MarkerArray):
        with self.lock:
            self.marker_rx += 1
            for m in msg.markers:
                if m.ns == 'rejected_oversized':
                    self.total_rejected_markers += 1

    def send_and_time(self, msg: PointCloud2, timeout: float = 2.0) -> float:
        with self.lock:
            prev = self.det_rx
        t0 = time.perf_counter()
        self.pub.publish(msg)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.002)
            with self.lock:
                if self.det_rx > prev:
                    return (time.perf_counter() - t0) * 1000.0
        return -1.0


def main():
    parser = argparse.ArgumentParser(description='Read-only real Mid-360 rosbag smoke test')
    parser.add_argument('--bag', default=DEFAULT_BAG_DB3, help='Path to rosbag2 .db3 file')
    parser.add_argument('--max-frames', type=int, default=120, help='Number of frames to replay')
    parser.add_argument('--inverted', default='true', choices=['true', 'false'])
    args = parser.parse_args()

    if not os.path.exists(args.bag):
        print(f'ERROR: bag file not found: {args.bag}', file=sys.stderr)
        sys.exit(1)

    # Strictly read-only SQLite access (never creates journal/wal files next to bag)
    con = sqlite3.connect(f'file:{args.bag}?mode=ro&immutable=1', uri=True)
    cur = con.cursor()
    topic_row = cur.execute(
        "SELECT id, name FROM topics WHERE type='sensor_msgs/msg/PointCloud2' LIMIT 1"
    ).fetchone()
    if topic_row is None:
        con.close()
        print('ERROR: no PointCloud2 topic found in bag', file=sys.stderr)
        sys.exit(1)

    topic_id, topic_name = topic_row
    rows = cur.execute(
        'SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp ASC LIMIT ?',
        (topic_id, args.max_frames),
    ).fetchall()
    con.close()
    print(f'Loaded {len(rows)} real Mid-360 frames from {args.bag} (topic={topic_name})')

    # Launch compiled C++ midtobox_detector_node directly
    # (avoids ros2 run wrapper orphan on SIGINT)
    from ament_index_python.packages import get_package_prefix
    exe = os.path.join(
        get_package_prefix('midtobox_detector'),
        'lib', 'midtobox_detector', 'midtobox_detector_node',
    )
    cmd = [
        exe, '--ros-args',
        '-p', f'input_topic:={topic_name}',
        '-p', 'input_format:=pointcloud2',
        '-p', 'qos_reliability:=reliable',
        '-p', f'mounting.inverted:={args.inverted}',
        '-p', 'mounting.z:=0.70',
        '-p', 'publish_static_tf:=true',
    ]
    det_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    rclpy.init()
    tester = BagSmokeNode(topic_name)

    latencies_ms = []
    raw_total = 0
    zero_total = 0
    tag_flagged_total = 0

    try:
        # Wait for detector subscription
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and tester.pub.get_subscription_count() == 0:
            rclpy.spin_once(tester, timeout_sec=0.05)
        if tester.pub.get_subscription_count() == 0:
            raise RuntimeError('midtobox_detector_node did not subscribe to bag topic')
        time.sleep(0.15)

        dt_livox = np.dtype({
            'names': ['x', 'y', 'z', 'i', 'tag', 'line', 'ts'],
            'formats': ['<f4', '<f4', '<f4', '<f4', 'u1', 'u1', '<f8'],
            'offsets': [0, 4, 8, 12, 16, 17, 18],
            'itemsize': 26,
        })

        t_wall_start = time.perf_counter()
        for idx, (blob,) in enumerate(rows):
            msg = deserialize_message(blob, PointCloud2)
            if msg.point_step == 26:
                arr = np.frombuffer(msg.data, dtype=dt_livox)
                raw_total += len(arr)
                zmask = (arr['x'] == 0.0) & (arr['y'] == 0.0) & (arr['z'] == 0.0)
                zero_total += int(zmask.sum())
                nz_tags = arr['tag'][~zmask]
                tag_flagged_total += int(((nz_tags & 0x3F) != 0).sum())

            lat_ms = tester.send_and_time(msg, timeout=2.0)
            if lat_ms < 0.0:
                raise RuntimeError(f'Timeout waiting for detector output on frame {idx}')
            latencies_ms.append(lat_ms)

        wall_elapsed = time.perf_counter() - t_wall_start
        # Drain any trailing messages
        for _ in range(10):
            rclpy.spin_once(tester, timeout_sec=0.01)

        node_alive = det_proc.poll() is None
    finally:
        tester.destroy_node()
        rclpy.shutdown()
        if det_proc.poll() is None:
            det_proc.send_signal(signal.SIGINT)
            try:
                det_proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                det_proc.kill()
                det_proc.wait(timeout=2.0)

    lat_arr = np.array(latencies_ms, dtype=np.float64)
    hz = len(rows) / max(wall_elapsed, 1e-6)
    print('\n' + '=' * 72)
    print('Real Mid-360 Rosbag Format & Throughput Smoke Test Summary')
    print('-' * 72)
    print(f'Frames replayed         : {len(rows)}')
    print(f'Detector stayed alive   : {node_alive}')
    print(f'Raw points inspected    : {raw_total:,}')
    if raw_total > 0:
        print(f'Zero (0,0,0) points     : {zero_total:,} ({100.0 * zero_total / raw_total:.1f}%)')
        nz_total = max(raw_total - zero_total, 1)
        pct_tag = 100.0 * tag_flagged_total / nz_total
        print(f'Non-zero tag!=0 (0x3F)  : {tag_flagged_total:,} ({pct_tag:.2f}%)')
    print(
        f'Outputs received        : detections={tester.det_rx}, '
        f'locked={tester.locked_rx}, markers={tester.marker_rx}'
    )
    print(
        f'E2E roundtrip latency   : mean={lat_arr.mean():.2f} ms, '
        f'p50={np.median(lat_arr):.2f} ms, '
        f'p95={np.percentile(lat_arr, 95):.2f} ms, max={lat_arr.max():.2f} ms'
    )
    print(f'Sustained throughput    : {hz:.1f} Hz (Mid-360 nominal = 10.0 Hz)')
    print('=' * 72)

    ok = (
        node_alive
        and tester.det_rx == len(rows)
        and tester.locked_rx == len(rows)
        and tester.marker_rx == len(rows)
        and lat_arr.mean() < 50.0
    )
    print('RESULT:', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

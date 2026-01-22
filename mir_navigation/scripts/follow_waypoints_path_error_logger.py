#!/usr/bin/env python3
import os
import csv
import math
import time
from datetime import datetime

import numpy as np
from scipy.spatial import cKDTree

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSPresetProfiles

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from nav2_msgs.action import FollowWaypoints


def quat_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class WaypointFollowerPathError(Node):

    def __init__(self):
        super().__init__('waypoint_follower_path_error')

        self._action_client = ActionClient(self, FollowWaypoints, 'follow_waypoints')

        # Parameters
        self.declare_parameter('csv_dir', os.path.expanduser('~/report_ccu/path_error/cont_MPPI/'))
        self.declare_parameter('min_interval', 0.1)

        self.min_interval = self.get_parameter('min_interval').value

        # State
        self._t0 = None
        self._recording = False
        self._rows = []
        self._last_write_time = 0.0

        # Path cache
        self._path_pts = None
        self._path_tree = None

        # Experiment control
        self.loop_count = 0
        self.max_loops = 10        # <-- SET THIS
        self.direction = 'forth'  # 'forth' or 'back'

        # Subscribers
        self.create_subscription(
            Path, '/plan', self._path_cb, QoSPresetProfiles.SENSOR_DATA.value
        )
        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self._amcl_cb,
            QoSPresetProfiles.SENSOR_DATA.value
        )
        self.create_subscription(
            Odometry, '/odom', self._odom_cb,
            QoSPresetProfiles.SENSOR_DATA.value
        )

        self._latest_vel = (0.0, 0.0)

        self._csv_path = self._build_csv_path()

        self.get_logger().info(f'Logging path error to: {self._csv_path}')

    # --------------------------------------------------
    def _build_csv_path(self):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = self.get_parameter('csv_dir').value

        dir_path = os.path.join(base_dir, self.direction)
        os.makedirs(dir_path, exist_ok=True)

        return os.path.join(
            dir_path,
            f'path_error_loop_{self.loop_count+1}_{self.direction}_{ts}.csv'
        )

    # --------------------------------------------------
    def _path_cb(self, msg: Path):
        if not msg.poses:
            return
        pts = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])
        self._path_pts = pts
        self._path_tree = cKDTree(pts)

    # --------------------------------------------------
    def _odom_cb(self, msg: Odometry):
        self._latest_vel = (
            max(0.0, msg.twist.twist.linear.x),
            msg.twist.twist.angular.z
        )

    # --------------------------------------------------
    def _amcl_cb(self, msg: PoseWithCovarianceStamped):
        # Log only while actively recording and path is available
        if not self._recording or self._path_tree is None or self._t0 is None:
            return

        now_wall = time.time()
        if now_wall - self._last_write_time < self.min_interval:
            return

        # -----------------------------
        # Robot pose (map frame)
        # -----------------------------
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # -----------------------------
        # Closest point on global path
        # -----------------------------
        dist, idx = self._path_tree.query([x, y])
        closest_pt = self._path_pts[idx]

        # -----------------------------
        # Path tangent (forward)
        # -----------------------------
        if idx < len(self._path_pts) - 1:
            next_pt = self._path_pts[idx + 1]
        else:
            next_pt = closest_pt

        tangent = next_pt - closest_pt
        t_norm = np.linalg.norm(tangent)
        if t_norm > 1e-6:
            tangent /= t_norm
        else:
            tangent = np.array([math.cos(yaw), math.sin(yaw)])

        # -----------------------------
        # Signed Cross-Track Error
        # -----------------------------
        error_vec = np.array([x, y]) - closest_pt
        cross = tangent[0] * error_vec[1] - tangent[1] * error_vec[0]
        signed_cte = math.copysign(np.linalg.norm(error_vec), cross)

        # -----------------------------
        # Heading error (robot vs path)
        # -----------------------------
        path_yaw = math.atan2(tangent[1], tangent[0])
        heading_error = math.atan2(
            math.sin(yaw - path_yaw),
            math.cos(yaw - path_yaw)
        )

        # -----------------------------
        # Timing
        # -----------------------------
        elapsed = (self.get_clock().now() - self._t0).nanoseconds / 1e9

        # -----------------------------
        # Velocity (from odom)
        # -----------------------------
        lin_x, ang_z = self._latest_vel

        # -----------------------------
        # Store row
        # -----------------------------
        self._rows.append([
            f'{elapsed:.6f}',
            f'{x:.6f}',
            f'{y:.6f}',
            f'{signed_cte:.6f}',
            f'{heading_error:.6f}',
            f'{lin_x:.6f}',
            f'{ang_z:.6f}',
            idx
        ])

        self._last_write_time = now_wall

    # --------------------------------------------------
    def define_waypoints(self):
        wp = PoseStamped()
        wp.header.frame_id = 'map'
        wp.pose.position.z = 0.0

        if self.direction == 'forth':
            wp.pose.position.x = 8.64
            wp.pose.position.y = 0.07
            yaw = math.pi / 2.0
        else:  # back
            wp.pose.position.x = 0.0
            wp.pose.position.y = 0.0
            yaw = 0.0

        wp.pose.orientation.z = math.sin(yaw / 2.0)
        wp.pose.orientation.w = math.cos(yaw / 2.0)

        return [wp]

    # --------------------------------------------------
    def send_goal(self):
        waypoints = self.define_waypoints()
        now = self.get_clock().now().to_msg()

        for w in waypoints:
            w.header.stamp = now

        # Prepare logging
        self._rows.clear()
        self._recording = True
        self._last_write_time = 0.0
        self._t0 = self.get_clock().now()

        goal = FollowWaypoints.Goal()
        goal.poses = waypoints

        self.get_logger().info('Waiting for FollowWaypoints action server...')
        self._action_client.wait_for_server()

        self.get_logger().info('Sending waypoint goal')
        self._send_goal_future = self._action_client.send_goal_async(
            goal,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Waypoint goal rejected')
            self.stop_and_save()
            return

        self.get_logger().info('Waypoint goal accepted')
        self._result_future = goal_handle.get_result_async()
        self._result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        idx = feedback_msg.feedback.current_waypoint
        self.get_logger().info(f'Currently heading to waypoint index {idx}')


    def get_result_callback(self, future):
        status = future.result().status
        self.get_logger().info(
            f'Finished {self.direction} | loop {self.loop_count} | status {status}'
        )

        self.stop_and_save()

        # ---- SWITCH DIRECTION ----
        if self.direction == 'forth':
            self.direction = 'back'
        else:
            self.direction = 'forth'
            self.loop_count += 1

        # ---- CHECK END CONDITION ----
        if self.loop_count >= self.max_loops:
            self.get_logger().info('Experiment completed')
            rclpy.shutdown()
            return

        # ---- PREPARE NEXT RUN ----
        self._csv_path = self._build_csv_path()
        self._rows.clear()
        self._t0 = None

        time.sleep(1.0)
        self.send_goal()

    # --------------------------------------------------
    def stop_and_save(self):
        self._recording = False

        if not self._rows:
            self.get_logger().warn('No data recorded, CSV not written')
            return

        # -----------------------------
        # Time offset correction
        # -----------------------------
        t0 = float(self._rows[0][0])  # first logged time

        corrected_rows = []
        for row in self._rows:
            corrected_time = float(row[0]) - t0
            corrected_rows.append([
                f'{corrected_time:.6f}',  # corrected time
                *row[1:]                  # rest unchanged
            ])

        header = [
            'time_s',
            'robot_x',
            'robot_y',
            'signed_cte',
            'heading_error_rad',
            'linear_x',
            'angular_z',
            'closest_path_idx'
        ]

        with open(self._csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(corrected_rows)

        self.get_logger().info(
            f'Path error CSV written ({len(corrected_rows)} rows, time offset applied)'
        )

def main(args=None):
    rclpy.init(args=args)
    node = WaypointFollowerPathError()

    # Give Nav2 time to publish global path
    time.sleep(2.0)

    # Start first goal (looping handled internally)
    node.send_goal()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn('Interrupted by user')

        # Save only if currently recording
        if node._recording:
            node.stop_and_save()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import os
import csv
import math
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSPresetProfiles

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from nav2_msgs.action import FollowWaypoints


def yaw_to_quat(yaw_rad: float):
    half = 0.5 * yaw_rad
    return (0.0, 0.0, math.sin(half), math.cos(half))


class WaypointFollowerOdom(Node):
    """
    Follow a waypoint and record *actual* velocities from /odom to CSV.

    - Timing starts at the moment we SEND the FollowWaypoints goal,
      so we measure from "start sending goal" to goal completion.
    - First row is time=0 with linear_x = 0, angular_z = 0.
    - New rows are logged at most every `min_interval` seconds.
    - Velocities are taken from:
        odom.twist.twist.linear.x
        odom.twist.twist.angular.z
    """

    def __init__(self):
        super().__init__('waypoint_follower_odom')
        self._action_client = ActionClient(self, FollowWaypoints, 'follow_waypoints')

        # Parameters
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('csv_dir', os.path.expanduser('~/report_ccu/controller_data/DWB/speed'))
        self.declare_parameter('csv_filename', '')     # if empty => auto timestamp name
        self.declare_parameter('min_interval', 0.5)    # seconds between logs

        # Load parameters
        self.min_interval = self.get_parameter(
            'min_interval'
        ).get_parameter_value().double_value

        # Recording state
        self._recording = False
        self._t0 = None          # ROS time WHEN WE START (when sending goal)
        self._rows = []          # [time_s, linear_x, angular_z]
        self._csv_path = self._build_csv_path()

        # Throttle state (wall time)
        self._last_write_time = 0.0

        # Subscriber to /odom
        odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self._odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self._odom_cb,
            QoSPresetProfiles.SENSOR_DATA.value
        )

        self.get_logger().info(
            f'WaypointFollowerOdom initialized.\n'
            f'  odom_topic:   {odom_topic}\n'
            f'  csv_path:     {self._csv_path}\n'
            f'  min_interval: {self.min_interval}s'
        )

    # ------------------------------------------------------------------ #
    # Utility: CSV path
    # ------------------------------------------------------------------ #
    def _build_csv_path(self):
        csv_dir = self.get_parameter('csv_dir').get_parameter_value().string_value
        name = self.get_parameter('csv_filename').get_parameter_value().string_value
        if not name:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            name = f'odom_vel_log_{ts}.csv'
        os.makedirs(csv_dir, exist_ok=True)
        return os.path.join(csv_dir, name)

    # ------------------------------------------------------------------ #
    # Waypoints definition (same style as before)
    # ------------------------------------------------------------------ #
    def define_waypoints(self):
        waypoints = []

        wp1 = PoseStamped()
        wp1.header.frame_id = 'map'
        wp1.pose.position.x = 8.64
        wp1.pose.position.y = 0.07
        wp1.pose.position.z = 0.0

        qx, qy, qz, qw = yaw_to_quat(-math.pi / 2.0)
        wp1.pose.orientation.x = qx
        wp1.pose.orientation.y = qy
        wp1.pose.orientation.z = qz
        wp1.pose.orientation.w = qw

        waypoints.append(wp1)
        return waypoints

    # ------------------------------------------------------------------ #
    # Action handling
    # ------------------------------------------------------------------ #
    def send_goal(self):
        waypoints = self.define_waypoints()
        now_msg_time = self.get_clock().now().to_msg()
        for w in waypoints:
            w.header.stamp = now_msg_time

        # Prepare for recording
        self._rows.clear()
        self._recording = True
        self._t0 = None
        self._last_write_time = 0.0

        goal_msg = FollowWaypoints.Goal()
        goal_msg.poses = waypoints

        self.get_logger().info('Waiting for follow_waypoints action server...')
        self._action_client.wait_for_server()

        self.get_logger().info('Sending waypoint(s) goal…')

        # TIME ORIGIN: at the moment we send the goal
        self._t0 = self.get_clock().now()

        # First row: time = 0, velocities = 0
        self._rows.append([
            f'{0.0:.6f}',
            f'{0.0:.6f}',  # linear_x
            f'{0.0:.6f}',  # angular_z
        ])

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().warn('Goal rejected.')
            self._stop_and_write_csv(final_status='REJECTED')
            rclpy.shutdown()
            return

        self.get_logger().info('Goal accepted. Timing already started in send_goal().')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        current_waypoint = feedback_msg.feedback.current_waypoint
        self.get_logger().info(f'Navigating to waypoint index {current_waypoint}')

    def get_result_callback(self, future):
        result = future.result()
        status = getattr(result, 'status', None)
        if status == 4:
            self.get_logger().info('Waypoint following completed: SUCCEEDED.')
            self._stop_and_write_csv(final_status='SUCCEEDED')
        else:
            self.get_logger().warn(f'Waypoint following finished with status: {status}')
            self._stop_and_write_csv(final_status=f'STATUS_{status}')
        rclpy.shutdown()

    # ------------------------------------------------------------------ #
    # /odom callback: logging with min_interval
    # ------------------------------------------------------------------ #
    def _odom_cb(self, msg: Odometry):
        # Only log while recording and after we have a start time
        if not self._recording or self._t0 is None:
            return

        current_time = time.time()
        if current_time - self._last_write_time < self.min_interval:
            return  # too soon since last write

        now_ros = self.get_clock().now()
        elapsed = (now_ros - self._t0).nanoseconds / 1e9  # seconds since we sent the goal

        lin_x = msg.twist.twist.linear.x
        if lin_x < 0.0:
            lin_x = 0.0  # clamp negative forward velocity to 0

        ang_z = msg.twist.twist.angular.z


        row = [
            f'{elapsed:.6f}',
            f'{lin_x:.6f}',
            f'{ang_z:.6f}',
        ]
        self._rows.append(row)
        self._last_write_time = current_time

    # ------------------------------------------------------------------ #
    # CSV writing
    # ------------------------------------------------------------------ #
    def _stop_and_write_csv(self, final_status='UNKNOWN'):
        if not self._recording:
            return
        self._recording = False

        header = [
            'time_s',
            'linear_x',
            'angular_z',
        ]

        try:
            with open(self._csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(self._rows)
            self.get_logger().info(
                f'odom vel CSV written: {self._csv_path} '
                f'(status={final_status}, rows={len(self._rows)})'
            )
        except Exception as e:
            self.get_logger().error(f'Failed to write CSV: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = WaypointFollowerOdom()
    node.send_goal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, writing partial CSV.')
        node._stop_and_write_csv(final_status='INTERRUPT')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

# import os
# import csv
# import math
# import time
# from datetime import datetime

# import rclpy
# from rclpy.node import Node
# from rclpy.action import ActionClient
# from rclpy.qos import QoSPresetProfiles

# from geometry_msgs.msg import PoseStamped, Twist
# from nav2_msgs.action import FollowWaypoints


# def yaw_to_quat(yaw_rad: float):
#     half = 0.5 * yaw_rad
#     return (0.0, 0.0, math.sin(half), math.cos(half))


# class WaypointFollowerCmdVel(Node):
#     """
#     Follow a waypoint and record /cmd_vel to CSV.

#     - Timing starts at the moment we SEND the FollowWaypoints goal,
#       so we measure from "start sending goal" to goal completion.
#     - First row is time=0 with cmd_vel = 0,0.
#     - New rows are logged at most every `min_interval` seconds.
#     """

#     def __init__(self):
#         super().__init__('waypoint_follower_cmd_vel')
#         self._action_client = ActionClient(self, FollowWaypoints, 'follow_waypoints')

#         # Parameters
#         self.declare_parameter('cmd_vel_topic', '/cmd_vel')
#         self.declare_parameter('csv_dir', os.path.expanduser('~/cmd_vel_data/speed_1_5'))
#         self.declare_parameter('csv_filename', '')     # if empty => auto timestamp name
#         self.declare_parameter('min_interval', 0.15)    # seconds between logs

#         # Load parameters
#         self.min_interval = self.get_parameter(
#             'min_interval'
#         ).get_parameter_value().double_value

#         # Recording state
#         self._recording = False
#         self._t0 = None          # ROS time WHEN WE START (when sending goal)
#         self._rows = []          # [time_s, linear_x, angular_z]
#         self._csv_path = self._build_csv_path()

#         # Throttle state (wall time)
#         self._last_write_time = 0.0

#         # Subscriber to cmd_vel
#         cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
#         self._cmd_vel_sub = self.create_subscription(
#             Twist,
#             cmd_vel_topic,
#             self._cmd_vel_cb,
#             QoSPresetProfiles.SENSOR_DATA.value
#         )

#         self.get_logger().info(
#             f'WaypointFollowerCmdVel initialized. Logging {cmd_vel_topic} to {self._csv_path}, '
#             f'min_interval={self.min_interval}s'
#         )

#     # ------------------------------------------------------------------ #
#     # Utility: CSV path
#     # ------------------------------------------------------------------ #
#     def _build_csv_path(self):
#         csv_dir = self.get_parameter('csv_dir').get_parameter_value().string_value
#         name = self.get_parameter('csv_filename').get_parameter_value().string_value
#         if not name:
#             ts = datetime.now().strftime('%Y%m%d_%H%M%S')
#             name = f'cmd_vel_log_{ts}.csv'
#         os.makedirs(csv_dir, exist_ok=True)
#         return os.path.join(csv_dir, name)

#     # ------------------------------------------------------------------ #
#     # Waypoints definition (same style as your existing file)
#     # ------------------------------------------------------------------ #
#     def define_waypoints(self):
#         waypoints = []

#         wp1 = PoseStamped()
#         wp1.header.frame_id = 'map'
#         wp1.pose.position.x = 8.64
#         wp1.pose.position.y = 0.07
#         wp1.pose.position.z = 0.0

#         qx, qy, qz, qw = yaw_to_quat(-math.pi / 2.0)
#         wp1.pose.orientation.x = qx
#         wp1.pose.orientation.y = qy
#         wp1.pose.orientation.z = qz
#         wp1.pose.orientation.w = qw

#         waypoints.append(wp1)
#         return waypoints

#     # ------------------------------------------------------------------ #
#     # Action handling
#     # ------------------------------------------------------------------ #
#     def send_goal(self):
#         waypoints = self.define_waypoints()
#         now_msg_time = self.get_clock().now().to_msg()
#         for w in waypoints:
#             w.header.stamp = now_msg_time

#         # Prepare for recording
#         self._rows.clear()
#         self._recording = True
#         self._t0 = None
#         self._last_write_time = 0.0  # reset throttle

#         goal_msg = FollowWaypoints.Goal()
#         goal_msg.poses = waypoints

#         self.get_logger().info('Waiting for follow_waypoints action server...')
#         self._action_client.wait_for_server()

#         self.get_logger().info('Sending waypoint(s) goal…')

#         # TIME ORIGIN: at the moment we send the goal
#         self._t0 = self.get_clock().now()

#         # First row: time = 0, cmd_vel = 0
#         self._rows.append([
#             f'{0.0:.6f}',
#             f'{0.0:.6f}',
#             f'{0.0:.6f}',
#         ])

#         self._send_goal_future = self._action_client.send_goal_async(
#             goal_msg,
#             feedback_callback=self.feedback_callback
#         )
#         self._send_goal_future.add_done_callback(self.goal_response_callback)

#     def goal_response_callback(self, future):
#         goal_handle = future.result()
#         if not goal_handle or not goal_handle.accepted:
#             self.get_logger().warn('Goal rejected.')
#             self._stop_and_write_csv(final_status='REJECTED')
#             rclpy.shutdown()
#             return

#         self.get_logger().info('Goal accepted. Timing already started in send_goal().')

#         self._get_result_future = goal_handle.get_result_async()
#         self._get_result_future.add_done_callback(self.get_result_callback)

#     def feedback_callback(self, feedback_msg):
#         current_waypoint = feedback_msg.feedback.current_waypoint
#         self.get_logger().info(f'Navigating to waypoint index {current_waypoint}')

#     def get_result_callback(self, future):
#         result = future.result()
#         status = getattr(result, 'status', None)
#         if status == 4:
#             self.get_logger().info('Waypoint following completed: SUCCEEDED.')
#             self._stop_and_write_csv(final_status='SUCCEEDED')
#         else:
#             self.get_logger().warn(f'Waypoint following finished with status: {status}')
#             self._stop_and_write_csv(final_status=f'STATUS_{status}')
#         rclpy.shutdown()

#     # ------------------------------------------------------------------ #
#     # /cmd_vel callback: logging with min_interval
#     # ------------------------------------------------------------------ #
#     def _cmd_vel_cb(self, msg: Twist):
#         # Only log while recording and after we have a start time
#         if not self._recording or self._t0 is None:
#             return

#         current_time = time.time()
#         if current_time - self._last_write_time < self.min_interval:
#             return  # skip this message (too soon since last write)

#         now_ros = self.get_clock().now()
#         elapsed = (now_ros - self._t0).nanoseconds / 1e9  # seconds since we sent the goal

#         row = [
#             f'{elapsed:.6f}',
#             f'{msg.linear.x:.6f}',
#             f'{msg.angular.z:.6f}',
#         ]
#         self._rows.append(row)

#         self._last_write_time = current_time

#     # ------------------------------------------------------------------ #
#     # CSV writing
#     # ------------------------------------------------------------------ #
#     def _stop_and_write_csv(self, final_status='UNKNOWN'):
#         if not self._recording:
#             return
#         self._recording = False

#         header = [
#             'time_s',
#             'linear_x',
#             'angular_z',
#         ]

#         try:
#             with open(self._csv_path, 'w', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(header)
#                 writer.writerows(self._rows)
#             self.get_logger().info(
#                 f'cmd_vel CSV written: {self._csv_path} '
#                 f'(status={final_status}, rows={len(self._rows)})'
#             )
#         except Exception as e:
#             self.get_logger().error(f'Failed to write cmd_vel CSV: {e}')


# def main(args=None):
#     rclpy.init(args=args)
#     node = WaypointFollowerCmdVel()
#     node.send_goal()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.get_logger().info('KeyboardInterrupt, writing partial cmd_vel CSV.')
#         node._stop_and_write_csv(final_status='INTERRUPT')
#     finally:
#         node.destroy_node()
#         rclpy.try_shutdown()


# if __name__ == '__main__':
#     main()

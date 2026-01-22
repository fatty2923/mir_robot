#!/usr/bin/env python3
import os
import csv
import math
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSPresetProfiles

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import FollowWaypoints
from std_msgs.msg import Float32


def yaw_to_quat(yaw_rad: float):
    half = 0.5 * yaw_rad
    return (0.0, 0.0, math.sin(half), math.cos(half))


class WaypointFollower(Node):
    def __init__(self):
        super().__init__('waypoint_follower')
        self._action_client = ActionClient(self, FollowWaypoints, 'follow_waypoints')

        # Params
        self.declare_parameter('nearest_topic', '/local_costmap/nearest_obstacle_distance')
        self.declare_parameter('csv_dir', os.path.expanduser('~/report_ccu/distance_data/normal_test_speed_1_0'))
        self.declare_parameter('csv_filename', '')  # if empty => auto timestamp name

        # Recording state
        self._recording = False
        self._t0 = None  # set WHEN GOAL IS ACCEPTED
        self._rows = []  # [elapsed_s, distance_m]
        self._csv_path = self._build_csv_path()
        self._offset = None  # first-sample elapsed; used to rebase time to start at 0

        # Subscribe to nearest distance
        nearest_topic = self.get_parameter('nearest_topic').get_parameter_value().string_value
        self._nearest_sub = self.create_subscription(
            Float32, nearest_topic, self._nearest_cb, QoSPresetProfiles.SENSOR_DATA.value
        )

    def _build_csv_path(self):
        csv_dir = self.get_parameter('csv_dir').get_parameter_value().string_value
        name = self.get_parameter('csv_filename').get_parameter_value().string_value
        if not name:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            name = f'nearest_obstacle_log_{ts}.csv'
        os.makedirs(csv_dir, exist_ok=True)
        return os.path.join(csv_dir, name)

    def define_waypoints(self):
        waypoints = []
        wp1 = PoseStamped()
        wp1.header.frame_id = 'map'
        wp1.pose.position.x = 8.64
        wp1.pose.position.y = 0.07
        wp1.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quat(-math.pi/2)
        wp1.pose.orientation.x = qx
        wp1.pose.orientation.y = qy
        wp1.pose.orientation.z = qz
        wp1.pose.orientation.w = qw
        waypoints.append(wp1)
        return waypoints

    def send_goal(self):
        waypoints = self.define_waypoints()
        now_msg_time = self.get_clock().now().to_msg()
        for w in waypoints:
            w.header.stamp = now_msg_time

        # Prepare for recording; start flag ON, but t0 set at GOAL ACCEPTED
        self._rows.clear()
        self._t0 = None
        self._recording = True

        goal_msg = FollowWaypoints.Goal()
        goal_msg.poses = waypoints

        self.get_logger().info('Waiting for follow_waypoints action server...')
        self._action_client.wait_for_server()
        self.get_logger().info('Sending waypoint(s) goal…')

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().warn('Goal rejected.')
            self._stop_and_write_csv(final_status='REJECTED')
            rclpy.shutdown()
            return

        # >>> Option A: start timing when GOAL IS ACCEPTED
        self._t0 = self.get_clock().now()
        self.get_logger().info('Goal accepted. Timing started. Waiting for result…')

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

    def _nearest_cb(self, msg: Float32):
        # Only log after goal acceptance (t0 set)
        if not self._recording or self._t0 is None:
            return

        now = self.get_clock().now()
        elapsed = (now - self._t0).nanoseconds / 1e9  # seconds since GOAL ACCEPTED

        # Rebase: make the first recorded sample start at 0.0 s
        if self._offset is None:
            self._offset = elapsed
        rebased = elapsed - self._offset

        self._rows.append([f'{rebased:.6f}', f'{msg.data:.6f}'])


    def _stop_and_write_csv(self, final_status='UNKNOWN'):
        if not self._recording:
            return
        self._recording = False

        header = ['time_s', 'nearest_distance_m']
        try:
            with open(self._csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(self._rows)
            self.get_logger().info(f'CSV written: {self._csv_path} (status={final_status})')
        except Exception as e:
            self.get_logger().error(f'Failed to write CSV: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = WaypointFollower()
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

# import math
# import rclpy
# from rclpy.node import Node
# from rclpy.action import ActionClient
# from geometry_msgs.msg import PoseStamped
# from nav2_msgs.action import FollowWaypoints


# def yaw_to_quat(yaw_rad: float):
#     # roll = pitch = 0; only yaw
#     half = 0.5 * yaw_rad
#     return (0.0, 0.0, math.sin(half), math.cos(half))


# class WaypointFollower(Node):
#     def __init__(self):
#         super().__init__('waypoint_follower')
#         self._action_client = ActionClient(self, FollowWaypoints, 'follow_waypoints')

#     def define_waypoints(self):
#         waypoints = []

#         # Waypoint 1 (8.64, 0.07)
#         wp1 = PoseStamped()
#         wp1.header.frame_id = 'map'
#         wp1.pose.position.x = 8.64
#         wp1.pose.position.y = 0.07
#         wp1.pose.position.z = 0.0
#         qx, qy, qz, qw = yaw_to_quat(-math.pi/2)
#         wp1.pose.orientation.x = qx
#         wp1.pose.orientation.y = qy
#         wp1.pose.orientation.z = qz
#         wp1.pose.orientation.w = qw
#         waypoints.append(wp1)

#         # (Add more waypoints here later if needed, same style as above)

#         return waypoints

#     def send_goal(self):
#         waypoints = self.define_waypoints()
#         for w in waypoints:
#             w.header.stamp = self.get_clock().now().to_msg()

#         goal_msg = FollowWaypoints.Goal()
#         goal_msg.poses = waypoints

#         self.get_logger().info('Waiting for follow_waypoints action server...')
#         self._action_client.wait_for_server()
#         self.get_logger().info('Sending waypoint(s) goal...')

#         self._send_goal_future = self._action_client.send_goal_async(
#             goal_msg, feedback_callback=self.feedback_callback
#         )
#         self._send_goal_future.add_done_callback(self.goal_response_callback)

#     def goal_response_callback(self, future):
#         goal_handle = future.result()
#         if not goal_handle or not goal_handle.accepted:
#             self.get_logger().info('Goal rejected.')
#             rclpy.shutdown()
#             return

#         self.get_logger().info('Goal accepted. Waiting for result...')
#         self._get_result_future = goal_handle.get_result_async()
#         self._get_result_future.add_done_callback(self.get_result_callback)

#     def feedback_callback(self, feedback_msg):
#         feedback = feedback_msg.feedback
#         current_waypoint = feedback.current_waypoint
#         self.get_logger().info(f'Navigating to waypoint index {current_waypoint}')

#     def get_result_callback(self, future):
#         result = future.result()
#         if result is None:
#             self.get_logger().error('No result received.')
#         else:
#             self.get_logger().info('Waypoint following completed.')
#         rclpy.shutdown()


# def main(args=None):
#     rclpy.init(args=args)
#     waypoint_follower = WaypointFollower()
#     waypoint_follower.send_goal()
#     rclpy.spin(waypoint_follower)


# if __name__ == '__main__':
#     main()

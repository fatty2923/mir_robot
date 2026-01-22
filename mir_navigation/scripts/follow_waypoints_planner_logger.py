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
from nav_msgs.msg import Odometry, Path
from nav2_msgs.action import FollowWaypoints
from std_msgs.msg import Float32


def yaw_to_quat(yaw_rad: float):
    half = 0.5 * yaw_rad
    return (0.0, 0.0, math.sin(half), math.cos(half))


class WaypointFollowerOdomNearest(Node):
    """
    Follow a waypoint and record:
      - Actual velocities from /odom
      - Nearest obstacle distance from /local_costmap/nearest_obstacle_distance
      - Global planner evaluation metrics in a separate CSV

    Files (same directory: csv_dir):
      1) odom_nearest_log_YYYYmmdd_HHMMSS.csv
         - time_s, linear_x, angular_z, nearest_distance_m

      2) planner_evaluation_results.csv (append mode)
         - total_exc_time_s
         - total_path_length_m
         - planning_time_s
         - average_speed_m_s
         - final_position_x
         - final_position_y
    """

    def __init__(self):
        super().__init__('waypoint_follower_odom_nearest')
        self._action_client = ActionClient(self, FollowWaypoints, 'follow_waypoints')

        # Parameters
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('nearest_topic', '/local_costmap/nearest_obstacle_distance')
        self.declare_parameter('plan_topic', '/plan')
        self.declare_parameter('csv_dir', os.path.expanduser('~/report_ccu/global_planner_data/theta_star_planner'))
        self.declare_parameter('csv_filename', '')   # if empty => auto timestamp name
        self.declare_parameter('min_interval', 0.5)  # seconds between logs

        # Load parameters
        self.min_interval = self.get_parameter(
            'min_interval'
        ).get_parameter_value().double_value

        odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        nearest_topic = self.get_parameter('nearest_topic').get_parameter_value().string_value
        plan_topic = self.get_parameter('plan_topic').get_parameter_value().string_value

        # Paths
        self._csv_path = self._build_csv_path()             # odom + nearest log (per run)
        self._eval_csv_path = self._build_eval_csv_path()   # planner evaluation log (all runs)
        self._ensure_eval_csv_header()

        # Recording state for odom+nearest CSV
        self._recording = False
        self._t0 = None                    # ROS time WHEN WE START (send goal)
        self._t_end = None                 # ROS time when we finish
        self._rows = []                    # [time_s, linear_x, angular_z, nearest_distance_m]

        # Latest values
        self._last_odom_linear_x = 0.0
        self._last_odom_angular_z = 0.0
        self._last_nearest_distance = 0.0

        # Throttle state (wall time)
        self._last_write_time = 0.0

        # Planner evaluation state
        self._total_path_length = 0.0
        self._last_pose = None
        self._robot_pose = None
        self._planning_start_wall = None
        self._planning_end_wall = None
        self._planned_path = None
        self._last_waypoint_index = -1
        self._waypoints_count = 0

        # Subscribers
        self._odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self._odom_cb,
            QoSPresetProfiles.SENSOR_DATA.value
        )

        self._nearest_sub = self.create_subscription(
            Float32,
            nearest_topic,
            self._nearest_cb,
            QoSPresetProfiles.SENSOR_DATA.value
        )

        self._plan_sub = self.create_subscription(
            Path,
            plan_topic,
            self._plan_cb,
            QoSPresetProfiles.SENSOR_DATA.value
        )

        self.get_logger().info(
            f'WaypointFollowerOdomNearest initialized.\n'
            f'  odom_topic:       {odom_topic}\n'
            f'  nearest_topic:    {nearest_topic}\n'
            f'  plan_topic:       {plan_topic}\n'
            f'  csv_path (odom):  {self._csv_path}\n'
            f'  eval_csv_path:    {self._eval_csv_path}\n'
            f'  min_interval:     {self.min_interval}s'
        )

    # ------------------------------------------------------------------ #
    # Utility: CSV paths
    # ------------------------------------------------------------------ #
    def _build_csv_path(self):
        csv_dir = self.get_parameter('csv_dir').get_parameter_value().string_value
        name = self.get_parameter('csv_filename').get_parameter_value().string_value
        if not name:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            name = f'odom_nearest_log_{ts}.csv'
        os.makedirs(csv_dir, exist_ok=True)
        return os.path.join(csv_dir, name)

    def _build_eval_csv_path(self):
        csv_dir = self.get_parameter('csv_dir').get_parameter_value().string_value
        os.makedirs(csv_dir, exist_ok=True)
        # Fixed name, all runs append here
        return os.path.join(csv_dir, 'planner_evaluation_results.csv')

    def _ensure_eval_csv_header(self):
        if not os.path.exists(self._eval_csv_path):
            try:
                with open(self._eval_csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'total_exc_time_s',
                        'total_path_length_m',
                        'planning_time_s',
                        'average_speed_m_s',
                        'final_position_x',
                        'final_position_y',
                    ])
                self.get_logger().info(f'Created planner evaluation CSV: {self._eval_csv_path}')
            except Exception as e:
                self.get_logger().error(f'Failed to create eval CSV: {e}')

    # ------------------------------------------------------------------ #
    # Waypoints definition
    # ------------------------------------------------------------------ #
    def define_waypoints(self):
        waypoints = []

        wp1 = PoseStamped()
        wp1.header.frame_id = 'map'
        wp1.pose.position.x = 8.15
        wp1.pose.position.y = -8.56
        wp1.pose.position.z = 0.0

        qx, qy, qz, qw = yaw_to_quat(0.0)
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

        self._waypoints_count = len(waypoints)

        # Prepare for recording (both odom+nearest and planner eval)
        self._rows.clear()
        self._recording = True
        self._t0 = None
        self._t_end = None
        self._last_write_time = 0.0

        # Reset latest values
        self._last_odom_linear_x = 0.0
        self._last_odom_angular_z = 0.0
        self._last_nearest_distance = 0.0

        # Reset planner evaluation state
        self._total_path_length = 0.0
        self._last_pose = None
        self._robot_pose = None
        self._planned_path = None
        self._last_waypoint_index = -1
        self._planning_start_wall = time.time()  # planning starts "now" when we send goal
        self._planning_end_wall = None

        goal_msg = FollowWaypoints.Goal()
        goal_msg.poses = waypoints

        self.get_logger().info('Waiting for follow_waypoints action server...')
        self._action_client.wait_for_server()

        self.get_logger().info('Sending waypoint(s) goal…')

        # TIME ORIGIN (ROS time): at the moment we send the goal
        self._t0 = self.get_clock().now()

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().warn('Goal rejected.')
            # No result callback, so finalize here
            self._t_end = self.get_clock().now()
            self._stop_and_write_csv(final_status='REJECTED')
            rclpy.shutdown()
            return

        self.get_logger().info('Goal accepted. Timing already started in send_goal().')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        current_waypoint = feedback_msg.feedback.current_waypoint
        self._last_waypoint_index = int(current_waypoint)
        self.get_logger().info(f'Navigating to waypoint index {current_waypoint}')

    def get_result_callback(self, future):
        result = future.result()
        status = getattr(result, 'status', None)

        # Record end time in ROS clock
        self._t_end = self.get_clock().now()

        if status == 4:
            self.get_logger().info('Waypoint following completed: SUCCEEDED.')
            self._stop_and_write_csv(final_status='SUCCEEDED')
        else:
            self.get_logger().warn(f'Waypoint following finished with status: {status}')
            self._stop_and_write_csv(final_status=f'STATUS_{status}')
        rclpy.shutdown()

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    def _odom_cb(self, msg: Odometry):
        # Update latest velocities
        lin_x = msg.twist.twist.linear.x
        if lin_x < 0.0:
            lin_x = 0.0  # clamp negative to 0

        ang_z = msg.twist.twist.angular.z

        self._last_odom_linear_x = lin_x
        self._last_odom_angular_z = ang_z

        # For path length + final pose
        pose = msg.pose.pose
        self._robot_pose = pose

        if self._recording:
            if self._last_pose is not None:
                dx = pose.position.x - self._last_pose.position.x
                dy = pose.position.y - self._last_pose.position.y
                dz = pose.position.z - self._last_pose.position.z
                distance = math.sqrt(dx * dx + dy * dy + dz * dz)
                self._total_path_length += distance

            self._last_pose = pose

    def _nearest_cb(self, msg: Float32):
        # Log each nearest-distance sample together with latest odom velocity,
        # but obey min_interval for throttling.
        if not self._recording or self._t0 is None:
            return

        current_time = time.time()
        if current_time - self._last_write_time < self.min_interval:
            return  # too soon since last write

        now_ros = self.get_clock().now()
        elapsed = (now_ros - self._t0).nanoseconds / 1e9  # seconds since we sent the goal

        self._last_nearest_distance = msg.data

        if not self._rows:
            # First row:
            #   time_s = 0.0
            #   linear_x = 0.0
            #   angular_z = 0.0
            #   nearest_distance_m = first actual value
            row = [
                f'{0.0:.6f}',
                f'{0.0:.6f}',  # linear_x
                f'{0.0:.6f}',  # angular_z
                f'{self._last_nearest_distance:.6f}',
            ]
        else:
            # Subsequent rows: use elapsed time and current odom velocities
            row = [
                f'{elapsed:.6f}',
                f'{self._last_odom_linear_x:.6f}',
                f'{self._last_odom_angular_z:.6f}',
                f'{self._last_nearest_distance:.6f}',
            ]

        self._rows.append(row)
        self._last_write_time = current_time

    def _plan_cb(self, msg: Path):
        """
        Capture planning_time as:
          planning_time = time_when_first_plan_received - time_when_goal_sent

        We only take the FIRST path after a goal is sent, and only once per run.
        """
        if not self._recording:
            return

        # Store the path once if you ever want to use it later
        if self._planned_path is None:
            self._planned_path = msg

        # planning_start_wall is set in send_goal(); here we mark the end
        if self._planning_end_wall is None and self._planning_start_wall is not None:
            self._planning_end_wall = time.time()

    # ------------------------------------------------------------------ #
    # CSV writing
    # ------------------------------------------------------------------ #
    def _stop_and_write_csv(self, final_status='UNKNOWN'):
        if not self._recording:
            return
        self._recording = False

        # 1) Write odom+nearest CSV (per-run file)
        header = [
            'time_s',
            'linear_x',
            'angular_z',
            'nearest_distance_m',
        ]

        try:
            with open(self._csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(self._rows)
            self.get_logger().info(
                f'odom+nearest CSV written: {self._csv_path} '
                f'(status={final_status}, rows={len(self._rows)})'
            )
        except Exception as e:
            self.get_logger().error(f'Failed to write odom+nearest CSV: {e}')

        # 2) Append planner evaluation row to planner_evaluation_results.csv
        self._write_eval_csv_row(final_status)

    def _write_eval_csv_row(self, final_status: str):
        # Total execution time (ROS time)
        if self._t0 is None:
            total_exc_time = 0.0
        else:
            if self._t_end is None:
                self._t_end = self.get_clock().now()
            dt_ns = (self._t_end - self._t0).nanoseconds
            total_exc_time = max(0.0, dt_ns / 1e9)

        # Planning time (wall-clock, from send_goal to first /plan)
        planning_time = 0.0
        if self._planning_start_wall is not None and self._planning_end_wall is not None:
            planning_time = max(0.0, self._planning_end_wall - self._planning_start_wall)

        # Total path length (meters) already accumulated from odom
        total_path_length = max(0.0, self._total_path_length)

        # Average speed
        if total_exc_time > 0.0:
            average_speed = total_path_length / total_exc_time
        else:
            average_speed = 0.0

        # Final position
        if self._robot_pose is not None:
            final_x = float(self._robot_pose.position.x)
            final_y = float(self._robot_pose.position.y)
        else:
            final_x = 0.0
            final_y = 0.0

        row = [
            f'{total_exc_time:.3f}',
            f'{total_path_length:.3f}',
            f'{planning_time:.3f}',
            f'{average_speed:.3f}',
            f'{final_x:.3f}',
            f'{final_y:.3f}',
        ]

        try:
            with open(self._eval_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            self.get_logger().info(
                f'Planner evaluation row appended to: {self._eval_csv_path}'
            )
        except Exception as e:
            self.get_logger().error(f'Failed to append planner eval row: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = WaypointFollowerOdomNearest()
    node.send_goal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, writing partial CSV.')
        node._t_end = node.get_clock().now()
        node._stop_and_write_csv(final_status='INTERRUPT')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

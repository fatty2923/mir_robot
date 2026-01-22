#!/usr/bin/env python3
import os
import csv
import math
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSPresetProfiles
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import FollowWaypoints
from nav_msgs.msg import Path
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
        self.declare_parameter('csv_dir', os.path.expanduser('~/report_ccu/controller_data/RPP/maze_obstacle/trial_6'))
        self.declare_parameter('csv_filename', '')  # if empty => auto timestamp name

        # Timing-related params
        self.declare_parameter('plan_topic', '/plan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('distance_trigger', 0.8)  # m, when we consider obstacle "close"
        self._distance_trigger = (
            self.get_parameter('distance_trigger').get_parameter_value().double_value
        )

        # Recording state
        self._recording = False
        self._t0 = None          # set WHEN GOAL IS ACCEPTED
        self._offset = None      # first-sample elapsed; used to rebase time to start at 0
        self._rows = []          # nearest-obstacle [time_s, distance_m]

        # CSV paths
        self._csv_path = self._build_csv_path()             # nearest-obstacle CSV
        self._exec_csv_path = self._build_exec_csv_path()   # execution-time CSV

        # --- Timing accumulators ---

        # Obstacle -> replan
        self._prev_distance = None
        self._waiting_for_replan = False
        self._t_obstacle = None
        self._last_plan_stamp = None
        self._replan_sum = 0.0
        self._replan_count = 0

        # Replan -> first cmd_vel (our "cmd_vel compute time")
        self._waiting_for_cmd_after_replan = False
        self._t_replan = None
        self._cmd_gen_sum = 0.0
        self._cmd_gen_count = 0

        # cmd_vel dt (control period)
        self._last_cmd_time = None
        self._cmd_vel_sum = 0.0
        self._cmd_vel_count = 0

        # Subscriptions
        nearest_topic = self.get_parameter('nearest_topic').get_parameter_value().string_value
        self._nearest_sub = self.create_subscription(
            Float32, nearest_topic, self._nearest_cb, QoSPresetProfiles.SENSOR_DATA.value
        )

        plan_topic = self.get_parameter('plan_topic').get_parameter_value().string_value
        self._plan_sub = self.create_subscription(Path, plan_topic, self._plan_cb, 10)

        cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        self._cmd_vel_sub = self.create_subscription(
            Twist, cmd_vel_topic, self._cmd_vel_cb, QoSPresetProfiles.SENSOR_DATA.value
        )

    # ---------- CSV path helpers ----------

    def _build_csv_path(self):
        """
        Nearest-obstacle CSV path (existing behavior):
        - if csv_filename param is set, use that exactly
        - else: nearest_obstacle_log_<ts>.csv
        """
        csv_dir = self.get_parameter('csv_dir').get_parameter_value().string_value
        name = self.get_parameter('csv_filename').get_parameter_value().string_value
        if not name:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            name = f'nearest_obstacle_log_{ts}.csv'
        os.makedirs(csv_dir, exist_ok=True)
        return os.path.join(csv_dir, name)

    def _build_exec_csv_path(self):
        """
        Execution-time CSV path:
        - if csv_filename param is set, prepend 'execution_time_' to it
        - else: execution_time_log_<ts>.csv
        """
        csv_dir = self.get_parameter('csv_dir').get_parameter_value().string_value
        name = self.get_parameter('csv_filename').get_parameter_value().string_value
        if name:
            exec_name = f'execution_time_{name}'
        else:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            exec_name = f'execution_time_log_{ts}.csv'
        os.makedirs(csv_dir, exist_ok=True)
        return os.path.join(csv_dir, exec_name)

    # ---------- Waypoints / action handling ----------

    def define_waypoints(self):
        waypoints = []
        wp1 = PoseStamped()
        wp1.header.frame_id = 'map'
        wp1.pose.position.x = 6.65
        wp1.pose.position.y = -6.65
        wp1.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quat(-math.pi / 4)
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

        # Prepare for recording
        self._rows.clear()
        self._t0 = None
        self._offset = None
        self._recording = True

        # Reset timing state
        self._prev_distance = None
        self._waiting_for_replan = False
        self._t_obstacle = None
        self._last_plan_stamp = None

        self._waiting_for_cmd_after_replan = False
        self._t_replan = None

        self._replan_sum = 0.0
        self._replan_count = 0
        self._cmd_gen_sum = 0.0
        self._cmd_gen_count = 0

        self._last_cmd_time = None
        self._cmd_vel_sum = 0.0
        self._cmd_vel_count = 0

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

    # ---------- Time helper ----------

    def _get_rebased_time(self, now=None):
        if not self._recording or self._t0 is None:
            return None
        if now is None:
            now = self.get_clock().now()
        elapsed = (now - self._t0).nanoseconds / 1e9  # seconds since GOAL ACCEPTED
        if self._offset is None:
            self._offset = elapsed
        return elapsed - self._offset

    # ---------- Callbacks ----------

    def _nearest_cb(self, msg: Float32):
        # Only log after goal acceptance (t0 set)
        if not self._recording or self._t0 is None:
            return

        now = self.get_clock().now()
        rebased = self._get_rebased_time(now)
        if rebased is None:
            return

        # 1) log nearest distance vs time (first CSV)
        self._rows.append([f'{rebased:.6f}', f'{msg.data:.6f}'])

        # 2) detect obstacle entering "danger zone" for obstacle->replan timing
        d = msg.data
        if self._prev_distance is not None:
            if (self._prev_distance > self._distance_trigger and
                    d <= self._distance_trigger and
                    not self._waiting_for_replan):
                self._t_obstacle = now
                self._waiting_for_replan = True
                self.get_logger().info(
                    f'Obstacle entered trigger zone (d={d:.3f} m), waiting for replan...'
                )
        self._prev_distance = d

    def _plan_cb(self, msg: Path):
        if not self._recording or self._t0 is None:
            return

        # consider only "new" plans
        plan_stamp = Time.from_msg(msg.header.stamp)
        if self._last_plan_stamp is not None and plan_stamp <= self._last_plan_stamp:
            return
        self._last_plan_stamp = plan_stamp

        now = self.get_clock().now()

        # 1) obstacle -> replan latency
        if self._waiting_for_replan and self._t_obstacle is not None:
            dt = (now - self._t_obstacle).nanoseconds / 1e9
            self._replan_sum += dt
            self._replan_count += 1
            self.get_logger().info(
                f'Replan after obstacle: {dt:.3f} s (total events: {self._replan_count})'
            )
            self._waiting_for_replan = False
            self._t_obstacle = None

        # 2) start measuring replan -> first cmd_vel
        self._t_replan = now
        self._waiting_for_cmd_after_replan = True

    def _cmd_vel_cb(self, msg: Twist):
        if not self._recording or self._t0 is None:
            return

        now = self.get_clock().now()
        now_sec = now.nanoseconds / 1e9

        # 1) cmd_vel dt (control period)
        if self._last_cmd_time is not None:
            dt = now_sec - self._last_cmd_time
            if 0.0 < dt < 1.0:
                self._cmd_vel_sum += dt
                self._cmd_vel_count += 1
        self._last_cmd_time = now_sec

        # 2) replan -> first cmd_vel (our "cmd_vel compute time")
        if self._waiting_for_cmd_after_replan and self._t_replan is not None:
            dt_gen = (now - self._t_replan).nanoseconds / 1e9
            self._cmd_gen_sum += dt_gen
            self._cmd_gen_count += 1
            self.get_logger().info(
                f'First cmd_vel after replan: {dt_gen:.3f} s (total events: {self._cmd_gen_count})'
            )
            self._waiting_for_cmd_after_replan = False
            self._t_replan = None

    # ---------- CSV writing ----------

    def _stop_and_write_csv(self, final_status='UNKNOWN'):
        if not self._recording:
            return
        self._recording = False

        # 1) nearest obstacle CSV (unchanged style)
        header = ['time_s', 'nearest_distance_m']
        try:
            with open(self._csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(self._rows)
            self.get_logger().info(f'CSV written: {self._csv_path} (status={final_status})')
        except Exception as e:
            self.get_logger().error(f'Failed to write nearest CSV: {e}')

        # 2) execution-time CSV with only 3 metrics (one row)
        # compute averages safely
        avg_obstacle_to_replan = (
            self._replan_sum / self._replan_count if self._replan_count > 0 else 0.0
        )
        avg_cmd_gen = (
            self._cmd_gen_sum / self._cmd_gen_count if self._cmd_gen_count > 0 else 0.0
        )
        avg_cmd_dt = (
            self._cmd_vel_sum / self._cmd_vel_count if self._cmd_vel_count > 0 else 0.0
        )

        exec_header = [
            'avg_obstacle_to_replan_s',
            'avg_cmd_vel_compute_s',   # interpreted as replan -> first cmd_vel
            'avg_cmd_vel_dt_s'
        ]
        exec_row = [
            f'{avg_obstacle_to_replan:.6f}',
            f'{avg_cmd_gen:.6f}',
            f'{avg_cmd_dt:.6f}'
        ]

        try:
            with open(self._exec_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(exec_header)
                writer.writerow(exec_row)
            self.get_logger().info(f'CSV written: {self._exec_csv_path} (status={final_status})')
        except Exception as e:
            self.get_logger().error(f'Failed to write execution-time CSV: {e}')

        # Also log to console for quick check
        self.get_logger().info(
            f'avg_obstacle_to_replan_s = {avg_obstacle_to_replan:.4f} '
            f'({self._replan_count} events)'
        )
        self.get_logger().info(
            f'avg_cmd_vel_compute_s (replan->cmd_vel) = {avg_cmd_gen:.4f} '
            f'({self._cmd_gen_count} events)'
        )
        self.get_logger().info(
            f'avg_cmd_vel_dt_s = {avg_cmd_dt:.4f} '
            f'({self._cmd_vel_count} samples)'
        )


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

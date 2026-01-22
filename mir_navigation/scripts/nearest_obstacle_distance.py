#!/usr/bin/env python3
import math
import threading
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.time import Time

from std_msgs.msg import Float32
from geometry_msgs.msg import PolygonStamped
from nav2_msgs.msg import Costmap
from nav_msgs.msg import OccupancyGrid

import tf2_ros
from tf2_ros import TransformException


class NearestObstacleDistance(Node):
    def __init__(self):
        super().__init__('nearest_obstacle_distance')

        # --- Parameters ---
        self.declare_parameter('costmap_topic', '/local_costmap/costmap')  # Changed default, can use /local_costmap/costmap_raw
        self.declare_parameter('footprint_topic', '/local_costmap/published_footprint')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('lethal_threshold', 100)  # For OccupancyGrid: 0-100 scale
        self.declare_parameter('subtract_inscribed', True)
        self.declare_parameter('robot_radius', 0.3)
        self.declare_parameter('tf_timeout_sec', 0.25)
        self.declare_parameter('use_occupancy_grid', True)  # New parameter
        self.declare_parameter('footprint_epsilon', 0.03)  # m, small safety margin
        self.footprint_epsilon = float(self.get_parameter('footprint_epsilon').get_parameter_value().double_value)

        costmap_topic = self.get_parameter('costmap_topic').get_parameter_value().string_value
        footprint_topic = self.get_parameter('footprint_topic').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.lethal_threshold = int(self.get_parameter('lethal_threshold').get_parameter_value().integer_value)
        self.subtract_inscribed = self.get_parameter('subtract_inscribed').get_parameter_value().bool_value
        self.fallback_robot_radius = float(self.get_parameter('robot_radius').get_parameter_value().double_value)
        self.tf_timeout = Duration(seconds=float(self.get_parameter('tf_timeout_sec').get_parameter_value().double_value))
        self.use_occupancy_grid = self.get_parameter('use_occupancy_grid').get_parameter_value().bool_value

        # --- TF buffer/listener ---
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)

        # --- QoS ---
        costmap_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )

        footprint_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST
        )

        # --- Subs ---
        if self.use_occupancy_grid:
            self.costmap_sub = self.create_subscription(
                OccupancyGrid, costmap_topic, self._on_occupancy_grid, costmap_qos)
            self.get_logger().info(f'Subscribed to OccupancyGrid on: {costmap_topic}')
        else:
            self.costmap_sub = self.create_subscription(
                Costmap, costmap_topic, self._on_costmap, costmap_qos)
            self.get_logger().info(f'Subscribed to Costmap on: {costmap_topic}')

        self.footprint_sub = self.create_subscription(
            PolygonStamped, footprint_topic, self._on_footprint, footprint_qos)

        # --- Pub ---
        self.distance_pub = self.create_publisher(Float32, '/local_costmap/nearest_obstacle_distance', 10)

        # Footprint storage
        self._footprint_msg: Optional[PolygonStamped] = None
        self._footprint_mutex = threading.Lock()

        # Debug counters
        self.costmap_count = 0
        self.tf_fail_count = 0
        
        self.get_logger().info('NearestObstacleDistance initialized, waiting for costmap data...')

    def _on_footprint(self, msg: PolygonStamped):
        with self._footprint_mutex:
            self._footprint_msg = msg
        self.get_logger().debug(f'Received footprint with {len(msg.polygon.points)} points')

    # ... (keep your existing footprint calculation methods) ...

    @staticmethod
    def _point_segment_distance(px, py, ax, ay, bx, by):
        # Distance from point P(px,py) to segment AB(ax,ay)-(bx,by)
        apx, apy = px - ax, py - ay
        abx, aby = bx - ax, by - ay
        ab2 = abx * abx + aby * aby
        if ab2 == 0.0:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
        cx, cy = ax + t * abx, ay + t * aby
        return math.hypot(px - cx, py - cy)

    def _compute_inscribed_radius_at_point(self, poly_xy, px, py) -> float:
        # Minimum distance from (px,py) to polygon edges
        n = len(poly_xy)
        if n < 2:
            return float('inf')
        min_d = float('inf')
        for i in range(n):
            ax, ay = poly_xy[i]
            bx, by = poly_xy[(i + 1) % n]
            d = self._point_segment_distance(px, py, ax, ay, bx, by)
            if d < min_d:
                min_d = d
        return min_d

    def _get_inscribed_radius(self) -> float:
        """Compute inscribed radius at the robot base pose using latest footprint.
        Returns fallback radius if footprint/TF unavailable or subtract_inscribed is False."""
        if not self.subtract_inscribed:
            return 0.0

        with self._footprint_mutex:
            fp = self._footprint_msg

        if fp is None:
            return self.fallback_robot_radius if self.fallback_robot_radius > 0.0 else 0.0

        try:
            # Get base pose in the footprint frame
            fp_frame = fp.header.frame_id
            tf = self.tf_buffer.lookup_transform(
                target_frame=fp_frame,
                source_frame=self.base_frame,
                time=Time(),  # latest
                timeout=self.tf_timeout
            )
            rx = tf.transform.translation.x
            ry = tf.transform.translation.y

            pts = [(p.x, p.y) for p in fp.polygon.points]
            if len(pts) < 3:
                return self.fallback_robot_radius if self.fallback_robot_radius > 0.0 else 0.0

            return self._compute_inscribed_radius_at_point(pts, rx, ry)

        except TransformException as ex:
            self.get_logger().warn(f'Footprint TF failed ({self.base_frame} -> {fp.header.frame_id}): {ex}')
            return self.fallback_robot_radius if self.fallback_robot_radius > 0.0 else 0.0

    def _on_occupancy_grid(self, msg: OccupancyGrid):
        """Handle nav_msgs/msg/OccupancyGrid costmap"""
        self.costmap_count += 1
        self.get_logger().debug(f"OccupancyGrid #{self.costmap_count}: frame={msg.header.frame_id}, size={len(msg.data)}")
        
        # Check for empty costmap
        if len(msg.data) == 0:
            self.get_logger().warn("Empty OccupancyGrid received")
            return

        # TF lookup
        costmap_frame = msg.header.frame_id
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame=costmap_frame,
                source_frame=self.base_frame,
                time=Time(),  # latest
                timeout=self.tf_timeout
            )
            self.get_logger().debug(f"TF OK: {self.base_frame} -> {costmap_frame}")
        except TransformException as ex:
            self.tf_fail_count += 1
            self.get_logger().warn(f'TF lookup failed {self.base_frame} -> {costmap_frame}: {ex}')
            return

        rx = tf.transform.translation.x
        ry = tf.transform.translation.y

        # OccupancyGrid metadata
        res = msg.info.resolution
        size_x = msg.info.width
        size_y = msg.info.height
        origin = msg.info.origin
        ox = origin.position.x
        oy = origin.position.y

        # Validate data length
        data = np.asarray(msg.data, dtype=np.int8)
        if data.size != size_x * size_y:
            self.get_logger().warn(f'Unexpected OccupancyGrid data size: {data.size} (expected {size_x * size_y})')
            return
            
        # Reshape and process
        grid = data.reshape((size_y, size_x))

        # Robot indices in grid coordinates (float)
        cx = (rx - ox) / res
        cy = (ry - oy) / res

        if cx < -0.5 or cy < -0.5 or cx > size_x - 0.5 or cy > size_y - 0.5:
            self.get_logger().warn('Robot outside local costmap bounds; cannot compute distance.')
            return

        # Lethal cells (OccupancyGrid uses 0-100, where 100 is occupied)
        lethal_mask = grid >= self.lethal_threshold
        if not lethal_mask.any():
            # No lethal cells inside this costmap window
            self.distance_pub.publish(Float32(data=float('inf')))
            self.get_logger().debug('No lethal cells present; distance = inf')
            return

        # Robot indices in grid coordinates (float) – already computed above
        # cx, cy

        # --- Compute distance to all lethal cells ---
        ys, xs = np.where(lethal_mask)
        dx_pix = xs.astype(np.float32) + 0.5 - cx
        dy_pix = ys.astype(np.float32) + 0.5 - cy
        d_pix = np.hypot(dx_pix, dy_pix)

        # --- Remove lethal cells that are inside / touching robot footprint ---
        inscribed = self._get_inscribed_radius()

        if self.subtract_inscribed and inscribed > 0.0 and math.isfinite(inscribed):
            robot_pix_radius = (inscribed + self.footprint_epsilon) / res

            # Keep only cells strictly outside the robot
            outside_mask = d_pix > robot_pix_radius

            if outside_mask.any():
                d_pix = d_pix[outside_mask]
            else:
                # All lethal cells are inside / touching the footprint
                # => treat as collision (clearance 0), but log it
                self.get_logger().warn(
                    'All lethal cells lie inside/touching the robot footprint. '
                    'Publishing clearance = 0.0 m (collision or extremely close).'
                )
                self.distance_pub.publish(Float32(data=0.0))
                return

        # # --- Now compute minimum clearance to obstacle *boundary* ---
        # nearest_pix_dist = float(d_pix.min())
        # d_center = nearest_pix_dist * res  # distance: robot center -> cell center

        # # Approximate cell "radius": center to furthest corner (conservative)
        # cell_radius = res * math.sqrt(2.0) * 0.5

        # if self.subtract_inscribed and inscribed > 0.0 and math.isfinite(inscribed):
        #     # Clearance between robot footprint and obstacle cell boundary
        #     clearance = d_center - inscribed - cell_radius
        # else:
        #     # Distance from robot center to obstacle cell boundary
        #     clearance = d_center - cell_radius

        # # Clamp: treat overlap as 0.0 clearance (you could keep negative if you want penetration depth)
        # clearance = max(0.0, clearance)

        # out = Float32()
        # out.data = float(clearance)
        # self.distance_pub.publish(out)
        
        # --- Now compute minimum distance from remaining cells ---
        nearest_pix_dist = float(d_pix.min())
        dist_m = nearest_pix_dist * res

        # Subtract inscribed radius if requested
        if self.subtract_inscribed and inscribed > 0.0 and math.isfinite(inscribed):
            clearance = max(0.0, dist_m - inscribed)
        else:
            clearance = dist_m

        out = Float32()
        out.data = clearance
        self.distance_pub.publish(out)

        # self.get_logger().info(f'Published distance: {out.data:.3f} m')

    def _on_costmap(self, msg: Costmap):
        """Handle nav2_msgs/msg/Costmap (your original method)"""
        self.costmap_count += 1
        self.get_logger().debug(f"Costmap #{self.costmap_count}: frame={msg.header.frame_id}, size={len(msg.data)}")
        
        if len(msg.data) == 0:
            self.get_logger().warn("Empty costmap received")
            return

        costmap_frame = msg.header.frame_id
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame=costmap_frame,
                source_frame=self.base_frame,
                time=Time(),
                timeout=self.tf_timeout
            )
            self.get_logger().debug(f"TF OK: {self.base_frame} -> {costmap_frame}")
        except TransformException as ex:
            self.tf_fail_count += 1
            self.get_logger().warn(f'TF lookup failed {self.base_frame} -> {costmap_frame}: {ex}')
            return

        rx = tf.transform.translation.x
        ry = tf.transform.translation.y

        meta = msg.metadata
        res = float(meta.resolution)
        size_x = int(meta.size_x)
        size_y = int(meta.size_y)
        origin = meta.origin
        ox = origin.position.x
        oy = origin.position.y

        data = np.asarray(msg.data, dtype=np.uint8)
        if data.size != size_x * size_y:
            self.get_logger().warn(f'Unexpected costmap data size: {data.size} (expected {size_x * size_y})')
            return
            
        grid = data.reshape((size_y, size_x))

        cx = (rx - ox) / res
        cy = (ry - oy) / res

        if cx < -0.5 or cy < -0.5 or cx > size_x - 0.5 or cy > size_y - 0.5:
            self.get_logger().warn('Robot outside local costmap bounds; cannot compute distance.')
            return

        # Lethal cells (OccupancyGrid uses 0-100, where 100 is occupied)
        lethal_mask = grid >= self.lethal_threshold
        if not lethal_mask.any():
            # No lethal cells inside this costmap window
            self.distance_pub.publish(Float32(data=float('inf')))
            self.get_logger().debug('No lethal cells present; distance = inf')
            return

        # Robot indices in grid coordinates (float) – already computed above
        # cx, cy

        # --- Compute distance to all lethal cells ---
        ys, xs = np.where(lethal_mask)
        dx_pix = xs.astype(np.float32) + 0.5 - cx
        dy_pix = ys.astype(np.float32) + 0.5 - cy
        d_pix = np.hypot(dx_pix, dy_pix)

        # --- Remove lethal cells that are inside / touching robot footprint ---
        inscribed = self._get_inscribed_radius()

        if self.subtract_inscribed and inscribed > 0.0 and math.isfinite(inscribed):
            robot_pix_radius = (inscribed + self.footprint_epsilon) / res

            # Keep only cells strictly outside the robot
            outside_mask = d_pix > robot_pix_radius

            if outside_mask.any():
                d_pix = d_pix[outside_mask]
            else:
                # All lethal cells are inside / touching the footprint
                # => treat as collision (clearance 0), but log it
                self.get_logger().warn(
                    'All lethal cells lie inside/touching the robot footprint. '
                    'Publishing clearance = 0.0 m (collision or extremely close).'
                )
                self.distance_pub.publish(Float32(data=0.0))
                return

        # --- Now compute minimum clearance to obstacle *boundary* ---
        nearest_pix_dist = float(d_pix.min())
        d_center = nearest_pix_dist * res  # distance: robot center -> cell center

        cell_radius = res * math.sqrt(2.0) * 0.5

        if self.subtract_inscribed and inscribed > 0.0 and math.isfinite(inscribed):
            clearance = d_center - inscribed - cell_radius
        else:
            clearance = d_center - cell_radius

        clearance = max(0.0, clearance)

        out = Float32()
        out.data = float(clearance)
        self.distance_pub.publish(out)
        # self.get_logger().info(f'Published distance: {out.data:.3f} m')

def main():
    rclpy.init()
    node = NearestObstacleDistance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
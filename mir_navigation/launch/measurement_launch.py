from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # # Start topic /local_costmap/nearest_obstacle_distance to get obstacle distance in local costmap
        # Node(
        #     package='mir_navigation',
        #     executable='nearest_obstacle_distance.py',
        #     name='nearest_obstacle_distance',
        #     output='screen',
        #     parameters=[
        #         {'costmap_topic': '/local_costmap/costmap'},
        #         # {'footprint_topic': '/local_costmap/published_footprint'},
        #         {'base_frame': 'base_footprint'},
        #         # {'subtract_inscribed': True},
        #         # {'robot_radius': 0.55},
        #         {'use_occupancy_grid': True},
        #         {'lethal_threshold': 80},
        #     ],
        # ),

        # # This points follower measure obstacle distance along the path
        # Node(
        #     package='mir_navigation',
        #     executable='follow_waypoints_measure_obstacle.py',
        #     name='follow_waypoints',
        #     output='screen',
        # )

        # # This points follower measure velocity by /odom along the path
        # Node(
        #     package='mir_navigation',
        #     executable='follow_waypoints_odom_vel_logger.py',
        #     name='follow_waypoints',
        #     output='screen',
        # )

        # # This points follower measure velocity and nearest distance along the path
        # Node(
        #     package='mir_navigation',
        #     executable='follow_waypoints_planner_logger.py',
        #     name='follow_waypoints',
        #     output='screen',
        # )

        # # This points follower measure nearest distance and average command execution time along the path
        # Node(
        #     package='mir_navigation',
        #     executable='follow_waypoints_controller_logger.py',
        #     name='follow_waypoints',
        #     output='screen',
        # )

        # # This points follower measure nearest distance and average command execution time along the path
        # Node(
        #     package='mir_navigation',
        #     executable='follow_waypoints_controller_human_logger.py',
        #     name='follow_waypoints',
        #     output='screen',
        # )

        # This points follower measure path error along the path
        Node(
            package='mir_navigation',
            executable='follow_waypoints_path_error_logger.py',
            name='follow_waypoints',
            output='screen',
        )
    ])

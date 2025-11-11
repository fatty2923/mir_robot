from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mir_navigation',
            executable='nearest_obstacle_distance.py',
            name='nearest_obstacle_distance',
            output='screen',
            parameters=[
                {'costmap_topic': '/local_costmap/costmap'},
                # {'footprint_topic': '/local_costmap/published_footprint'},
                {'base_frame': 'base_footprint'},
                # {'subtract_inscribed': True},
                # {'robot_radius': 0.25},
                {'use_occupancy_grid': True},
                {'lethal_threshold': 80},
            ],
        )
    ])

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument(
            'namespace',
            default_value='',
            description='Namespace to push all topics into.'),

        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description=''),

        # DeclareLaunchArgument(
        #     'mir_hostname',
        #     default_value='192.168.12.20',
        #     description=''),

        Node(
            package='ira_laser_tools',
            name='mir_laser_scan_merger',
            executable='laserscan_multi_merger',
            parameters=[{'laserscan_topics': "b_scan f_scan",
                        #  'destination_frame': "base_link",
                         'destination_frame': "virtual_laser_link",
                         'scan_destination_topic': 'scan',
                         'cloud_destination_topic': 'scan_cloud',
                        #  'max_angle': 2.35619449615,
                        #  'min_angle': -2.35619449615,
                         'min_height': -0.25,
                         'max_merge_time_diff': 0.005,
                         # driver (msg converter) delay
                         'max_delay_scan_time': 2.5,
                         'max_completion_time': 0.1,
                         'alow_scan_delay': True,
                         'use_sim_time': use_sim_time,
                         'best_effort': False}],
            namespace=LaunchConfiguration('namespace'),
            output='screen')

    ])

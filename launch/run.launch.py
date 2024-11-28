import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node


def generate_launch_description():

    share_dir = get_package_share_directory('lio_sam')
    parameter_file = LaunchConfiguration('params_file')
    parameter_navsat_file = LaunchConfiguration('navsat_transform_file')
    xacro_path = os.path.join(share_dir, 'config', 'robot.urdf.xacro')
    #rviz_config_file = os.path.join(share_dir, 'config', 'rviz2.rviz')
    rviz_config_file = os.path.join(share_dir, 'config', 'base.rviz')
    #rviz_config_file = os.path.join(share_dir, 'config', 'video.rviz')

    params_declare = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(
            share_dir, 'config', 'params.yaml'),
        description='FPath to the ROS2 parameters file to use.')

    print("urdf_file_name : {}".format(xacro_path))

    return LaunchDescription([

        DeclareLaunchArgument(
            'navsat_transform_file',
            default_value=os.path.join(
               share_dir, 'config', 'navsat_transform.yaml'),
           description='Navsat transform configuration file'),
        
        params_declare,
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments='0.0 0.0 0.0 0.0 0.0 0.0 map odom'.split(' '),
            parameters=[parameter_file],
            output='screen'
            ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': Command(['xacro', ' ', xacro_path])
            }]
        ),
        Node(
            package='lio_sam',
            executable='lio_sam_imuPreintegration',
            name='lio_sam_imuPreintegration',
            parameters=[parameter_file],
            output='screen'
        ),
        Node(
            package='lio_sam',
            executable='lio_sam_imageProjection',
            name='lio_sam_imageProjection',
            parameters=[parameter_file],
            output='screen'
        ),
        Node(
            package='lio_sam',
            executable='lio_sam_featureExtraction',
            name='lio_sam_featureExtraction',
            parameters=[parameter_file],
            output='screen'
        ),
        Node(
            package='lio_sam',
            executable='lio_sam_mapOptimization',
            name='lio_sam_mapOptimization',
            parameters=[parameter_file],
            output='screen'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            output='screen'
        ),
        # Aggiunta del nodo EKF GPS
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_gps',
            respawn=True,
            output='screen',
            parameters=[parameter_navsat_file],
            remappings=[
                ('odometry/filtered', '/odometry/navsat'),
            ],
            #arguments=['--ros-args', '--log-level', 'debug']
        ),
        Node(
            package='robot_localization',
            executable='navsat_transform_node',
            name='navsat_transform_node',
            output='screen',
            #parameters=[{'frequency': 50.0, 'delay': 0.0,'magnetic_declination_radians': 0.0,
            #             'yaw_offset': 0.0, 'zero_altitude': False, 'publish_filtered_gps': False,
            #             'broadcast_utm_transform': False, 'use_odometry_yaw': False, 'wait_for_datum': False,
            #             'broadcast_utm_transform_as_parent_frame': False, 'transform_timeout': 0.0}],
            parameters=[parameter_navsat_file],
            remappings=[
                ('imu', '/oxts/imu'),  # Input IMU
                ('gps/fix', '/oxts/nav_sat_fix'),  # Input GPS
                ('odometry/filtered', '/odometry/navsat'),  # Input Odometry
                ('gps/filtered', 'gps/filtered'),  # GPS filtrato (output)
                ('odometry/gps', 'odometry/gps'),  # Odometry GPS (output)
            ]
        ),
    ])
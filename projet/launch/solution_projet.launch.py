import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():

    pkg_projet2025 = get_package_share_directory('projet2025')

    # IncludeLaunchDescription pour appeler projet.launch.py
    gazebo_and_robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_projet2025, 'launch', 'projet.launch.py')
        )
    )

    line_detector_node = Node(
        package='projet', 
        executable='line_detector',
        name='line_detector',
        output='screen'
    )

    superviseur_node = Node(
        package='projet',
        executable='superviseur',
        name='superviseur',
        output='screen'
    )

    ld = LaunchDescription()
    
    ld.add_action(gazebo_and_robot_launch)
    
    
    ld.add_action(line_detector_node)
    ld.add_action(superviseur_node)
    

    return ld
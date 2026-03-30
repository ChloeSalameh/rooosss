import os

import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():

    # Tue tous les processus fantomes du nouveau Gazebo (Ignition) 
    # et des ponts ROS (bridges) avant de relancer l'environnement.
    try:
        subprocess.run(['pkill', '-9', '-f', 'ign gazebo'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(['pkill', '-9', '-f', 'gz sim'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(['pkill', '-9', '-f', 'parameter_bridge'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(['pkill', '-9', '-f', 'ruby'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[INFO] Nettoyage simulation et demarrage de Gazebo")
    except Exception:
        pass

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

    line_follower_node = Node(
        package='projet',
        executable='line_follower',
        name='line_follower',
        output='screen'
    )

    challenge1_node = Node(
        package='projet',
        executable='challenge1', 
        name='challenge1',
        output='screen'
    )

    ld = LaunchDescription()
    
    ld.add_action(gazebo_and_robot_launch)    
    ld.add_action(line_detector_node)
    ld.add_action(superviseur_node)

    ld.add_action(line_follower_node)
    ld.add_action(challenge1_node)
    

    return ld
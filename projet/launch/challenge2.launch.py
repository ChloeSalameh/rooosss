import os
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():

    # Nettoyage de l'environnement
    try:
        subprocess.run(['pkill', '-9', '-f', 'ign gazebo'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(['pkill', '-9', '-f', 'gz sim'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(['pkill', '-9', '-f', 'parameter_bridge'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(['pkill', '-9', '-f', 'ruby'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[INFO] Nettoyage simulation et demarrage de Gazebo pour Challenge 2")
    except Exception:
        pass

    pkg_projet2025 = get_package_share_directory('projet2025')

    gazebo_and_robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_projet2025, 'launch', 'projet.launch.py')
        ),
        launch_arguments={
            'x_pose': '-0.42',
            'y_pose': '0.64',
            'yaw_angle': '2.4'
        }.items()
    )

    line_detector_node = Node(
        package='projet', 
        executable='line_detector',
        name='line_detector',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # Superviseur paramétré pour démarrer à l'état 2
    superviseur_node = Node(
        package='projet',
        executable='superviseur',
        name='superviseur',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    line_follower_node = Node(
        package='projet',
        executable='line_follower',
        name='line_follower',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # Nœud dédié au challenge 2 (à créer par la suite)
    challenge2_node = Node(
        package='projet',
        executable='challenge2', 
        name='challenge2',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    ld = LaunchDescription()
    
    ld.add_action(gazebo_and_robot_launch)    
    ld.add_action(line_detector_node)
    ld.add_action(superviseur_node)
    ld.add_action(line_follower_node)
    ld.add_action(challenge2_node)

    return ld
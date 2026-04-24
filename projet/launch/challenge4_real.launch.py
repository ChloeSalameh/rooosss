from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    line_detector_node = Node(
        package='projet',
        executable='line_detector',
        name='line_detector',
        output='screen',
        parameters=[{'use_sim_time': False}]
    )

    line_follower_node = Node(
        package='projet',
        executable='line_follower',
        name='line_follower',
        output='screen',
        parameters=[{'use_sim_time': False}]
    )

    superviseur_node = Node(
        package='projet',
        executable='superviseur',
        name='superviseur',
        output='screen',
        parameters=[{'use_sim_time': False, 'initial_state': 4}]
    )

    challenge4_node = Node(
        package='projet',
        executable='challenge4',
        name='challenge4',
        output='screen',
        parameters=[{'use_sim_time': False}]
    )

    ld = LaunchDescription()
    ld.add_action(line_detector_node)
    ld.add_action(line_follower_node)
    ld.add_action(superviseur_node)
    #ld.add_action(challenge4_node)

    return ld
import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool
from geometry_msgs.msg import Twist

class Superviseur(Node):
    def __init__(self):
        super().__init__('superviseur')
        
        # 1 = Suivi de ligne, 2 = Obstacles, 3 = Couloir, 4 = Foot, 5 = Controle humain
        self.current_state = 1
        
        # Variables pour sauvegarder les "suggestions" de chaque challenge
        self.last_twist_ch1 = Twist()
        self.last_twist_ch2 = Twist()
        self.last_twist_ch3 = Twist()
        self.last_twist_ch4 = Twist()
        self.last_twist_ch5 = Twist()

        # deection de la ligne bleue
        self.sub_blue_line = self.create_subscription(
            Bool,
            '/blue_line_crossed',
            self.state_callback,
            10 # Taille de la file d'attente (QoS)
        )

        # On s'abonne aux topics intermediaires de chaque challenge
        self.sub_cmd_ch1 = self.create_subscription(Twist, '/cmd_vel_challenge_1', self.cmd_ch1_callback, 10)
        self.sub_cmd_ch2 = self.create_subscription(Twist, '/cmd_vel_challenge_2', self.cmd_ch2_callback, 10)
        self.sub_cmd_ch3 = self.create_subscription(Twist, '/cmd_vel_challenge_3', self.cmd_ch3_callback, 10)
        self.sub_cmd_ch4 = self.create_subscription(Twist, '/cmd_vel_challenge_4', self.cmd_ch4_callback, 10)
        self.sub_cmd_ch5 = self.create_subscription(Twist, '/cmd_vel_challenge_5', self.cmd_ch5_callback, 10)

        # vrai topic pour le Turtlebot dans Gazebo et irl
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        timer_period = 0.1 
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info('Challenge : 1')
    

    def state_callback(self, msg):
        """se declanche quand on publie sur /blue_line_crossed"""
        if msg.data == True:
            self.current_state += 1
            self.get_logger().info(f"Passage au challenge {self.current_state}")

    def cmd_ch1_callback(self, msg):
        """Sauvegarde la commande du Challenge 1"""
        self.last_twist_ch1 = msg

    def cmd_ch2_callback(self, msg):
        """Sauvegarde la commande du Challenge 2"""
        self.last_twist_ch2 = msg

    def cmd_ch3_callback(self, msg):
        """Sauvegarde la commande du Challenge 3"""
        self.last_twist_ch3 = msg

    def cmd_ch4_callback(self, msg):
        """Sauvegarde la commande du Challenge 4"""
        self.last_twist_ch4 = msg

    def cmd_ch5_callback(self, msg):
        """Sauvegarde la commande du Challenge 5"""
        self.last_twist_ch5 = msg
    

    def timer_callback(self):
        """ boucle a 10Hz decide de quoi publier."""
        twist_to_publish = Twist()
        
        if self.current_state == 1:
            twist_to_publish = self.last_twist_ch1
        elif self.current_state == 2:
            twist_to_publish = self.last_twist_ch2
        elif self.current_state == 3:
            twist_to_publish = self.last_twist_ch3
        elif self.current_state == 4:
            twist_to_publish = self.last_twist_ch4
        elif self.current_state >= 5:
            twist_to_publish = self.last_twist_ch5
            
        # Publication sur le vrai topic du robot
        self.publisher_.publish(twist_to_publish)

def main(args=None):
    rclpy.init(args=args)
    node = Superviseur()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
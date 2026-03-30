import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32
import numpy as np

class Challenge1Strategy(Node):

    def __init__(self):

        super().__init__('challenge1')

        self.declare_parameter('roundabout', 'right') # pour changer le sens sans relancer le code quand on a un roundabout
        
        # pour ecouter le lidar
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        # pour ecouter la trajectoire
        self.sub_cmd_raw = self.create_subscription(Twist, '/cmd_vel_line_raw', self.cmd_callback, 10)
        
        # 3. Optionnel : On écoute la position de la ligne pour savoir si on est dans le rond-point
        self.sub_red_pos = self.create_subscription(Int32, '/red_line_pos', self.red_pos_callback, 10)

        self.publisher_ = self.create_publisher(Twist, '/cmd_vel_challenge_1', 10)

        self.emergency_stop = False
        self.current_red_x = -1
    
    def scan_callback(self, msg):

        """ Logique de securite Lidar : Cone frontal de 30 degres """
        # Sur le Turtlebot3 : 0degres est devant, les angles vont de 0 a 359
        # On regarde de 345deg a 15deg (le cone devant)
        angles_devant = list(range(0, 15)) + list(range(345, 360))
        
        distances = []
        for angle in angles_devant:
            dist = msg.ranges[angle]
            if dist > 0.01:
                distances.append(dist)
        
        if distances and min(distances) < 0.25: # Seuil de 25cm
            if not self.emergency_stop:
                self.get_logger().info("Obstacle detecte")
            self.emergency_stop = True
        else:
            self.emergency_stop = False

    def red_pos_callback(self, msg):
        self.current_red_x = msg.data

    def cmd_callback(self, msg_a):
        """ Decision de trajectoire """
        final_twist = Twist()

        # Si obstacle, on force l'arret
        if self.emergency_stop:
            final_twist.linear.x = 0.0
            final_twist.angular.z = 0.0
        
        # Si la voie est libre, on applique le biais du rond-point
        else:
            # On recupere le parametre actuel (droite ou gauche)
            direction = self.get_parameter('roundabout').get_parameter_value().string_value
            
            final_twist.linear.x = msg_a.linear.x
            
            # BIAIS DU ROND-POINT
            # Si on ne voit plus la ligne (-1) ou si on veut forcer le passage
            # on ajoute un leger decalage (offset) a la rotation
            if direction == "right":
                final_twist.angular.z = msg_a.angular.z - 0.15 # On "pousse" vers la droite
            else:
                final_twist.angular.z = msg_a.angular.z + 0.15 # On "pousse" vers la gauche
            
            # Si le robot perd la ligne rouge (msg_a.linear.x sera 0.0 selon le code de A)
            # On peut forcer une petite vitesse pour chercher
            if msg_a.linear.x == 0.0:
                final_twist.linear.x = 0.05 


        self.publisher_.publish(final_twist)


def main(args=None):
    rclpy.init(args=args)
    node = Challenge1Strategy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
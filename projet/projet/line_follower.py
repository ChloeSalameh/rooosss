import rclpy
from rclpy.node import Node

from std_msgs.msg import Int32
from geometry_msgs.msg import Twist

class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower_node')

        # 1. Initialisation des variables d'etat (-1 signifie "non detecte")
        self.cx_red = -1
        self.cx_green = -1

        # 2. Parametres de l'asservissement (a ajuster lors des tests en simulation)
        self.image_center = 320.0       # Centre ideal si l'image fait 640px de large
        self.offset_pixels = 150.0      # Decalage estime quand on ne voit qu'une seule ligne
        self.kp = 0.005                 # Gain Proportionnel (Volant)
        self.kd = 0.01                  # Gain Derive (Amortisseur)
        self.last_error = 0.0           # memoire
        self.base_speed = 0.1

        # 3. Abonnements (Subscribers) aux positions des lignes
        self.sub_red = self.create_subscription(
            Int32,
            '/red_line_pos',
            self.red_callback,
            10
        )
        self.sub_green = self.create_subscription(
            Int32,
            '/green_line_pos',
            self.green_callback,
            10
        )

        # 4. Publication (Publisher) de la commande brute
        self.pub_cmd = self.create_publisher(
            Twist, 
            '/cmd_vel_line_raw', 
            10
        )

        self.get_logger().info("Noeud Line Follower (P-Controller) demarre.")

    def red_callback(self, msg):
        """se declanche a chaque fois que la position de la ligne rouge est recue."""
        self.cx_red = msg.data
        self.compute_and_publish_command()

    def green_callback(self, msg):
        """se declanche a chaque fois que la position de la ligne verte est recue."""
        self.cx_green = msg.data
        self.compute_and_publish_command()

    def compute_and_publish_command(self):
        """Logique centrale : calcule le centre de la route et genere la commande."""
        twist = Twist()

        # CAS CRITIQUE : Aucune ligne n'est vue
        if self.cx_red == -1 and self.cx_green == -1:
            twist.linear.x = 0.0
            twist.angular.z = 0.2  # Rotation lente pour chercher la ligne
            self.pub_cmd.publish(twist)
            return  # On arrete la fonction ici

        # Calcul du centre de la route selon les lignes visibles
        if self.cx_red != -1 and self.cx_green != -1:
            # Les deux lignes sont vues
            centre_route = (self.cx_red + self.cx_green) / 2.0
        
        elif self.cx_red != -1:
            # Si on ne voit que la ROUGE (qui est a droite), le centre est a sa gauche (-)
            centre_route = self.cx_red - self.offset_pixels 
            
        else:
            # Si on ne voit que la VERTE (qui est a gauche), le centre est a sa droite (+)
            centre_route = self.cx_green + self.offset_pixels

        # Calcul de l'erreur (difference entre le centre de l'image et le centre de la route)
        # Si le centre de la route est a droite (ex: 400), l'erreur sera negative (320 - 400 = -80)
        # Un angular.z negatif fait tourner le robot vers la droite, c'est ce qu'on veut !
        # Calcul de l'erreur (difference entre le centre de l'image et le centre de la route)
        erreur = self.image_center - centre_route

        # La derivee est simplement la variation de l'erreur depuis la derniere fois
        derivation = erreur - self.last_error
        
        # Creation de la commande avec le correcteur PD
        twist.linear.x = self.base_speed
        twist.angular.z = float((self.kp * erreur) + (self.kd * derivation))

        # Sauvegarde de l'erreur pour le prochain calcul
        self.last_error = erreur

        # Publication
        self.pub_cmd.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
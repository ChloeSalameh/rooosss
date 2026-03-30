import rclpy
from rclpy.node import Node
#from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import numpy as np
import cv2

class LineDetector(Node):
    def __init__(self):
        super().__init__('line_detector')
        
        # securite anti spam
        self.blue_detected_previously = False

        # recoit l'image brute
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.listener_callback,
            10
        )
        self.subscription  # to prevent unused variable warning

        #self.publisher = self.create_publisher(
         #   Twist, 
          #  '/cmd_vel_line_raw', # noeud utilise un publisher pour envoyer des messages sur ce topic
           # 10)

        self.publisher_blue = self.create_publisher(Bool, '/blue_line_crossed', 10)
        
        # position de la ligne rouge
        self.publisher_red_pos = self.create_publisher(Int32, '/red_line_pos', 10)

        # position de la ligne vert
        self.publisher_green_pos = self.create_publisher(Int32, '/green_line_pos', 10)


        self.bridge = CvBridge()
        self.get_logger().info("Detection de lignes")

    def listener_callback(self, msg):

        # conversion Image ROS a OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Recuperation des dimensions de l'image
        h, w, d = cv_image.shape

        # Pretraitement
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV) # HSV PAS RGB

        # H : hue/ teinte
        # S : saturation
        # V : value / valeur

        # On calcule la ligne de coupe (ex: on garde seulement le 1/3 inferieur)
        # Si ca tourne encore trop tot, essayez int(3*h/4) pour ne garder que le quart inferieur
        horizon_asservissement = int(2 * h / 3)

        # MASQUE ROUGE
        rouge_bas1, rouge_haut1 = np.array([0, 120, 70]), np.array([10, 255, 255])
        rouge_bas2, rouge_haut2 = np.array([170, 120, 70]), np.array([180, 255, 255])
        mask_red = cv2.bitwise_or(cv2.inRange(hsv, rouge_bas1, rouge_haut1), 
                                  cv2.inRange(hsv, rouge_bas2, rouge_haut2))
        
        mask_red[0:horizon_asservissement, 0:w] = 0 # On masque le haut

        # MASQUE VERT
        vert_bas = np.array([40, 100, 50])
        vert_haut = np.array([80, 255, 255])
        mask_green = cv2.inRange(hsv, vert_bas, vert_haut)
        
        mask_green[0:horizon_asservissement, 0:w] = 0 # On masque le haut

        # Masque Bleu : filtre qui utilise que le bas de l'image pour 
        bleu_bas, bleu_haut = np.array([100, 150, 50]), np.array([140, 255, 255])
        mask_blue = cv2.inRange(hsv, bleu_bas, bleu_haut)

        # On met a 0 tout le haut de l'image pour ne garder que le sol.
        # on coupe les 3/4 de l'image haute
        mask_blue[0:int(3*h/4), 0:w] = 0

        # Logique anti spam
        # On utilise countNonZero qui compte le VRAI nombre de pixels
        pixels_bleus = cv2.countNonZero(mask_blue)
        is_blue_now = bool(pixels_bleus > 2000) # Seuil de 2000 pixels (a ajuster si besoin)

        if is_blue_now and not self.blue_detected_previously:
            # On ne publie que si on vient de decouvrir la ligne (Front Montant)
            blue_msg = Bool()
            blue_msg.data = True
            self.publisher_blue.publish(blue_msg)
            self.get_logger().info("Ligne bleue detectee! Envoi du signal au Superviseur.")
        
        # Mise a jour de l'etat pour la prochaine frame
        self.blue_detected_previously = is_blue_now

        # renvoie la position de la ligne rouge
        red_msg = Int32()
        M = cv2.moments(mask_red)
        if M['m00'] > 0:
            red_msg.data = int(M['m10'] / M['m00']) # On envoie juste le centre X
        else:
            red_msg.data = -1 # -1 signifie "aucune ligne rouge detectee"
        self.publisher_red_pos.publish(red_msg)

        # renvoie la position de la ligne verte
        green_msg = Int32()
        M_green = cv2.moments(mask_green)
        if M_green['m00'] > 500:
            green_msg.data = int(M_green['m10'] / M_green['m00'])
        else:
            green_msg.data = -1
        self.publisher_green_pos.publish(green_msg)

        # Affichage (comme dans cv_plot.py)
        cv2.imshow("Masque Rouge", mask_red)
        cv2.imshow("Masque Bleu", mask_blue)
        cv2.imshow("Masque Vert", mask_green)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = LineDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
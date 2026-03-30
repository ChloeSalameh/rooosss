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
        super().__init__('line_detector_node')

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
        self.get_logger().info("Détection de lignes")

    def listener_callback(self, msg):

        # conversion Image ROS a OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Pretraitement
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV) # HSV PAS RGB

        # H : hue/ teinte
        # S : saturation
        # V : value / valeur

        # Masque rouge : filtre qui ne laisse passer que le rouge 
        # sur le cercle chromatique, c'est les deux extremites du cercle (0 et 180 pour H, 120 et 255 pour saturation, 70 et 255 pour V)
        rouge_bas1, rouge_haut1 = np.array([0, 120, 70]), np.array([10, 255, 255]) # on definit les deux extremitees avec leurs seuils
        rouge_bas2, rouge_haut2 = np.array([170, 120, 70]), np.array([180, 255, 255])
        mask_red = cv2.bitwise_or(cv2.inRange(hsv, rouge_bas1, rouge_haut1), 
                                  cv2.inRange(hsv, rouge_bas2, rouge_haut2))
        
        # Masque bleu : le bleu se situe environ entre 100 et 140
        bleu_bas, bleu_haut = np.array([100, 150, 50]), np.array([140, 255, 255])
        mask_blue = cv2.inRange(hsv, bleu_bas, bleu_haut)

        # Masque vert
        # Teinte (H) entre 40 et 80 pour le vert
        vert_bas = np.array([40, 100, 50])
        vert_haut = np.array([80, 255, 255])
        mask_green = cv2.inRange(hsv, vert_bas, vert_haut)

        # Convertir les données compressées en tableau numpy
        #np_arr = np.frombuffer(msg.data, np.uint8)
        # Décoder l'image
        #image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        #if image is not None:
         #   cv2.imshow("Compressed Image", image)
          #  cv2.waitKey(1)
        #else:
         #   self.get_logger().warn("Failed to decode compressed image")

        # Condition si le robot rencontre une ligne bleue + on le publie sur le topic
        blue_msg = Bool()

        # Si on détecte une masse bleue significative
        blue_msg.data = bool(np.sum(mask_blue) > 10000) 
        self.publisher_blue.publish(blue_msg)

        # renvoie la position de la ligne rouge
        red_msg = Int32()
        M = cv2.moments(mask_red)
        if M['m00'] > 0:
            red_msg.data = int(M['m10'] / M['m00']) # On envoie juste le centre X
        else:
            red_msg.data = -1 # -1 signifie "aucune ligne rouge détectée"
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
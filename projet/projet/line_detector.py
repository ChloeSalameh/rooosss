import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int32
from cv_bridge import CvBridge
import numpy as np
import cv2

class LineDetector(Node):
    def __init__(self):
        super().__init__('line_detector')
        self.blue_detected_previously = False
        self.bridge = CvBridge()

        # Offsets dynamiques auto-calibrés
        self.offset_near_dyn = -1.0
        self.offset_far_dyn  = -1.0

        # Abonnement unique à la caméra
        self.subscription = self.create_subscription(Image, '/image_raw', self.listener_callback, 10)

        # Publications pour la FSM et le Follower
        self.pub_blue      = self.create_publisher(Bool,  '/blue_line_crossed',  10)
        self.pub_red_pos   = self.create_publisher(Int32, '/red_line_pos',       10)
        self.pub_green_pos = self.create_publisher(Int32, '/green_line_pos',     10)
        self.pub_red_far   = self.create_publisher(Int32, '/red_line_pos_far',   10)
        self.pub_green_far = self.create_publisher(Int32, '/green_line_pos_far', 10)

        # Nouvelles publications de configuration dynamique
        self.pub_cam_width = self.create_publisher(Int32, '/camera_width', 10)
        self.pub_off_near  = self.create_publisher(Int32, '/offset_near',  10)
        self.pub_off_far   = self.create_publisher(Int32, '/offset_far',   10)

        self.get_logger().info("LineDetector démarré — Mode Auto-Calibration Actif")

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w, _ = frame.shape
        img_area = w * h

        # 1. On publie immédiatement la largeur réelle de la caméra
        self.pub_cam_width.publish(Int32(data=w))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Définition des couleurs
        r_lo1 = np.array([0,   100, 80]);  r_hi1 = np.array([10,  255, 255])
        r_lo2 = np.array([165, 100, 80]);  r_hi2 = np.array([180, 255, 255])
        g_lo  = np.array([38,   80, 50]);  g_hi  = np.array([85,  255, 255])
        b_lo  = np.array([100, 120, 60]);  b_hi  = np.array([140, 255, 255])

        mask_r_full = cv2.bitwise_or(cv2.inRange(hsv, r_lo1, r_hi1), cv2.inRange(hsv, r_lo2, r_hi2))
        mask_g_full = cv2.inRange(hsv, g_lo, g_hi)
        mask_blue   = cv2.inRange(hsv, b_lo, b_hi)

        # Découpage de l'image
        y_near = 6 * h // 10
        y_far  = 4 * h // 10
        y_blue = 3 * h // 4

        mask_r_near = mask_r_full.copy(); mask_r_near[0:y_near, :] = 0
        mask_g_near = mask_g_full.copy(); mask_g_near[0:y_near, :] = 0

        mask_r_far = mask_r_full.copy(); mask_r_far[0:y_far, :] = 0; mask_r_far[y_near:h, :] = 0
        mask_g_far = mask_g_full.copy(); mask_g_far[0:y_far, :] = 0; mask_g_far[y_near:h, :] = 0

        mask_blue[0:y_blue, :] = 0
        
        # Détection Ligne Bleue
        pixels_bleus = cv2.countNonZero(mask_blue)
        is_blue_now  = pixels_bleus > (img_area * 0.005)
        if is_blue_now and not self.blue_detected_previously:
            self.pub_blue.publish(Bool(data=True))
        self.blue_detected_previously = is_blue_now

        # Calcul des centres (moments)
        min_area_near = img_area * 0.0010
        min_area_far  = img_area * 0.0006

        cx_r_near = self._get_cx(mask_r_near, min_area_near)
        cx_g_near = self._get_cx(mask_g_near, min_area_near)
        cx_r_far  = self._get_cx(mask_r_far,  min_area_far)
        cx_g_far  = self._get_cx(mask_g_far,  min_area_far)

        # -------------------------------------------------------------
        # AUTO-CALIBRATION DES OFFSETS
        # Si on voit les DEUX lignes, on apprend la largeur de la piste
        # -------------------------------------------------------------
        if cx_r_near != -1 and cx_g_near != -1:
            mesure_actuelle = (cx_r_near - cx_g_near) / 2.0
            if self.offset_near_dyn == -1.0:
                self.offset_near_dyn = mesure_actuelle
            else:
                # Moyenne glissante : 90% d'historique, 10% de nouveauté
                self.offset_near_dyn = 0.90 * self.offset_near_dyn + 0.10 * mesure_actuelle
            self.pub_off_near.publish(Int32(data=int(self.offset_near_dyn)))

        if cx_r_far != -1 and cx_g_far != -1:
            mesure_actuelle = (cx_r_far - cx_g_far) / 2.0
            if self.offset_far_dyn == -1.0:
                self.offset_far_dyn = mesure_actuelle
            else:
                self.offset_far_dyn = 0.90 * self.offset_far_dyn + 0.10 * mesure_actuelle
            self.pub_off_far.publish(Int32(data=int(self.offset_far_dyn)))

        # Publication des positions
        self.pub_red_pos.publish(Int32(data=cx_r_near))
        self.pub_green_pos.publish(Int32(data=cx_g_near))
        self.pub_red_far.publish(Int32(data=cx_r_far))
        self.pub_green_far.publish(Int32(data=cx_g_far))

        cv2.imshow("Vision Nette", cv2.bitwise_or(mask_r_near, mask_g_near))
        cv2.waitKey(1)

    def _get_cx(self, mask, min_area):
        M = cv2.moments(mask)
        if M['m00'] > min_area:
            return int(M['m10'] / M['m00'])
        return -1

def main(args=None):
    rclpy.init(args=args)
    node = LineDetector()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__': main()
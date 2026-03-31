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
        self.subscription = self.create_subscription(Image, '/image_raw', self.listener_callback, 10)

        self.pub_blue      = self.create_publisher(Bool,  '/blue_line_crossed',  10)
        self.pub_red_pos   = self.create_publisher(Int32, '/red_line_pos',        10)
        self.pub_green_pos = self.create_publisher(Int32, '/green_line_pos',      10)
        self.pub_red_far   = self.create_publisher(Int32, '/red_line_pos_far',    10)
        self.pub_green_far = self.create_publisher(Int32, '/green_line_pos_far',  10)

        self.bridge = CvBridge()
        self.get_logger().info("LineDetector démarré")

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w, _ = frame.shape
        img_area = w * h  # Aire totale de l'image
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        r_lo1 = np.array([0,   100, 80]);  r_hi1 = np.array([10,  255, 255])
        r_lo2 = np.array([165, 100, 80]);  r_hi2 = np.array([180, 255, 255])
        g_lo  = np.array([38,   80, 50]);  g_hi  = np.array([85,  255, 255])
        b_lo  = np.array([100, 120, 60]);  b_hi  = np.array([140, 255, 255])

        mask_r_full = cv2.bitwise_or(cv2.inRange(hsv, r_lo1, r_hi1), cv2.inRange(hsv, r_lo2, r_hi2))
        mask_g_full = cv2.inRange(hsv, g_lo, g_hi)
        mask_blue   = cv2.inRange(hsv, b_lo, b_hi)

        y_near = h // 2  # On regarde des le milieu de l'image vers le bas
        y_far  = h // 4  # On regarde encore plus loin pour la trajectoire globale
        y_blue = 3 * h // 4

        mask_r_near = mask_r_full.copy(); mask_r_near[0:y_near, :] = 0
        mask_g_near = mask_g_full.copy(); mask_g_near[0:y_near, :] = 0

        mask_r_far = mask_r_full.copy()
        mask_r_far[0:y_far,  :] = 0
        mask_r_far[y_near:h, :] = 0
        mask_g_far = mask_g_full.copy()
        mask_g_far[0:y_far,  :] = 0
        mask_g_far[y_near:h, :] = 0

        mask_blue[0:y_blue, :] = 0
        
        # Seuils adaptatifs en fonction de la taille de l'image
        pixels_bleus = cv2.countNonZero(mask_blue)
        is_blue_now  = pixels_bleus > (img_area * 0.005) # eq. 1500px pour 640x480
        
        if is_blue_now and not self.blue_detected_previously:
            self.pub_blue.publish(Bool(data=True))
            self.get_logger().info("Ligne bleue → signal superviseur")
        self.blue_detected_previously = is_blue_now

        min_area_near = img_area * 0.0010  # egale a 300px pour 640x480
        min_area_far  = img_area * 0.0006  # egale a 180px pour 640x480

        self._pub_cx(mask_r_near, min_area_near, self.pub_red_pos)
        self._pub_cx(mask_g_near, min_area_near, self.pub_green_pos)
        self._pub_cx(mask_r_far,  min_area_far,  self.pub_red_far)
        self._pub_cx(mask_g_far,  min_area_far,  self.pub_green_far)

        cv2.imshow("Rouge near", mask_r_near)
        cv2.imshow("Rouge far",  mask_r_far)
        cv2.imshow("Vert  near", mask_g_near)
        cv2.imshow("Vert  far",  mask_g_far)
        cv2.waitKey(1)

    def _pub_cx(self, mask, min_area, publisher):
        M = cv2.moments(mask)
        msg = Int32()
        msg.data = int(M['m10'] / M['m00']) if M['m00'] > min_area else -1
        publisher.publish(msg)

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
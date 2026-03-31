import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from geometry_msgs.msg import Twist

class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower_node')

        self.cx_red_near   = -1
        self.cx_green_near = -1
        self.cx_red_far    = -1
        self.cx_green_far  = -1

        # Parametre ROS pour la largeur de la camera
        self.declare_parameter('camera_width', 640)
        cam_width = self.get_parameter('camera_width').value
        
        self.image_width  = float(cam_width)
        self.image_center = self.image_width / 2.0

        # Ratios de largeur
        self.offset_near   = 0.12 * self.image_width
        self.offset_far    = 0.13 * self.image_width
        self.SAFETY_MARGIN = 0.08 * self.image_width

        # Poids des lignes en fonction de si elles sont loin ou po
        self.w_near = 0.70
        self.w_far  = 0.30

        # --- PID ---
        self.kp           = 3.0  
        self.kd           = 12.0  

        self.base_speed   = 0.10    # m/s
        self.speed_min    = 0.05    # m/s

        self.alpha        = 0.40
        self.filtered_err = 0.0
        self.last_error   = 0.0

        # Abonnements
        self.create_subscription(Int32, '/red_line_pos',       self.cb_r_near, 10)
        self.create_subscription(Int32, '/green_line_pos',     self.cb_g_near, 10)
        self.create_subscription(Int32, '/red_line_pos_far',   self.cb_r_far,  10)
        self.create_subscription(Int32, '/green_line_pos_far', self.cb_g_far,  10)

        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel_line_raw', 10)
        self.create_timer(0.05, self.compute_and_publish)  # 20 Hz

        self.get_logger().info(f"Parcours : challenge1, camera : Width={cam_width}px")

    def cb_r_near(self, msg): self.cx_red_near   = msg.data
    def cb_g_near(self, msg): self.cx_green_near = msg.data
    def cb_r_far (self, msg): self.cx_red_far    = msg.data
    def cb_g_far (self, msg): self.cx_green_far  = msg.data

    def centre_voie(self, cx_red, cx_green, offset):
        m = self.SAFETY_MARGIN
        if cx_red != -1 and cx_green != -1:
            centre = (cx_red + cx_green) / 2.0
            lo = cx_green + m
            hi = cx_red   - m
            if lo < hi:
                centre = max(lo, min(hi, centre))
            return centre

        elif cx_red != -1:
            centre = cx_red - offset
            centre = min(centre, cx_red - m)
            centre = max(m, centre)
            return centre

        elif cx_green != -1:
            centre = cx_green + offset
            centre = max(centre, cx_green + m)
            centre = min(centre, self.image_width - m)
            return centre
        return None

    def compute_and_publish(self):
        twist = Twist()

        c_near = self.centre_voie(self.cx_red_near, self.cx_green_near, self.offset_near)
        c_far  = self.centre_voie(self.cx_red_far,  self.cx_green_far,  self.offset_far)

        # Si on perd toutes les lignes, on pivote doucement pour les chercher
        if c_near is None and c_far is None:
            twist.linear.x  = 0.0
            twist.angular.z = 0.25
            self.pub_cmd.publish(twist)
            return

        # Calcul du centre souhaite
        if c_near is not None and c_far is not None:
            centre = self.w_near * c_near + self.w_far * c_far
        elif c_near is not None:
            centre = c_near
        else:
            centre = c_far

        # PID Classique
        erreur_pixels = self.image_center - centre
        erreur_brute  = erreur_pixels / self.image_width 

        self.filtered_err = (self.alpha * self.filtered_err + (1.0 - self.alpha) * erreur_brute)
        erreur = self.filtered_err

        derivation    = erreur - self.last_error
        self.last_error = erreur

        omega = float(self.kp * erreur + self.kd * derivation)
        vitesse = self.base_speed # Vitesse nominale

        marge_strict = self.image_width * 0.30
        
        # Si la ligne verte depasse 'center - 30%', le robot est trop a gauche
        seuil_vert = self.image_center - marge_strict
        # Si la ligne rouge passe sous 'center + 30%', le robot est trop a droite
        seuil_rouge = self.image_center + marge_strict

        mord_verte = (self.cx_green_near != -1) and (self.cx_green_near > seuil_vert)
        mord_rouge = (self.cx_red_near != -1) and (self.cx_red_near < seuil_rouge)

        if mord_verte:
            vitesse = 0.0     # ARRET TOTAL de la progression
            omega = -2.0      # ROTATION BRUTALE vers la droite (oppose a la verte)
        elif mord_rouge:
            vitesse = 0.0     # ARRET TOTAL
            omega = 2.0       # ROTATION BRUTALE vers la gauche

        twist.linear.x = vitesse
        twist.angular.z = omega
        self.pub_cmd.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': main()
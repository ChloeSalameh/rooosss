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

        self.image_width = 640.0
        self.offset_near = 75.0
        self.offset_far  = 75.0

        self.w_near = 0.90
        self.w_far  = 0.10

        self.kp = 3.0
        self.kd = 12.0
        self.base_speed = 0.10
        self.speed_min  = 0.05

        self.alpha        = 0.40
        self.filtered_err = 0.0
        self.last_error   = 0.0

        # =====================================================================
        # CORRECTION 3 : Inertie de trajectoire
        # Quand aucune ligne n'est visible (cx all == -1), au lieu de tourner
        # sur place à l'aveugle (omega=0.25), on conserve la dernière commande
        # valide ET on continue d'avancer légèrement pour retrouver la piste.
        #
        # - last_omega_valid  : dernier omega calculé sur la base d'une ligne réelle
        # - inertia_speed     : vitesse linéaire réduite pendant la phase d'inertie
        # - inertia_decay     : facteur de décroissance de l'inertie à chaque tick
        #   (0.85 = l'omega est atténué de 15% par cycle pour éviter un virage infini)
        # =====================================================================
        self.last_omega_valid = 0.0
        self.inertia_speed    = 0.05   # m/s pendant la perte de ligne
        self.inertia_decay    = 0.85   # décroissance de l'omega d'inertie par tick

        self.create_subscription(Int32, '/red_line_pos',       self.cb_r_near, 10)
        self.create_subscription(Int32, '/green_line_pos',     self.cb_g_near, 10)
        self.create_subscription(Int32, '/red_line_pos_far',   self.cb_r_far,  10)
        self.create_subscription(Int32, '/green_line_pos_far', self.cb_g_far,  10)

        self.create_subscription(Int32, '/camera_width', self.cb_cam_width, 10)
        self.create_subscription(Int32, '/offset_near',  self.cb_off_near,  10)
        self.create_subscription(Int32, '/offset_far',   self.cb_off_far,   10)

        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel_line_raw', 10)
        self.create_timer(0.05, self.compute_and_publish)

        self.get_logger().info("LineFollower démarré — Inertie de trajectoire active")

    def cb_r_near(self, msg): self.cx_red_near   = msg.data
    def cb_g_near(self, msg): self.cx_green_near = msg.data
    def cb_r_far (self, msg): self.cx_red_far    = msg.data
    def cb_g_far (self, msg): self.cx_green_far  = msg.data

    def cb_cam_width(self, msg): self.image_width = float(msg.data)
    def cb_off_near(self, msg):  self.offset_near = float(msg.data)
    def cb_off_far(self, msg):   self.offset_far  = float(msg.data)

    def centre_voie(self, cx_red, cx_green, offset, safety_margin):
        if cx_red != -1 and cx_green != -1:
            centre = (cx_red + cx_green) / 2.0
            lo = cx_green + safety_margin
            hi = cx_red   - safety_margin
            if lo < hi:
                centre = max(lo, min(hi, centre))
            return centre
        elif cx_red != -1:
            centre = cx_red - offset
            centre = min(centre, cx_red - safety_margin)
            centre = max(safety_margin, centre)
            return centre
        elif cx_green != -1:
            centre = cx_green + offset
            centre = max(centre, cx_green + safety_margin)
            centre = min(centre, self.image_width - safety_margin)
            return centre
        return None

    def compute_and_publish(self):
        if self.image_width == 0.0:
            return

        twist = Twist()

        image_center  = self.image_width / 2.0
        safety_margin = 0.08 * self.image_width
        marge_strict  = 0.40 * self.image_width

        c_near = self.centre_voie(self.cx_red_near, self.cx_green_near, self.offset_near, safety_margin)
        c_far  = self.centre_voie(self.cx_red_far,  self.cx_green_far,  self.offset_far,  safety_margin)

        # =====================================================================
        # CORRECTION 3 appliquée : comportement quand aucune ligne n'est visible
        # =====================================================================
        if c_near is None and c_far is None:
            # Inertie : on conserve le dernier omega valide (atténué) et on avance
            # doucement. Cela évite la rotation sur place aveugle qui désoriente
            # le robot et rend la reprise de ligne aléatoire.
            self.last_omega_valid *= self.inertia_decay

            twist.linear.x  = self.inertia_speed
            twist.angular.z = self.last_omega_valid
            self.pub_cmd.publish(twist)
            return
        # =====================================================================

        if c_near is not None and c_far is not None:
            centre = self.w_near * c_near + self.w_far * c_far
        elif c_near is not None:
            centre = c_near
        else:
            centre = c_far

        erreur_pixels = image_center - centre
        erreur_brute  = erreur_pixels / self.image_width

        self.filtered_err = (self.alpha * self.filtered_err + (1.0 - self.alpha) * erreur_brute)
        erreur = self.filtered_err

        derivation = erreur - self.last_error
        self.last_error = erreur

        omega   = float(self.kp * erreur + self.kd * derivation)
        vitesse = self.base_speed

        seuil_vert  = image_center - marge_strict
        seuil_rouge = image_center + marge_strict

        mord_verte = (self.cx_green_near != -1) and (self.cx_green_near > seuil_vert)
        mord_rouge = (self.cx_red_near != -1) and (self.cx_red_near < seuil_rouge)

        if mord_verte:
            vitesse = 0.05
            omega   = -1.0
        elif mord_rouge:
            vitesse = 0.05
            omega   = 1.0

        # Mémorise l'omega valide pour l'inertie (uniquement hors cas d'urgence)
        if not mord_verte and not mord_rouge:
            self.last_omega_valid = omega

        twist.linear.x  = vitesse
        twist.angular.z = omega
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

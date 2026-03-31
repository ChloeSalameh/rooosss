import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32
import time

STATE_FOLLOW     = 'FOLLOW'
STATE_ROUNDABOUT = 'ROUNDABOUT'
STATE_EXITING    = 'EXITING'

FRAMES_CONFIRMATION = 5
DUREE_MIN_ROUNDABOUT = 4.0
BIAIS = 0.50                 # sert a choisir la direction dans laquelle on prend le rond point mais a voir

class Challenge1(Node):
    def __init__(self):
        super().__init__('challenge1')

        self.declare_parameter('roundabout', 'right')
        self.declare_parameter('camera_width', 640)
        cam_width = self.get_parameter('camera_width').value

        self.seuil_ecart_near = 0.25 * cam_width  
        self.seuil_ecart_far  = 0.12 * cam_width
        self.seuil_sortie     = 0.31 * cam_width 

        self.create_subscription(LaserScan, '/scan', self.cb_scan, 10)
        self.create_subscription(Twist, '/cmd_vel_line_raw', self.cb_cmd, 10)
        self.create_subscription(Int32, '/red_line_pos', self.cb_red_near, 10)
        self.create_subscription(Int32, '/green_line_pos', self.cb_green_near, 10)
        self.create_subscription(Int32, '/red_line_pos_far', self.cb_red_far, 10)
        self.create_subscription(Int32, '/green_line_pos_far', self.cb_green_far, 10)

        self.publisher = self.create_publisher(Twist, '/cmd_vel_challenge_1', 10)

        self.state   = STATE_FOLLOW
        self.t_state = time.time()
        self.emergency = False

        self.cx_red_near   = -1
        self.cx_green_near = -1
        self.cx_red_far    = -1
        self.cx_green_far  = -1
        self.cpt_entree = 0

    def cb_scan(self, msg):
        angles = list(range(0, 21)) + list(range(340, 360))
        dists  = [msg.ranges[a] for a in angles if 0.01 < msg.ranges[a] < 3.0]
        prev   = self.emergency
        self.emergency = bool(dists and min(dists) < 0.25)
        if self.emergency and not prev:
            self.get_logger().warn("Obstacle : arrêt d'urgence")

    def cb_red_near(self,  msg): self.cx_red_near   = msg.data
    def cb_green_near(self,msg): self.cx_green_near = msg.data
    def cb_red_far(self,   msg): self.cx_red_far    = msg.data
    def cb_green_far(self, msg): self.cx_green_far  = msg.data

    def cb_cmd(self, cmd_raw: Twist):
        if self.emergency:
            self.publisher.publish(Twist())
            return
        direction = self.get_parameter('roundabout').get_parameter_value().string_value
        self.publisher.publish(self.fsm_step(cmd_raw, direction))

    def fsm_step(self, cmd: Twist, direction: str) -> Twist:
        dt  = time.time() - self.t_state

        ecart_near = self.ecart(self.cx_red_near, self.cx_green_near)
        ecart_far  = self.ecart(self.cx_red_far,  self.cx_green_far)

        if self.state == STATE_FOLLOW:
            if self.critere_entree(ecart_far):
                self.cpt_entree += 1
            else:
                self.cpt_entree = max(0, self.cpt_entree - 1)

            if self.cpt_entree >= FRAMES_CONFIRMATION:
                self.cpt_entree = 0
                self.transition(STATE_ROUNDABOUT)
                return self.cmd(cmd, self.biais(direction), reduire_vitesse=True)

            return self.cmd(cmd, 0.0)

        elif self.state == STATE_ROUNDABOUT:
            if dt > DUREE_MIN_ROUNDABOUT and self.critere_sortie(ecart_near):
                self.transition(STATE_EXITING)
                return self.cmd(cmd, 0.0)
            
            # Pendant les premieres secondes du rond-point, on applique le biais fortement
            # Une fois "insere" dans le rond point, le LineFollower se debrouille avec les lignes
            biais_actif = self.biais(direction) if dt < 1.5 else 0.0
            return self.cmd(cmd, biais_actif)

        elif self.state == STATE_EXITING:
            if dt > 1.5:
                self.transition(STATE_FOLLOW)
            return self.cmd(cmd, 0.0)

        return self.cmd(cmd, 0.0)

    def ecart(self, cx_red, cx_green):
        if cx_red == -1 or cx_green == -1: return None
        # Si les couleurs se croisent visuellement (ilot central), l'ecart devient negatif
        return cx_red - cx_green

    def critere_entree(self, ecart_far) -> bool:
        if ecart_far is None: return False
        # Le rond-point est detecte au loin : la separation fait chuter brutalement l'ecart far
        # Parfois cet ecart devient negatif car le rouge de l'elot passe a gauche du vert.
        ilot_vu_au_loin = ecart_far < self.seuil_ecart_far
        return ilot_vu_au_loin

    def critere_sortie(self, ecart_near) -> bool:
        if ecart_near is None: return False
        return ecart_near > self.seuil_sortie

    def biais(self, direction: str) -> float:
        return -BIAIS if direction == 'right' else BIAIS

    def cmd(self, cmd: Twist, biais: float, reduire_vitesse=False) -> Twist:
        t = Twist()
        # On ralentit au moment ou on doit s'inserer pour que le biais ait plus d'impact
        t.linear.x  = max(cmd.linear.x, 0.05) if not reduire_vitesse else 0.05
        t.angular.z = cmd.angular.z + biais
        return t

    def transition(self, new_state: str):
        if new_state != self.state:
            self.get_logger().info(f" {self.state} => {new_state}")
            self.state   = new_state
            self.t_state = time.time()

def main(args=None):
    rclpy.init(args=args)
    node = Challenge1()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': main()
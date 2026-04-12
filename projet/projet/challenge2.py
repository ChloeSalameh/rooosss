import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Int32
import math

STATE_FOLLOW   = 'FOLLOW'
STATE_HUGGING  = 'HUGGING'    
STATE_RECENTER = 'RECENTER'

# ── LiDAR : Découpage sans angle mort
CONE_FRONT_C  = 15   # 345° à 15°   (Plein centre)
CONE_FRONT_L  = 75   # 15° à 75°    (Avant Gauche)
CONE_SIDE_L   = 135  # 75° à 135°   (Côté Gauche)

CONE_FRONT_R  = 345  # 285° à 345°  (Avant Droite)
CONE_SIDE_R   = 285  # 225° à 285°  (Côté Droite)

# ── Seuils d'anticipation et de dégagement
DIST_ANTICIPATION = 0.90  # (On garde 0.70, c'était la bonne idée)
DIST_CLEAR_FRONT  = 0.30  # BAISSÉ : L'obstacle n'est plus pile devant
DIST_CLEAR_SIDE   = 0.20  # BAISSÉ (Crucial) : Permet de valider le dépassement même sur piste étroite

# ── Paramètres Visuels (Marge de sécurité anti-franchissement)
# RAPPROCHÉ DU CENTRE : 0.58 et 0.42 gardent fermement le robot sur la piste
PCT_TARGET_RED   = 0.90
PCT_TARGET_GREEN = 0.10

KP_VIS = 0.008  # Un peu plus nerveux pour s'aligner vite
KD_VIS = 0.015
V_HUGGING = 0.08

class Challenge2(Node):
    def __init__(self):
        super().__init__('challenge2')

        self.state     = STATE_FOLLOW
        self.is_active = False
        
        self.target_line = None 

        self.front_min  = 999.0
        self.front_left = 999.0
        self.side_left  = 999.0
        self.front_right= 999.0
        self.side_right = 999.0

        self.cx_red_near   = -1
        self.cx_green_near = -1
        self.cam_width     = 640.0

        self.last_err_vis = 0.0
        self.cpt_clear    = 0
        self.cpt_recenter = 0

        self.create_subscription(LaserScan, '/scan',                self.cb_scan,       10)
        self.create_subscription(Twist,     '/cmd_vel_challenge_1', self.cb_cmd,        10)
        self.create_subscription(Bool,      '/blue_line_crossed',   self.cb_blue_line,  10)
        self.create_subscription(Int32,     '/red_line_pos',        self.cb_red_near,   10)
        self.create_subscription(Int32,     '/green_line_pos',      self.cb_green_near, 10)
        self.create_subscription(Int32,     '/camera_width',        self.cb_cam_width,  10)

        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel_challenge_2', 10)
        self.get_logger().info("Challenge 2 — Anticipation 0.7m & Marges Sécurisées")

    def cb_blue_line(self, msg: Bool):
        if msg.data and not self.is_active:
            self.is_active = True
            self.get_logger().info("Ligne bleue ✓ — Challenge 2 ACTIF")

    def cb_cam_width(self, msg: Int32):   self.cam_width = float(msg.data) if msg.data > 0 else 640.0
    def cb_red_near(self,   msg: Int32):  self.cx_red_near   = msg.data
    def cb_green_near(self, msg: Int32):  self.cx_green_near = msg.data

    def cb_scan(self, msg: LaserScan):
        n = len(msg.ranges)
        def safe_min(start_deg, end_deg):
            vals = []
            for i in range(start_deg, end_deg + 1):
                d = msg.ranges[i % n]
                if 0.05 < d < 3.0 and not math.isinf(d): vals.append(d)
            return min(vals) if vals else 999.0

        self.front_min   = safe_min(360 - CONE_FRONT_C, CONE_FRONT_C)
        self.front_left  = safe_min(CONE_FRONT_C, CONE_FRONT_L)
        self.side_left   = safe_min(CONE_FRONT_L, CONE_SIDE_L)
        self.side_right  = safe_min(CONE_SIDE_R, CONE_FRONT_R)
        self.front_right = safe_min(CONE_FRONT_R, 360 - CONE_FRONT_C)

    def cb_cmd(self, cmd_raw: Twist):
        if not self.is_active:
            self.pub_cmd.publish(cmd_raw)
            return
        self.pub_cmd.publish(self.fsm_step(cmd_raw))

    def fsm_step(self, cmd_raw: Twist) -> Twist:
        danger_ahead = min(self.front_min, self.front_left, self.front_right) < DIST_ANTICIPATION

        if self.state == STATE_FOLLOW or self.state == STATE_RECENTER:
            if danger_ahead:
                self._lock_target_line()
                self.transition(STATE_HUGGING)
                return self._state_hugging()
            
            if self.state == STATE_RECENTER:
                return self._state_recenter(cmd_raw)
            return cmd_raw

        elif self.state == STATE_HUGGING:
            return self._state_hugging()

        return cmd_raw

    def _lock_target_line(self):
        if self.front_left < self.front_right:
            self.target_line = 'red'
            c_str = "ROUGE (Droite)"
        else:
            self.target_line = 'green'
            c_str = "VERTE (Gauche)"
            
        self.last_err_vis = 0.0 
        self.cpt_clear = 0
        self.cpt_recenter = 0
        self.get_logger().info(f"🎯 Obstacle à {min(self.front_left, self.front_right):.2f}m -> Je colle {c_str}")

    def _is_current_obstacle_cleared(self):
        if self.target_line == 'red':
            return self.front_left > DIST_CLEAR_FRONT and self.side_left > DIST_CLEAR_SIDE
        else:
            return self.front_right > DIST_CLEAR_FRONT and self.side_right > DIST_CLEAR_SIDE

    def _state_hugging(self) -> Twist:
        err_vis = 0.0
        omega_vis = 0.0
        
        if self.target_line == 'red':
            if self.cx_red_near != -1:
                cible = self.cam_width * PCT_TARGET_RED
                err_vis = cible - self.cx_red_near
                d_err = err_vis - self.last_err_vis
                self.last_err_vis = err_vis
                omega_vis = (KP_VIS * err_vis) + (KD_VIS * d_err)
            else:
                omega_vis = -0.4 
                self.last_err_vis = 0.0

        elif self.target_line == 'green':
            if self.cx_green_near != -1:
                cible = self.cam_width * PCT_TARGET_GREEN
                err_vis = cible - self.cx_green_near
                d_err = err_vis - self.last_err_vis
                self.last_err_vis = err_vis
                omega_vis = (KP_VIS * err_vis) + (KD_VIS * d_err)
            else:
                omega_vis = 0.4
                self.last_err_vis = 0.0

        omega_total = max(-0.8, min(0.8, omega_vis))

        if self._is_current_obstacle_cleared():
            self.cpt_clear += 1
        else:
            self.cpt_clear = 0

        if self.cpt_clear >= 5:
            self.cpt_clear = 0
            self.get_logger().info("✅ Pilier franchi -> Retour au centre")
            self.transition(STATE_RECENTER)

        out = Twist()
        out.linear.x = V_HUGGING * max(0.6, 1.0 - abs(omega_total))
        out.angular.z = omega_total
        return out

    def _state_recenter(self, cmd_raw: Twist) -> Twist:
        out = Twist()
        out.linear.x = cmd_raw.linear.x * 0.9 
        
        # On donne un peu plus de puissance au recentrage pour qu'il le fasse VITE
        out.angular.z = max(-1.0, min(1.0, cmd_raw.angular.z * 1.2))
        
        self.cpt_recenter += 1
        
        if self.cpt_recenter > 20:
            self.cpt_recenter = 0
            self.get_logger().info("Milieu de piste atteint -> FOLLOW")
            self.transition(STATE_FOLLOW)
            
        return out

    def transition(self, new_state: str):
        self.state = new_state
        self.get_logger().info(f"FSM → {new_state}")

def main(args=None):
    rclpy.init(args=args)
    node = Challenge2()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32
import numpy as np
import queue
import time

# =============================================================================
# PARAMÈTRES AJUSTABLES (CONFIGURATION)
# =============================================================================

# --- Contrôleur Latéral (Évitement gauche/droite) ---
LAT_KP = 4.0
LAT_KI = 0.0
LAT_KD = 0.8
LAT_KS = 10    # Taille de la fenêtre d'intégration (Saturation)

# --- Contrôleur Longitudinal (Vitesse d'approche) ---
LON_KP = 1.2
LON_KI = 0.0
LON_KD = 0.05
LON_KS = 10

# --- Seuils de Distance (en mètres) ---
SEUIL_EVITEMENT   = 0.65   # Distance de détection de l'obstacle
TARGET_CLEARANCE  = 0.35   # Distance latérale à maintenir avec le pilier

# --- Limites de Vitesse ---
LIN_SPEED_MIN     = 0.04   # Vitesse minimale pendant l'évitement
LIN_SPEED_MAX     = 0.07   # Vitesse maximale pendant l'évitement
ANG_SPEED_LIMIT   = 1.5    # Vitesse angulaire max autorisée (rad/s)

# --- Pare-chocs Visuel (Protection de ligne) ---
BUMPER_MARGIN_PCT = 0.13   # Marge de sécurité par rapport au centre (0.13 = 13%)
BUMPER_GAIN       = 0.012  # Force de répulsion de la ligne colorée

# --- Fusion de Trajectoire ---
FOLLOW_WEIGHT     = 0.9    # Poids du suiveur de ligne pendant l'évitement (0.0 à 1.0)


# =============================================================================
# CLASSES ET LOGIQUE DE CONTRÔLE
# =============================================================================

class PIDController:
    def __init__(self, kP, kI, kD, kS):
        self.kP       = kP 
        self.kI       = kI 
        self.kD       = kD 
        self.kS       = kS 
        self.err_int  = 0.0 
        self.err_dif  = 0.0 
        self.err_prev = 0.0 
        self.err_hist = queue.Queue(self.kS) 
        self.t_prev   = 0.0 

    def control(self, err, t):
        if self.t_prev == 0.0:
            self.t_prev = t
            return self.kP * err

        dt = t - self.t_prev 
        if dt > 0.0:
            self.err_hist.put(err) 
            self.err_int += err * dt
            if self.err_hist.full(): 
                self.err_int -= self.err_hist.get() * dt
            self.err_dif = (err - self.err_prev) 
            u = (self.kP * err) + (self.kI * self.err_int) + (self.kD * self.err_dif / dt) 
            self.err_prev = err 
            self.t_prev = t 
            return u 
        return 0.0


class Challenge2(Node):
    def __init__(self):
        super().__init__('challenge2')

        # ── Abonnements (Les topics restent ici) ──────────────────────────────
        self.create_subscription(LaserScan, '/scan',             self.cb_scan,       10)
        self.create_subscription(Twist,     '/cmd_vel_line_raw', self.cb_cmd_line,   10)
        self.create_subscription(Int32,     '/red_line_pos',     self.cb_red_near,   10)
        self.create_subscription(Int32,     '/green_line_pos',   self.cb_green_near, 10)
        self.create_subscription(Int32,     '/camera_width',     self.cb_cam_width,  10)
        
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel_challenge_2', 10)

        # ── [MODIFIED] Initialisation de deux Contrôleurs Latéraux Indépendants ────────
        self.pid_lat_left  = PIDController(LAT_KP, LAT_KI, LAT_KD, LAT_KS)
        self.pid_lat_right = PIDController(LAT_KP, LAT_KI, LAT_KD, LAT_KS)
        self.pid_lon       = PIDController(LON_KP, LON_KI, LON_KD, LON_KS)

        # ── Variables d'état ──────────────────────────────────────────────────
        self.laserscan     = None
        self.cmd_line      = Twist()
        self.cam_width     = 640.0
        self.cx_red_near   = -1
        self.cx_green_near = -1
        self.data_available = False

        self.create_timer(0.05, self.compute_and_publish)
        self.get_logger().info("Challenge 2 prêt : Variables de configuration déportées en haut du fichier.")

    def cb_scan(self, msg):
        ranges = np.asarray(msg.ranges)
        ranges[np.isinf(ranges)] = 3.5
        ranges[ranges == 0.0] = 3.5
        ranges[ranges > 3.5] = 3.5
        self.laserscan = ranges
        self.data_available = True

    def cb_cmd_line(self, msg): self.cmd_line = msg
    def cb_red_near(self, msg): self.cx_red_near = msg.data
    def cb_green_near(self, msg): self.cx_green_near = msg.data
    def cb_cam_width(self, msg): self.cam_width = float(msg.data)

    def compute_and_publish(self):
        if not self.data_available or self.laserscan is None:
            return

        tstamp = time.time()
        cmd_out = Twist()

        # 1. [MODIFIED] CORRECTION LIDAR: Réduction de l'angle pour éviter les "obstacles fantômes"
        front_sector = np.concatenate((self.laserscan[0:25], self.laserscan[335:360]))
        left_sector  = self.laserscan[0:60]    # Réduit de 90 à 60
        right_sector = self.laserscan[300:360] # Réduit de 270 à 300

        min_front = np.min(front_sector)
        min_left  = np.min(left_sector)
        min_right = np.min(right_sector)

        # Détection d'obstacle
        if min(min_front, min_left, min_right) > SEUIL_EVITEMENT:
            self.pub_cmd.publish(self.cmd_line)
            return

        # --- [MODIFIED] CALCUL ÉVITEMENT CORRIGÉ (CHAMPS DE POTENTIELS) ---
        
        # Répulsion de l'obstacle à gauche (pousse vers la droite -> oméga négatif)
        error_left = max(0.0, TARGET_CLEARANCE - min_left)
        omega_left = -self.pid_lat_left.control(error_left, tstamp)

        # Répulsion de l'obstacle à droite (pousse vers la gauche -> oméga positif)
        error_right = max(0.0, TARGET_CLEARANCE - min_right)
        omega_right = self.pid_lat_right.control(error_right, tstamp)

        # On additionne les forces au lieu d'utiliser un if/else ! 
        # Cela empêche les pics de dérivation et gère les doubles obstacles naturellement.
        omega_pid = omega_left + omega_right

        # Vitesse linéaire bridée
        v_lin = self.pid_lon.control(min_front, tstamp)
        v_lin = max(LIN_SPEED_MIN, min(LIN_SPEED_MAX, v_lin))

        # --- PARE-CHOCS VISUEL CORRIGÉ ---
        margin = self.cam_width * BUMPER_MARGIN_PCT
        center = self.cam_width / 2.0
        
        safe_red_min   = center + margin
        safe_green_max = center - margin

        omega_final = omega_pid
        bumper_active = False
        omega_bumper_total = 0.0

        # 3. SYMÉTRIE DU BUMPER: On additionne les forces au lieu d'utiliser min/max
        if self.cx_red_near != -1 and self.cx_red_near < safe_red_min:
            omega_bumper_total += BUMPER_GAIN * (safe_red_min - self.cx_red_near)
            bumper_active = True

        if self.cx_green_near != -1 and self.cx_green_near > safe_green_max:
            omega_bumper_total += BUMPER_GAIN * (safe_green_max - self.cx_green_near)
            bumper_active = True

        # Application de la force du pare-chocs ou du suiveur de ligne
        if bumper_active:
            omega_final += omega_bumper_total
        else:
            omega_final += FOLLOW_WEIGHT * self.cmd_line.angular.z

        # Commande finale avec limite stricte
        cmd_out.linear.x  = float(v_lin)
        cmd_out.angular.z = float(max(-ANG_SPEED_LIMIT, min(ANG_SPEED_LIMIT, omega_final)))

        self.pub_cmd.publish(cmd_out)

def main(args=None):
    rclpy.init(args=args)
    node = Challenge2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
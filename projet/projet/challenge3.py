#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import queue
import time

# =============================================================================
# CONFIGURATION ET PARAMÈTRES AJUSTABLES (Challenge 3 - Spécial U-Turn)
# =============================================================================

# --- Gains du PID Latéral (Centrage) ---
LAT_KP = 1.5    # Augmenté pour tourner plus agressivement dans le U
LAT_KI = 0.005
LAT_KD = 0.6    # Damping pour éviter de zigzaguer en ligne droite
LAT_KS = 10

# --- Gains du PID Longitudinal (Vitesse) ---
LON_KP = 0.2
LON_KI = 0.0
LON_KD = 0.05
LON_KS = 10

# --- Paramètres LIDAR (Anticipation du virage) ---
MAX_LIDAR_RANGE = 3.5
LAT_CROP_RANGE  = 1.0   # Très important : ignore le vide au-delà de 1m pour ne pas dériver
FRONT_CONE_DEG  = 15    # Cône frontal pour la vitesse (+/- 15°)

# NOUVEAU : Angles "Look-ahead" pour anticiper le virage
# Au lieu de regarder à 90°, on regarde en diagonale avant (ex: 30° à 70°)
ANGLE_LAT_MIN = 30
ANGLE_LAT_MAX = 70

# --- Limites de Vitesse ---
V_LIN_MIN = 0.03        # Permet de pivoter doucement
V_LIN_MAX = 0.12        # Vitesse max modérée pour un couloir étroit
V_ANG_MAX = 1.5         # Capacité de rotation rapide requise pour le U-turn

# --- Seuils de Sécurité ---
DIST_FREINAGE = 0.35    # Distance au mur de face déclenchant le mode "virage serré"

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
            self.err_prev = err
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


class Challenge3(Node):
    def __init__(self):
        super().__init__('challenge3')

        self.create_subscription(LaserScan, '/scan', self.cb_scan, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel_challenge_3', 10)

        self.pid_lat = PIDController(kP=LAT_KP, kI=LAT_KI, kD=LAT_KD, kS=LAT_KS) 
        self.pid_lon = PIDController(kP=LON_KP, kI=LON_KI, kD=LON_KD, kS=LON_KS) 

        self.laserscan = None
        self.data_available = False

        self.create_timer(0.05, self.compute_and_publish)
        self.get_logger().info("Challenge 3 prêt : LIDAR en mode anticipation (Look-ahead) activé.")

    def cb_scan(self, msg):
        ranges = np.asarray(msg.ranges)
        ranges[np.isinf(ranges)] = MAX_LIDAR_RANGE
        ranges[np.isnan(ranges)] = MAX_LIDAR_RANGE
        ranges[ranges == 0.0]    = MAX_LIDAR_RANGE
        ranges[ranges > MAX_LIDAR_RANGE] = MAX_LIDAR_RANGE
        self.laserscan = ranges
        self.data_available = True

    def compute_and_publish(self):
        if not self.data_available or self.laserscan is None:
            return

        tstamp = time.time()
        cmd_out = Twist()
        N = len(self.laserscan)

        # 1. Calcul des indices basés sur les angles souhaités
        idx_front_right = int(N * FRONT_CONE_DEG / 360)
        idx_front_left  = N - int(N * FRONT_CONE_DEG / 360)
        
        idx_lat_l_start = int(N * ANGLE_LAT_MIN / 360)
        idx_lat_l_end   = int(N * ANGLE_LAT_MAX / 360)
        
        idx_lat_r_start = N - int(N * ANGLE_LAT_MAX / 360)
        idx_lat_r_end   = N - int(N * ANGLE_LAT_MIN / 360)

        # 2. Extraction des secteurs
        front_sector = np.concatenate((self.laserscan[0:idx_front_right], self.laserscan[idx_front_left:N]))
        min_front = np.min(front_sector)
        
        # Secteurs latéraux diagonaux (Look-ahead) écrêtés à LAT_CROP_RANGE
        left_sector  = np.clip(self.laserscan[idx_lat_l_start : idx_lat_l_end], 0.0, LAT_CROP_RANGE)
        right_sector = np.clip(self.laserscan[idx_lat_r_start : idx_lat_r_end], 0.0, LAT_CROP_RANGE)

        left_avg  = np.mean(left_sector)
        right_avg = np.mean(right_sector)

        # 3. Calcul de l'erreur (CTE)
        # Si le mur droit se rapproche, right_avg diminue, cte devient positif -> omega_pid positif (tourne à gauche)
        cte = left_avg - right_avg

        # Si on est au fond du virage et que le robot est bloqué face au mur,
        # on force une rotation basée sur l'espace disponible global pour s'extirper.
        if min_front < 0.20 and abs(cte) < 0.1:
            cte = 0.5 if np.mean(self.laserscan[0:N//2]) > np.mean(self.laserscan[N//2:N]) else -0.5

        # 4. Calcul PID
        omega_pid = self.pid_lat.control(cte, tstamp)
        v_lin_pid = self.pid_lon.control(min_front, tstamp)

        # 5. Application des limites de sécurité
        v_lin_safe = max(V_LIN_MIN, min(V_LIN_MAX, v_lin_pid))
        
        # Mode virage serré : si le mur approche, on ralentit la vitesse linéaire
        # pour laisser le temps au robot de pivoter.
        if min_front < DIST_FREINAGE:
            v_lin_safe = V_LIN_MIN

        omega_safe = max(-V_ANG_MAX, min(V_ANG_MAX, omega_pid))

        # 6. Publication
        cmd_out.linear.x  = float(v_lin_safe)
        cmd_out.angular.z = float(omega_safe)

        self.pub_cmd.publish(cmd_out)

def main(args=None):
    rclpy.init(args=args)
    node = Challenge3()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
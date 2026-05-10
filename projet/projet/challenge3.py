import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import queue
import time

# PARAMETRES A AJUSTER

# Controleur Lateral (evitement et centrage)
LAT_KP = 1.5    # force pour tourner sec dans le U
LAT_KI = 0.005
LAT_KD = 0.6    # on freine le mouvement pour pas zigzaguer en ligne droite
LAT_KS = 10

# Controleur Longitudinal (pour la vitesse)
LON_KP = 0.2
LON_KI = 0.0
LON_KD = 0.05
LON_KS = 10

# Les reglages du laser (on anticipe le virage)
MAX_LIDAR_RANGE = 3.5
LAT_CROP_RANGE  = 1.0   # on ignore le vide apres 1m sinon le robot derive
FRONT_CONE_DEG  = 15    # le cone de vision juste devant (+/- 15 degres)

# on regarde de travers pour voir le virage arriver plus tot
# au lieu de regarder a 90 degres on regarde en diagonale
ANGLE_LAT_MIN = 30
ANGLE_LAT_MAX = 70

# Limites de Vitesse
V_LIN_MIN = 0.03        # pour tourner doucement sans foncer
V_LIN_MAX = 0.12        # vitesse tranquille parce que c'est etroit
V_ANG_MAX = 1.5         # pour pouvoir tourner vite sur place dans le U

# Securite
DIST_FREINAGE = 0.35    # distance du mur pour declencher le virage serre

# LOGIQUE DE CONTROLE

class PIDController:
    """
    Controleur pour gerer les mouvements du robot, le meme que dans le challenge 2.
    """
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

        # Abonnements aux topics
        self.create_subscription(LaserScan, '/scan', self.cb_scan, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel_challenge_3', 10)

        # Initialisation des controleurs
        self.pid_lat = PIDController(kP=LAT_KP, kI=LAT_KI, kD=LAT_KD, kS=LAT_KS) 
        self.pid_lon = PIDController(kP=LON_KP, kI=LON_KI, kD=LON_KD, kS=LON_KS) 

        # variables d'etat
        self.laserscan = None
        self.data_available = False

        self.create_timer(0.05, self.compute_and_publish)
        self.get_logger().info("Challenge 3 prêt : LIDAR activé.")

    def cb_scan(self, msg):
        ranges = np.asarray(msg.ranges)

        # on nettoie les valeurs bizarres du laser (trop loin ou si le laser bug)
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

        # on calcule les index avec les angles qu'on a choisi
        idx_front_right = int(N * FRONT_CONE_DEG / 360)
        idx_front_left  = N - int(N * FRONT_CONE_DEG / 360)
        
        idx_lat_l_start = int(N * ANGLE_LAT_MIN / 360)
        idx_lat_l_end   = int(N * ANGLE_LAT_MAX / 360)
        
        idx_lat_r_start = N - int(N * ANGLE_LAT_MAX / 360)
        idx_lat_r_end   = N - int(N * ANGLE_LAT_MIN / 360)

        # on decoupe la vision du laser
        front_sector = np.concatenate((self.laserscan[0:idx_front_right], self.laserscan[idx_front_left:N]))
        min_front = np.min(front_sector)
        
        # les cotes en diagonale on les coupe a notre limite
        left_sector  = np.clip(self.laserscan[idx_lat_l_start : idx_lat_l_end], 0.0, LAT_CROP_RANGE)
        right_sector = np.clip(self.laserscan[idx_lat_r_start : idx_lat_r_end], 0.0, LAT_CROP_RANGE)

        left_avg  = np.mean(left_sector)
        right_avg = np.mean(right_sector)

        # calcul de l'erreur : si le mur de droite est trop pres l'erreur devient positive et on tourne a gauche
        cte = left_avg - right_avg

        # si on arrive au bout et qu'on est bloque face au mur
        # on l'oblige a tourner du cote ou y a de la place
        if min_front < 0.20 and abs(cte) < 0.1:
            cte = 0.5 if np.mean(self.laserscan[0:N//2]) > np.mean(self.laserscan[N//2:N]) else -0.5

        # calcul de la force du moteur
        omega_pid = self.pid_lat.control(cte, tstamp)
        v_lin_pid = self.pid_lon.control(min_front, tstamp)

        # on limite les vitesses pour pas crasher
        v_lin_safe = max(V_LIN_MIN, min(V_LIN_MAX, v_lin_pid))
        
        # s'il y a un mur en face on ralentit a fond comme ca il a le temps de tourner
        if min_front < DIST_FREINAGE:
            v_lin_safe = V_LIN_MIN

        omega_safe = max(-V_ANG_MAX, min(V_ANG_MAX, omega_pid))

        # on envoie l'ordre
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
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32
import numpy as np
import queue
import time

# PARAMETRES A AJUSTER

# Controleur Lateral (evitement gauche/droite pour tourner)
LAT_KP = 4.0 # force de reaction immediate
LAT_KI = 0.0 #correction sur le long terme
LAT_KD = 0.8 # Anticipation(freiner le mouvement)
LAT_KS = 10    #memoire du robot pour lisser ses mouvements

# Controleur Longitudinal (gestion de la vitesse)
LON_KP = 1.2
LON_KI = 0.0
LON_KD = 0.05
LON_KS = 10

# Seuils de Distance (en m)
SEUIL_EVITEMENT   = 0.65   # Distance de detection de l'obstacle
TARGET_DIST  = 0.35   # Distance laterale a maintenir avec le pilier

# Limites de Vitesse
LIN_SPEED_MIN     = 0.04   # Vitesse minimale pendant l'evitement
LIN_SPEED_MAX     = 0.07   # Vitesse maximale
ANG_SPEED_LIMIT   = 1.5    # Vitesse angulaire max (rad/s)

# Securite visuelle (Pour ne pas sortir de la route) / bumper : par-chocs
BUMPER_MARGIN_PCT = 0.13   # Marge de securite par rapport au centre (13%)
BUMPER_GAIN       = 0.012  # Force de repulsion de la ligne coloree

FOLLOW_WEIGHT     = 0.9    # Importance donnee au suivi de ligne pendant qu'on esquive (90%)


# LOGIQUE DE CONTROLE

class PIDController:
    """
    Controleur pour gerer les mouvments du robot
    Le systeme calcule la bonne force a appliquer en fonction de l'erreur (la distance qui nous separe de notre but)
    """
    def __init__(self, kP, kI, kD, kS):
        self.kP       = kP 
        self.kI       = kI 
        self.kD       = kD 
        self.kS       = kS 
        self.err_int  = 0.0 
        self.err_dif  = 0.0 
        self.err_prev = 0.0 
        self.err_hist = queue.Queue(self.kS) # pour que la memoire n'accumule pas des erreurs trop vieilles
        self.t_prev   = 0.0 # Heure du dernier calcul

    def control(self, err, t):
        if self.t_prev == 0.0: # S'il vient juste d'etre allume, on applique juste une force proportionnelle simple
            self.t_prev = t
            return self.kP * err

        dt = t - self.t_prev # Temps ecoule depuis le dernier calcul
        if dt > 0.0:

            # Gestion de la memoire (L\integrale)
            self.err_hist.put(err) # On ajoute l'erreur actuelle dans la file
            self.err_int += err * dt # On l'ajoute a notre total d'erreurs
            if self.err_hist.full(): 
                self.err_int -= self.err_hist.get() * dt # Si la memoire est pleine on enleve la plus vieille erreur du total

            # Gestion de l'anticipation (La Derivee)
            self.err_dif = (err - self.err_prev)  # On regarde si l'erreur grandit ou diminue

            # P + I + D
            u = (self.kP * err) + (self.kI * self.err_int) + (self.kD * self.err_dif / dt) 

            # On met a jour les memoires pour le prochain tour
            self.err_prev = err 
            self.t_prev = t 

            # On renvoie la force a appliquer
            return u 
        return 0.0


class Challenge2(Node):
    def __init__(self):
        super().__init__('challenge2')

        # Abonnements aux topics
        self.create_subscription(LaserScan, '/scan',             self.cb_scan,       10)
        self.create_subscription(Twist,     '/cmd_vel_line_raw', self.cb_cmd_line,   10)
        self.create_subscription(Int32,     '/red_line_pos',     self.cb_red_near,   10)
        self.create_subscription(Int32,     '/green_line_pos',   self.cb_green_near, 10)
        self.create_subscription(Int32,     '/camera_width',     self.cb_cam_width,  10)
        
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel_challenge_2', 10)

        # Initialisation des controleurs
        self.pid_lat_left  = PIDController(LAT_KP, LAT_KI, LAT_KD, LAT_KS) # Gere l'esquive vers la droite (si obstacle a gauche)
        self.pid_lat_right = PIDController(LAT_KP, LAT_KI, LAT_KD, LAT_KS) # Gere l'esquive vers la gauche (si obstacle a droite)
        self.pid_lon       = PIDController(LON_KP, LON_KI, LON_KD, LON_KS) # Gere l'acceleration et le freinage

        # variables d'etat
        self.laserscan     = None
        self.cmd_line      = Twist()
        self.cam_width     = 640.0
        self.cx_red_near   = -1
        self.cx_green_near = -1
        self.data_available = False

        self.create_timer(0.05, self.compute_and_publish) # Fait reflechir le robot 20 fois par seconde (0.05s)
        self.get_logger().info("Challenge 2 pret")

    #les fonctions cb servent juste a ranger les donnees
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
        if not self.data_available or self.laserscan is None: # Si le laser n'est pas encore allumé, on attend
            return

        tstamp = time.time()
        cmd_out = Twist() #la commande vide qu'on va remplir

        # On decoupe la vision laser en 3 morceaux pour chercher les dangers : devant, gauche, droite
        front_sector = np.concatenate((self.laserscan[0:25], self.laserscan[335:360]))
        left_sector  = self.laserscan[0:60]    
        right_sector = self.laserscan[300:360] 

        # On trouve l'obstacle le plus proche dans chaque zone
        min_front = np.min(front_sector)
        min_left  = np.min(left_sector)
        min_right = np.min(right_sector)

        # Si tout est loin (pas de danger),on suit juste la ligne
        if min(min_front, min_left, min_right) > SEUIL_EVITEMENT:
            self.pub_cmd.publish(self.cmd_line)
            return

        
        # Si un truc est a gauche, pousse vers la droite (omega negatif)
        error_left = max(0.0, TARGET_DIST - min_left)
        omega_left = -self.pid_lat_left.control(error_left, tstamp)

        # Si un truc est a droite, pousse vers la gauche (omega positif)
        error_right = max(0.0, TARGET_DIST - min_right)
        omega_right = self.pid_lat_right.control(error_right, tstamp)

        # On additionne les forces 
        omega_pid = omega_left + omega_right

        # On calcule la vitesse pour avancer : le PID avant freine si le mur de face se rapproche
        v_lin = self.pid_lon.control(min_front, tstamp)
        v_lin = max(LIN_SPEED_MIN, min(LIN_SPEED_MAX, v_lin))

        # On definit des murs invisibles sur les bords de l'image camera(pour ne pas sortir de la route)
        margin = self.cam_width * BUMPER_MARGIN_PCT
        center = self.cam_width / 2.0
        safe_red_min   = center + margin
        safe_green_max = center - margin

        omega_final = omega_pid
        bumper_active = False
        omega_bumper_total = 0.0

        # Si le bord rouge se rapproche dangereusement, on cree une force pour nous repousser vers le centre
        if self.cx_red_near != -1 and self.cx_red_near < safe_red_min:
            omega_bumper_total += BUMPER_GAIN * (safe_red_min - self.cx_red_near)
            bumper_active = True

        #pareil pour le bord vert
        if self.cx_green_near != -1 and self.cx_green_near > safe_green_max:
            omega_bumper_total += BUMPER_GAIN * (safe_green_max - self.cx_green_near)
            bumper_active = True

        # Application de la force du pare-chocs ou du suiveur de ligne
        if bumper_active:
            omega_final += omega_bumper_total #n ignore la ligne et on se repousse vers le centre
        else:
            omega_final += FOLLOW_WEIGHT * self.cmd_line.angular.z #on ajoute le suivi de ligne

        # On remplit l'ordre final pour les moteurs (avec une limite de vitesse pour tourner)
        cmd_out.linear.x  = float(v_lin)
        cmd_out.angular.z = float(max(-ANG_SPEED_LIMIT, min(ANG_SPEED_LIMIT, omega_final)))

        self.pub_cmd.publish(cmd_out) # on envoie l'ordre aux roues

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
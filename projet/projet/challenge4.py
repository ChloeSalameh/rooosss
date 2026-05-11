import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, CompressedImage, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import time


MODE = "IRL"   # "IRL" ou "SIMULATION"

# ETATS DE LA MACHINE A ETATS (FSM) :

STATE_SEARCH_BALL          = 'SEARCH_BALL'
STATE_MEMORIZE_BALL        = 'MEMORIZE_BALL'
STATE_SEARCH_GOAL          = 'SEARCH_GOAL'
STATE_COMPUTE_TRAJECTORY   = 'COMPUTE_TRAJECTORY'
STATE_NAVIGATE_TO_WAYPOINT = 'NAVIGATE_TO_WAYPOINT'
STATE_ALIGN_AND_PUSH       = 'ALIGN_AND_PUSH'
# etat final quand on a marque, on bloque le robot
STATE_DONE                 = 'DONE'

# on regroupe les etats de la phase 2 (quand on a deja lock la balle)
POST_MEMO_STATES = {
    STATE_SEARCH_GOAL,
    STATE_COMPUTE_TRAJECTORY,
    STATE_NAVIGATE_TO_WAYPOINT,
    STATE_ALIGN_AND_PUSH,
    STATE_DONE,
}


# =============================================================================
# PARAMETRES PAR DEFAUT
# =============================================================================

# -- Couleurs balle (jaune/vert) -----------------------------------------
if MODE == "IRL":
    HSV_BALL = [20, 50, 60, 255, 60, 255]
else:
    HSV_BALL = [28, 65, 80, 255, 60, 255]

BALL_MIN_AREA     = 400      # taille mini du blob pour pas prendre un pixel au pif

# -- Couleurs poteaux (rouge) -------------------------------------------
if MODE == "IRL":
    HSV_RED1 = [  0,  10,  50, 255,  50, 255]
    HSV_RED2 = [160, 180,  50, 255,  50, 255]
else:
    HSV_RED1 = [  0,  10, 100, 255,  80, 255]
    HSV_RED2 = [160, 180, 100, 255,  80, 255]

POST_MIN_AREA     = 300      
POST_ASPECT_MIN   = 1.5      # on s'assure que c'est un rectangle vertical (hauteur > largeur)

# -- PD pour centrer la vision --------------------------------
ANG_KP            = 3.5
ANG_KD            = 1.0
ALPHA_FILT        = 0.70

# -- PID pour aller au waypoint -------------------------------
NAV_ANG_KP        = 2.0      # correction pour s'aligner vers le point
NAV_ANG_KI        = 0.02     
NAV_ANG_KD        = 0.5      
NAV_LIN_KP        = 0.6      # correction pour avancer
NAV_LIN_MAX       = 0.12     # on bride un peu la vitesse max
NAV_LIN_MIN       = 0.03     
NAV_ARRIVAL_DIST  = 0.12     # a quelle distance on considere qu'on est arrive

# -- Vitesses de base -------------------------------------------------------
SEARCH_OMEGA      = 0.35     # vitesse quand il tourne sur lui meme
PUSH_V            = 0.10     # vitesse quand il fonce dans la balle
OMEGA_MAX         = 1.4      

# -- Validations -------------------------------------------------------
BALL_STABLE_FRAMES = 8       # nb de frames ok de suite avant de valider la balle
BALL_STABLE_TOL   = 0.04     # tolerance pour considerer qu'on est bien centre
BLIND_SPOT_FRAMES = 4        

# -- Bidouilles pour estimer les distances sans laser -------------------------
# formule optique de base avec la focale
BALL_REAL_DIAM_M  = 0.065    
CAMERA_FOCAL_EST  = 530.0    
CAMERA_FOV_RAD    = math.radians(60)

# on corrige le laser parce qu'il tape sur le bord des objets et pas au centre
DIST_OFFSET       = 0.45     

# -- Distances de calcul ----------------------------------------------------
WAYPOINT_OFFSET   = 0.20     # on se place a 20cm derriere la balle pour preparer le tir

# -- Secu --------------------------------------------------------
SAFETY_DIST       = 0.17     # si mur trop pres on stop
PUSH_TIMEOUT      = 30.0     # au bout de 30s de poussee on abandonne
NAV_TIMEOUT       = 30.0     

# -- Point final (End Point) --------------------------------------------------
# on pousse la balle 30cm au dela de la ligne de but pour etre sur
PUSH_OVERSHOOT    = 0.30     

# -- Filtres laser pour les poteaux --------------------------------------------------
LIDAR_MAX_RANGE   = 2.0
CLUSTER_GAP       = 0.15
CLUSTER_MIN_PTS   = 3
POST_WIDTH_MAX    = 0.12
GOAL_DIST_MIN     = 0.35
GOAL_DIST_MAX     = 1.30

# -- Validation vision cage ----------------------------------------------------------
CONFIRM_FRAMES    = 2        

# -- Balayage ----------------------------------------------------
# on regarde que devant (allers-retours de 90 degres de chaque cote)
SWEEP_HALF_ANGLE  = math.pi / 2.0   


# =============================================================================
# OUTILS MATHS
# =============================================================================

def normalize_angle(a: float) -> float:
    # remet l'angle proprement entre -pi et pi
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a

def robot_to_world(rx: float, ry: float, ryaw: float,
                   lx: float, ly: float) -> tuple:
    # convertit les coordonnees du robot (relatif) en coordonnees map (odom)
    cos_y = math.cos(ryaw)
    sin_y = math.sin(ryaw)
    wx = rx + cos_y * lx - sin_y * ly
    wy = ry + sin_y * lx + cos_y * ly
    return wx, wy

def polar_to_local(dist: float, angle_rad: float) -> tuple:
    # convertit angle+distance du capteur en x,y par rapport au robot
    return dist * math.cos(angle_rad), dist * math.sin(angle_rad)

def dist2d(ax: float, ay: float, bx: float, by: float) -> float:
    return math.sqrt((ax - bx)**2 + (ay - by)**2)


# =============================================================================
# LE NOEUD PRINCIPAL
# =============================================================================

class Challenge4(Node):

    def __init__(self):
        super().__init__('challenge4')

        if MODE not in ("IRL", "SIMULATION"):
            raise ValueError(f"MODE invalide : '{MODE}'. Choisir 'IRL' ou 'SIMULATION'.")

        # on setup nos parametres ros2 pour pouvoir tuner sans relancer le code
        self.declare_parameter('ball_h_min',        float(HSV_BALL[0]))
        self.declare_parameter('ball_h_max',        float(HSV_BALL[1]))
        self.declare_parameter('ball_s_min',        float(HSV_BALL[2]))
        self.declare_parameter('ball_s_max',        float(HSV_BALL[3]))
        self.declare_parameter('ball_v_min',        float(HSV_BALL[4]))
        self.declare_parameter('ball_v_max',        float(HSV_BALL[5]))
        self.declare_parameter('ball_min_area',     float(BALL_MIN_AREA))
        
        self.declare_parameter('red1_h_min',        float(HSV_RED1[0]))
        self.declare_parameter('red1_h_max',        float(HSV_RED1[1]))
        self.declare_parameter('red1_s_min',        float(HSV_RED1[2]))
        self.declare_parameter('red2_h_min',        float(HSV_RED2[0]))
        self.declare_parameter('red2_h_max',        float(HSV_RED2[1]))
        self.declare_parameter('red2_s_min',        float(HSV_RED2[2]))
        self.declare_parameter('post_min_area',     float(POST_MIN_AREA))
        self.declare_parameter('post_aspect_min',   float(POST_ASPECT_MIN))
        
        self.declare_parameter('ang_kp',            float(ANG_KP))
        self.declare_parameter('ang_kd',            float(ANG_KD))
        self.declare_parameter('nav_ang_kp',        float(NAV_ANG_KP))
        self.declare_parameter('nav_ang_ki',        float(NAV_ANG_KI))
        self.declare_parameter('nav_ang_kd',        float(NAV_ANG_KD))
        self.declare_parameter('nav_lin_kp',        float(NAV_LIN_KP))
        self.declare_parameter('nav_lin_max',       float(NAV_LIN_MAX))
        self.declare_parameter('nav_arrival_dist',  float(NAV_ARRIVAL_DIST))
        
        self.declare_parameter('search_omega',      float(SEARCH_OMEGA))
        self.declare_parameter('push_v',            float(PUSH_V))
        self.declare_parameter('omega_max',         float(OMEGA_MAX))
        
        self.declare_parameter('waypoint_offset',   float(WAYPOINT_OFFSET))
        
        self.declare_parameter('align_err_thresh',  float(BALL_STABLE_TOL))
        self.declare_parameter('align_frames_ok',   int(BALL_STABLE_FRAMES))
        self.declare_parameter('confirm_frames',    int(CONFIRM_FRAMES))
        self.declare_parameter('safety_dist',       float(SAFETY_DIST))
        self.declare_parameter('push_timeout',      float(PUSH_TIMEOUT))
        self.declare_parameter('nav_timeout',       float(NAV_TIMEOUT))
        self.declare_parameter('goal_dist_min',     float(GOAL_DIST_MIN))
        self.declare_parameter('goal_dist_max',     float(GOAL_DIST_MAX))
        
        self.declare_parameter('push_overshoot',    float(PUSH_OVERSHOOT))
        self.declare_parameter('dist_offset',       float(DIST_OFFSET))

        self.bridge = CvBridge()

        # on start direct sur la recherche
        self.state = STATE_SEARCH_BALL

        self.laserscan    = None
        self.image_width  = 640.0
        self.image_height = 480.0

        # odometrie 
        self.robot_x   = 0.0    
        self.robot_y   = 0.0    
        self.robot_yaw = 0.0    
        self.odom_ok   = False  

        # sert pour le scan a 180 degres (on bloque la direction initiale)
        self.yaw_initial   = None   
        self.sweep_direction = -1.0  

        # verrou quand on a trouve la balle pour arreter de faire chauffer le cpu
        self.ball_locked = False

        # memoire balle
        self.ball_dist  = 0.30    
        self.ball_angle = 0.0     
        self.ball_cx    = 320     
        self.ball_world_x = None  
        self.ball_world_y = None  
        self.stable_frames = 0

        # memoire cage
        self.goal_world_x  = None  
        self.goal_world_y  = None  
        self.goal_cx_vision = self.image_width / 2.0  
        self.confirm_frames_count = 0

        # le fameux point ou on doit se placer pour tirer
        self.waypoint_x = None    
        self.waypoint_y = None    

        # la ligne d'arrivee
        self.end_point_x = None   
        self.end_point_y = None   

        # vecteur pour pousser tout droit
        self.push_vx = 1.0    
        self.push_vy = 0.0    

        # variables pour nos pids
        self.err_ang_prev  = 0.0
        self.err_ang_filt  = 0.0
        self.frames_aligned = 0
        self.blind_frames  = 0

        self.nav_err_prev     = 0.0
        self.nav_err_integral = 0.0
        self.nav_t_prev       = 0.0

        # chronos
        self.push_start_time  = 0.0
        self.nav_start_time   = 0.0

        self.cmd_pending = Twist()

        # topics
        self.create_subscription(LaserScan, '/scan', self.cb_scan, 10)
        self.create_subscription(Odometry,  '/odom', self.cb_odom, 10)
        
        if MODE == "IRL":
            self.create_subscription(CompressedImage, '/camera/image_raw/compressed', self.cb_image, 10)
        else:
            self.create_subscription(Image, '/image_raw', self.cb_image, 10)

        self.pub_cmd   = self.create_publisher(Twist, '/cmd_vel_challenge_4', 10)
        self.pub_debug = self.create_publisher(Image, '/debug/challenge4',    10)
        self.create_timer(0.05, self._publish_cmd)   

        self.get_logger().info("Challenge 4 pret.")

    # =========================================================================
    # Callbacks capteurs
    # =========================================================================

    def cb_scan(self, msg: LaserScan):
        # on coupe tout ce qui est derriere le robot pour pas etre gene
        r = np.asarray(msg.ranges, dtype=np.float32)
        r = np.where(np.isinf(r) | np.isnan(r) | (r == 0.0), 3.5, r)
        r = np.clip(r, 0.0, 3.5)
        r[91:270] = 3.5
        self.laserscan = r

    def cb_odom(self, msg: Odometry):
        # maj de notre position absolue sur la carte
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        # passage quaternion en angle d'euler
        self.robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        self.odom_ok = True

        # on garde notre angle de depart pour le balayage
        if self.yaw_initial is None:
            self.yaw_initial = self.robot_yaw

    def cb_image(self, msg):
        # boucle principale qui va appeler la machine a etat
        try:
            if MODE == "IRL":
                frame = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            else:
                frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return

        self.image_height, self.image_width, _ = frame.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # si on a pas encore lock la balle on continue de la chercher
        if not self.ball_locked:
            ball_data = self._detect_ball(hsv)
            if ball_data is not None:
                self._refresh_ball_memory(ball_data)
                self.blind_frames = 0
            else:
                self.blind_frames += 1
        else:
            # opti cpu, on cherche plus la balle
            ball_data = None   

        # on cherche toujours les poteaux par contre
        posts_data = self._detect_red_posts(hsv)

        # interface graphique
        debug = self._render_debug(frame, ball_data, posts_data)
        try:
            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(debug, 'bgr8'))
        except Exception:
            pass
        cv2.imshow("Challenge 4 v5 - Debug", debug)
        cv2.waitKey(1)

        # on actualise la commande moteur
        self.cmd_pending = self._fsm_step(ball_data, posts_data)

    # =========================================================================
    # Gestion memoire balle
    # =========================================================================

    def _refresh_ball_memory(self, ball_data: dict):
        # met a jour les infos de la balle a chaque image
        cx   = ball_data['cx']
        err_norm    = (cx - self.image_width / 2.0) / self.image_width
        self.ball_cx    = cx
        self.ball_angle = -err_norm * CAMERA_FOV_RAD

        d_lidar = self._min_front_dist()
        
        # si le laser bug on estime a la louche avec la camera
        if d_lidar is not None and d_lidar < 2.0:
            offset = self.get_parameter('dist_offset').value
            self.ball_dist = max(0.05, d_lidar - offset)
        else:
            bx, by, bw, bh = ball_data['bbox']
            if bw > 0:
                self.ball_dist = (BALL_REAL_DIAM_M * CAMERA_FOCAL_EST) / bw

    def _lock_ball_and_convert_to_world(self):
        # on sauvegarde definitivement la balle en coordonnees odom 
        self.ball_locked = True

        if not self.odom_ok:
            # bidouille si odom plante
            lx, ly = polar_to_local(self.ball_dist, self.ball_angle)
            self.ball_world_x = lx
            self.ball_world_y = ly
        else:
            lx, ly = polar_to_local(self.ball_dist, self.ball_angle)
            self.ball_world_x, self.ball_world_y = robot_to_world(
                self.robot_x, self.robot_y, self.robot_yaw, lx, ly)


    # =========================================================================
    # Calculs geometriques
    # =========================================================================

    def _convert_goal_to_world(self, lidar_result: dict) -> tuple:
        lx, ly = polar_to_local(lidar_result['center_dist'],
                                 lidar_result['center_angle'])
        return robot_to_world(self.robot_x, self.robot_y, self.robot_yaw, lx, ly)

    def _compute_waypoint(self) -> tuple:
        # calcule le point juste derriere la balle par rapport a la cage
        bx, by   = self.ball_world_x, self.ball_world_y
        gx, gy   = self.goal_world_x, self.goal_world_y
        offset   = self.get_parameter('waypoint_offset').value

        dx = gx - bx
        dy = gy - by
        d  = math.sqrt(dx*dx + dy*dy)
        
        if d < 1e-6:
            return bx, by

        ux = dx / d
        uy = dy / d

        self.push_vx = ux
        self.push_vy = uy

        wpx = bx - offset * ux
        wpy = by - offset * uy

        return wpx, wpy

    # =========================================================================
    # Vision camera
    # =========================================================================

    def _detect_ball(self, hsv: np.ndarray):
        # cherche le gros blob jaune/vert
        lo = np.array([self.get_parameter('ball_h_min').value,
                       self.get_parameter('ball_s_min').value,
                       self.get_parameter('ball_v_min').value], dtype=np.uint8)
        hi = np.array([self.get_parameter('ball_h_max').value,
                       self.get_parameter('ball_s_max').value,
                       self.get_parameter('ball_v_max').value], dtype=np.uint8)
        min_area = self.get_parameter('ball_min_area').value

        mask = cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
            
        best = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(best)
        if area < min_area:
            return None
            
        M = cv2.moments(best)
        if M['m00'] == 0:
            return None
            
        return {
            'cx': int(M['m10'] / M['m00']), 'cy': int(M['m01'] / M['m00']),
            'area': area, 'contour': best, 'bbox': cv2.boundingRect(best),
        }


    def _detect_red_posts(self, hsv: np.ndarray) -> list:
        # cherche les deux rectangles rouges bien verticaux
        lo1 = np.array([self.get_parameter('red1_h_min').value,
                        self.get_parameter('red1_s_min').value, 50], dtype=np.uint8)
        hi1 = np.array([self.get_parameter('red1_h_max').value, 255, 255], dtype=np.uint8)
        lo2 = np.array([self.get_parameter('red2_h_min').value,
                        self.get_parameter('red2_s_min').value, 50], dtype=np.uint8)
        hi2 = np.array([self.get_parameter('red2_h_max').value, 255, 255], dtype=np.uint8)

        post_area  = self.get_parameter('post_min_area').value
        aspect_min = self.get_parameter('post_aspect_min').value

        mask = cv2.bitwise_or(cv2.inRange(hsv, lo1, hi1),
                              cv2.inRange(hsv, lo2, hi2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        posts = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < post_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            
            # filtre ratio hauteur/largeur
            if bh <= bw or bw == 0 or (bh / bw) < aspect_min:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            posts.append({
                'cx': int(M['m10'] / M['m00']), 'cy': int(M['m01'] / M['m00']),
                'area': area, 'bbox': (bx, by, bw, bh), 'aspect': bh / bw,
            })
            
        posts.sort(key=lambda p: p['cx'])
        return posts

    # =========================================================================
    # Scan Laser 
    # =========================================================================

    def _lidar_find_goal_candidates(self):
        # trouve 2 clusters au laser qui ressemblent a nos poteaux
        if self.laserscan is None:
            return None

        goal_dist_min = self.get_parameter('goal_dist_min').value
        goal_dist_max = self.get_parameter('goal_dist_max').value

        cart = []
        for idx in list(range(0, 91)) + list(range(270, 360)):
            d = float(self.laserscan[idx])
            if d >= LIDAR_MAX_RANGE:
                continue
            a = math.radians(idx) if idx <= 180 else math.radians(idx - 360)
            cart.append((d * math.cos(a), d * math.sin(a), a, d))

        if len(cart) < 2:
            return None

        cart.sort(key=lambda p: p[2])

        clusters, cur = [], [cart[0]]
        for i in range(1, len(cart)):
            dx = cart[i][0] - cart[i-1][0]
            dy = cart[i][1] - cart[i-1][1]
            if math.sqrt(dx*dx + dy*dy) < CLUSTER_GAP:
                cur.append(cart[i])
            else:
                clusters.append(cur)
                cur = [cart[i]]
        clusters.append(cur)

        valid = []
        for cl in clusters:
            if len(cl) < CLUSTER_MIN_PTS:
                continue
            xs    = [p[0] for p in cl]
            ys    = [p[1] for p in cl]
            width = math.sqrt((max(xs) - min(xs))**2 + (max(ys) - min(ys))**2)
            if width > POST_WIDTH_MAX:
                continue
            cx_cl = sum(xs) / len(cl)
            cy_cl = sum(ys) / len(cl)
            valid.append({
                'lx': cx_cl, 'ly': cy_cl,
                'dist':  math.sqrt(cx_cl**2 + cy_cl**2),
                'angle': math.atan2(cy_cl, cx_cl),
            })

        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                c1, c2  = valid[i], valid[j]
                dx, dy  = c1['lx'] - c2['lx'], c1['ly'] - c2['ly']
                inter_d = math.sqrt(dx*dx + dy*dy)
                if goal_dist_min <= inter_d <= goal_dist_max:
                    mx = (c1['lx'] + c2['lx']) / 2.0
                    my = (c1['ly'] + c2['ly']) / 2.0
                    return {
                        'center_angle': math.atan2(my, mx),
                        'center_dist':  math.sqrt(mx**2 + my**2),
                        'inter_dist':   inter_d,
                        'mid_lx': mx, 'mid_ly': my,        
                    }
        return None

    # =========================================================================
    # L'Aiguilleur des etats (le switch central)
    # =========================================================================

    def _fsm_step(self, ball_data, posts_data: list) -> Twist:
        # securite de base si on fonce dans un mur pendant la phase d'approche
        security_states = {STATE_SEARCH_BALL, STATE_MEMORIZE_BALL,
                           STATE_NAVIGATE_TO_WAYPOINT}
        if self.state in security_states and self._obstacle_front():
            self.get_logger().warn("ARRET URGENCE")
            return Twist()

        if self.state == STATE_SEARCH_BALL:
            return self._state_search_ball(ball_data)
        elif self.state == STATE_MEMORIZE_BALL:
            return self._state_memorize_ball(ball_data)
        elif self.state == STATE_SEARCH_GOAL:
            return self._state_search_goal(posts_data)
        elif self.state == STATE_COMPUTE_TRAJECTORY:
            return self._state_compute_trajectory(posts_data)
        elif self.state == STATE_NAVIGATE_TO_WAYPOINT:
            return self._state_navigate_to_waypoint()
        elif self.state == STATE_ALIGN_AND_PUSH:
            return self._state_align_and_push(posts_data)
        elif self.state == STATE_DONE:
            return Twist()
        return Twist()

    # =========================================================================
    # ETATS 
    # =========================================================================

    def _state_search_ball(self, ball_data) -> Twist:
        # on tourne en cherchant la balle
        if ball_data is not None:
            self._reset_pd()
            self.stable_frames = 0
            self.frames_aligned = 0
            self.blind_frames   = 0
            self._transition(STATE_MEMORIZE_BALL)
            return self._pd_cmd(ball_data['cx'])
        return self._spin_cmd()

    def _state_memorize_ball(self, ball_data) -> Twist:
        # on se centre sur la balle et on attends que ca stabilise
        if ball_data is None:
            self.blind_frames += 1
            if self.blind_frames > BLIND_SPOT_FRAMES:
                self.stable_frames  = 0
                self.blind_frames   = 0
                self.frames_aligned = 0
                self._reset_pd()
                self._transition(STATE_SEARCH_BALL)
                return self._spin_cmd()
            return self._pd_cmd(self.ball_cx)

        self.blind_frames = 0
        cx       = ball_data['cx']
        err_norm = abs(cx - self.image_width / 2.0) / self.image_width
        thresh   = self.get_parameter('align_err_thresh').value
        n_frames = self.get_parameter('align_frames_ok').value

        if err_norm < thresh:
            self.stable_frames += 1
        else:
            self.stable_frames = max(0, self.stable_frames - 1)

        # c'est bon, on est lock
        if self.stable_frames >= n_frames:
            self._lock_ball_and_convert_to_world()
            self.stable_frames     = 0
            self.frames_aligned    = 0
            self.confirm_frames_count = 0
            self._reset_pd()
            self._reset_nav_pid()
            self._transition(STATE_SEARCH_GOAL)
            return self._spin_cmd()

        return self._pd_cmd(cx)


    def _state_search_goal(self, posts_data: list) -> Twist:
        # on balaie de gauche a droite pour trouver les cages
        if len(posts_data) >= 2:
            self.goal_cx_vision = (posts_data[0]['cx'] + posts_data[1]['cx']) / 2.0

            err_norm_goal = (self.goal_cx_vision - self.image_width / 2.0) / self.image_width
            goal_angle = -err_norm_goal * CAMERA_FOV_RAD   

            lidar_goal = self._lidar_find_goal_candidates()
            if lidar_goal is not None:
                offset = self.get_parameter('dist_offset').value
                goal_dist = max(0.10, lidar_goal['center_dist'] - offset)
            else:
                POST_REAL_HEIGHT_M = 0.30
                avg_height_px = (posts_data[0]['bbox'][3] + posts_data[1]['bbox'][3]) / 2.0
                if avg_height_px > 0:
                    goal_dist = (POST_REAL_HEIGHT_M * CAMERA_FOCAL_EST) / avg_height_px
                else:
                    goal_dist = 1.0   

            mid_lx = goal_dist * math.cos(goal_angle)
            mid_ly = goal_dist * math.sin(goal_angle)
            vision_goal = {
                'center_angle': goal_angle,
                'center_dist':  goal_dist,
                'mid_lx':       mid_lx,
                'mid_ly':       mid_ly,
            }

            self.confirm_frames_count += 1

            n_conf = self.get_parameter('confirm_frames').value
            if self.confirm_frames_count >= n_conf:
                # poteaux valides
                self._last_lidar_goal = lidar_goal if lidar_goal is not None else vision_goal
                self.confirm_frames_count = 0
                self._reset_pd()
                self._reset_nav_pid()
                self._transition(STATE_COMPUTE_TRAJECTORY)
                return Twist()
        else:
            self.confirm_frames_count = max(0, self.confirm_frames_count - 1)

        return self._spin_cmd()

    def _state_compute_trajectory(self, posts_data: list) -> Twist:
        # les gros calculs d'angles et point cible
        if not self.odom_ok:
            return Twist()

        if self.ball_world_x is None or self.ball_world_y is None:
            self._transition(STATE_SEARCH_BALL)
            return Twist()

        gx, gy = robot_to_world(
            self.robot_x, self.robot_y, self.robot_yaw,
            self._last_lidar_goal['mid_lx'],
            self._last_lidar_goal['mid_ly']
        )
        self.goal_world_x = gx
        self.goal_world_y = gy

        self.waypoint_x, self.waypoint_y = self._compute_waypoint()

        # on calcule ou on va s'arreter (apres la ligne)
        overshoot = self.get_parameter('push_overshoot').value
        self.end_point_x = gx + overshoot * self.push_vx
        self.end_point_y = gy + overshoot * self.push_vy

        self.nav_start_time = time.time()
        self._reset_nav_pid()
        self._transition(STATE_NAVIGATE_TO_WAYPOINT)
        return Twist()

    def _state_navigate_to_waypoint(self) -> Twist:
        # on fonce vers le waypoint derriere la balle au pid
        if self.waypoint_x is None:
            self._transition(STATE_SEARCH_BALL)
            return Twist()

        elapsed = time.time() - self.nav_start_time
        nav_timeout = self.get_parameter('nav_timeout').value
        
        # si on bloque trop longtemps on force le tir quand meme
        if elapsed > nav_timeout:
            self.push_start_time = time.time()
            self._transition(STATE_ALIGN_AND_PUSH)
            return Twist()

        d_wp = dist2d(self.robot_x, self.robot_y, self.waypoint_x, self.waypoint_y)
        arrival = self.get_parameter('nav_arrival_dist').value

        # on y est !
        if d_wp < arrival:
            self.push_start_time = time.time()
            self._transition(STATE_ALIGN_AND_PUSH)
            return Twist()

        cap_cible = math.atan2(
            self.waypoint_y - self.robot_y,
            self.waypoint_x - self.robot_x
        )
        err_angle = normalize_angle(cap_cible - self.robot_yaw)

        t_now = time.time()
        dt    = t_now - self.nav_t_prev if self.nav_t_prev > 0.0 else 0.05
        self.nav_t_prev = t_now

        kp_a = self.get_parameter('nav_ang_kp').value
        ki_a = self.get_parameter('nav_ang_ki').value
        kd_a = self.get_parameter('nav_ang_kd').value

        self.nav_err_integral += err_angle * dt
        self.nav_err_integral = float(np.clip(self.nav_err_integral, -1.0, 1.0))
        derr = (err_angle - self.nav_err_prev) / dt if dt > 0 else 0.0
        self.nav_err_prev = err_angle

        omega = float(np.clip(
            kp_a * err_angle + ki_a * self.nav_err_integral + kd_a * derr,
            -self.get_parameter('omega_max').value,
            self.get_parameter('omega_max').value
        ))

        kp_l  = self.get_parameter('nav_lin_kp').value
        v_max = self.get_parameter('nav_lin_max').value
        v_lin = float(np.clip(kp_l * d_wp, NAV_LIN_MIN, v_max))

        # on gere l'approche en douceur (on tourne d'abord, on avance apres)
        if abs(err_angle) > math.radians(45):
            v_lin = 0.0
        elif abs(err_angle) > math.radians(20):
            v_lin *= 0.3

        cmd = Twist()
        cmd.linear.x  = v_lin
        cmd.angular.z = omega
        return cmd

    def _state_align_and_push(self, posts_data: list) -> Twist:
        # derniere ligne droite, on defonce la balle
        push_timeout = self.get_parameter('push_timeout').value
        push_v       = self.get_parameter('push_v').value
        omega_max    = self.get_parameter('omega_max').value
        kp_a         = self.get_parameter('nav_ang_kp').value

        elapsed = time.time() - self.push_start_time

        if elapsed > push_timeout:
            self._transition(STATE_DONE)
            return Twist()

        # on verifie si on a franchi la ligne avec l'odom
        if self.end_point_x is not None:
            d_end = dist2d(self.robot_x, self.robot_y,
                           self.end_point_x, self.end_point_y)
            arrival = self.get_parameter('nav_arrival_dist').value
            if d_end < arrival:
                self._transition(STATE_DONE)
                return Twist()

        if len(posts_data) >= 2:
            self.goal_cx_vision = (posts_data[0]['cx'] + posts_data[1]['cx']) / 2.0

        cap_push = math.atan2(self.push_vy, self.push_vx)
        err_push = normalize_angle(cap_push - self.robot_yaw)

        # mix correction geo et visuelle
        err_vis_norm = (self.goal_cx_vision - self.image_width / 2.0) / self.image_width
        omega_geo = kp_a * err_push
        omega_vis = -(err_vis_norm * self.get_parameter('ang_kp').value)
        omega = float(np.clip(0.7 * omega_geo + 0.3 * omega_vis, -omega_max, omega_max))

        cmd = Twist()
        cmd.linear.x  = push_v
        cmd.angular.z = omega
        return cmd

    # =========================================================================
    # Helpers
    # =========================================================================

    def _pd_cmd(self, cx: int) -> Twist:
        # simple pd pour se centrer sur qqch a l'ecran
        kp = self.get_parameter('ang_kp').value
        kd = self.get_parameter('ang_kd').value
        om = self.get_parameter('omega_max').value

        err_raw = (cx - self.image_width / 2.0) / self.image_width
        self.err_ang_filt = ALPHA_FILT * self.err_ang_filt + (1.0 - ALPHA_FILT) * err_raw
        derr = self.err_ang_filt - self.err_ang_prev
        self.err_ang_prev = self.err_ang_filt

        t = Twist()
        t.angular.z = float(np.clip(-(kp * self.err_ang_filt + kd * derr), -om, om))
        return t

    def _spin_cmd(self) -> Twist:
        # notre balayage radar a 180 degres (comme un essuie glace)
        omega = self.get_parameter('search_omega').value
        t = Twist()

        if self.yaw_initial is None:
            t.angular.z = omega
            return t

        deviation = normalize_angle(self.robot_yaw - self.yaw_initial)

        # on tape sur les bords = on repart dans l'autre sens
        if deviation >= SWEEP_HALF_ANGLE and self.sweep_direction > 0:
            self.sweep_direction = -1.0
        elif deviation <= -SWEEP_HALF_ANGLE and self.sweep_direction < 0:
            self.sweep_direction = 1.0

        t.angular.z = self.sweep_direction * omega
        return t

    def _reset_pd(self):
        self.err_ang_prev = 0.0
        self.err_ang_filt = 0.0

    def _reset_nav_pid(self):
        self.nav_err_prev     = 0.0
        self.nav_err_integral = 0.0
        self.nav_t_prev       = 0.0

    def _full_reset(self):
        # fonction de reset total en cas de bug
        self.ball_locked   = False
        self.ball_world_x  = None
        self.ball_world_y  = None
        self.goal_world_x  = None
        self.goal_world_y  = None
        self.waypoint_x    = None
        self.waypoint_y    = None
        self.stable_frames = 0
        self.blind_frames  = 0
        self.confirm_frames_count = 0
        self.sweep_direction = -1.0
        self._reset_pd()
        self._reset_nav_pid()
        self._transition(STATE_SEARCH_BALL)

    def _min_front_dist(self):
        # regarde juste devant pour pas s'emplafonner
        if self.laserscan is None:
            return None
        front = np.concatenate((self.laserscan[0:20], self.laserscan[340:360]))
        return float(np.min(front))

    def _obstacle_front(self) -> bool:
        d = self._min_front_dist()
        return d is not None and d < self.get_parameter('safety_dist').value

    def _transition(self, new_state: str):
        self.get_logger().info(f"Changement d'etat : {self.state} -> {new_state}")
        self.state = new_state

    # =========================================================================
    # Rendu debug (camera + minimap)
    # =========================================================================

    def _render_debug(self, frame: np.ndarray,
                      ball_data, posts_data: list) -> np.ndarray:
        # trace les carres et les infos sur le retour camera 
        debug = frame.copy()
        h_img, w_img = debug.shape[:2]

        cv2.line(debug, (w_img // 2, 0), (w_img // 2, h_img), (255, 120, 0), 1)

        if ball_data is not None:
            cx, cy = ball_data['cx'], ball_data['cy']
            bx, by, bw, bh = ball_data['bbox']
            cv2.drawContours(debug, [ball_data['contour']], -1, (0, 255, 0), 2)
            cv2.rectangle(debug, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
            cv2.drawMarker(debug, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.line(debug, (cx, cy), (w_img // 2, cy), (0, 180, 255), 1)
            cv2.putText(debug, f"Balle ({ball_data['area']:.0f}px) d~{self.ball_dist:.2f}m",
                        (bx, max(by - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        elif self.ball_locked:
            cv2.putText(debug, "BALLE lock (odom)",
                        (10, h_img - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2)
        elif not self.ball_locked and self.blind_frames > 0:
            cv2.putText(debug, f"BALLE pas la ({self.blind_frames})",
                        (10, h_img - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        if self.state == STATE_MEMORIZE_BALL:
            n_req = self.get_parameter('align_frames_ok').value
            pct   = int(self.stable_frames / max(n_req, 1) * 100)
            bar_w = int(w_img * 0.4 * pct / 100)
            cv2.rectangle(debug, (10, 55), (10 + int(w_img * 0.4), 70),
                          (50, 50, 50), -1)
            cv2.rectangle(debug, (10, 55), (10 + bar_w, 70), (0, 255, 0), -1)
            cv2.putText(debug, f"Load: {pct}%",
                        (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        for i, post in enumerate(posts_data):
            bx, by, bw, bh = post['bbox']
            cv2.rectangle(debug, (bx, by), (bx + bw, by + bh), (255, 0, 255), 2)
            cv2.drawMarker(debug, (post['cx'], post['cy']),
                           (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(debug, f"P{i} h/w={post['aspect']:.1f}",
                        (bx, max(by - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 0, 255), 1)

        if len(posts_data) >= 2:
            p0  = (posts_data[0]['cx'], posts_data[0]['cy'])
            p1  = (posts_data[1]['cx'], posts_data[1]['cy'])
            mid = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
            cv2.line(debug, p0, p1, (255, 0, 255), 1)
            cv2.circle(debug, mid, 6, (0, 255, 255), -1)
            cv2.putText(debug, "CAGE OK",
                        (mid[0] - 35, mid[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        state_colors = {
            STATE_SEARCH_BALL:          (0, 200, 255),
            STATE_MEMORIZE_BALL:        (0, 255, 100),
            STATE_SEARCH_GOAL:          (0, 165, 255),
            STATE_COMPUTE_TRAJECTORY:   (255, 200, 0),
            STATE_NAVIGATE_TO_WAYPOINT: (200, 100, 255),
            STATE_ALIGN_AND_PUSH:       (0, 0, 255),
        }
        cv2.putText(debug, self.state, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                    state_colors.get(self.state, (255, 255, 255)), 2)
        if self.ball_locked:
            cv2.putText(debug, "[LOCK]", (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 165, 255), 1)

        h_min = int(self.get_parameter('ball_h_min').value)
        h_max = int(self.get_parameter('ball_h_max').value)
        s_min = int(self.get_parameter('ball_s_min').value)
        v_min = int(self.get_parameter('ball_v_min').value)
        asp   = self.get_parameter('post_aspect_min').value
        odom_str = (f"odom ({self.robot_x:.2f},{self.robot_y:.2f}) "
                    f"yaw={math.degrees(self.robot_yaw):.0f}deg"
                    if self.odom_ok else "odom: WAIT")
        cv2.putText(debug,
                    f"HSV H[{h_min},{h_max}] S>={s_min} V>={v_min} | "
                    f"h/w>={asp:.1f} | {odom_str}",
                    (10, h_img - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1)
        ball_str = (f"ball=({self.ball_world_x:.2f},{self.ball_world_y:.2f})"
                    if self.ball_world_x is not None
                    else f"ball d={self.ball_dist:.2f}m a={math.degrees(self.ball_angle):.0f}d")
        goal_str = (f" goal=({self.goal_world_x:.2f},{self.goal_world_y:.2f})"
                    if self.goal_world_x is not None else "")
        wp_str   = (f" wp=({self.waypoint_x:.2f},{self.waypoint_y:.2f})"
                    if self.waypoint_x is not None else "")
        cv2.putText(debug, ball_str + goal_str + wp_str,
                    (10, h_img - 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1)

        debug = self._render_minimap(debug)

        return debug

    def _render_minimap(self, debug: np.ndarray) -> np.ndarray:
        # petite minimap radar pour voir si nos coordonnees odom font n'imp
        MAP_SIZE  = 160    
        MAP_SCALE = 0.04   
        cx_map    = MAP_SIZE // 2
        cy_map    = MAP_SIZE // 2

        overlay = debug.copy()
        cv2.rectangle(overlay, (0, 0), (MAP_SIZE, MAP_SIZE), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, debug, 0.25, 0, debug)

        for gx in range(0, MAP_SIZE, 50):
            cv2.line(debug, (gx, 0), (gx, MAP_SIZE), (40, 40, 40), 1)
        for gy in range(0, MAP_SIZE, 50):
            cv2.line(debug, (0, gy), (MAP_SIZE, gy), (40, 40, 40), 1)

        def w2m(wx: float, wy: float) -> tuple:
            # passe le repere map sur le petit radar
            dx = wx - self.robot_x   
            dy = wy - self.robot_y
            cos_y = math.cos(-self.robot_yaw)
            sin_y = math.sin(-self.robot_yaw)
            rx =  cos_y * dx - sin_y * dy
            ry =  sin_y * dx + cos_y * dy
            px = cx_map - int(ry / MAP_SCALE)   
            py = cy_map - int(rx / MAP_SCALE)   
            return px, py

        cv2.rectangle(debug, (0, 0), (MAP_SIZE - 1, MAP_SIZE - 1), (100, 100, 100), 1)

        wp_rad_px = int(self.get_parameter('waypoint_offset').value / MAP_SCALE)
        cv2.circle(debug, (cx_map, cy_map), wp_rad_px, (60, 60, 60), 1)

        fwd_px = cx_map
        fwd_py = cy_map - 20
        cv2.arrowedLine(debug, (cx_map, cy_map), (fwd_px, fwd_py),
                        (200, 200, 200), 1, tipLength=0.3)

        tri = np.array([[cx_map, cy_map - 8],
                        [cx_map - 5, cy_map + 5],
                        [cx_map + 5, cy_map + 5]], dtype=np.int32)
        cv2.fillPoly(debug, [tri], (220, 220, 220))

        if self.ball_world_x is not None:
            bpx, bpy = w2m(self.ball_world_x, self.ball_world_y)
            if 0 <= bpx < MAP_SIZE and 0 <= bpy < MAP_SIZE:
                cv2.circle(debug, (bpx, bpy), 5, (0, 220, 0), -1)
                cv2.putText(debug, "B", (bpx + 6, bpy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 220, 0), 1)

        if self.goal_world_x is not None:
            gpx, gpy = w2m(self.goal_world_x, self.goal_world_y)
            if 0 <= gpx < MAP_SIZE and 0 <= gpy < MAP_SIZE:
                diamond = np.array([[gpx, gpy - 6], [gpx + 5, gpy],
                                    [gpx, gpy + 6], [gpx - 5, gpy]], dtype=np.int32)
                cv2.polylines(debug, [diamond], True, (0, 255, 255), 1)
                cv2.putText(debug, "G", (gpx + 7, gpy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 255), 1)

        if self.waypoint_x is not None:
            wpx_px, wpy_px = w2m(self.waypoint_x, self.waypoint_y)
            if 0 <= wpx_px < MAP_SIZE and 0 <= wpy_px < MAP_SIZE:
                for angle_star in range(0, 360, 72):
                    a_rad = math.radians(angle_star)
                    ex = wpx_px + int(6 * math.cos(a_rad))
                    ey = wpy_px + int(6 * math.sin(a_rad))
                    cv2.line(debug, (wpx_px, wpy_px), (ex, ey), (0, 220, 220), 1)
                cv2.putText(debug, "W", (wpx_px + 7, wpy_px + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 220, 220), 1)

        if self.end_point_x is not None:
            epx_px, epy_px = w2m(self.end_point_x, self.end_point_y)
            if 0 <= epx_px < MAP_SIZE and 0 <= epy_px < MAP_SIZE:
                cv2.rectangle(debug,
                              (epx_px - 4, epy_px - 4),
                              (epx_px + 4, epy_px + 4),
                              (0, 140, 255), 1)   
                cv2.putText(debug, "E", (epx_px + 6, epy_px + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 140, 255), 1)

        if self.ball_world_x is not None and self.goal_world_x is not None:
            bpx, bpy = w2m(self.ball_world_x, self.ball_world_y)
            gpx, gpy = w2m(self.goal_world_x, self.goal_world_y)
            bpx_c = max(0, min(MAP_SIZE - 1, bpx))
            bpy_c = max(0, min(MAP_SIZE - 1, bpy))
            gpx_c = max(0, min(MAP_SIZE - 1, gpx))
            gpy_c = max(0, min(MAP_SIZE - 1, gpy))
            cv2.arrowedLine(debug, (bpx_c, bpy_c), (gpx_c, gpy_c),
                            (0, 0, 200), 1, tipLength=0.15)

        cv2.putText(debug, "B=balle G=cage W=wp E=end",
                    (2, MAP_SIZE - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (120, 120, 120), 1)

        return debug


    def _publish_cmd(self):
        self.pub_cmd.publish(self.cmd_pending)

def main(args=None):
    rclpy.init(args=args)
    node = Challenge4()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
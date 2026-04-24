import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

# =============================================================================
# MODE
# =============================================================================
MODE = "IRL"   # "IRL" ou "SIMULATION"

# =============================================================================
# PHASE 1 — OBSERVATION : paramètres de confirmation
# =============================================================================

# Nombre de frames consécutives où balle ET but doivent être détectés
# avant de figer la scène et passer en planification.
# Plus la valeur est haute, plus la mesure est stable mais lente.
OBS_CONFIRM_FRAMES = 15   # frames — ↑ = plus stable, ↓ = plus rapide

# Durée (en secondes) pendant laquelle la balle peut être absente en OBSERVE
# avant de retourner en SCAN. Évite qu'une seule frame de bruit relance la rotation.
BALL_LOST_TIMEOUT = 1.5   # s — ↑ = plus tolérant au bruit, ↓ = réagit plus vite

# Distance max de la balle acceptée en observation (évite les confusions lointaines)
OBS_BALL_MAX_DIST = 2.5   # m  (estimée depuis le rayon pixel + focale approx.)

# Vitesse de rotation pendant la recherche (SCAN)
SEARCH_OMEGA     = 0.40   # rad/s
SEARCH_MAX_ANGLE = math.radians(90)   # balayage ±90°

# =============================================================================
# PHASE 2 — PLANIFICATION : géométrie du waypoint
# =============================================================================

# Distance derrière la balle (sur l'axe But→Balle) où le robot doit se placer
# avant de pousser. Plus c'est grand, plus l'alignement est précis mais long.
APPROACH_DIST = 0.35   # m  — ↑ = plus de recul, ↓ = approche plus directe

# Rayon de la balle de tennis (~6.5 cm) — utilisé pour estimer la distance balle
# depuis le LIDAR (le LIDAR mesure la surface de la balle, pas son centre)
BALL_RADIUS_M = 0.033   # m  (rayon réel ≈ 3.3 cm)

# Distance focale estimée de la caméra TurtleBot3 (en pixels).
# Utilisée pour convertir le rayon pixel → distance métrique.
# f_px ≈ f_mm * (résolution_px / taille_capteur_mm)
# Valeur par défaut pour la caméra 160° grand-angle du TurtleBot3 :
# À calibrer si possible (ros2 run camera_calibration cameracalibrator)
CAMERA_FOCAL_PX = 280.0   # px — ↑ = caméra téléobjectif, ↓ = grand-angle

# Facteur de correction empirique de la distance balle.
# La formule optique seule (CAMERA_FOCAL_PX) est souvent imprécise IRL.
# Ce multiplicateur ajuste la distance estimée sans toucher au paramètre physique.
#   = 1.0 → pas de correction
#   > 1.0 → robot pense que la balle est plus loin (à augmenter si trop proche)
#   < 1.0 → robot pense que la balle est plus près
# Méthode : regarder la distance affichée en vert dans la fenêtre debug,
# mesurer la vraie distance avec un mètre, puis :
#   BALL_DIST_SCALE = distance_réelle / distance_affichée
BALL_DIST_SCALE = 1.0   # ← ajuster ici jusqu'à ce que l'affichage soit correct

# =============================================================================
# PHASE 3 — EXÉCUTION : contrôle de navigation
# =============================================================================

# --- Sous-phase GOTO_WAYPOINT : aller au point d'approche ---
# Tolérance de position pour considérer le waypoint atteint
GOTO_POS_TOL   = 0.08   # m  — ↑ = s'arrête plus loin du waypoint
# Gain proportionnel angulaire pendant la navigation vers le waypoint
GOTO_KP_ANG    = 1.8    # rad/s par rad d'erreur
# Vitesse de navigation (réduite si virage serré)
GOTO_SPEED_MAX = 0.12   # m/s
GOTO_SPEED_MIN = 0.04   # m/s

# --- Sous-phase FACE_GOAL : pivoter pour faire face au but ---
# Tolérance angulaire pour considérer l'alignement atteint
FACE_TOL = math.radians(5)   # rad — ↑ = alignement plus approximatif
# Gain proportionnel de rotation sur place
FACE_KP   = 1.5

# --- Sous-phase PUSH : avancer pour pousser la balle ---
# Distance totale à parcourir pendant la poussée
# = distance robot→balle estimée + marge pour traverser le but
PUSH_EXTRA_DIST = 0.30   # m  — ↑ = pousse plus loin après la balle
PUSH_SPEED      = 0.14   # m/s — vitesse de poussée (ne pas dépasser 0.20)
# Sécurité : si le LIDAR voit un obstacle inattendu à moins de X m, on stoppe
PUSH_STOP_DIST  = 0.08   # m

# =============================================================================
# PARAMÈTRES BALLE — détection vision
# =============================================================================
BALL_HSV = dict(H_min=22, H_max=48, S_min=60, S_max=255, V_min=100, V_max=255)

BALL_CIRCULARITY_MIN = 0.65
BALL_RADIUS_MIN      = 10    # px
BALL_RADIUS_MAX      = 150   # px
# Pas de ROI : on analyse l'image entière pour ne pas rater la balle

# =============================================================================
# PARAMÈTRES CAGE — détection LIDAR
# =============================================================================
GOAL_HALF_ANGLE     = math.radians(90)
GOAL_GAP_MIN        = 0.35   # m
GOAL_GAP_MAX        = 0.45   # m
GOAL_POST_MAX_WIDTH = 0.10   # m
GOAL_MAX_DIST       = 2.5    # m
GOAL_POST_MIN_PTS   = 2
GOAL_POST_MAX_PTS   = 12
LIDAR_CLUSTER_GAP   = 0.10   # m
LIDAR_MIN_VALID     = 0.12   # m — filtre les 0.0 du firmware TurtleBot3

# =============================================================================
# ÉTATS DE LA MACHINE
# =============================================================================
# Phase 1 — Observation
STATE_SCAN    = "SCAN"       # tourne pour trouver balle + but
STATE_OBSERVE = "OBSERVE"    # accumule les mesures pour les stabiliser

# Phase 2 — Planification (instantanée, pas d'état dédié — transition directe)

# Phase 3 — Exécution
STATE_GOTO    = "GOTO"       # navigue vers le waypoint d'approche
STATE_FACE    = "FACE"       # pivote pour faire face au but
STATE_PUSH    = "PUSH"       # avance et pousse la balle
STATE_DONE    = "DONE"       # tir terminé

WIN_DEBUG = "Challenge 4 — Debug"
WIN_TUNER = "Challenge 4 — Tuner Balle HSV"

def _null(_): pass

def _angle_diff(a, b):
    """Différence angulaire normalisée dans [-π, π]."""
    d = a - b
    while d >  math.pi: d -= 2 * math.pi
    while d < -math.pi: d += 2 * math.pi
    return d

def _yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


# =============================================================================
# NŒUD PRINCIPAL
# =============================================================================

class Challenge4(Node):

    def __init__(self):
        super().__init__('challenge4')

        # ── Publications ──────────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_challenge_4', 10)
        self.bridge  = CvBridge()

        # ── Abonnements ───────────────────────────────────────────────────────
        if MODE == "IRL":
            self.create_subscription(
                CompressedImage, '/camera/image_raw/compressed',
                self._img_cb, 10)
        else:
            self.create_subscription(Image, '/image_raw', self._img_cb_raw, 10)

        self.create_subscription(LaserScan, '/scan',  self._scan_cb, 10)
        self.create_subscription(Odometry,  '/odom',  self._odom_cb, 10)

        # ── Données capteurs brutes ───────────────────────────────────────────
        # Vision balle
        self.ball_cx     = -1
        self.ball_cy     = -1
        self.ball_radius = 0
        self.img_w       = 640
        self.img_h       = 480

        # LIDAR cage
        self.goal_angle    = None    # angle vers le milieu du but (rad, ROS: + = gauche)
        self.goal_dist     = None    # distance estimée jusqu'au milieu du but (m)
        self.goal_detected = False

        # LIDAR distance frontale
        self.front_dist = 9.0

        # Odométrie
        self.odom_x   = None   # position x (m)
        self.odom_y   = None   # position y (m)
        self.odom_yaw = None   # cap (rad)

        # ── Phase 1 : buffers d'observation ───────────────────────────────────
        # On accumule les mesures de balle (angle vision) et de but (angle LIDAR)
        # sur OBS_CONFIRM_FRAMES frames consécutives valides.
        self._obs_ball_angles = []   # angles balle dans image (rad depuis axe optique)
        self._obs_ball_dists  = []   # distances balle estimées (m)
        self._obs_goal_angles = []   # angles but LIDAR (rad)
        self._obs_goal_dists  = []   # distances but LIDAR (m)
        self._obs_frames      = 0    # compteur de frames valides consécutives

        # ── Phase 2 : plan calculé ────────────────────────────────────────────
        # Toutes les coordonnées sont dans le repère MONDE (frame odom)
        self.plan_ball_x    = None   # position estimée balle (m, monde)
        self.plan_ball_y    = None
        self.plan_goal_x    = None   # position estimée but (m, monde)
        self.plan_goal_y    = None
        self.plan_wp_x      = None   # waypoint d'approche (m, monde)
        self.plan_wp_y      = None
        self.plan_face_yaw  = None   # cap à tenir pendant la poussée (rad, monde)
        self.plan_push_dist = None   # distance totale à parcourir en poussée (m)

        # ── Phase 3 : état d'exécution ────────────────────────────────────────
        self._push_start_x  = None   # position x au début de la poussée
        self._push_start_y  = None   # position y au début de la poussée
        self._push_dist_done = 0.0   # distance parcourue depuis le début de la poussée

        # ── Machine à états ───────────────────────────────────────────────────
        self.state      = STATE_SCAN
        self._scan_dir  = 1.0
        self._scan_cum  = 0.0

        # Horodatage de la dernière frame où la balle était visible.
        # En OBSERVE, on ne retourne en SCAN que si la balle est absente
        # pendant plus de BALL_LOST_TIMEOUT secondes consécutives.
        # Cela évite qu'une seule frame manquante relance la rotation.
        self._last_ball_seen = None   # initialisé au premier tick

        # ── Fenêtres OpenCV ───────────────────────────────────────────────────
        cv2.namedWindow(WIN_DEBUG, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_DEBUG, 1000, 480)
        cv2.namedWindow(WIN_TUNER, cv2.WINDOW_NORMAL)
        for name, val, mx in [
            ('H_min', BALL_HSV['H_min'], 179), ('H_max', BALL_HSV['H_max'], 179),
            ('S_min', BALL_HSV['S_min'], 255), ('S_max', BALL_HSV['S_max'], 255),
            ('V_min', BALL_HSV['V_min'], 255), ('V_max', BALL_HSV['V_max'], 255),
        ]:
            cv2.createTrackbar(name, WIN_TUNER, val, mx, _null)

        self.create_timer(0.05, self._control_loop)
        self.get_logger().info(f"Challenge4 — Observer/Planifier/Exécuter [MODE={MODE}]")

    # =========================================================================
    # CALLBACKS CAPTEURS
    # =========================================================================

    def _img_cb(self, msg):
        self._process(self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8'))

    def _img_cb_raw(self, msg):
        self._process(self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8'))

    def _odom_cb(self, msg):
        p = msg.pose.pose
        self.odom_x   = p.position.x
        self.odom_y   = p.position.y
        self.odom_yaw = _yaw_from_quat(p.orientation)

    def _scan_cb(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        invalid = (~np.isfinite(ranges)) | (ranges < LIDAR_MIN_VALID)
        ranges[invalid] = 9.0

        front = np.concatenate((ranges[:30], ranges[-30:]))
        self.front_dist = float(np.min(front))

        angle, dist, detected = self._find_goal(ranges, msg.angle_min, msg.angle_increment)
        self.goal_angle    = angle
        self.goal_dist     = dist
        self.goal_detected = detected

    # =========================================================================
    # TRAITEMENT IMAGE
    # =========================================================================

    def _process(self, frame):
        h, w = frame.shape[:2]
        self.img_w, self.img_h = w, h

        lo = np.array([cv2.getTrackbarPos('H_min', WIN_TUNER),
                       cv2.getTrackbarPos('S_min', WIN_TUNER),
                       cv2.getTrackbarPos('V_min', WIN_TUNER)], dtype=np.uint8)
        hi = np.array([cv2.getTrackbarPos('H_max', WIN_TUNER),
                       cv2.getTrackbarPos('S_max', WIN_TUNER),
                       cv2.getTrackbarPos('V_max', WIN_TUNER)], dtype=np.uint8)

        # Image entière — pas de ROI restrictive
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask  = cv2.inRange(hsv, lo, hi)
        ker   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  ker)
        mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)

        cx, cy, radius = self._detect_ball(mask)
        if cx != -1:
            self.ball_cx     = cx
            self.ball_cy     = cy   # pas de décalage ROI
            self.ball_radius = radius
            self._last_ball_seen = self.get_clock().now()   # horodatage dernière détection
        else:
            self.ball_cx = self.ball_cy = -1
            self.ball_radius = 0

        self._draw_debug(frame, mask)

    # =========================================================================
    # DÉTECTION BALLE PAR CIRCULARITÉ
    # =========================================================================

    def _detect_ball(self, mask):
        """C = 4π·A/P²  → retourne (cx, cy, rayon) dans le repère de mask, ou (-1,-1,0)"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_r, best = -1, (-1, -1, 0)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < math.pi * BALL_RADIUS_MIN**2:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri < 1.0:
                continue
            if 12.566 * area / (peri * peri) < BALL_CIRCULARITY_MIN:
                continue
            _, radius = cv2.minEnclosingCircle(cnt)
            radius = int(radius)
            if not (BALL_RADIUS_MIN <= radius <= BALL_RADIUS_MAX):
                continue
            if radius > best_r:
                best_r = radius
                M  = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00']) if M['m00'] > 0 else int(cnt[0][0][0])
                cy = int(M['m01'] / M['m00']) if M['m00'] > 0 else int(cnt[0][0][1])
                best = (cx, cy, radius)
        return best

    def _ball_pixel_to_angle_dist(self):
        """
        Convertit la détection pixel de la balle en (angle_rad, distance_m)
        dans le repère robot.

        Angle horizontal :
            angle = atan2( (cx - w/2) , f_px )
            Positif = balle à droite de l'axe optique.
            Convention ROS : on l'inverse → positif = gauche.

        Distance estimée :
            d = (BALL_RADIUS_M * CAMERA_FOCAL_PX) / ball_radius_px
            C'est la formule de la taille angulaire apparente (thin-lens).
            Précision ±20 % sans calibration — suffisant pour le waypoint.
        """
        if self.ball_cx == -1 or self.ball_radius == 0:
            return None, None

        cx_centered = self.ball_cx - self.img_w / 2.0
        # Angle depuis l'axe optique (positif = droite dans l'image = droite du robot)
        # On l'inverse pour la convention ROS (positif = gauche)
        angle = -math.atan2(cx_centered, CAMERA_FOCAL_PX)

        # Distance par similitude de triangles, corrigée par BALL_DIST_SCALE
        dist = (BALL_RADIUS_M * CAMERA_FOCAL_PX) / max(self.ball_radius, 1)
        dist = dist * BALL_DIST_SCALE          # ← correction empirique terrain
        dist = min(dist, OBS_BALL_MAX_DIST)

        return angle, dist

    # =========================================================================
    # DÉTECTION CAGE PAR LIDAR — retourne aussi la distance
    # =========================================================================

    def _find_goal(self, ranges, angle_min, angle_inc):
        """
        Cherche deux pieds de chaise dans le demi-cercle frontal.
        Retourne (angle_milieu, distance_milieu, True) ou (None, None, False).
        """
        pts = []
        for i, r in enumerate(ranges):
            if r > GOAL_MAX_DIST:
                continue
            angle = angle_min + i * angle_inc
            angle = (angle + math.pi) % (2 * math.pi) - math.pi
            if abs(angle) > GOAL_HALF_ANGLE:
                continue
            pts.append((angle, r, r * math.cos(angle), r * math.sin(angle)))

        if len(pts) < 4:
            return None, None, False

        pts.sort(key=lambda p: p[0])
        clusters, cur = [], [pts[0]]
        for i in range(1, len(pts)):
            prev, curr = cur[-1], pts[i]
            da = curr[0] - prev[0]
            d  = math.sqrt(prev[1]**2 + curr[1]**2 - 2*prev[1]*curr[1]*math.cos(da))
            if d > LIDAR_CLUSTER_GAP:
                if len(cur) >= GOAL_POST_MIN_PTS:
                    clusters.append(cur)
                cur = [curr]
            else:
                cur.append(curr)
        if len(cur) >= GOAL_POST_MIN_PTS:
            clusters.append(cur)

        if len(clusters) < 2:
            return None, None, False

        def centroid(cl):
            xs = [p[2] for p in cl]
            ys = [p[3] for p in cl]
            cx, cy = sum(xs)/len(xs), sum(ys)/len(ys)
            width  = math.sqrt((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2)
            return cx, cy, width

        posts = []
        for cl in clusters:
            if GOAL_POST_MIN_PTS <= len(cl) <= GOAL_POST_MAX_PTS:
                cx, cy, w = centroid(cl)
                if w <= GOAL_POST_MAX_WIDTH:
                    posts.append((cx, cy))

        if len(posts) < 2:
            return None, None, False

        best_angle, best_dist, best_score = None, None, float('inf')
        for i in range(len(posts)):
            for j in range(i+1, len(posts)):
                cx1, cy1 = posts[i]
                cx2, cy2 = posts[j]
                gap = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                if not (GOAL_GAP_MIN <= gap <= GOAL_GAP_MAX):
                    continue
                mid_x = (cx1+cx2) / 2.0
                mid_y = (cy1+cy2) / 2.0
                mid_angle = math.atan2(mid_y, mid_x)
                mid_dist  = math.sqrt(mid_x**2 + mid_y**2)
                if abs(mid_angle) < best_score:
                    best_score = abs(mid_angle)
                    best_angle = mid_angle
                    best_dist  = mid_dist

        if best_angle is None:
            return None, None, False
        return best_angle, best_dist, True

    # =========================================================================
    # PHASE 2 : CALCUL DU PLAN GÉOMÉTRIQUE
    # =========================================================================

    def _compute_plan(self):
        """
        Calcule le waypoint d'approche et le cap de poussée à partir des
        positions moyennes observées de la balle et du but.

        Repère de travail : MONDE (frame odom), avec :
            x_monde = odom_x + dist * cos(odom_yaw + angle_relatif)
            y_monde = odom_y + dist * sin(odom_yaw + angle_relatif)

        Schéma :
                        [BUT]
                          ↑
                    vecteur de poussée
                          |
                        [BALLE]
                          |
                    ← APPROACH_DIST →
                          |
                      [WAYPOINT]   ← robot s'y rend d'abord

        Calcul du waypoint :
            vec_but_balle = balle - but   (vecteur unitaire)
            waypoint = balle + vec_but_balle * APPROACH_DIST

        Le robot, depuis le waypoint, fait face au but en calculant :
            cap_poussée = atan2(but_y - wp_y, but_x - wp_x)
        """
        if self.odom_x is None:
            return False

        # Moyennes des observations
        ball_angle_mean = sum(self._obs_ball_angles) / len(self._obs_ball_angles)
        ball_dist_mean  = sum(self._obs_ball_dists)  / len(self._obs_ball_dists)
        goal_angle_mean = sum(self._obs_goal_angles) / len(self._obs_goal_angles)
        goal_dist_mean  = sum(self._obs_goal_dists)  / len(self._obs_goal_dists)

        rob_x   = self.odom_x
        rob_y   = self.odom_y
        rob_yaw = self.odom_yaw

        # Conversion polaire → cartésien MONDE
        # angle_relatif est dans le repère robot (convention ROS : + = gauche)
        # → dans le repère monde : x_monde += dist*cos(yaw + angle), y += dist*sin(...)
        ball_world_x = rob_x + ball_dist_mean * math.cos(rob_yaw + ball_angle_mean)
        ball_world_y = rob_y + ball_dist_mean * math.sin(rob_yaw + ball_angle_mean)

        goal_world_x = rob_x + goal_dist_mean * math.cos(rob_yaw + goal_angle_mean)
        goal_world_y = rob_y + goal_dist_mean * math.sin(rob_yaw + goal_angle_mean)

        # Vecteur unitaire du but vers la balle (axe de poussée)
        dx = ball_world_x - goal_world_x
        dy = ball_world_y - goal_world_y
        norm = math.sqrt(dx**2 + dy**2)
        if norm < 0.05:
            self.get_logger().warn("Plan: but et balle trop proches, abandon.")
            return False

        ux = dx / norm   # composante unitaire x de l'axe But→Balle
        uy = dy / norm   # composante unitaire y

        # Waypoint = balle + vecteur unitaire * APPROACH_DIST
        # (on recule APPROACH_DIST derrière la balle, dans la direction opposée au but)
        wp_x = ball_world_x + ux * APPROACH_DIST
        wp_y = ball_world_y + uy * APPROACH_DIST

        # Cap de poussée = direction du waypoint vers le but
        # (= direction opposée au vecteur unitaire, soit l'angle de -ux, -uy)
        face_yaw = math.atan2(-uy, -ux)

        # Distance de poussée = distance waypoint→balle + marge supplémentaire
        # = APPROACH_DIST (par construction) + PUSH_EXTRA_DIST
        push_dist = APPROACH_DIST + PUSH_EXTRA_DIST

        # Enregistrement du plan
        self.plan_ball_x   = ball_world_x
        self.plan_ball_y   = ball_world_y
        self.plan_goal_x   = goal_world_x
        self.plan_goal_y   = goal_world_y
        self.plan_wp_x     = wp_x
        self.plan_wp_y     = wp_y
        self.plan_face_yaw = face_yaw
        self.plan_push_dist = push_dist

        self.get_logger().info(
            f"[PLAN] balle=({ball_world_x:.2f},{ball_world_y:.2f})m  "
            f"but=({goal_world_x:.2f},{goal_world_y:.2f})m  "
            f"wp=({wp_x:.2f},{wp_y:.2f})m  "
            f"cap={math.degrees(face_yaw):.1f}°  "
            f"pousse={push_dist:.2f}m"
        )
        return True

    # =========================================================================
    # FENÊTRE DE DEBUG
    # =========================================================================

    def _draw_debug(self, frame, mask):
        h, w = frame.shape[:2]
        debug = frame.copy()

        cv2.line(debug, (w//2, 0), (w//2, h), (80, 80, 80), 1)

        # Balle
        if self.ball_cx != -1:
            cv2.circle(debug, (self.ball_cx, self.ball_cy), self.ball_radius,
                       (0, 255, 80), 2)
            cv2.drawMarker(debug, (self.ball_cx, self.ball_cy),
                           (0, 255, 80), cv2.MARKER_CROSS, 18, 2)
            # Affichage distance estimée
            _, bd = self._ball_pixel_to_angle_dist()
            if bd:
                cv2.putText(debug, f"{bd:.2f}m",
                            (self.ball_cx + self.ball_radius + 4, self.ball_cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 80), 1)

        # But LIDAR
        if self.goal_detected and self.goal_angle is not None:
            L  = 65
            ax = int(w/2 - math.sin(self.goal_angle) * L)
            ay = int(h - 40 - math.cos(self.goal_angle) * L)
            ax, ay = max(10, min(w-10, ax)), max(10, min(h-10, ay))
            cv2.arrowedLine(debug, (w//2, h-40), (ax, ay), (0, 100, 255), 3, tipLength=0.3)
            gd_str = f"{self.goal_dist:.2f}m" if self.goal_dist else "?"
            cv2.putText(debug, f"BUT {math.degrees(self.goal_angle):.0f}d {gd_str}",
                        (ax+4, ay), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1)

        # Barre de progression de l'observation
        if self.state == STATE_OBSERVE and OBS_CONFIRM_FRAMES > 0:
            pct   = self._obs_frames / OBS_CONFIRM_FRAMES
            bar_w = int(w * pct)
            cv2.rectangle(debug, (0, h-10), (bar_w, h), (0, 200, 255), -1)
            cv2.putText(debug, f"OBS {self._obs_frames}/{OBS_CONFIRM_FRAMES}",
                        (4, h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        # Plan calculé (waypoint + cap)
        if self.plan_wp_x is not None and self.odom_x is not None:
            cv2.putText(debug,
                        f"WP=({self.plan_wp_x:.2f},{self.plan_wp_y:.2f})  "
                        f"cap={math.degrees(self.plan_face_yaw):.0f}d",
                        (4, h-28), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 0), 1)

        # État + dist frontale
        cv2.putText(debug, f"{self.state}  front={self.front_dist:.2f}m",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2)

        # Masque pleine image colorisé en vert
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_vis[mask > 0] = (0, 200, 80)

        cv2.imshow(WIN_DEBUG, np.hstack([debug, mask_vis]))
        cv2.waitKey(1)

    # =========================================================================
    # BOUCLE DE CONTRÔLE (20 Hz)
    # =========================================================================

    def _control_loop(self):
        if self.odom_x is None:
            return   # attendre la première mesure odométrique

        twist = Twist()

        # ─── PHASE 1a : SCAN ─────────────────────────────────────────────────
        # Le robot tourne UNIQUEMENT pour trouver la balle.
        # Dès qu'elle est visible (même une seule frame), il s'arrête
        # et passe en OBSERVE pour accumuler les mesures immobile.
        # Le but (LIDAR) sera attendu pendant OBSERVE, pas ici.

        if self.state == STATE_SCAN:
            if self.ball_cx != -1:
                # Balle trouvée → s'arrêter immédiatement, initialiser les buffers
                self._obs_frames = 0
                self._obs_ball_angles.clear()
                self._obs_ball_dists.clear()
                self._obs_goal_angles.clear()
                self._obs_goal_dists.clear()
                self._transition(STATE_OBSERVE)
                # twist reste à zéro → arrêt immédiat
            else:
                # Balayage oscillant ±SEARCH_MAX_ANGLE
                self._scan_cum += 0.05 * SEARCH_OMEGA
                if self._scan_cum > SEARCH_MAX_ANGLE:
                    self._scan_cum = 0.0
                    self._scan_dir = -self._scan_dir
                twist.angular.z = SEARCH_OMEGA * self._scan_dir

        # ─── PHASE 1b : OBSERVE ──────────────────────────────────────────────
        # Le robot est IMMOBILE. Il attend balle + but pour accumuler
        # OBS_CONFIRM_FRAMES mesures stables.
        #
        # Règle clé : on ne retourne en SCAN que si la balle est VRAIMENT
        # perdue (absente depuis > BALL_LOST_TIMEOUT secondes).
        # Une ou deux frames manquantes = bruit normal → on ignore et on attend.

        elif self.state == STATE_OBSERVE:
            now = self.get_clock().now()

            # Initialiser le timer au premier tick de OBSERVE
            if self._last_ball_seen is None:
                self._last_ball_seen = now

            ball_angle, ball_dist = self._ball_pixel_to_angle_dist()

            # Mise à jour de l'horodatage si la balle est visible cette frame
            if ball_angle is not None:
                self._last_ball_seen = now

            # Calcul du temps écoulé sans balle
            age_ball = (now - self._last_ball_seen).nanoseconds / 1e9

            if age_ball > BALL_LOST_TIMEOUT:
                # Balle vraiment perdue (pas juste une frame de bruit) → SCAN
                self.get_logger().warn(
                    f"[OBSERVE] Balle perdue depuis {age_ball:.1f}s — retour SCAN")
                self._obs_frames = 0
                self._obs_ball_angles.clear()
                self._obs_ball_dists.clear()
                self._obs_goal_angles.clear()
                self._obs_goal_dists.clear()
                self._transition(STATE_SCAN)

            elif ball_angle is not None and self.goal_detected \
                    and self.goal_angle is not None and self.goal_dist is not None:
                # Balle ET but visibles cette frame → accumulation
                self._obs_ball_angles.append(ball_angle)
                self._obs_ball_dists.append(ball_dist)
                self._obs_goal_angles.append(self.goal_angle)
                self._obs_goal_dists.append(self.goal_dist)
                self._obs_frames += 1

                if self._obs_frames >= OBS_CONFIRM_FRAMES:
                    # ── PHASE 2 : PLANIFICATION ───────────────────────────────
                    if self._compute_plan():
                        self._transition(STATE_GOTO)
                    else:
                        self._obs_frames = 0
                        self._obs_ball_angles.clear()
                        self._obs_ball_dists.clear()
                        self._obs_goal_angles.clear()
                        self._obs_goal_dists.clear()
                        self._transition(STATE_SCAN)
            # else : balle ou but absents mais dans le délai toléré → on attend
            # (twist reste à zéro → robot immobile)

        # ─── PHASE 3a : GOTO ─────────────────────────────────────────────────
        # Navigation vers le waypoint d'approche (en coordonnées monde).
        #
        # À chaque tick :
        #   1. Calculer le vecteur (wp - robot) dans le repère monde
        #   2. En déduire l'angle cible dans le repère robot : atan2(dy, dx) - yaw
        #   3. Commande proportionnelle : omega = KP * angle_err
        #                                  v    = proportionnel à cos(angle_err)

        elif self.state == STATE_GOTO:
            dx = self.plan_wp_x - self.odom_x
            dy = self.plan_wp_y - self.odom_y
            dist_to_wp = math.sqrt(dx**2 + dy**2)

            if dist_to_wp < GOTO_POS_TOL:
                self._transition(STATE_FACE)
            else:
                # Angle vers le waypoint dans le repère monde
                target_yaw = math.atan2(dy, dx)
                # Erreur angulaire dans le repère robot
                angle_err  = _angle_diff(target_yaw, self.odom_yaw)

                twist.angular.z = GOTO_KP_ANG * angle_err

                # Vitesse : réduite si virage serré (cos passe de 1 à 0 pour 90°)
                # On avance moins vite quand on est très décalé angulairement
                fwd = math.cos(angle_err)    # [-1, 1]
                fwd = max(0.0, fwd)          # on n'avance pas à reculons
                twist.linear.x = GOTO_SPEED_MIN + (GOTO_SPEED_MAX - GOTO_SPEED_MIN) * fwd

        # ─── PHASE 3b : FACE ─────────────────────────────────────────────────
        # Rotation sur place pour s'aligner sur le cap de poussée.
        # Le robot est au waypoint, il pivote jusqu'à faire face au but.

        elif self.state == STATE_FACE:
            angle_err = _angle_diff(self.plan_face_yaw, self.odom_yaw)

            if abs(angle_err) < FACE_TOL:
                # Aligné — mémoriser la position de départ de la poussée
                self._push_start_x    = self.odom_x
                self._push_start_y    = self.odom_y
                self._push_dist_done  = 0.0
                self._transition(STATE_PUSH)
            else:
                twist.angular.z = FACE_KP * angle_err
                twist.linear.x  = 0.0

        # ─── PHASE 3c : PUSH ─────────────────────────────────────────────────
        # Avance en ligne droite sur plan_push_dist.
        # Correction angulaire douce pour rester sur le cap.
        # Arrêt dès que la distance cible est parcourue, ou obstacle trop proche.

        elif self.state == STATE_PUSH:
            # Distance parcourue depuis le début de la poussée (odométrie)
            ddx = self.odom_x - self._push_start_x
            ddy = self.odom_y - self._push_start_y
            self._push_dist_done = math.sqrt(ddx**2 + ddy**2)

            if self._push_dist_done >= self.plan_push_dist:
                self.get_logger().info(
                    f"[PUSH] Distance atteinte ({self._push_dist_done:.2f}m) — DONE")
                self._transition(STATE_DONE)
            elif self.front_dist < PUSH_STOP_DIST:
                self.get_logger().warn(
                    f"[PUSH] Obstacle frontal à {self.front_dist:.2f}m — DONE")
                self._transition(STATE_DONE)
            else:
                # Correction angulaire pour rester sur le cap
                angle_err       = _angle_diff(self.plan_face_yaw, self.odom_yaw)
                twist.angular.z = FACE_KP * angle_err
                twist.linear.x  = PUSH_SPEED

        # ─── DONE ────────────────────────────────────────────────────────────
        elif self.state == STATE_DONE:
            pass   # twist reste à zéro → robot immobile

        self.cmd_pub.publish(twist)

    # =========================================================================
    # UTILITAIRE : transition d'état avec log
    # =========================================================================

    def _transition(self, new_state):
        if new_state != self.state:
            self.get_logger().info(f"[ÉTAT] {self.state} → {new_state}")
            self.state = new_state


# =============================================================================
# MAIN
# =============================================================================

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
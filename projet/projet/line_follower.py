import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Int32MultiArray
from geometry_msgs.msg import Twist
import numpy as np

# PID Suiveur de ligne : on lit les topics du detecteur pour avoir les positions (cx) des lignes

# Poids des bandes : indice 0 = tout en haut (loin), N-1 = tout en bas (pres)
BAND_WEIGHTS = [0.05, 0.10, 0.20, 0.30, 0.35]

# a partir de combien de bandes valides on tente une regression lineaire
POLY_MIN_POINTS = 3


class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower_node')

        # memoire des lignes (rempli par les callbacks avec des tableaux)
        # -1 ca veut dire qu'on a rien vu dans cette bande
        self.cx_red_bands   = []
        self.cx_green_bands = []

        self.image_width = 640.0

        # params du PID
        self.kp = 1.5
        self.kd = 6.0
        self.base_speed = 0.10
        self.speed_min  = 0.05

        # filtre pour lisser l'erreur et eviter que le robot tremble
        # alpha grand = beaucoup d'inertie
        self.alpha        = 0.80
        self.filtered_err = 0.0
        self.last_error   = 0.0

        # l'ecart en pixels entre la ligne et le centre du robot
        # on le met a jour en direct quand on voit les deux lignes
        self.offset_dynamique = 75.0 

        # si on perd la piste on garde notre dernier omega valide et on le diminue doucement (decay)
        self.last_omega_valid = 0.0
        self.inertia_speed    = 0.05
        self.inertia_decay    = 0.85

        # Abonnements aux topics de la vision
        self.create_subscription(Int32MultiArray, '/red_bands_cx',   self.cb_red_bands,   10)
        self.create_subscription(Int32MultiArray, '/green_bands_cx', self.cb_green_bands, 10)
        self.create_subscription(Int32,           '/camera_width',   self.cb_cam_width,   10)

        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel_line_raw', 10)
        
        # on fait tourner la boucle de controle a 20Hz
        self.create_timer(0.05, self.compute_and_publish)

        self.get_logger().info(
            "LineFollower demarre (mode PID multicouche)"
        )

    # lectures des topics
    def cb_red_bands(self,   msg): self.cx_red_bands   = list(msg.data)
    def cb_green_bands(self, msg): self.cx_green_bands = list(msg.data)
    def cb_cam_width(self,   msg): self.image_width     = float(msg.data)


    def centre_voie_bande(self, cx_r, cx_g, safety_margin):
        """
        Trouve le milieu de la piste sur une bande precise.
        Si on voit les deux lignes c'est parfait on fait la moyenne.
        Si on en voit qu'une, on utilise notre offset dynamique pour deviner ou est l'autre.
        """
        if cx_r != -1 and cx_g != -1:
            # on ajuste notre offset vu qu'on a la vraie info
            self.offset_dynamique = 0.90 * self.offset_dynamique + 0.10 * abs(cx_r - cx_g) / 2.0

            centre = (cx_r + cx_g) / 2.0
            
            # on verifie qu'on est pas en dehors des limites
            lo = cx_g + safety_margin
            hi = cx_r - safety_margin
            if lo < hi:
                centre = max(lo, min(hi, centre))
            return centre

        elif cx_r != -1:
            # on a que la ligne rouge (droite)
            centre = cx_r - self.offset_dynamique
            centre = max(safety_margin, min(centre, self.image_width - safety_margin))
            return centre

        elif cx_g != -1:
            # on a que la verte (gauche)
            centre = cx_g + self.offset_dynamique
            centre = max(safety_margin, min(centre, self.image_width - safety_margin))
            return centre

        return None


    def calcul_centre_pondere(self, safety_margin):
        """
        Mixte de toutes les bandes pour avoir un seul point cible.
        Si on voit assez de bouts de ligne (>= POLY_MIN_POINTS), on trace une droite 
        avec polyfit pour lisser le bruit de la camera et anticiper.
        Sinon on fait juste une moyenne ponderee avec les poids definis en haut.
        """
        n = len(self.cx_red_bands)
        if n == 0:
            return None

        # securite si jamais on change la taille du tableau en cours de route
        weights = BAND_WEIGHTS[:n]
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        centres_par_bande = []

        # on calcule le centre pour chaque bande qu'on a recu
        for i in range(n):
            cx_r = self.cx_red_bands[i]   if i < len(self.cx_red_bands)   else -1
            cx_g = self.cx_green_bands[i] if i < len(self.cx_green_bands) else -1

            c = self.centre_voie_bande(cx_r, cx_g, safety_margin)
            if c is not None:
                centres_par_bande.append((i, c))

        if not centres_par_bande:
            return None

        # Si on a assez de points on fait la regression lineaire
        if len(centres_par_bande) >= POLY_MIN_POINTS:
            indices  = np.array([p[0] for p in centres_par_bande], dtype=float)
            centres  = np.array([p[1] for p in centres_par_bande], dtype=float)

            # on recupere l'equation de la droite (degre 1)
            coeffs = np.polyfit(indices, centres, 1)
            poly   = np.poly1d(coeffs)

            # on extrapole la ou le robot devrait aller
            centre_proche = poly(n - 1)          
            centre_milieu = poly(n * 0.5)        

            # on mixe (70% pour le present, 30% d'anticipation)
            centre = 0.70 * centre_proche + 0.30 * centre_milieu

            centre = max(safety_margin, min(self.image_width - safety_margin, centre))
            return centre

        # Plan B : simple moyenne ponderee si le polyfit marche pas
        total_weight = 0.0
        total_centre = 0.0
        for (i, c) in centres_par_bande:
            w = weights[i]
            total_centre += w * c
            total_weight += w

        return total_centre / total_weight if total_weight > 0 else None


    def compute_and_publish(self):
        if self.image_width == 0.0:
            return

        twist = Twist()

        image_center  = self.image_width / 2.0
        safety_margin = 0.08 * self.image_width
        marge_strict  = 0.40 * self.image_width

        centre = self.calcul_centre_pondere(safety_margin)

        # si on perd completement la piste, on active l'inertie
        # on garde notre elan en tournant de moins en moins fort
        if centre is None:
            self.last_omega_valid *= self.inertia_decay
            twist.linear.x  = self.inertia_speed
            twist.angular.z = self.last_omega_valid
            self.pub_cmd.publish(twist)
            return

        # calcul de l'erreur pour le correcteur
        erreur_pixels = image_center - centre
        erreur_brute  = erreur_pixels / self.image_width

        # on lisse un peu l'erreur
        self.filtered_err = (self.alpha * self.filtered_err
                             + (1.0 - self.alpha) * erreur_brute)
        erreur = self.filtered_err

        # calcul de la derivee
        derivation  = erreur - self.last_error
        self.last_error = erreur

        omega   = float(self.kp * erreur + self.kd * derivation)
        vitesse = self.base_speed

        # URGENCE : on regarde la toute derniere bande en bas de l'ecran
        # si on est sur le point d'ecraser une ligne, on braque a fond
        cx_r_near = self.cx_red_bands[-1]   if self.cx_red_bands   else -1
        cx_g_near = self.cx_green_bands[-1] if self.cx_green_bands else -1

        seuil_vert  = image_center - marge_strict
        seuil_rouge = image_center + marge_strict

        mord_verte = (cx_g_near != -1) and (cx_g_near > seuil_vert)
        mord_rouge = (cx_r_near != -1) and (cx_r_near < seuil_rouge)

        if mord_verte:
            vitesse = 0.05
            omega   = -1.0   # on braque a droite
        elif mord_rouge:
            vitesse = 0.05
            omega   = 1.0    # on braque a gauche

        # on sauvegarde notre rotation seulement si on n'etait pas en urgence
        if not mord_verte and not mord_rouge:
            self.last_omega_valid = omega

        # on envoie aux moteurs
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
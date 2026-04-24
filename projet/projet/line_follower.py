import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Int32MultiArray
from geometry_msgs.msg import Twist
import numpy as np

# =============================================================================
# LineFollower — Régulateur PD multicouche avec anticipation par le lointain
# =============================================================================
#
# Ce nœud consomme les tableaux publiés par line_detector.py :
#   /red_bands_cx   : Int32MultiArray — cx rouge par bande (0=loin, N-1=proche)
#   /green_bands_cx : Int32MultiArray — cx vert  par bande
#
# STRATÉGIE DE PONDÉRATION CROISSANTE :
#   La bande la plus proche (indice N-1) a le poids le plus élevé : elle dit
#   au robot "où est la piste maintenant" et évite de mordre la ligne.
#   Les bandes supérieures (loin, petit indice) ont un poids plus faible mais
#   contribuent à "tirer" la trajectoire en anticipant les virages.
#
#   Poids par défaut (modifiables) :
#     BAND_WEIGHTS = [0.05, 0.10, 0.20, 0.30, 0.35]   (N=5, somme = 1.0)
#     → indice 0 (loin) : 5%  de l'influence totale
#     → indice 4 (proche) : 35% de l'influence totale
#
# RÉGRESSION POLYNOMIALE (optionnel) :
#   Quand suffisamment de bandes sont valides (au moins POLY_MIN_POINTS),
#   on ajuste un polynôme de degré 1 (droite) sur les centres de voie de
#   chaque bande et on extrapole la position cible à la bande proche.
#   Cela lisse les mesures bruitées et donne une commande plus douce.
#   Si trop peu de bandes sont valides, on retombe sur la somme pondérée simple.
#
# INERTIE DE TRAJECTOIRE (inchangée) :
#   Quand aucune bande ne fournit d'information, on conserve le dernier omega
#   valide atténué par inertia_decay, ce qui évite la rotation sur place aveugle.
#
# COMPATIBILITÉ :
#   Les anciens topics /camera_width, /offset_near, /offset_far ne sont plus
#   utilisés pour le calcul principal, mais /camera_width est toujours écouté
#   pour la normalisation de l'erreur.
# =============================================================================

# Poids par bande — indice 0 = bande la plus loin, indice N-1 = plus proche.
# La somme doit valoir 1.0. Augmenter les derniers indices renforce le "où je suis",
# augmenter les premiers indices renforce l'"anticipation virage".
BAND_WEIGHTS = [0.05, 0.10, 0.20, 0.30, 0.35]   # N=5 — ajuster si N_BANDS change

# Nombre minimum de bandes valides (cx != -1) pour activer la régression
POLY_MIN_POINTS = 3


class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower_node')

        # ── Données multicouches reçues depuis line_detector ─────────────────
        # Listes de taille N_BANDS ; valeur -1 = bande invalide (pas de ligne)
        self.cx_red_bands   = []   # sera initialisé à la première réception
        self.cx_green_bands = []

        # ── Métadonnées caméra ────────────────────────────────────────────────
        self.image_width = 640.0

        # ── Paramètres PD ─────────────────────────────────────────────────────
        self.kp = 1.5
        self.kd = 6.0
        self.base_speed = 0.10
        self.speed_min  = 0.05

        # ── Filtre exponentiel sur l'erreur ───────────────────────────────────
        # alpha élevé = forte inertie (lisse les oscillations rapides)
        # alpha faible = réactivité maximale
        self.alpha        = 0.80
        self.filtered_err = 0.0
        self.last_error   = 0.0

        # ── Largeur de voie estimée (offset) ──────────────────────────────────
        # Utilisée quand une seule des deux lignes est visible pour estimer
        # le centre de voie : centre = cx_visible ± offset
        # Valeur par défaut ; sera mise à jour dynamiquement si les deux
        # lignes sont visibles simultanément dans au moins une bande.
        self.offset_dynamique = 75.0   # pixels — mis à jour en ligne

        # ── Inertie de trajectoire ─────────────────────────────────────────────
        # Quand aucune ligne n'est visible (toutes les bandes retournent -1),
        # au lieu de tourner sur place à l'aveugle (omega=0.25), on conserve
        # le dernier omega calculé sur une ligne réelle et on l'atténue
        # progressivement. Cela permet de retrouver la piste en continuant
        # d'avancer plutôt qu'en tournant en rond.
        #
        # - last_omega_valid  : dernier omega calculé sur la base d'une ligne réelle
        # - inertia_speed     : vitesse linéaire réduite pendant la phase d'inertie
        # - inertia_decay     : facteur de décroissance de l'omega d'inertie par tick
        #   (0.85 = l'omega est atténué de 15 % par cycle)
        self.last_omega_valid = 0.0
        self.inertia_speed    = 0.05
        self.inertia_decay    = 0.85

        # ── Abonnements ───────────────────────────────────────────────────────
        self.create_subscription(Int32MultiArray, '/red_bands_cx',   self.cb_red_bands,   10)
        self.create_subscription(Int32MultiArray, '/green_bands_cx', self.cb_green_bands, 10)
        self.create_subscription(Int32,           '/camera_width',   self.cb_cam_width,   10)

        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel_line_raw', 10)
        self.create_timer(0.05, self.compute_and_publish)   # 20 Hz

        self.get_logger().info(
            "LineFollower démarré — multicouche PD + anticipation lointain"
        )

    # =========================================================================
    # Callbacks
    # =========================================================================

    def cb_red_bands(self,   msg): self.cx_red_bands   = list(msg.data)
    def cb_green_bands(self, msg): self.cx_green_bands = list(msg.data)
    def cb_cam_width(self,   msg): self.image_width     = float(msg.data)

    # =========================================================================
    # Calcul du centre de voie pour une bande donnée
    # =========================================================================

    def centre_voie_bande(self, cx_r, cx_g, safety_margin):
        """
        Retourne le centre estimé de la voie pour une bande.

        Priorité 1 : les deux lignes sont visibles → centre géométrique.
        Priorité 2 : une seule ligne visible → estimation par offset.
        Retourne None si aucune ligne n'est visible dans cette bande.
        """
        if cx_r != -1 and cx_g != -1:
            # Mise à jour dynamique de l'offset (moyenne des deux lignes)
            self.offset_dynamique = 0.90 * self.offset_dynamique + 0.10 * abs(cx_r - cx_g) / 2.0

            centre = (cx_r + cx_g) / 2.0
            # Clamp entre les marges de sécurité
            lo = cx_g + safety_margin
            hi = cx_r - safety_margin
            if lo < hi:
                centre = max(lo, min(hi, centre))
            return centre

        elif cx_r != -1:
            # Seule la ligne rouge (droite) est visible
            centre = cx_r - self.offset_dynamique
            centre = max(safety_margin, min(centre, self.image_width - safety_margin))
            return centre

        elif cx_g != -1:
            # Seule la ligne verte (gauche) est visible
            centre = cx_g + self.offset_dynamique
            centre = max(safety_margin, min(centre, self.image_width - safety_margin))
            return centre

        return None   # bande invalide

    # =========================================================================
    # Algorithme de centre pondéré avec anticipation
    # =========================================================================

    def calcul_centre_pondere(self, safety_margin):
        """
        Fusionne les N bandes pour obtenir un unique centre de voie cible.

        Étape 1 — Calcul du centre de voie de chaque bande valide.
        Étape 2 — Si assez de bandes sont valides (≥ POLY_MIN_POINTS) :
                  régression linéaire (np.polyfit deg=1) sur les centres
                  et extrapolation à la bande la plus proche. Cela lisse
                  les mesures et génère un signal anticipatif naturel.
        Étape 3 — Sinon : somme pondérée classique (BAND_WEIGHTS).

        Retourne None si aucune bande n'est exploitable.
        """
        n = len(self.cx_red_bands)
        if n == 0:
            return None

        # On tronque les poids si N_BANDS a changé
        weights = BAND_WEIGHTS[:n]
        # Renormalisation au cas où la somme ne vaut pas exactement 1
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        centres_par_bande = []   # liste de (indice_bande, centre)

        for i in range(n):
            cx_r = self.cx_red_bands[i]   if i < len(self.cx_red_bands)   else -1
            cx_g = self.cx_green_bands[i] if i < len(self.cx_green_bands) else -1

            c = self.centre_voie_bande(cx_r, cx_g, safety_margin)
            if c is not None:
                centres_par_bande.append((i, c))

        if not centres_par_bande:
            return None

        # ── Régression polynomiale (deg 1) si assez de points ────────────────
        # On utilise l'indice de bande comme abscisse :
        #   indice 0 = loin (haut de la zone analysée)
        #   indice N-1 = proche (bas de l'image, devant le robot)
        # La régression lisse les mesures bruitées et l'extrapolation à
        # l'indice N-1 donne le "où le robot sera si la trajectoire actuelle
        # se prolonge" — c'est l'anticipation feed-forward.
        if len(centres_par_bande) >= POLY_MIN_POINTS:
            indices  = np.array([p[0] for p in centres_par_bande], dtype=float)
            centres  = np.array([p[1] for p in centres_par_bande], dtype=float)

            # Régression linéaire : centre ≈ a * indice + b
            coeffs = np.polyfit(indices, centres, 1)
            poly   = np.poly1d(coeffs)

            # Extrapolation au niveau de la bande la plus proche (indice N-1)
            # et à mi-chemin entre la bande la plus loin et la bande proche
            # pour un mélange "anticipation + présent".
            centre_proche = poly(n - 1)          # extrapolé vers le bas
            centre_milieu = poly(n * 0.5)        # milieu de la piste

            # Mélange : 70 % proche, 30 % anticipation lointaine
            centre = 0.70 * centre_proche + 0.30 * centre_milieu

            # Clamp pour éviter les extrapolations hors champ
            centre = max(safety_margin, min(self.image_width - safety_margin, centre))
            return centre

        # ── Fallback : somme pondérée simple ─────────────────────────────────
        total_weight = 0.0
        total_centre = 0.0
        for (i, c) in centres_par_bande:
            w = weights[i]
            total_centre += w * c
            total_weight += w

        return total_centre / total_weight if total_weight > 0 else None

    # =========================================================================
    # Boucle de contrôle principale
    # =========================================================================

    def compute_and_publish(self):
        if self.image_width == 0.0:
            return

        twist = Twist()

        image_center  = self.image_width / 2.0
        safety_margin = 0.08 * self.image_width
        marge_strict  = 0.40 * self.image_width

        centre = self.calcul_centre_pondere(safety_margin)

        # ── Inertie de trajectoire (aucune bande valide) ──────────────────────
        # Quand aucune ligne n'est visible, on conserve le dernier omega valide
        # (atténué) et on avance doucement pour retrouver la piste.
        if centre is None:
            self.last_omega_valid *= self.inertia_decay
            twist.linear.x  = self.inertia_speed
            twist.angular.z = self.last_omega_valid
            self.pub_cmd.publish(twist)
            return

        # ── Calcul de l'erreur normalisée ─────────────────────────────────────
        erreur_pixels = image_center - centre
        erreur_brute  = erreur_pixels / self.image_width

        # Filtre exponentiel — atténue les oscillations rapides
        self.filtered_err = (self.alpha * self.filtered_err
                             + (1.0 - self.alpha) * erreur_brute)
        erreur = self.filtered_err

        # Terme dérivé (anticipe les variations d'erreur)
        derivation  = erreur - self.last_error
        self.last_error = erreur

        omega   = float(self.kp * erreur + self.kd * derivation)
        vitesse = self.base_speed

        # ── Cas d'urgence : mordre la ligne ───────────────────────────────────
        # Utilise uniquement la bande la plus proche (indice -1)
        # pour détecter si le robot est en train de mordre la bordure.
        cx_r_near = self.cx_red_bands[-1]   if self.cx_red_bands   else -1
        cx_g_near = self.cx_green_bands[-1] if self.cx_green_bands else -1

        seuil_vert  = image_center - marge_strict
        seuil_rouge = image_center + marge_strict

        mord_verte = (cx_g_near != -1) and (cx_g_near > seuil_vert)
        mord_rouge = (cx_r_near != -1) and (cx_r_near < seuil_rouge)

        if mord_verte:
            vitesse = 0.05
            omega   = -1.0   # correction braquage plein droite
        elif mord_rouge:
            vitesse = 0.05
            omega   = 1.0    # correction braquage plein gauche

        # Mémorise l'omega valide pour l'inertie (uniquement hors urgence)
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
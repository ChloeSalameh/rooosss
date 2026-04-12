import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32

# =============================================================================
# MACHINE À ÉTATS — Challenge 1 : Suivi de ligne + Rond-point
# =============================================================================
#
# Zéro minuterie pour les manœuvres.
# Chaque transition est déclenchée par un REPÈRE VISUEL confirmé sur
# plusieurs images consécutives (FRAMES_CONFIRMATION), ce qui rend le
# comportement indépendant de la vitesse CPU, du FPS caméra et de la
# charge de la machine.
#
# =============================================================================

# --- États ---
STATE_FOLLOW         = 'FOLLOW'
STATE_ENTRY_TURN     = 'ENTRY_TURN'      # Rotation jusqu'à la ligne directrice
STATE_ROUNDABOUT_NAV = 'ROUNDABOUT_NAV'  # Navigation dans le rond-point
STATE_EXITING        = 'EXITING'         # Sortie, retour au suivi normal

# --- Paramètres de confirmation visuelle (nombre de frames consécutives) ---
# Augmenter ces valeurs rend le robot plus prudent mais légèrement plus lent
# à transitionner. Réduire les rend plus réactif mais moins robuste au bruit.
FRAMES_ENTREE_DETECT    = 5   # Frames pour détecter l'approche du rond-point
FRAMES_LIGNE_ALIGNEE    = 3   # Frames avec la ligne directrice dans le bon cadran
FRAMES_SORTIE_DETECT    = 6   # Frames avec les 2 lignes bien écartées (sortie)
FRAMES_SUIVI_STABLE     = 8   # Frames de suivi nominal pour valider la fin de sortie


class Challenge1(Node):
    def __init__(self):
        super().__init__('challenge1')

        print("\n" + "="*60)
        choix_direction = ""
        while choix_direction not in ['left', 'right']:
            choix_direction = input("Direction rond-point 'left' ou 'right' : ").strip().lower()
        print("="*60 + "\n")

        self.declare_parameter('roundabout', choix_direction)
        self.declare_parameter('camera_width', 640)
        self.cam_width = float(self.get_parameter('camera_width').value)

        # --- Horloge ROS (use_sim_time compatible) ---
        self._clock = self.get_clock()

        # ── Rotation d'insertion ──────────────────────────────────────────────
        # Vitesse angulaire pour la rotation sur place.
        # La rotation s'arrête quand la ligne directrice est visible dans le
        # bon cadran — plus besoin de durée ni de phase d'avance séparée.
        self.vitesse_rotation = 0.8      # rad/s

        # ── Détection d'entrée du rond-point ─────────────────────────────────
        # PRINCIPE : dans tout le circuit, la ligne verte est à GAUCHE du robot
        # et la ligne rouge à DROITE, donc cx_green < cx_red.
        # À l'approche du rond-point, le robot voit les deux arcs de l'îlot
        # central de face : la géométrie s'INVERSE — la ligne rouge apparaît
        # à gauche et la verte à droite dans l'image, donc cx_red < cx_green.
        # Ce signe est INVARIANT : il ne dépend ni de la résolution, ni du FPS,
        # ni de la charge CPU — c'est un fait géométrique pur.
        #
        # On ajoute un garde-fou optionnel : les deux lignes doivent être
        # effectivement visibles (pas juste du bruit sur une seule), et leur
        # écart doit être suffisant pour exclure un artefact de toute petite taille.
        self.seuil_ecart_inversion_min = 0.03   # écart minimal (fraction cam_width)
                                                  # pour que l'inversion soit réelle

        # ── Seuil d'alignement de fin de rotation ────────────────────────────
        # La rotation s'arrête quand la ligne directrice est bien positionnée
        # dans son cadran naturel de suivi, c'est-à-dire :
        #   'right' → cx_red   dans le tiers DROIT  de l'image (> 0.60)
        #   'left'  → cx_green dans le tiers GAUCHE de l'image (< 0.40)
        # Ces seuils sont asymétriques par rapport à 0.5 pour garantir que
        # le robot est déjà orienté dans la bonne direction avant de démarrer
        # le suivi PD — évitant une correction initiale trop brutale.
        self.seuil_alignement_droite = 0.60   # cx_red  / cam_width  > seuil
        self.seuil_alignement_gauche = 0.40   # cx_green / cam_width < seuil

        # ── Seuil de détection de sortie du rond-point ───────────────────────
        # On détecte la sortie quand les 2 lignes sont à nouveau bien visibles
        # et bien écartées (on retrouve la piste normale).
        self.seuil_ecart_sortie = 0.30   # écart > 30% de la largeur = sortie

        # ── Seuil de suivi nominal (validation fin de sortie) ─────────────────
        # L'écart "normal" entre les 2 lignes en suivi stable.
        self.seuil_ecart_nominal_min = 0.28
        self.seuil_ecart_nominal_max = 0.60

        # ── Contrôleur PD pour la navigation dans le rond-point ──────────────
        self.kp_round = 0.008
        self.kd_round = 0.008
        self.v_angulaire_base = 0.25
        self.v_lineaire_round = 0.08

        # Position cible de la ligne directrice (très près du bord de l'image)
        self.pct_cible_rouge_droite = 0.98
        self.pct_cible_verte_gauche = 0.02

        self.last_err_round = 0.0

        # ── Abonnements ───────────────────────────────────────────────────────
        self.create_subscription(LaserScan, '/scan',             self.cb_scan,      10)
        self.create_subscription(Twist,     '/cmd_vel_line_raw', self.cb_cmd,       10)
        self.create_subscription(Int32,     '/red_line_pos',     self.cb_red_near,  10)
        self.create_subscription(Int32,     '/green_line_pos',   self.cb_green_near,10)
        self.create_subscription(Int32,     '/camera_width',     self.cb_cam_width, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel_challenge_1', 10)

        # ── Variables d'état ──────────────────────────────────────────────────
        self.state         = STATE_FOLLOW
        self.emergency     = False
        self.cx_red_near   = -1
        self.cx_green_near = -1

        # Compteurs de confirmation visuelle (remplacent les minuteries)
        self.cpt_entree          = 0   # inversion détectée (approche rond-point)
        self.cpt_ligne_alignee   = 0   # ligne directrice dans le bon cadran
        self.cpt_sortie          = 0   # 2 lignes bien écartées (détection sortie)
        self.cpt_suivi_stable    = 0   # suivi nominal stable (fin EXITING)

        self.get_logger().info(
            f"Challenge1 démarré — direction={choix_direction} — mode VISUEL (sans minuterie)"
        )

    # =========================================================================
    # Callbacks
    # =========================================================================

    def cb_cam_width(self, msg):
        self.cam_width = float(msg.data)

    def cb_scan(self, msg):
        angles = list(range(0, 15)) + list(range(345, 360))
        dists  = [msg.ranges[a] for a in angles if 0.01 < msg.ranges[a] < 3.0]
        self.emergency = bool(dists and min(dists) < 0.22)

    def cb_red_near(self,   msg): self.cx_red_near   = msg.data
    def cb_green_near(self, msg): self.cx_green_near = msg.data

    def cb_cmd(self, cmd_raw: Twist):
        """Point d'entrée principal — déclenché à chaque nouvelle image."""
        if self.emergency:
            self.publisher.publish(Twist())
            return
        direction = self.get_parameter('roundabout').value
        self.publisher.publish(self.fsm_step(cmd_raw, direction))

    # =========================================================================
    # Machine à états
    # =========================================================================

    def fsm_step(self, cmd_raw: Twist, direction: str) -> Twist:

        # Calcul de l'écart normalisé entre les deux lignes
        # (utilisé dans ROUNDABOUT_NAV et EXITING, pas dans FOLLOW)
        ecart_near = None
        if self.cx_red_near != -1 and self.cx_green_near != -1:
            ecart_near = abs(self.cx_red_near - self.cx_green_near) / self.cam_width

        # ── ÉTAT 1 : Suivi de ligne normal ────────────────────────────────────
        if self.state == STATE_FOLLOW:
            return self._state_follow(cmd_raw, direction, ecart_near)

        # ── ÉTAT 2 : Rotation d'insertion ─────────────────────────────────────
        elif self.state == STATE_ENTRY_TURN:
            return self._state_entry_turn(direction)

        # ── ÉTAT 3 : Navigation dans le rond-point ────────────────────────────
        elif self.state == STATE_ROUNDABOUT_NAV:
            return self._state_roundabout_nav(cmd_raw, direction, ecart_near)

        # ── ÉTAT 4 : Sortie ───────────────────────────────────────────────────
        elif self.state == STATE_EXITING:
            return self._state_exiting(cmd_raw, ecart_near)

        return cmd_raw

    # =========================================================================
    # Implémentations des états
    # =========================================================================

    def _state_follow(self, cmd_raw: Twist, direction: str, ecart_near) -> Twist:
        """
        Suivi normal. Détecte l'entrée dans le rond-point via l'INVERSION
        des côtés rouge et vert dans l'image caméra.

        Partout sur le circuit : cx_green < cx_red  (vert à gauche, rouge à droite).
        Face à l'îlot central :  cx_red   < cx_green (inversion géométrique).

        Cette condition est universelle — elle ne dépend d'aucun seuil
        arbitraire lié à la résolution, au FPS ou à la charge CPU.

        Garde-fous :
          1. Les deux lignes doivent être visibles (cx != -1).
          2. L'écart entre elles doit dépasser seuil_ecart_inversion_min
             pour exclure un artefact ponctuel (deux pixels proches).
          3. La condition doit être confirmée sur FRAMES_ENTREE_DETECT
             frames consécutives (filtre anti-rebond).
        """
        inversion_detectee = False

        if (self.cx_red_near != -1 and self.cx_green_near != -1):
            ecart_px = self.cx_green_near - self.cx_red_near   # positif si inversion
            ecart_frac = ecart_px / self.cam_width
            # Inversion = rouge à gauche du vert ET écart assez grand pour être réel
            inversion_detectee = (ecart_frac > self.seuil_ecart_inversion_min)

        if inversion_detectee:
            self.cpt_entree += 1
        else:
            # Décrémentation progressive : une frame de bruit ne remet pas
            # le compteur à zéro — il faut autant de frames fausses pour
            # défaire que de frames vraies pour confirmer.
            self.cpt_entree = max(0, self.cpt_entree - 1)

        if self.cpt_entree >= FRAMES_ENTREE_DETECT:
            self.cpt_entree        = 0
            self.cpt_ligne_alignee = 0
            self.transition(STATE_ENTRY_TURN)
            return self.cmd_turn(direction)

        return cmd_raw

    def _state_entry_turn(self, direction: str) -> Twist:
        """
        Rotation d'insertion dans le rond-point.

        PRINCIPE : le robot tourne dans la direction demandée et s'arrête
        dès que la ligne directrice qu'il va suivre apparaît bien positionnée
        dans son cadran naturel — c'est-à-dire à la position où le régulateur
        PD de ROUNDABOUT_NAV peut immédiatement la travailler sans correction
        initiale excessive.

          direction='right' → tourne à droite (CW, angular.z < 0)
            ARRÊT quand cx_red  > seuil_alignement_droite * cam_width
            (la ligne rouge est bien dans le tiers droit de l'image)

          direction='left'  → tourne à gauche (CCW, angular.z > 0)
            ARRÊT quand cx_green < seuil_alignement_gauche * cam_width
            (la ligne verte est bien dans le tiers gauche de l'image)

        Le compteur anti-rebond (décrémentation progressive) absorbe les
        frames parasites sans repartir à zéro sur un pixel de bruit.
        """
        aligne = False

        if direction == 'right' and self.cx_red_near != -1:
            aligne = (self.cx_red_near / self.cam_width > self.seuil_alignement_droite)
        elif direction == 'left' and self.cx_green_near != -1:
            aligne = (self.cx_green_near / self.cam_width < self.seuil_alignement_gauche)

        if aligne:
            self.cpt_ligne_alignee += 1
        else:
            self.cpt_ligne_alignee = max(0, self.cpt_ligne_alignee - 1)

        if self.cpt_ligne_alignee >= FRAMES_LIGNE_ALIGNEE:
            # Ligne directrice stable dans le bon cadran → on démarre le suivi
            self.cpt_ligne_alignee = 0
            self.last_err_round    = 0.0
            self.cpt_sortie        = 0
            self.transition(STATE_ROUNDABOUT_NAV)
            return self.cmd_roundabout_nav(direction)

        # La ligne n'est pas encore en place : on continue de tourner
        return self.cmd_turn(direction)

    def _state_roundabout_nav(self, cmd_raw: Twist, direction: str, ecart_near) -> Twist:
        """
        Navigation dans le rond-point avec suivi PD mono-ligne.

        REPÈRE DE SORTIE : les deux lignes (rouge ET verte) sont à nouveau
        simultanément visibles et leur écart dépasse le seuil de sortie
        (on retrouve la piste normale à l'approche de la sortie du rond-point).

        Ce critère est bien plus robuste qu'une minuterie : si le robot
        prend plus longtemps à traverser le rond-point (chargement CPU),
        il ne sortira pas prématurément.

        Protection anti-rebond : on exige N frames consécutives confirmatoires.
        """
        # Détection de sortie : 2 lignes visibles + grand écart
        if (ecart_near is not None and ecart_near > self.seuil_ecart_sortie):
            self.cpt_sortie += 1
        else:
            self.cpt_sortie = max(0, self.cpt_sortie - 1)

        if self.cpt_sortie >= FRAMES_SORTIE_DETECT:
            self.cpt_sortie      = 0
            self.cpt_suivi_stable = 0
            self.transition(STATE_EXITING)
            return cmd_raw

        return self.cmd_roundabout_nav(direction)

    def _state_exiting(self, cmd_raw: Twist, ecart_near) -> Twist:
        """
        Sortie du rond-point : on rend la main au line_follower (cmd_raw).

        REPÈRE DE FIN : le suivi est stabilisé, c'est-à-dire que les deux
        lignes sont visibles avec un écart dans la plage "normale" de la piste
        pendant N frames consécutives.

        Ce critère garantit que le robot est bien réinséré dans la piste avant
        de revenir à STATE_FOLLOW, évitant un nouveau déclenchement parasite
        de la détection d'entrée.
        """
        suivi_nominal = (
            ecart_near is not None
            and self.seuil_ecart_nominal_min < ecart_near < self.seuil_ecart_nominal_max
        )

        if suivi_nominal:
            self.cpt_suivi_stable += 1
        else:
            self.cpt_suivi_stable = max(0, self.cpt_suivi_stable - 1)

        if self.cpt_suivi_stable >= FRAMES_SUIVI_STABLE:
            self.cpt_suivi_stable = 0
            self.cpt_entree       = 0   # Remise à zéro anti-rebond
            self.transition(STATE_FOLLOW)

        return cmd_raw

    # =========================================================================
    # Commandes de base
    # =========================================================================

    def cmd_turn(self, direction: str) -> Twist:
        """Rotation sur place, sens dépendant de la direction choisie."""
        t = Twist()
        t.linear.x  = 0.0
        t.angular.z = self.vitesse_rotation if direction == 'left' else -self.vitesse_rotation
        return t

    def cmd_roundabout_nav(self, direction: str) -> Twist:
        """
        Suivi fluide à une seule ligne avec régulateur PD.
        La ligne directrice est maintenue très près du bord de l'image
        (pct_cible = 0.02 ou 0.98) pour que le robot suive la courbure
        du rond-point.
        """
        t = Twist()
        t.linear.x = self.v_lineaire_round

        erreur = 0.0
        v_base = 0.0

        if direction == 'right':
            cible  = self.cam_width * self.pct_cible_rouge_droite
            v_base = -self.v_angulaire_base
            if self.cx_red_near != -1:
                erreur = cible - self.cx_red_near
            else:
                t.angular.z = v_base
                return t

        elif direction == 'left':
            cible  = self.cam_width * self.pct_cible_verte_gauche
            v_base = self.v_angulaire_base
            if self.cx_green_near != -1:
                erreur = cible - self.cx_green_near
            else:
                t.angular.z = v_base
                return t

        derivee = erreur - self.last_err_round
        self.last_err_round = erreur
        correction = (self.kp_round * erreur) + (self.kd_round * derivee)
        t.angular.z = v_base + correction
        return t

    def transition(self, new_state: str):
        self.get_logger().info(f"FSM: {self.state} → {new_state}")
        self.state = new_state


def main(args=None):
    rclpy.init(args=args)
    node = Challenge1()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
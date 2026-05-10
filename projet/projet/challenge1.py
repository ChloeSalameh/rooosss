import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32

# les 4 etapes du rond-point (LA MACHINE A ETATS)
STATE_FOLLOW         = 'FOLLOW' #on roule normalement sur la piste
STATE_ENTRY_TURN     = 'ENTRY_TURN'      # On a vu le rond-point, on braque pour s'inserer
STATE_ROUNDABOUT_NAV = 'ROUNDABOUT_NAV'  # On est dans le rond-point, on longe la ligne exterieure
STATE_EXITING        = 'EXITING'         # On a vu la sortie, on se remet droit sur la piste

FRAMES_ENTREE_DETECT    = 5   # Frames pour detecter l'approche du rond-point
FRAMES_LIGNE_ALIGNEE    = 3   # Frames avec la ligne directrice dans le bon cadran
FRAMES_SORTIE_DETECT    = 6   # Frames avec les 2 lignes bien ecartees (sortie)
FRAMES_SUIVI_STABLE     = 8   # Frames de suivi nominal pour valider la fin de sortie


class Challenge1(Node):
    """
    Le programme ecoute la camera et decide quand quitter le suivi de ligne normal 
    pour executer l'insertion et la sortie
    """
    def __init__(self):
        super().__init__('challenge1')

        # Choix du sens du rond-point au lancement
        print("\n" + "="*60)
        choix_direction = ""
        while choix_direction not in ['left', 'right']:
            choix_direction = input("Direction rond-point 'left' ou 'right' : ").strip().lower()
        print("="*60 + "\n")

        self.declare_parameter('roundabout', choix_direction)
        self.declare_parameter('camera_width', 640)
        self.cam_width = float(self.get_parameter('camera_width').value)

        # Horloge du robot
        self._clock = self.get_clock()

        # Parametres de Conduite
        self.vitesse_rotation = 0.8      #vitesse quand il tourne sur place pour s'inserer

        # seuils visuels (Comment le robot interprete ce qu'il voit)
        self.seuil_ecart_inversion_min = 0.03   # les lignes se croisent (debut du rond point)
        self.seuil_alignement_droite = 0.75   # la ligne rouge est bien calee a droite
        self.seuil_alignement_gauche = 0.25   # la ligne rouge est bien calee a gauche
        self.seuil_ecart_sortie = 0.45   # les lignes s'ecartent beaucoup (c'est la sortie)

        # L'ecart ideal entre la ligne rouge et la verte en ligne droite
        self.seuil_ecart_nominal_min = 0.28
        self.seuil_ecart_nominal_max = 0.60

        # Controleur PD
        self.kp_round = 0.008
        self.kd_round = 0.008
        self.v_angulaire_base = 0.25
        self.v_lineaire_round = 0.08

        # Ou le robot veut-il placer la ligne sur son ecran ? 
        self.pct_cible_rouge_droite = 0.98
        self.pct_cible_verte_gauche = 0.02

        self.last_err_round = 0.0 #memoire de l'erreur precedente

        # Abonnements ROS
        self.create_subscription(LaserScan, '/scan',             self.cb_scan,      10)
        self.create_subscription(Twist,     '/cmd_vel_line_raw', self.cb_cmd,       10)
        self.create_subscription(Int32,     '/red_line_pos',     self.cb_red_near,  10)
        self.create_subscription(Int32,     '/green_line_pos',   self.cb_green_near,10)
        self.create_subscription(Int32,     '/camera_width',     self.cb_cam_width, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel_challenge_1', 10)

        # Variables d'etat
        self.state         = STATE_FOLLOW
        self.emergency     = False
        self.cx_red_near   = -1
        self.cx_green_near = -1

        # Les compteurs remplacent les chronometres
        self.cpt_entree          = 0   # inversion detectee (approche rond-point)
        self.cpt_ligne_alignee   = 0   #ligne directrice dans le bon cadran
        self.cpt_sortie          = 0   # 2 lignes bien ecartees (detection sortie)
        self.cpt_suivi_stable    = 0   # Suivi nominal stable

        self.get_logger().info(
            f"Challenge1 demarre — direction={choix_direction} — mode VISUEL (sans minuterie)"
        )

    # lecture des capteurs
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
        direction = self.get_parameter('roundabout').value # on passe dans notre "Machine a etats" pour savoir quoi faire
        self.publisher.publish(self.fsm_step(cmd_raw, direction))

    def fsm_step(self, cmd_raw: Twist, direction: str) -> Twist:
        """
        Il regarde dans quelle etape on est et renvoie l'ordre approprie pour les roues
        """

        # On calcule l'ecart entre les deux lignes sur l'ecran
        ecart_near = None
        if self.cx_red_near != -1 and self.cx_green_near != -1:
            ecart_near = abs(self.cx_red_near - self.cx_green_near) / self.cam_width

        # Etat1: Suivi de ligne normal
        if self.state == STATE_FOLLOW:
            return self._state_follow(cmd_raw, direction, ecart_near)

        # Etat2 : Rotation d'insertion
        elif self.state == STATE_ENTRY_TURN:
            return self._state_entry_turn(direction)

        # Etat3 : Navigation dans le rond-point
        elif self.state == STATE_ROUNDABOUT_NAV:
            return self._state_roundabout_nav(cmd_raw, direction, ecart_near)

        # Etat4 : Sortie
        elif self.state == STATE_EXITING:
            return self._state_exiting(cmd_raw, ecart_near)

        return cmd_raw

    # ETAPE1 : SURVEILLER L'ENTREE
    def _state_follow(self, cmd_raw: Twist, direction: str, ecart_near) -> Twist:
        inversion_detectee = False

        if (self.cx_red_near != -1 and self.cx_green_near != -1):
            ecart_px = self.cx_green_near - self.cx_red_near   # positif si inversion
            ecart_frac = ecart_px / self.cam_width
            # Inversion = rouge a gauche du vert ET ecart assez grand pour etre reel
            inversion_detectee = (ecart_frac > self.seuil_ecart_inversion_min)

        if inversion_detectee:
            self.cpt_entree += 1 # On commence à compter..
        else:
            self.cpt_entree = max(0, self.cpt_entree - 1) # Si c'était une fausse alerte, on rebaisse le compteur doucement

        # Si on est sûr que c'est le rond-point (5 images de suite)
        if self.cpt_entree >= FRAMES_ENTREE_DETECT:
            self.cpt_entree        = 0
            self.cpt_ligne_alignee = 0
            self.transition(STATE_ENTRY_TURN) # On change d'étape
            return self.cmd_turn(direction) # Et on commence à tourner

        # Tant qu'on n'a rien vu, on suit la ligne normalement
        return cmd_raw

    # EAPE 2 : S'INSÉRER 
    def _state_entry_turn(self, direction: str) -> Twist:
        
        aligne = False

        # On tourne jusqu'à ce que la ligne qu'on veut longer soit tout au bord de l'écran
        if direction == 'right' and self.cx_red_near != -1:
            aligne = (self.cx_red_near / self.cam_width > self.seuil_alignement_droite)
        elif direction == 'left' and self.cx_green_near != -1:
            aligne = (self.cx_green_near / self.cam_width < self.seuil_alignement_gauche)

        if aligne:
            self.cpt_ligne_alignee += 1
        else:
            self.cpt_ligne_alignee = max(0, self.cpt_ligne_alignee - 1)

        # Si la ligne est bien calée sur le côté (3 images de suite)
        if self.cpt_ligne_alignee >= FRAMES_LIGNE_ALIGNEE:
            
            self.cpt_ligne_alignee = 0
            self.last_err_round    = 0.0
            self.cpt_sortie        = 0
            self.transition(STATE_ROUNDABOUT_NAV) # On s'engage dans le rond-point
            return self.cmd_roundabout_nav(direction)

        # La ligne n'est pas encore en place : on continue de tourner
        return self.cmd_turn(direction)

    # ETAPE 3 : LONGER LE ROND-POINT
    def _state_roundabout_nav(self, cmd_raw: Twist, direction: str, ecart_near) -> Twist:
       
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

    # ETAPE 4 : SE REMETTRE DROIT 
    def _state_exiting(self, cmd_raw: Twist, ecart_near) -> Twist:
        
        # On vérifie que l'écart entre les lignes est redevenu "normal" (comme sur une ligne droite)
        suivi_nominal = (
            ecart_near is not None
            and self.seuil_ecart_nominal_min < ecart_near < self.seuil_ecart_nominal_max
        )

        if suivi_nominal:
            self.cpt_suivi_stable += 1
        else:
            self.cpt_suivi_stable = max(0, self.cpt_suivi_stable - 1)

        # Si ça fait un petit moment qu'on roule droit (8 images de suite)
        if self.cpt_suivi_stable >= FRAMES_SUIVI_STABLE:
            self.cpt_suivi_stable = 0
            self.cpt_entree       = 0   # Remise à zéro anti-rebond
            self.transition(STATE_FOLLOW)

        return cmd_raw

# Les actions physiques
# Comment le robot bouge vraiment

    def cmd_turn(self, direction: str) -> Twist:
        """Rotation sur place, sens dépendant de la direction choisie."""
        t = Twist()
        t.linear.x  = 0.0
        t.angular.z = self.vitesse_rotation if direction == 'left' else -self.vitesse_rotation
        return t

    def cmd_roundabout_nav(self, direction: str) -> Twist:

        """
        Avancer en longeant la courbe
        Utilise un calcul (PD) pour corriger doucement la trajectoire
        """
     
        t = Twist()
        t.linear.x = self.v_lineaire_round # On avance doucement

        erreur = 0.0
        v_base = 0.0

        # On calcule la distance (l'erreur) entre l'endroit ou on voudrait voir la ligne, et l'endroit ou la ligne est vraiment sur l'ecran.
        if direction == 'right':
            cible  = self.cam_width * self.pct_cible_rouge_droite # On veut la ligne tout à droite
            v_base = -self.v_angulaire_base
            if self.cx_red_near != -1:
                erreur = cible - self.cx_red_near
            else:
                t.angular.z = v_base
                return t

        elif direction == 'left':
            cible  = self.cam_width * self.pct_cible_verte_gauche # On veut la ligne tout à gauche
            v_base = self.v_angulaire_base
            if self.cx_green_near != -1:
                erreur = cible - self.cx_green_near
            else:
                t.angular.z = v_base
                return t

        # Le calcul du correcteur PD
        derivee = erreur - self.last_err_round
        self.last_err_round = erreur
        # On ajuste le volant en fonction de l'erreur actuelle (P) et de son evolution (D)
        correction = (self.kp_round * erreur) + (self.kd_round * derivee)
        t.angular.z = v_base + correction
        return t

    def transition(self, new_state: str):
        """fonction pour afficher quand le robot change d'étape."""
        self.get_logger().info(f"Changement d'étape : {self.state} → {new_state}")
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
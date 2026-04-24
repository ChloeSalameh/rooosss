import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool
from geometry_msgs.msg import Twist

# =============================================================================
# CHOISIR LE MODE ICI
#   MODE = "IRL"        → horloge système, pas d'attente Gazebo
#   MODE = "SIMULATION" → horloge Gazebo (use_sim_time=True), attente synchro
# =============================================================================
MODE = "IRL"   # IRL ou SIMULATION


if MODE not in ("IRL", "SIMULATION"):
    raise ValueError(f"MODE invalide : '{MODE}'. Choisir 'IRL' ou 'SIMULATION'.")

USE_SIM_TIME = (MODE == "SIMULATION")

# =============================================================================
# MACHINE À ÉTATS DU PARCOURS
# =============================================================================
#
# Séquence exacte des lignes bleues détectées :
#
#   Ligne bleue 1 : CH1        → CH2
#   Ligne bleue 2 : CH2        → CH3
#   Ligne bleue 3 : CH3        → LINE_FOLLOW_ONLY  (virage sur piste normale)
#   Ligne bleue 4 : LINE_FOLLOW_ONLY → CH4         (dernier challenge)
#   Ligne bleue 5+: CH4        → ignorée           (parcours terminé)
#
# Les états sont des chaînes nommées pour la lisibilité des logs.
# =============================================================================

STATE_CH1              = 'CHALLENGE_1'
STATE_CH2              = 'CHALLENGE_2'
STATE_CH3              = 'CHALLENGE_3'
STATE_LINE_FOLLOW_ONLY = 'LINE_FOLLOW_ONLY'   # état intermédiaire entre CH3 et CH4
STATE_CH4              = 'CHALLENGE_4'        # état final — jamais dépassé

# Table de transition : état actuel → état suivant lors d'une ligne bleue valide
TRANSITIONS = {
    STATE_CH1:              STATE_CH2,
    STATE_CH2:              STATE_CH3,
    STATE_CH3:              STATE_LINE_FOLLOW_ONLY,
    STATE_LINE_FOLLOW_ONLY: STATE_CH4,
    STATE_CH4:              None,   # état terminal, aucune transition possible
}

# Correspondance entre le paramètre initial_state (entier) et les états nommés.
# Permet de démarrer directement sur n'importe quel challenge via le launch file.
INT_TO_STATE = {
    1: STATE_CH1,
    2: STATE_CH2,
    3: STATE_CH3,
    4: STATE_CH4,
}


class Superviseur(Node):
    def __init__(self):
        super().__init__('superviseur')

        # initial_state accepte 1, 2, 3 ou 4 (correspondance via INT_TO_STATE)
        self.declare_parameter('initial_state', 1)
        initial_int = self.get_parameter('initial_state').value

        if initial_int not in INT_TO_STATE:
            raise ValueError(
                f"initial_state={initial_int} invalide. Valeurs acceptées : {list(INT_TO_STATE.keys())}"
            )
        self.current_state = INT_TO_STATE[initial_int]

        # En mode IRL, l'horloge est disponible immédiatement : on initialise
        # last_transition_time avec un décalage de -40 s pour que la sécurité
        # anti-rebond soit déjà "expirée" dès le démarrage.
        # En SIMULATION, on attend la synchro de l'horloge Gazebo (-1.0 = non initialisé).
        if MODE == "IRL":
            self.last_transition_time = self.get_clock().now().nanoseconds / 1e9 - 40.0
        else:
            self.last_transition_time = -1.0

        # Dernières commandes reçues de chaque nœud challenge
        self.last_twist_ch1 = Twist()
        self.last_twist_ch2 = Twist()
        self.last_twist_ch3 = Twist()
        self.last_twist_ch4 = Twist()
        self.last_twist_line = Twist()   # line_follower seul (état intermédiaire)

        # Détection de la ligne bleue
        self.sub_blue_line = self.create_subscription(
            Bool, '/blue_line_crossed', self.state_callback, 10)

        # Abonnements aux topics intermédiaires de chaque challenge (1 à 4)
        self.sub_cmd_ch1  = self.create_subscription(Twist, '/cmd_vel_challenge_1', self.cmd_ch1_callback,  10)
        self.sub_cmd_ch2  = self.create_subscription(Twist, '/cmd_vel_challenge_2', self.cmd_ch2_callback,  10)
        self.sub_cmd_ch3  = self.create_subscription(Twist, '/cmd_vel_challenge_3', self.cmd_ch3_callback,  10)
        self.sub_cmd_ch4  = self.create_subscription(Twist, '/cmd_vel_challenge_4', self.cmd_ch4_callback,  10)
        self.sub_cmd_line = self.create_subscription(Twist, '/cmd_vel_line_raw',    self.cmd_line_callback, 10)

        # Vrai topic pour le Turtlebot dans Gazebo et IRL
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(
            f'Superviseur démarré — MODE={MODE} — État initial : {self.current_state}'
        )

    # =========================================================================
    # Callbacks
    # =========================================================================

    def state_callback(self, msg):
        """Se déclenche quand on publie sur /blue_line_crossed."""
        if not msg.data:
            return

        # État terminal : la ligne bleue est ignorée, parcours terminé
        next_state = TRANSITIONS.get(self.current_state)
        if next_state is None:
            self.get_logger().info(
                "Ligne bleue détectée mais état terminal atteint (CHALLENGE_4) — parcours terminé."
            )
            return

        # En SIMULATION, on attend que l'horloge soit synchronisée
        if self.last_transition_time == -1.0:
            return

        current_time = self.get_clock().now().nanoseconds / 1e9
        duree_ecoulee = current_time - self.last_transition_time

        if duree_ecoulee > 3.0:
            previous_state     = self.current_state
            self.current_state = next_state
            self.last_transition_time = current_time
            self.get_logger().info(
                f"======== LIGNE BLEUE ! {previous_state} → {self.current_state} ========"
            )
        else:
            temps_restant = 3.0 - duree_ecoulee
            self.get_logger().warn(
                f"Objet bleu ignoré. Fin de sécurité dans {temps_restant:.1f}s"
            )

    def cmd_ch1_callback(self,  msg): self.last_twist_ch1  = msg
    def cmd_ch2_callback(self,  msg): self.last_twist_ch2  = msg
    def cmd_ch3_callback(self,  msg): self.last_twist_ch3  = msg
    def cmd_ch4_callback(self,  msg): self.last_twist_ch4  = msg
    def cmd_line_callback(self, msg): self.last_twist_line = msg

    # =========================================================================
    # Boucle de publication
    # =========================================================================

    def timer_callback(self):
        """Boucle à 10 Hz — sélectionne et publie la commande de l'état actuel."""

        # En SIMULATION uniquement : synchronisation de l'horloge Gazebo
        if MODE == "SIMULATION" and self.last_transition_time == -1.0:
            current_time = self.get_clock().now().nanoseconds / 1e9
            if current_time > 0:
                self.last_transition_time = current_time - 40.0
                self.get_logger().info("Horloge simulée synchronisée, timer de sécurité activé.")

        if self.current_state == STATE_CH1:
            twist_to_publish = self.last_twist_ch1
        elif self.current_state == STATE_CH2:
            twist_to_publish = self.last_twist_ch2
        elif self.current_state == STATE_CH3:
            twist_to_publish = self.last_twist_ch3
        elif self.current_state == STATE_LINE_FOLLOW_ONLY:
            # Virage intermédiaire entre CH3 et CH4 : line_follower seul
            twist_to_publish = self.last_twist_line
        elif self.current_state == STATE_CH4:
            twist_to_publish = self.last_twist_ch4
        else:
            twist_to_publish = Twist()   # sécurité : état inconnu → stop

        self.publisher_.publish(twist_to_publish)


def main(args=None):
    rclpy.init(args=args)
    node = Superviseur()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
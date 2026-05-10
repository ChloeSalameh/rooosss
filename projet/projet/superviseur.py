import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool
from geometry_msgs.msg import Twist


# On choisit le mode
# IRL pour le vrai robot, SIMULATION pour gazebo
MODE = "IRL"   


if MODE not in ("IRL", "SIMULATION"):
    raise ValueError(f"MODE invalide : '{MODE}'. Choisir 'IRL' ou 'SIMULATION'.")

USE_SIM_TIME = (MODE == "SIMULATION")

# L'ordre du parcours et les changements d'etats
# A chaque fois qu'on croise une ligne bleue on passe au challenge suivant
# Ligne bleue 1 : on passe au challenge 2 (evitement)
# Ligne bleue 2 : on passe au challenge 3 (u-turn)
# Ligne bleue 3 : on suit juste la ligne pour le grand virage
# Ligne bleue 4 : on passe au challenge 4 
# apres ca on ignore les autres lignes bleues on a fini le tour

STATE_CH1              = 'CHALLENGE_1'
STATE_CH2              = 'CHALLENGE_2'
STATE_CH3              = 'CHALLENGE_3'
STATE_LINE_FOLLOW_ONLY = 'LINE_FOLLOW_ONLY'   # just un suivi de ligne normal pour le virage
STATE_CH4              = 'CHALLENGE_4'        # la fin

# Dictionnaire pour savoir ou on va apres chaque ligne bleue
TRANSITIONS = {
    STATE_CH1:              STATE_CH2,
    STATE_CH2:              STATE_CH3,
    STATE_CH3:              STATE_LINE_FOLLOW_ONLY,
    STATE_LINE_FOLLOW_ONLY: STATE_CH4,
    STATE_CH4:              None,   # on s'arrete la
}

# ca nous permet de lancer le robot direct sur un challenge precis depuis le launch file
# sans se taper tout le circuit a chaque fois qu'on teste
INT_TO_STATE = {
    1: STATE_CH1,
    2: STATE_CH2,
    3: STATE_CH3,
    4: STATE_CH4,
}


class Superviseur(Node):
    def __init__(self):
        super().__init__('superviseur')

        # on recupere le parametre de depart (1 par defaut)
        self.declare_parameter('initial_state', 1)
        initial_int = self.get_parameter('initial_state').value

        if initial_int not in INT_TO_STATE:
            raise ValueError(
                f"initial_state={initial_int} invalide. Valeurs acceptées : {list(INT_TO_STATE.keys())}"
            )
        self.current_state = INT_TO_STATE[initial_int]

        # Gestion du temps pour pas detecter la meme ligne bleue 2 fois de suite (anti rebond)
        # on decale le temps de depart pour pas etre bloque direct au lancement
        # Sur gazebo on met -1 en attendant que la simu demarre vraiment
        if MODE == "IRL":
            self.last_transition_time = self.get_clock().now().nanoseconds / 1e9 - 40.0
        else:
            self.last_transition_time = -1.0

        # on stocke les ordres en attente de chaque noeud
        self.last_twist_ch1 = Twist()
        self.last_twist_ch2 = Twist()
        self.last_twist_ch3 = Twist()
        self.last_twist_ch4 = Twist()
        self.last_twist_line = Twist()   # line_follower tout seul

        # on ecoute le detecteur de ligne pour le bleu
        self.sub_blue_line = self.create_subscription(
            Bool, '/blue_line_crossed', self.state_callback, 10)

        # on s'abonne aux topics de tous nos challenges
        self.sub_cmd_ch1  = self.create_subscription(Twist, '/cmd_vel_challenge_1', self.cmd_ch1_callback,  10)
        self.sub_cmd_ch2  = self.create_subscription(Twist, '/cmd_vel_challenge_2', self.cmd_ch2_callback,  10)
        self.sub_cmd_ch3  = self.create_subscription(Twist, '/cmd_vel_challenge_3', self.cmd_ch3_callback,  10)
        self.sub_cmd_ch4  = self.create_subscription(Twist, '/cmd_vel_challenge_4', self.cmd_ch4_callback,  10)
        self.sub_cmd_line = self.create_subscription(Twist, '/cmd_vel_line_raw',    self.cmd_line_callback, 10)

        # le publisher final qui envoie vraiment les vitesses aux roues du turtlebot
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(
            f'Superviseur démarré — MODE={MODE} — État initial : {self.current_state}'
        )


    def state_callback(self, msg):
        """
        callback declenche quand la vision voit la ligne bleue
        """
        if not msg.data:
            return

        # si on est a la fin du circuit on ignore la ligne bleue
        next_state = TRANSITIONS.get(self.current_state)
        if next_state is None:
            self.get_logger().info(
                "Ligne bleue détectée mais état terminal atteint (CHALLENGE_4) — parcours terminé."
            )
            return

        # protection si gazebo a pas encore synchronise son horloge
        if self.last_transition_time == -1.0:
            return

        current_time = self.get_clock().now().nanoseconds / 1e9
        duree_ecoulee = current_time - self.last_transition_time

        # il faut qu'il se passe au moins 3 secondes entre deux lignes bleues
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

    # on met a jour les variables des qu'un noeud publie qqchose
    def cmd_ch1_callback(self,  msg): self.last_twist_ch1  = msg
    def cmd_ch2_callback(self,  msg): self.last_twist_ch2  = msg
    def cmd_ch3_callback(self,  msg): self.last_twist_ch3  = msg
    def cmd_ch4_callback(self,  msg): self.last_twist_ch4  = msg
    def cmd_line_callback(self, msg): self.last_twist_line = msg


    def timer_callback(self):
        """
        boucle principale (10 fois par seconde)
        on regarde dans quel etat on est et on laisse passer que l'ordre de ce challenge
        """

        # pour gazebo on attend le premier tick de l'horloge
        if MODE == "SIMULATION" and self.last_transition_time == -1.0:
            current_time = self.get_clock().now().nanoseconds / 1e9
            if current_time > 0:
                self.last_transition_time = current_time - 40.0
                self.get_logger().info("Horloge simulée synchronisée, timer de sécurité activé.")

        # le switch pour donner les manettes au bon script
        if self.current_state == STATE_CH1:
            twist_to_publish = self.last_twist_ch1
        elif self.current_state == STATE_CH2:
            twist_to_publish = self.last_twist_ch2
        elif self.current_state == STATE_CH3:
            twist_to_publish = self.last_twist_ch3
        elif self.current_state == STATE_LINE_FOLLOW_ONLY:
            # pour la portion de circuit normale on ecoute juste le line follower de base
            twist_to_publish = self.last_twist_line
        elif self.current_state == STATE_CH4:
            twist_to_publish = self.last_twist_ch4
        else:
            twist_to_publish = Twist()   # en cas de bug on stop le robot par securite

        # on envoie
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
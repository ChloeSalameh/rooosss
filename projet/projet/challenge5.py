import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import threading

class Challenge5Gestures(Node):
    """
    Ce programme lit la video de la main, devine le geste,et donne les ordres de conduite au robot
    """
    def __init__(self):
        super().__init__('challenge5_gestures')
        
        # ROS publishers
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_debug = self.create_publisher(Image, '/challenge5_debug', 10) # Pour afficher ce que le robot voit (avec le squelette de la main dessiné)
        
        self.bridge = CvBridge() # Traducteur entre les images d'OpenCV et les images de ROS
        
        # PREPARATION DE L'IA (MediaPipe)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1, # on ne regarde qu'une seule main a la fois
            min_detection_confidence=0.75, # l'IA doit etre sure a 75% que c'est une main pour l'accepter
            min_tracking_confidence=0.75
        )
        self.mp_draw = mp.solutions.drawing_utils # Outil pour dessiner les points sur les doigts
        
        # LECTURE DE LA VIDEO ET GESTION DU LAG
        # on lit le flux video de notre serveur "streaming.py"
        self.cap = cv2.VideoCapture("http://host.docker.internal:8080/video")
        self.latest_frame = None
        self.running = True
        
        # Si on lit la video au meme endroit qu'on calcule l'IA, la video va prendre du retard (lag)
        # On cree donc un mini-programme en arriere-plan (thread) dont le SEUL but est de telecharger la toute derniere image disponible, sans jamais s'arreter
        self.thread = threading.Thread(target=self._reader_thread)
        self.thread.daemon = True
        self.thread.start()
        
        # Timer pour le traitement (30 FPS environ, 30 fois par seconde, on regarde l'image et on agit)
        self.timer = self.create_timer(0.03, self.timer_callback)
        
        # Vitesse du robot
        self.v_linear = 0.15 # pour avancer/reculer
        self.v_angular = 0.6 # pour tourner
        
        self.get_logger().info("Challenge 5")

    def _reader_thread(self):
        """Lit les images en continu pour vider le buffer d'OpenCV. Il recupere les images le plus vite possible 
        pour qu'on ait toujours la vue la plus recente de la camera
        """
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.latest_frame = frame

    def get_gesture_label(self, hand_landmarks):
        """
        Le Dictionnaire des gestes.
        Il regarde quels doigts sont leves et decide de l'action a faire
        """
        tip_ids = [4, 8, 12, 16, 20] #les points qui correspondent au bout de chaque doigt
        fingers = []

        # est-ce que le Pouce est leve? (C'est un peu different des autres doigts a cause de sa position)
        if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
            fingers.append(1) # 1 = levé
        else:
            fingers.append(0) # 0 = baissé

        # est-ce que les 4 autres doigts sont levés ?
        for id in range(1, 5):
            if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)

        total_fingers = fingers.count(1) #nb total de doigts levés 
        
        #la traduction des gestes en ordres pour le robot
        if total_fingers == 0: return "FIST", "STOP"
        elif fingers == [0, 1, 0, 0, 0]: return "INDEX UP", "AVANCER"
        elif fingers == [0, 1, 1, 0, 0]: return "PEACE", "RECULER"
        elif fingers[0] == 1 and total_fingers == 1: return "POUCE", "TOURNER_GAUCHE"
        elif fingers == [0, 0, 0, 0, 1]: return "PINKY UP", "TOURNER_DROITE"
        elif total_fingers == 5: return "MAIN OUVERTE", "STOP"
            
        return "UNKNOWN", "STOP" # Si on fait n'importe quoi avec nos doigts, il s'arrete

    def timer_callback(self):
        """
        La boucle principale qui s'active 30 fois par seconde.
        """
        # On verifie si on a recu au moins une image
        if self.latest_frame is None:
            return
            
        # On fait une copie de la derniere image recue par le thread
        frame = self.latest_frame.copy()
        
        # On fait un effet miroir sur l'image et on change les couleurs pour que l'IA puisse la comprendre
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(img_rgb) #on demande a l'IA s'il y a une main
        
        cmd = Twist() # La commande vide pour le robot
        action = "STOP" # Par defaut, on s'arrete
        gesture = "NONE" # Par defaut, aucun geste
        
        if results.multi_hand_landmarks: # Si l'IA trouve une main
            for hand_landmarks in results.multi_hand_landmarks:
                # On dessine le squelette de la main sur l'image pour l'utilisateur
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # On devine l'action a faire
                gesture, action = self.get_gesture_label(hand_landmarks)
                
                # On remplit la commande de mouvement selon l'action voulue
                if action == "AVANCER": cmd.linear.x = self.v_linear
                elif action == "RECULER": cmd.linear.x = -self.v_linear
                elif action == "TOURNER_GAUCHE": cmd.angular.z = self.v_angular
                elif action == "TOURNER_DROITE": cmd.angular.z = -self.v_angular

        # On ecrit du texte sur l'image
        cv2.putText(frame, f"Gesture: {gesture}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Action: {action}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # On envoie l'ordre aux roues du robot
        self.pub_cmd.publish(cmd)
        
        # On partage l'image sur ROS pour pouvoir la regarder
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.pub_debug.publish(debug_msg)
        except Exception:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = Challenge5Gestures()
    try: 
        rclpy.spin(node)
    except KeyboardInterrupt: 
        pass
    finally:
        node.running = False
        node.cap.release()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
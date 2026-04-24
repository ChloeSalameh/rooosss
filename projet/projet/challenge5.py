import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import threading

class Challenge5Gestures(Node):
    def __init__(self):
        super().__init__('challenge5_gestures')
        
        # 1. ROS 2 Publishers
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_debug = self.create_publisher(Image, '/challenge5_debug', 10)
        
        self.bridge = CvBridge()
        
        # 2. MediaPipe Initialization
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # 3. Video Capture & Threading for Latency
        self.cap = cv2.VideoCapture("http://host.docker.internal:8080/video")
        self.latest_frame = None
        self.running = True
        
        # Lancement du thread de lecture
        self.thread = threading.Thread(target=self._reader_thread)
        self.thread.daemon = True
        self.thread.start()
        
        # Timer pour le traitement (30 FPS environ)
        self.timer = self.create_timer(0.03, self.timer_callback)
        
        # Speeds
        self.v_linear = 0.15
        self.v_angular = 0.6
        
        self.get_logger().info("Challenge 5: Streaming Gesture Control Started (No-Latency Mode)!")

    def _reader_thread(self):
        """Lit les images en continu pour vider le buffer d'OpenCV."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.latest_frame = frame

    def get_gesture_label(self, hand_landmarks):
        tip_ids = [4, 8, 12, 16, 20]
        fingers = []

        # Thumb
        if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
            fingers.append(1) 
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)

        total_fingers = fingers.count(1)
        
        if total_fingers == 0: return "FIST", "STOP"
        elif fingers == [0, 1, 0, 0, 0]: return "INDEX UP", "FORWARD"
        elif fingers == [0, 1, 1, 0, 0]: return "PEACE", "BACKWARD"
        elif fingers[0] == 1 and total_fingers == 1: return "THUMB LEFT", "TURN_LEFT"
        elif fingers == [0, 0, 0, 0, 1]: return "PINKY UP", "TURN_RIGHT"
        elif total_fingers == 5: return "OPEN PALM", "STOP"
            
        return "UNKNOWN", "STOP"

    def timer_callback(self):
        # On vérifie si on a reçu au moins une image
        if self.latest_frame is None:
            return
            
        # On fait une copie de la dernière image reçue par le thread
        frame = self.latest_frame.copy()
            
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        cmd = Twist()
        action = "STOP"
        gesture = "NONE"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                gesture, action = self.get_gesture_label(hand_landmarks)
                
                if action == "FORWARD": cmd.linear.x = self.v_linear
                elif action == "BACKWARD": cmd.linear.x = -self.v_linear
                elif action == "TURN_LEFT": cmd.angular.z = self.v_angular
                elif action == "TURN_RIGHT": cmd.angular.z = -self.v_angular

        # UI
        cv2.putText(frame, f"Gesture: {gesture}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Action: {action}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        self.pub_cmd.publish(cmd)
        
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
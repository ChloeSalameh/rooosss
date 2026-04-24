import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Int32, Int32MultiArray
from cv_bridge import CvBridge
import numpy as np
import cv2


# MODE : choisir ici IRL ou SIMULATION :
#      -"IRL"        => topic CompressedImage /camera/image_raw/compressed
#      -"SIMULATION" => topic Image           /image_raw

MODE = "IRL"

# Nombre de bandes horizontales d'analyse
N_BANDS = 5

# ZONE D'ANALYSE :
ZONE_TOP_FRAC = 0.50   # fraction haute de la zone d'analyse (0.0 = tout en haut)
ZONE_BOT_FRAC = 1.00   # fraction basse de la zone d'analyse (1.0 = tout en bas)

# Pourcentage minimum de pixels dans une bande pour valider un centroide
# (0.001 = 0.1% — robuste au bruit, valeur exacte de l'antérieur)
MIN_AREA_PCT = 0.001

# Nombre minimum de pixels bleus pour déclencher la détection
BLUE_MIN_PX  = 500


# SEUILS HSV FORMAT : [H_min, H_max, S_min, S_max, V_min, V_max]

if MODE == "IRL" :
    HSV_DEFAULT_RED1 = [  0,   5,  20, 255, 120, 255]   # premier arc rouge (teintes basses)
    HSV_DEFAULT_RED2 = [130, 180,  28, 255, 54, 255]    # second  arc rouge (teintes hautes)
    HSV_DEFAULT_GREEN = [ 55, 92,  25, 255, 80, 255]    # vert
    HSV_DEFAULT_BLUE  = [ 95, 140, 113, 255, 120, 255]  # bleu

elif MODE == "SIMULATION":
    HSV_DEFAULT_RED1  = [  0,  10, 100, 255,  80, 255]
    HSV_DEFAULT_RED2  = [165, 180, 100, 255,  80, 255]
    HSV_DEFAULT_GREEN = [ 38,  85,  80, 255,  50, 255]
    HSV_DEFAULT_BLUE  = [100, 140, 120, 255,  60, 255]

else :
    raise ValueError(f"MODE invalide : '{MODE}'. Choisissez 'IRL' ou 'SIMULATION'.")


def _null(_): pass   # callback vide obligatoire pour cv2.createTrackbar


# TRACKBARS DES SEUILS HSV

def _create_hsv_window(win_name: str, defaults: list):
    """
    Crée une fenêtre OpenCV avec 6 trackbars H/S/V min/max.

    :param win_name:  nom de la fenêtre (affiché dans la barre de titre)
    :param defaults:  liste [H_min, H_max, S_min, S_max, V_min, V_max]
    """
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('H_min', win_name, defaults[0], 179, _null)
    cv2.createTrackbar('H_max', win_name, defaults[1], 179, _null)
    cv2.createTrackbar('S_min', win_name, defaults[2], 255, _null)
    cv2.createTrackbar('S_max', win_name, defaults[3], 255, _null)
    cv2.createTrackbar('V_min', win_name, defaults[4], 255, _null)
    cv2.createTrackbar('V_max', win_name, defaults[5], 255, _null)


def _read_hsv_window(win_name: str):
    """
    Lit les 6 trackbars d'une fenêtre et retourne (lo, hi) sous forme
    de np.array prêts pour cv2.inRange.
    """
    h_min = cv2.getTrackbarPos('H_min', win_name)
    h_max = cv2.getTrackbarPos('H_max', win_name)
    s_min = cv2.getTrackbarPos('S_min', win_name)
    s_max = cv2.getTrackbarPos('S_max', win_name)
    v_min = cv2.getTrackbarPos('V_min', win_name)
    v_max = cv2.getTrackbarPos('V_max', win_name)
    lo = np.array([h_min, s_min, v_min], dtype=np.uint8)
    hi = np.array([h_max, s_max, v_max], dtype=np.uint8)
    return lo, hi


# Noms des fenêtres trackbars (utilisés pour la création ET la lecture)
WIN_RED1  = "HSV - Rouge arc 1  (teintes basses  0-10)"
WIN_RED2  = "HSV - Rouge arc 2  (teintes hautes 165-180)"
WIN_GREEN = "HSV - Vert"
WIN_BLUE  = "HSV - Bleu"

class LineDetector(Node):
    def __init__(self):
        super().__init__('line_detector')
        self.blue_detected_previously = False
        self.bridge = CvBridge()

        # Validation du mode
        if MODE not in ("IRL", "SIMULATION"):
            raise ValueError(f"MODE invalide : '{MODE}'. Choisir 'IRL' ou 'SIMULATION'.")

        # Abonnement selon le mode
        if MODE == "IRL":
            self.subscription = self.create_subscription(
                CompressedImage, '/camera/image_raw/compressed',
                self.listener_callback, 10)
        else:
            self.subscription = self.create_subscription(
                Image, '/image_raw',
                self.listener_callback, 10)

        # Topics multicouches (identiques à l'antérieur)
        self.pub_red_bands   = self.create_publisher(Int32MultiArray, '/red_bands_cx',    10)
        self.pub_green_bands = self.create_publisher(Int32MultiArray, '/green_bands_cx',  10)

        # Topics legacy
        self.pub_red_pos   = self.create_publisher(Int32, '/red_line_pos',   10)
        self.pub_green_pos = self.create_publisher(Int32, '/green_line_pos', 10)

        # Topics ligne bleue et largeur caméra
        self.pub_blue      = self.create_publisher(Bool,  '/blue_line_crossed', 10)
        self.pub_cam_width = self.create_publisher(Int32, '/camera_width',      10)

        # Topics de débogage visuel
        self.pub_debug_nette = self.create_publisher(Image, '/debug/vision_nette', 10)
        self.pub_debug_blue  = self.create_publisher(Image, '/debug/mask_blue',    10)

        # Création des fenêtres trackbars HSV
        # Chaque couleur dispose de sa propre fenêtre avec 6 curseurs.
        # Les valeurs initiales correspondent aux seuils calibrés de l'antérieur.
        _create_hsv_window(WIN_RED1,  HSV_DEFAULT_RED1)
        _create_hsv_window(WIN_RED2,  HSV_DEFAULT_RED2)
        _create_hsv_window(WIN_GREEN, HSV_DEFAULT_GREEN)
        _create_hsv_window(WIN_BLUE,  HSV_DEFAULT_BLUE)

        self.get_logger().info(
            f"LineDetector démarré — MODE={MODE} | {N_BANDS} bandes "
            f"| zone [{ZONE_TOP_FRAC:.0%}–{ZONE_BOT_FRAC:.0%}] "
            f"| trackbars HSV actives"
        )

    def listener_callback(self, msg):

        # Décodage de l'image selon le mode
        if MODE == "IRL":
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        else:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        h, w, _ = frame.shape
        img_area = w * h

        self.pub_cam_width.publish(Int32(data=w))

        # Conversion en HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        cv2.imshow("Image Brute", frame)

        # Lecture des seuils HSV depuis les trackbars
        # Les masques se mettent à jour en temps réel à chaque frame.
        r_lo1, r_hi1 = _read_hsv_window(WIN_RED1)
        r_lo2, r_hi2 = _read_hsv_window(WIN_RED2)
        g_lo,  g_hi  = _read_hsv_window(WIN_GREEN)
        b_lo,  b_hi  = _read_hsv_window(WIN_BLUE)

        # Construction des masques couleur
        mask_r_full = cv2.bitwise_or(
            cv2.inRange(hsv, r_lo1, r_hi1),
            cv2.inRange(hsv, r_lo2, r_hi2)
        )
        mask_g_full = cv2.inRange(hsv, g_lo, g_hi)
        mask_blue   = cv2.inRange(hsv, b_lo, b_hi)

        # Segmentation en N_BANDS bandes
        y_start = int(h * ZONE_TOP_FRAC)
        y_end   = int(h * ZONE_BOT_FRAC)
        zone_height = (y_end - y_start) // N_BANDS

        # Détection LIGNE BLEUE restreinte à la bande la plus proche (dernière bande)
        y_blue_top    = y_start + (N_BANDS - 1) * zone_height
        y_blue_bottom = y_blue_top + zone_height

        mask_blue_roi = mask_blue.copy()
        mask_blue_roi[0:y_blue_top, :]    = 0
        mask_blue_roi[y_blue_bottom:h, :] = 0

        pixels_bleus = cv2.countNonZero(mask_blue_roi)
        is_blue_now  = pixels_bleus > BLUE_MIN_PX

        if is_blue_now and not self.blue_detected_previously:
            self.get_logger().info("🔵 Ligne Bleue Détectée !")
            self.pub_blue.publish(Bool(data=True))

        self.blue_detected_previously = is_blue_now

        zone_height = (y_end - y_start) // N_BANDS

        cx_red_bands   = []
        cx_green_bands = []

        band_area = w * zone_height
        min_area  = band_area * MIN_AREA_PCT

        for i in range(N_BANDS):
            y_top_band = y_start + i * zone_height
            y_bot_band = y_top_band + zone_height

            # Isolation de la bande (copie + effacement hors bande)
            mask_r_band = mask_r_full.copy()
            mask_r_band[0:y_top_band, :] = 0
            mask_r_band[y_bot_band:h, :] = 0

            mask_g_band = mask_g_full.copy()
            mask_g_band[0:y_top_band, :] = 0
            mask_g_band[y_bot_band:h, :] = 0

            cx_red_bands.append(self._get_cx(mask_r_band, min_area))
            cx_green_bands.append(self._get_cx(mask_g_band, min_area))

        # Publication multicouches
        msg_red         = Int32MultiArray()
        msg_red.data    = cx_red_bands
        msg_green       = Int32MultiArray()
        msg_green.data  = cx_green_bands

        self.pub_red_bands.publish(msg_red)
        self.pub_green_bands.publish(msg_green)

        # Publication legacy (bande la plus proche = indice N-1)
        self.pub_red_pos.publish(Int32(data=cx_red_bands[-1]))
        self.pub_green_pos.publish(Int32(data=cx_green_bands[-1]))

        # Rendu coloré du masque de débogage
        # Création d'une image BGR vide sur laquelle on colorie :
        #   - pixels rouges  → (0, 0, 255)  BGR rouge
        #   - pixels verts   → (0, 200, 0)  BGR vert
        #   - pixels bleus   → (255, 50, 0) BGR bleu  (depuis la ROI bleue)
        # Les séparateurs de bandes sont tracés en blanc.
        debug_color = np.zeros((h, w, 3), dtype=np.uint8)

        # On restreint l'affichage rouge/vert à la zone analysée
        roi_r = np.zeros_like(mask_r_full)
        roi_r[y_start:y_start + zone_height * N_BANDS, :] = \
            mask_r_full[y_start:y_start + zone_height * N_BANDS, :]

        roi_g = np.zeros_like(mask_g_full)
        roi_g[y_start:y_start + zone_height * N_BANDS, :] = \
            mask_g_full[y_start:y_start + zone_height * N_BANDS, :]

        debug_color[roi_r == 255]      = (  0,   0, 255)   # rouge
        debug_color[roi_g == 255]      = (  0, 200,   0)   # vert
        debug_color[mask_blue_roi == 255] = (255,  50,   0)   # bleu (ROI basse)

        # Séparateurs de bandes (blancs) + numérotation
        for i in range(1, N_BANDS):
            y_sep = y_start + i * zone_height
            cv2.line(debug_color, (0, y_sep), (w, y_sep), (255, 255, 255), 1)

        # Ligne jaune marquant la frontière haute de la zone analysée
        cv2.line(debug_color, (0, y_start), (w, y_start), (0, 255, 255), 2)

        # Numéro de bande (0 = loin, N-1 = proche)
        for i in range(N_BANDS):
            y_label = y_start + i * zone_height + zone_height // 2
            cv2.putText(debug_color, str(i), (4, y_label),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow("Vision du Masque (multicouche)", debug_color)
        cv2.waitKey(1)

        # Publication des images de débogage ROS (mono8 pour la rétrocompatibilité)
        mask_nette   = cv2.bitwise_or(roi_r, roi_g)
        msg_nette    = self.bridge.cv2_to_imgmsg(mask_nette, encoding="mono8")
        msg_blue_img = self.bridge.cv2_to_imgmsg(mask_blue,  encoding="mono8")

        self.pub_debug_nette.publish(msg_nette)
        self.pub_debug_blue.publish(msg_blue_img)

    def _get_cx(self, mask, min_area):
        M = cv2.moments(mask)
        if M['m00'] > min_area:
            return int(M['m10'] / M['m00'])
        return -1


def main(args=None):
    rclpy.init(args=args)
    node = LineDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
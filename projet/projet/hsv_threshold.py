import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# =============================================================================
# hsv_threshold.py — Calibration HSV en temps réel (outil autonome)
# =============================================================================
#
# Ce script s'exécute INDÉPENDAMMENT du pipeline de navigation.
# Il affiche quatre fenêtres de trackbars (Rouge arc1, Rouge arc2, Vert, Bleu)
# et trois fenêtres de résultat (masque, résultat bitwise_and) pour chaque
# couleur afin d'ajuster visuellement les seuils HSV.
#
# Une fois les valeurs satisfaisantes, recopier-les dans les constantes
# HSV_DEFAULT_* en haut de line_detector.py.
#
# USAGE :
#   ros2 run <package> hsv_threshold          # (si installé dans un package ROS 2)
#   python3 hsv_threshold.py                  # (exécution directe)
#
# TOPICS ÉCOUTES (selon MODE) :
#   IRL        : /camera/image_raw/compressed  (CompressedImage)
#   SIMULATION : /image_raw                    (Image)
#
# TOUCHES CLAVIER dans n'importe quelle fenêtre OpenCV :
#   's' : sauvegarde un snapshot de l'image brute (snapshot_<N>.png)
#   'q' : quitter proprement
# =============================================================================


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CHOISIR LE MODE ICI                                                   ║
# ║    MODE = "IRL"        → CompressedImage /camera/image_raw/compressed  ║
# ║    MODE = "SIMULATION" → Image           /image_raw                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
MODE = "IRL"   # <── MODIFIER ICI


# =============================================================================
# VALEURS INITIALES DES TRACKBARS
# =============================================================================
# Correspondent aux seuils calibrés IRL de line_detector_anterieur.py.
# Adapter si vous êtes en mode SIMULATION.
#
# FORMAT : [H_min, H_max, S_min, S_max, V_min, V_max]

INIT_RED1  = [  0,   5,  20, 255, 120, 255]   # Rouge — arc 1 (teintes  0–5)
INIT_RED2  = [165, 180,  20, 255, 120, 255]   # Rouge — arc 2 (teintes 165–180)
INIT_GREEN = [ 75, 105,  25, 255, 100, 255]   # Vert
INIT_BLUE  = [ 95, 120, 100, 255, 120, 255]   # Bleu

# Valeurs SIMULATION (décommenter et commenter celles au-dessus si besoin) :
# INIT_RED1  = [  0,  10, 100, 255,  80, 255]
# INIT_RED2  = [165, 180, 100, 255,  80, 255]
# INIT_GREEN = [ 38,  85,  80, 255,  50, 255]
# INIT_BLUE  = [100, 140, 120, 255,  60, 255]


# =============================================================================
# Noms des fenêtres (trackbars + résultats)
# =============================================================================

WIN_TB_R1    = "Trackbars — Rouge arc 1  (H 0–10)"
WIN_TB_R2    = "Trackbars — Rouge arc 2  (H 165–180)"
WIN_TB_GREEN = "Trackbars — Vert"
WIN_TB_BLUE  = "Trackbars — Bleu"

WIN_RES_RED   = "Résultat — Rouge (arc1 OR arc2)"
WIN_RES_GREEN = "Résultat — Vert"
WIN_RES_BLUE  = "Résultat — Bleu"
WIN_ORIGINAL  = "Image brute (HSV Calibration)"


# =============================================================================
# Helpers
# =============================================================================

def _null(_): pass


def _create_trackbars(win_name: str, defaults: list):
    """Crée une fenêtre avec 6 trackbars H/S/V min/max."""
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('H_min', win_name, defaults[0], 179, _null)
    cv2.createTrackbar('H_max', win_name, defaults[1], 179, _null)
    cv2.createTrackbar('S_min', win_name, defaults[2], 255, _null)
    cv2.createTrackbar('S_max', win_name, defaults[3], 255, _null)
    cv2.createTrackbar('V_min', win_name, defaults[4], 255, _null)
    cv2.createTrackbar('V_max', win_name, defaults[5], 255, _null)


def _read_trackbars(win_name: str):
    """Lit les 6 trackbars et retourne (lo, hi) pour cv2.inRange."""
    h_min = cv2.getTrackbarPos('H_min', win_name)
    h_max = cv2.getTrackbarPos('H_max', win_name)
    s_min = cv2.getTrackbarPos('S_min', win_name)
    s_max = cv2.getTrackbarPos('S_max', win_name)
    v_min = cv2.getTrackbarPos('V_min', win_name)
    v_max = cv2.getTrackbarPos('V_max', win_name)
    lo = np.array([h_min, s_min, v_min], dtype=np.uint8)
    hi = np.array([h_max, s_max, v_max], dtype=np.uint8)
    return lo, hi


def _print_values(label: str, lo: np.ndarray, hi: np.ndarray):
    """Affiche dans le terminal les valeurs courantes — pratique pour les copier."""
    print(f"  {label:<20}  lo={list(lo)}  hi={list(hi)}")


# =============================================================================
# Nœud de calibration
# =============================================================================

class HSVThreshold(Node):
    def __init__(self):
        super().__init__('hsv_threshold')

        if MODE not in ("IRL", "SIMULATION"):
            raise ValueError(f"MODE invalide : '{MODE}'. Choisir 'IRL' ou 'SIMULATION'.")

        self.bridge        = CvBridge()
        self._snap_counter = 0

        # ── Abonnement image selon le mode ────────────────────────────────────
        if MODE == "IRL":
            self.subscription = self.create_subscription(
                CompressedImage, '/camera/image_raw/compressed',
                self.image_callback, 10)
            self.get_logger().info("hsv_threshold — mode IRL (/camera/image_raw/compressed)")
        else:
            self.subscription = self.create_subscription(
                Image, '/image_raw',
                self.image_callback, 10)
            self.get_logger().info("hsv_threshold — mode SIMULATION (/image_raw)")

        # ── Création des fenêtres trackbars ───────────────────────────────────
        # Chaque couleur a sa propre fenêtre pour éviter les conflits de noms.
        _create_trackbars(WIN_TB_R1,    INIT_RED1)
        _create_trackbars(WIN_TB_R2,    INIT_RED2)
        _create_trackbars(WIN_TB_GREEN, INIT_GREEN)
        _create_trackbars(WIN_TB_BLUE,  INIT_BLUE)

        # Fenêtres de résultat (pas de trackbars)
        for win in (WIN_ORIGINAL, WIN_RES_RED, WIN_RES_GREEN, WIN_RES_BLUE):
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        self.get_logger().info(
            "Trackbars ouvertes. Appuyer sur 's' pour snapshot, 'q' pour quitter."
        )

    # =========================================================================
    # Callback image
    # =========================================================================

    def image_callback(self, msg):
        # Décodage
        if MODE == "IRL":
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        else:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        if frame is None:
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ── Lecture des trackbars pour chaque couleur ─────────────────────────
        r1_lo, r1_hi = _read_trackbars(WIN_TB_R1)
        r2_lo, r2_hi = _read_trackbars(WIN_TB_R2)
        g_lo,  g_hi  = _read_trackbars(WIN_TB_GREEN)
        b_lo,  b_hi  = _read_trackbars(WIN_TB_BLUE)

        # ── Calcul des masques ────────────────────────────────────────────────
        mask_r1   = cv2.inRange(hsv, r1_lo, r1_hi)
        mask_r2   = cv2.inRange(hsv, r2_lo, r2_hi)
        mask_red  = cv2.bitwise_or(mask_r1, mask_r2)   # union des deux arcs
        mask_g    = cv2.inRange(hsv, g_lo, g_hi)
        mask_b    = cv2.inRange(hsv, b_lo, b_hi)

        # ── Résultat bitwise_and (n'affiche que les pixels sélectionnés) ──────
        res_red   = cv2.bitwise_and(frame, frame, mask=mask_red)
        res_green = cv2.bitwise_and(frame, frame, mask=mask_g)
        res_blue  = cv2.bitwise_and(frame, frame, mask=mask_b)

        # ── Affichage ─────────────────────────────────────────────────────────
        cv2.imshow(WIN_ORIGINAL,  frame)
        cv2.imshow(WIN_RES_RED,   res_red)
        cv2.imshow(WIN_RES_GREEN, res_green)
        cv2.imshow(WIN_RES_BLUE,  res_blue)

        # ── Gestion clavier ───────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # Snapshot de l'image brute
            fname = f"snapshot_{self._snap_counter:04d}.png"
            cv2.imwrite(fname, frame)
            self.get_logger().info(f"Snapshot sauvegardé : {fname}")
            self._snap_counter += 1

        elif key == ord('p'):
            # Affichage des valeurs courantes dans le terminal pour les copier
            print("\n── Valeurs HSV actuelles ──────────────────────────────")
            _print_values("Rouge arc1",  r1_lo, r1_hi)
            _print_values("Rouge arc2",  r2_lo, r2_hi)
            _print_values("Vert",        g_lo,  g_hi)
            _print_values("Bleu",        b_lo,  b_hi)
            print("────────────────────────────────────────────────────────\n")

        elif key == ord('q'):
            self.get_logger().info("Arrêt demandé par l'utilisateur ('q').")
            raise KeyboardInterrupt


# =============================================================================
# Point d'entrée
# =============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = HSVThreshold()
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

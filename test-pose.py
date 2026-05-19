"""
test-pose.py
Runs MediaPipe Pose Landmarker, computes Euler XYZ angles (0-360) for
both legs (thigh, shin, foot) relative to their parent bones, writes
euler_degrees into pose.json. No quaternions anywhere.

Bone chain and parent-relative logic:
  torso   -> thighL / thighR   (rest: straight down from torso)
  thighL  -> shinL             (rest: continues along thigh direction)
  thighR  -> shinR             (rest: continues along thigh direction)
  shinL   -> footL             (rest: continues along shin direction)
  shinR   -> footR             (rest: continues along shin direction)

For each bone we build the PARENT's local coordinate frame, express
the child bone direction in that frame, then decompose the rotation
from the rest direction [0,-1,0] (or [0,0,-1] for feet) into Euler XYZ.
"""

import sys
import json
import urllib.request
import shutil
import os
import math

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)
MODEL_PATH         = "pose_landmarker.task"
DEFAULT_IMAGE_PATH = "test-image.png"
OUTPUT_JSON        = "pose.json"
OUTPUT_ANNOTATED   = "output_landmarks.jpg"

WHITE  = (255, 255, 255)
CYAN   = (0, 255, 255)
YELLOW = (0, 255, 255)
GREEN  = (0, 255, 0)
ORANGE = (0, 165, 255)
PINK   = (180, 105, 255)

POSE_CONNECTIONS = [
    (0, 7), (0, 8), (0, 9), (0, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]
HEAD_LANDMARKS  = {0, 7, 8, 9, 10}
ARM_LANDMARKS   = {11, 12, 13, 14, 15, 16, 19, 20}
BODY_LANDMARKS  = set(range(23, 33))
DRAWN_LANDMARKS = HEAD_LANDMARKS | ARM_LANDMARKS | BODY_LANDMARKS


# ── Helpers ────────────────────────────────────────────────────────────────────

def download_file(url, dest):
    if os.path.exists(dest):
        return
    print("[INFO] Downloading model...")
    urllib.request.urlretrieve(url, dest)


def mp_to_world(lm):
    """MediaPipe (+X right, +Y down, +Z toward cam) -> Y-up world."""
    return np.array([lm.x, -lm.y, -lm.z], dtype=np.float64)


def midpoint(a, b):
    class _Lm: pass
    m = _Lm()
    m.x = (a.x + b.x) / 2
    m.y = (a.y + b.y) / 2
    m.z = (a.z + b.z) / 2
    m.visibility = min(getattr(a, 'visibility', 1.0), getattr(b, 'visibility', 1.0))
    return m


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def build_local_frame(parent_dir, hint_right):
    """Build an orthonormal frame from a parent bone direction.

    Args:
        parent_dir: unit vector along the parent bone (proximal -> distal)
        hint_right: a hint vector for the "right" direction (e.g. hip line)

    Returns:
        (local_up, local_right, local_forward) orthonormal basis where
        local_up = parent_dir (the bone's primary axis)
    """
    up = normalize(parent_dir)

    # Gram-Schmidt: make hint_right perpendicular to up
    right = hint_right - np.dot(hint_right, up) * up
    n = np.linalg.norm(right)
    if n < 1e-6:
        # hint_right is parallel to up, pick an arbitrary perpendicular
        perp = np.array([1, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1, 0])
        right = perp - np.dot(perp, up) * up
    right = normalize(right)

    forward = normalize(np.cross(up, right))
    return up, right, forward


def bone_euler_from_parent(child_dir, parent_dir, hint_right, rest_dir_local=None):
    """Compute Euler XYZ angles (0-360) for a child bone relative to parent.

    The child bone direction is expressed in the parent's local frame,
    then the rotation from the rest direction to the observed direction
    is decomposed into Euler XYZ.

    Args:
        child_dir:  unit vector of the child bone in world space
        parent_dir: unit vector of the parent bone in world space
        hint_right: hint for building the parent's local frame
        rest_dir_local: rest direction in local frame (default: [0, -1, 0] = straight down)

    Returns:
        (x_deg, y_deg, z_deg) in 0-360 range
    """
    up, right, forward = build_local_frame(parent_dir, hint_right)

    # Project child direction into parent local frame
    cx = np.dot(child_dir, right)
    cy = np.dot(child_dir, up)
    cz = np.dot(child_dir, forward)

    if rest_dir_local is None:
        # Default rest: straight down in parent space = [0, -1, 0]
        # Only X rotation (forward/backward swing in Y-Z plane)
        x_deg = math.degrees(math.atan2(-cz, -cy)) % 360
    else:
        x_deg = math.degrees(math.atan2(-cz, -cy)) % 360

    return x_deg, 0.0, 0.0


def total_angle(v1, v2):
    """Total angle in degrees between two unit vectors."""
    return math.degrees(math.acos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


# ── Drawing ────────────────────────────────────────────────────────────────────

def draw_landmarks_on_image(rgb_image, detection_result):
    img = np.copy(rgb_image)
    h, w = img.shape[:2]

    if detection_result.segmentation_masks:
        mask   = np.squeeze(detection_result.segmentation_masks[0].numpy_view())
        blue   = np.zeros_like(img); blue[:] = (0, 0, 255)
        smooth = cv2.GaussianBlur(mask, (15, 15), 0)
        alpha  = (smooth[..., np.newaxis] * 0.6).astype(np.float32)
        img    = (img.astype(np.float32) * (1 - alpha) + blue.astype(np.float32) * alpha).astype(np.uint8)

    for pose_landmarks in detection_result.pose_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in pose_landmarks]
        for s, e in POSE_CONNECTIONS:
            if s < len(pts) and e < len(pts):
                cv2.line(img, pts[s], pts[e], WHITE, 3, cv2.LINE_AA)
        for idx in DRAWN_LANDMARKS:
            if idx < len(pts):
                cv2.circle(img, pts[idx], 6, WHITE, 2, cv2.LINE_AA)

        l_sh, r_sh   = pts[11], pts[12]
        l_hip, r_hip = pts[23], pts[24]
        torso_top    = ((l_sh[0]+r_sh[0])//2, (l_sh[1]+r_sh[1])//2)
        torso_bottom = ((l_hip[0]+r_hip[0])//2, (l_hip[1]+r_hip[1])//2)
        cv2.circle(img, torso_top,    6, CYAN, -1, cv2.LINE_AA)
        cv2.circle(img, torso_bottom, 6, CYAN, -1, cv2.LINE_AA)
        cv2.line(img, torso_bottom, torso_top, CYAN, 2, cv2.LINE_AA)
    return img


def draw_bone_angle(img, joint_px, parent_dir_px, child_px, label, color, font,
                    x_deg, z_deg, tot_deg):
    """Draw the angle annotation for a single bone joint."""
    h_img, w_img = img.shape[:2]
    parent_norm = normalize(parent_dir_px)

    # Dashed rest line
    for i in range(0, 80, 10):
        p1 = (joint_px + parent_norm * i).astype(int)
        p2 = (joint_px + parent_norm * min(i + 6, 80)).astype(int)
        cv2.line(img, tuple(p1), tuple(p2), GREEN, 1, cv2.LINE_AA)

    # Actual bone line
    cv2.line(img, tuple(joint_px), tuple(child_px), color, 2, cv2.LINE_AA)

    # Arc
    arc_r = 30
    rest_ang  = np.degrees(np.arctan2(parent_norm[1], parent_norm[0]))
    child_vec = child_px - joint_px
    child_ang = np.degrees(np.arctan2(child_vec[1], child_vec[0]))
    sa, ea = min(rest_ang, child_ang), max(rest_ang, child_ang)
    if ea - sa > 180:
        sa, ea = ea, sa + 360
    cv2.ellipse(img, tuple(joint_px), (arc_r, arc_r), 0, sa, ea, color, 1, cv2.LINE_AA)

    # Text
    pos = (joint_px[0] + 12, joint_px[1] - 8)
    text = f"{label} {tot_deg:.0f}d X:{x_deg:.0f} Z:{z_deg:.0f}"
    cv2.putText(img, text, pos, font, 0.35, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, 0.35, color,   1, cv2.LINE_AA)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(image_path=DEFAULT_IMAGE_PATH):
    download_file(MODEL_URL, MODEL_PATH)

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        output_segmentation_masks=False,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Could not load: {image_path}")
    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    elif raw.shape[2] == 4:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
    else:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=raw)
    result   = detector.detect(mp_image)

    if not result.pose_landmarks:
        print("[ERROR] No pose detected.")
        sys.exit(1)

    annotated     = draw_landmarks_on_image(mp_image.numpy_view(), result)
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    lm      = result.pose_landmarks[0]
    hip_mid = midpoint(lm[23], lm[24])
    sho_mid = midpoint(lm[11], lm[12])
    h_img, w_img = raw.shape[:2]

    # ── World-space positions ──────────────────────────────────────────────────
    # MediaPipe landmark indices:
    #   23=L hip, 24=R hip, 25=L knee, 26=R knee,
    #   27=L ankle, 28=R ankle, 29=L heel, 30=R heel, 31=L toe, 32=R toe
    w_hip_mid = mp_to_world(hip_mid)
    w_sho_mid = mp_to_world(sho_mid)

    w_l_hip   = mp_to_world(lm[23])
    w_r_hip   = mp_to_world(lm[24])
    w_l_knee  = mp_to_world(lm[25])
    w_r_knee  = mp_to_world(lm[26])
    w_l_ankle = mp_to_world(lm[27])
    w_r_ankle = mp_to_world(lm[28])
    w_l_toe   = mp_to_world(lm[31])
    w_r_toe   = mp_to_world(lm[32])

    # Arms: 11=L shoulder, 12=R shoulder, 13=L elbow, 14=R elbow, 15=L wrist, 16=R wrist
    w_l_shoulder = mp_to_world(lm[11])
    w_r_shoulder = mp_to_world(lm[12])
    w_l_elbow    = mp_to_world(lm[13])
    w_r_elbow    = mp_to_world(lm[14])
    w_l_wrist    = mp_to_world(lm[15])
    w_r_wrist    = mp_to_world(lm[16])

    # ── Torso frame ────────────────────────────────────────────────────────────
    torso_dir = normalize(w_sho_mid - w_hip_mid)   # torso points UP
    torso_down = -torso_dir                          # thigh rest direction

    hip_line = normalize(w_r_hip - w_l_hip)         # left-to-right across hips
    # Build torso local frame (up, right, forward)
    torso_right = normalize(hip_line - np.dot(hip_line, torso_dir) * torso_dir)
    torso_forward = normalize(np.cross(torso_dir, torso_right))

    print(f"[INFO] Torso frame: up={torso_dir}, right={torso_right}, fwd={torso_forward}")

    # ── Bone directions (world space, proximal -> distal) ──────────────────────
    dir_thigh_L = normalize(w_l_knee  - w_l_hip)
    dir_thigh_R = normalize(w_r_knee  - w_r_hip)
    dir_shin_L  = normalize(w_l_ankle - w_l_knee)
    dir_shin_R  = normalize(w_r_ankle - w_r_knee)
    dir_foot_L  = normalize(w_l_toe   - w_l_ankle)
    dir_foot_R  = normalize(w_r_toe   - w_r_ankle)

    # ── Compute euler angles for each bone ───────────────────────────────────────
    # TORSO: relative to ground (world up [0,1,0]). This is the only bone
    # measured against a global reference, not a parent bone.
    # Use 2D angle in the X-Y plane (ignore Z/depth) because MediaPipe's
    # Z estimates are noisy and inflate the angle far beyond what's visible.
    torso_2d = normalize(np.array([torso_dir[0], torso_dir[1]]))
    up_2d = np.array([0.0, 1.0])
    torso_tilt = math.degrees(math.acos(np.clip(np.dot(up_2d, torso_2d), -1.0, 1.0)))
    # Sign: positive = leaning forward (in image: shoulders right of hips)
    # Use atan2 for signed angle so we know direction
    torso_tilt_signed = math.degrees(math.atan2(torso_2d[0], torso_2d[1]))
    torso_tilt = abs(torso_tilt_signed)
    torso_xyz = (torso_tilt, 0.0, 0.0)
    print(f"[INFO] Torso tilt from vertical (2D): {torso_tilt:.1f}° (signed: {torso_tilt_signed:+.1f}°)")

    # THIGHS: parent = torso, rest = torso_down
    thighR_xyz = bone_euler_from_parent(dir_thigh_R, torso_dir, torso_right)
    thighL_xyz = bone_euler_from_parent(dir_thigh_L, torso_dir, torso_right)

    # SHINS: use the total angle between thigh direction and shin direction
    # (the value shown on the annotated image is confirmed correct)
    shinR_total = total_angle(dir_thigh_R, dir_shin_R)
    shinL_total = total_angle(dir_thigh_L, dir_shin_L)
    shinR_xyz = (shinR_total, 0.0, 0.0)
    shinL_xyz = (shinL_total, 0.0, 0.0)

    # FEET: the GLB bind pose already has feet at 90° to the shin,
    # so 0,0,0 = foot perpendicular to shin. The pose.json value
    # should be 90 - total_angle (deviation from shin continuation).
    footR_total = total_angle(dir_shin_R, dir_foot_R)
    footL_total = total_angle(dir_shin_L, dir_foot_L)
    footR_xyz = (90.0 - footR_total, 0.0, 0.0)
    footL_xyz = (90.0 - footL_total, 0.0, 0.0)

    # UPPER ARMS: parent = shoulder-to-hip line on the same side (the torso side edge).
    # 0° = arm along torso (straight down), 90° = arm out (T-pose), 180° = overhead.
    # Use 2D angle (X-Y plane only) to avoid MediaPipe Z-depth noise.
    def angle_2d(v1, v2):
        """Angle between two 3D vectors projected to X-Y plane (ignoring Z)."""
        a = normalize(np.array([v1[0], v1[1]]))
        b = normalize(np.array([v2[0], v2[1]]))
        return math.degrees(math.acos(np.clip(np.dot(a, b), -1.0, 1.0)))

    dir_torso_side_R = normalize(w_r_hip - w_r_shoulder)
    dir_torso_side_L = normalize(w_l_hip - w_l_shoulder)
    dir_upper_arm_R = normalize(w_r_elbow - w_r_shoulder)
    dir_upper_arm_L = normalize(w_l_elbow - w_l_shoulder)

    upper_armR_total = angle_2d(dir_torso_side_R, dir_upper_arm_R)
    upper_armL_total = angle_2d(dir_torso_side_L, dir_upper_arm_L)
    upper_armR_xyz = (0.0, upper_armR_total, 70.0)
    upper_armL_xyz = (0.0, 360.0 - upper_armL_total, 289.0)
    print(f"[INFO] Upper arm R angle from torso side (2D): {upper_armR_total:.1f}°")
    print(f"[INFO] Upper arm L angle from torso side (2D): {upper_armL_total:.1f}° (written as {360.0 - upper_armL_total:.1f}°)")

    # FOREARMS: parent = upper arm, angle = elbow bend.
    # Use 2D angle between upper arm direction and forearm direction.
    dir_forearm_R = normalize(w_r_wrist - w_r_elbow)
    dir_forearm_L = normalize(w_l_wrist - w_l_elbow)
    forearmR_total = angle_2d(dir_upper_arm_R, dir_forearm_R)
    forearmL_total = angle_2d(dir_upper_arm_L, dir_forearm_L)
    forearmR_xyz = (forearmR_total, 0.0, 0.0)
    forearmL_xyz = (forearmL_total, 0.0, 0.0)
    print(f"[INFO] Forearm R angle from upper arm (2D): {forearmR_total:.1f}°")
    print(f"[INFO] Forearm L angle from upper arm (2D): {forearmL_total:.1f}°")

    results = {
        "torso":      torso_xyz,
        "upper_armR": upper_armR_xyz,
        "upper_armL": upper_armL_xyz,
        "forearmR":   forearmR_xyz,
        "forearmL":   forearmL_xyz,
        "thighR":     thighR_xyz,
        "thighL":     thighL_xyz,
        "shinR":      shinR_xyz,
        "shinL":      shinL_xyz,
        "footR":      footR_xyz,
        "footL":      footL_xyz,
    }

    print(f"[INFO] Leg bone Euler angles (slider values):")
    for name, (xd, yd, zd) in results.items():
        td = total_angle(
            -torso_dir if "thigh" in name else
            (dir_thigh_R if name == "shinR" else
             dir_thigh_L if name == "shinL" else
             dir_shin_R if name == "footR" else dir_shin_L),
            dir_thigh_R if name == "thighR" else
            dir_thigh_L if name == "thighL" else
            dir_shin_R if name == "shinR" else
            dir_shin_L if name == "shinL" else
            dir_foot_R if name == "footR" else dir_foot_L
        ) if "thigh" not in name else total_angle(-torso_dir,
            dir_thigh_R if name == "thighR" else dir_thigh_L)
        print(f"  {name:8s}  X:{xd:6.1f}  Y:{yd:4.1f}  Z:{zd:6.1f}  (total:{td:.1f}°)")

    # ── Draw annotations for all leg bones ─────────────────────────────────────
    def to_px(landmark):
        return np.array([landmark.x * w_img, landmark.y * h_img])

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Pixel positions
    px_l_hip   = to_px(lm[23]).astype(int)
    px_r_hip   = to_px(lm[24]).astype(int)
    px_l_knee  = to_px(lm[25]).astype(int)
    px_r_knee  = to_px(lm[26]).astype(int)
    px_l_ankle = to_px(lm[27]).astype(int)
    px_r_ankle = to_px(lm[28]).astype(int)
    px_l_toe   = to_px(lm[31]).astype(int)
    px_r_toe   = to_px(lm[32]).astype(int)
    px_torso_top = to_px(sho_mid).astype(int)
    px_torso_bot = to_px(hip_mid).astype(int)
    px_r_shoulder = to_px(lm[12]).astype(int)
    px_r_elbow    = to_px(lm[14]).astype(int)
    px_l_shoulder = to_px(lm[11]).astype(int)
    px_l_elbow    = to_px(lm[13]).astype(int)

    torso_px_dir = normalize((px_torso_bot - px_torso_top).astype(float))

    # ── Draw torso angle (relative to ground vertical) ─────────────────────────
    # Rest = straight up in image = [0, -1] (image Y is inverted)
    vertical_px_dir = np.array([0.0, -1.0])
    # Draw dashed vertical rest line from hip_mid upward
    for i in range(0, 110, 12):
        p1 = (px_torso_bot.astype(float) + vertical_px_dir * i).astype(int)
        p2 = (px_torso_bot.astype(float) + vertical_px_dir * min(i + 7, 110)).astype(int)
        cv2.line(annotated_bgr, tuple(p1), tuple(p2), GREEN, 2, cv2.LINE_AA)
    # Actual torso line
    cv2.line(annotated_bgr, tuple(px_torso_bot), tuple(px_torso_top), (255, 200, 0), 3, cv2.LINE_AA)
    # Arc
    arc_r = 45
    rest_ang  = np.degrees(np.arctan2(vertical_px_dir[1], vertical_px_dir[0]))
    torso_vec_px = px_torso_top - px_torso_bot
    torso_ang = np.degrees(np.arctan2(torso_vec_px[1], torso_vec_px[0]))
    sa, ea = min(rest_ang, torso_ang), max(rest_ang, torso_ang)
    if ea - sa > 180:
        sa, ea = ea, sa + 360
    cv2.ellipse(annotated_bgr, tuple(px_torso_bot), (arc_r, arc_r), 0, sa, ea, (255, 200, 0), 2, cv2.LINE_AA)
    # Label
    tpos = (px_torso_bot[0] + 15, px_torso_bot[1] - 50)
    ttext = f"torso {torso_tilt:.0f}d"
    cv2.putText(annotated_bgr, ttext, tpos, font, 0.5, (0,0,0),       3, cv2.LINE_AA)
    cv2.putText(annotated_bgr, ttext, tpos, font, 0.5, (255, 200, 0), 1, cv2.LINE_AA)

    # ── Draw upper arm angles (relative to shoulder→hip torso side line) ───────
    LIGHT_BLUE = (255, 180, 100)
    for side, px_sho, px_elb, px_hip_side, arm_total, label in [
        ("R", px_r_shoulder, px_r_elbow, px_r_hip, upper_armR_total, "uaR"),
        ("L", px_l_shoulder, px_l_elbow, px_l_hip, upper_armL_total, "uaL"),
    ]:
        # Rest line = shoulder → hip direction (torso side edge)
        torso_side_px = normalize((px_hip_side - px_sho).astype(float))
        # Dashed rest line from shoulder along torso side
        for i in range(0, 80, 10):
            p1 = (px_sho.astype(float) + torso_side_px * i).astype(int)
            p2 = (px_sho.astype(float) + torso_side_px * min(i + 6, 80)).astype(int)
            cv2.line(annotated_bgr, tuple(p1), tuple(p2), GREEN, 1, cv2.LINE_AA)
        # Draw the torso side line itself (shoulder to hip)
        cv2.line(annotated_bgr, tuple(px_sho), tuple(px_hip_side), GREEN, 1, cv2.LINE_AA)
        # Actual upper arm line
        cv2.line(annotated_bgr, tuple(px_sho), tuple(px_elb), LIGHT_BLUE, 3, cv2.LINE_AA)
        # Arc
        arc_r = 35
        rest_ang = np.degrees(np.arctan2(torso_side_px[1], torso_side_px[0]))
        arm_vec_px = px_elb - px_sho
        arm_ang = np.degrees(np.arctan2(arm_vec_px[1], arm_vec_px[0]))
        sa, ea = min(rest_ang, arm_ang), max(rest_ang, arm_ang)
        if ea - sa > 180:
            sa, ea = ea, sa + 360
        cv2.ellipse(annotated_bgr, tuple(px_sho), (arc_r, arc_r), 0, sa, ea, LIGHT_BLUE, 2, cv2.LINE_AA)
        # Label
        apos = (px_sho[0] + 12, px_sho[1] - 12)
        atext = f"{label} {arm_total:.0f}d"
        cv2.putText(annotated_bgr, atext, apos, font, 0.45, (0,0,0),    3, cv2.LINE_AA)
        cv2.putText(annotated_bgr, atext, apos, font, 0.45, LIGHT_BLUE, 1, cv2.LINE_AA)

    # ── Draw forearm angles (relative to upper arm) ────────────────────────────
    px_r_wrist = to_px(lm[16]).astype(int)
    px_l_wrist = to_px(lm[15]).astype(int)
    PURPLE = (200, 100, 255)
    for side, px_elb, px_wri, px_sho, fa_total, label in [
        ("R", px_r_elbow, px_r_wrist, px_r_shoulder, forearmR_total, "faR"),
        ("L", px_l_elbow, px_l_wrist, px_l_shoulder, forearmL_total, "faL"),
    ]:
        # Rest line = upper arm direction continued (shoulder → elbow, extended)
        ua_px_dir = normalize((px_elb - px_sho).astype(float))
        # Dashed rest line from elbow along upper arm direction
        for i in range(0, 70, 10):
            p1 = (px_elb.astype(float) + ua_px_dir * i).astype(int)
            p2 = (px_elb.astype(float) + ua_px_dir * min(i + 6, 70)).astype(int)
            cv2.line(annotated_bgr, tuple(p1), tuple(p2), GREEN, 1, cv2.LINE_AA)
        # Actual forearm line
        cv2.line(annotated_bgr, tuple(px_elb), tuple(px_wri), PURPLE, 2, cv2.LINE_AA)
        # Arc
        arc_r = 30
        rest_ang = np.degrees(np.arctan2(ua_px_dir[1], ua_px_dir[0]))
        fa_vec_px = px_wri - px_elb
        fa_ang = np.degrees(np.arctan2(fa_vec_px[1], fa_vec_px[0]))
        sa, ea = min(rest_ang, fa_ang), max(rest_ang, fa_ang)
        if ea - sa > 180:
            sa, ea = ea, sa + 360
        cv2.ellipse(annotated_bgr, tuple(px_elb), (arc_r, arc_r), 0, sa, ea, PURPLE, 2, cv2.LINE_AA)
        # Label
        fpos = (px_elb[0] + 10, px_elb[1] - 10)
        ftext = f"{label} {fa_total:.0f}d"
        cv2.putText(annotated_bgr, ftext, fpos, font, 0.4, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(annotated_bgr, ftext, fpos, font, 0.4, PURPLE,  1, cv2.LINE_AA)

    # Annotation data: (joint_px, parent_dir_px, child_px, label, color, euler)
    annotations = [
        # Thighs: parent rest direction = torso down (in pixel space)
        (px_r_hip,   torso_px_dir,                             px_r_knee,  "thR", YELLOW,  thighR_xyz),
        (px_l_hip,   torso_px_dir,                             px_l_knee,  "thL", YELLOW,  thighL_xyz),
        # Shins: parent rest direction = thigh direction (in pixel space)
        (px_r_knee,  normalize((px_r_knee - px_r_hip).astype(float)),   px_r_ankle, "shR", ORANGE,  shinR_xyz),
        (px_l_knee,  normalize((px_l_knee - px_l_hip).astype(float)),   px_l_ankle, "shL", ORANGE,  shinL_xyz),
        # Feet: parent rest direction = shin direction (in pixel space)
        (px_r_ankle, normalize((px_r_ankle - px_r_knee).astype(float)), px_r_toe,   "ftR", PINK,    footR_xyz),
        (px_l_ankle, normalize((px_l_ankle - px_l_knee).astype(float)), px_l_toe,   "ftL", PINK,    footL_xyz),
    ]

    for joint_px, par_dir_px, child_px, label, color, (xd, yd, zd) in annotations:
        # Compute total angle for display
        par_dir_3d = {
            "thR": -torso_dir, "thL": -torso_dir,
            "shR": dir_thigh_R, "shL": dir_thigh_L,
            "ftR": dir_shin_R,  "ftL": dir_shin_L,
        }[label]
        child_dir_3d = {
            "thR": dir_thigh_R, "thL": dir_thigh_L,
            "shR": dir_shin_R,  "shL": dir_shin_L,
            "ftR": dir_foot_R,  "ftL": dir_foot_L,
        }[label]
        td = total_angle(par_dir_3d, child_dir_3d)
        draw_bone_angle(annotated_bgr, joint_px, par_dir_px, child_px,
                        label, color, font, xd, zd, td)

    cv2.imwrite(OUTPUT_ANNOTATED, annotated_bgr)
    print(f"[INFO] Saved annotated image -> '{OUTPUT_ANNOTATED}'")

    # ── Write pose.json ────────────────────────────────────────────────────────
    raw_landmarks = [
        {"x": l.x, "y": l.y, "z": l.z, "visibility": getattr(l, 'visibility', 1.0)}
        for l in lm
    ]

    all_bones = [
        "origin", "torso", "neck", "head",
        "shoulderL", "upper_armL", "forearmL", "handL",
        "shoulderR", "upper_armR", "forearmR", "handR",
        "thighL", "shinL", "footL",
        "thighR", "shinR", "footR",
    ]
    euler_degrees = {bone: [0, 0, 0] for bone in all_bones}

    for name, (xd, yd, zd) in results.items():
        euler_degrees[name] = [round(xd, 1), round(yd, 1), round(zd, 1)]

    pose = {
        "image": os.path.basename(image_path),
        "landmarks": raw_landmarks,
        "euler_degrees": euler_degrees,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(pose, f, indent=2)

    print(f"[INFO] Saved pose -> '{OUTPUT_JSON}'")
    for name in ["torso", "upper_armL", "upper_armR", "forearmL", "forearmR", "thighL", "shinL", "footL", "thighR", "shinR", "footR"]:
        d = euler_degrees[name]
        print(f"  {name:8s}  X:{d[0]:6.1f}  Y:{d[1]:4.1f}  Z:{d[2]:6.1f}")

    # ── Copy into manequinn/ ───────────────────────────────────────────────────
    viewer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manequinn")
    os.makedirs(viewer_dir, exist_ok=True)
    for src, dst_name in [
        (image_path,       os.path.basename(image_path)),
        (OUTPUT_ANNOTATED, OUTPUT_ANNOTATED),
        (OUTPUT_JSON,      OUTPUT_JSON),
    ]:
        shutil.copy2(src, os.path.join(viewer_dir, dst_name))
        print(f"[INFO] Copied '{src}' -> manequinn/{dst_name}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH)
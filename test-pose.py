"""
test-pose.py
Runs MediaPipe Pose Landmarker on an image OR video file.

IMAGE mode: produces a single-frame pose.json with rich annotated JPEG.

VIDEO mode:
  - Extracts every frame, runs pose detection
  - Annotates EACH frame with the full rich annotation set
  - Writes output_landmarks.mp4 (annotated) and copies it to manequinn/
  - Saves ALL keyframes into pose.json

Usage:
  python test-pose.py                      # auto-detect (prefers test-vid.mp4)
  python test-pose.py test-vid.mp4         # explicit video
  python test-pose.py test-image.png       # explicit image
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
MODEL_PATH              = "pose_landmarker.task"
DEFAULT_IMAGE_PATH      = "test-image.png"
DEFAULT_VIDEO_PATH      = "test-vid.mp4"
OUTPUT_JSON             = "pose.json"
OUTPUT_ANNOTATED        = "output_landmarks.jpg"
OUTPUT_ANNOTATED_VIDEO  = "output_landmarks.mp4"

WHITE      = (255, 255, 255)
CYAN       = (0, 255, 255)
GREEN      = (0, 255, 0)
ORANGE     = (0, 165, 255)
PINK       = (180, 105, 255)
LIGHT_BLUE = (255, 180, 100)
PURPLE     = (200, 100, 255)

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

ALL_BONES = [
    "origin", "torso", "neck", "head",
    "shoulderL", "upper_armL", "forearmL", "handL",
    "shoulderR", "upper_armR", "forearmR", "handR",
    "thighL", "shinL", "footL",
    "thighR", "shinR", "footR",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def download_file(url, dest):
    if os.path.exists(dest):
        return
    print("[INFO] Downloading model...")
    urllib.request.urlretrieve(url, dest)


def mp_to_world(lm):
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
    up = normalize(parent_dir)
    right = hint_right - np.dot(hint_right, up) * up
    n = np.linalg.norm(right)
    if n < 1e-6:
        perp = np.array([1, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1, 0])
        right = perp - np.dot(perp, up) * up
    right = normalize(right)
    forward = normalize(np.cross(up, right))
    return up, right, forward


def bone_euler_from_parent(child_dir, parent_dir, hint_right):
    up, right, forward = build_local_frame(parent_dir, hint_right)
    cy = np.dot(child_dir, up)
    cz = np.dot(child_dir, forward)
    x_deg = math.degrees(math.atan2(-cz, -cy)) % 360
    return x_deg, 0.0, 0.0


def total_angle(v1, v2):
    return math.degrees(math.acos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


def angle_2d(v1, v2):
    a = normalize(np.array([v1[0], v1[1]]))
    b = normalize(np.array([v2[0], v2[1]]))
    return math.degrees(math.acos(np.clip(np.dot(a, b), -1.0, 1.0)))


# ── Euler computation ─────────────────────────────────────────────────────────

def compute_euler_from_landmarks(lm):
    """Return (euler_degrees dict, world-space bone-direction dict)."""
    hip_mid = midpoint(lm[23], lm[24])
    sho_mid = midpoint(lm[11], lm[12])

    w_hip_mid    = mp_to_world(hip_mid)
    w_sho_mid    = mp_to_world(sho_mid)
    w_l_hip      = mp_to_world(lm[23])
    w_r_hip      = mp_to_world(lm[24])
    w_l_knee     = mp_to_world(lm[25])
    w_r_knee     = mp_to_world(lm[26])
    w_l_ankle    = mp_to_world(lm[27])
    w_r_ankle    = mp_to_world(lm[28])
    w_l_toe      = mp_to_world(lm[31])
    w_r_toe      = mp_to_world(lm[32])
    w_l_shoulder = mp_to_world(lm[11])
    w_r_shoulder = mp_to_world(lm[12])
    w_l_elbow    = mp_to_world(lm[13])
    w_r_elbow    = mp_to_world(lm[14])
    w_l_wrist    = mp_to_world(lm[15])
    w_r_wrist    = mp_to_world(lm[16])

    torso_dir   = normalize(w_sho_mid - w_hip_mid)
    hip_line    = normalize(w_r_hip - w_l_hip)
    torso_right = normalize(hip_line - np.dot(hip_line, torso_dir) * torso_dir)

    torso_2d          = normalize(np.array([torso_dir[0], torso_dir[1]]))
    torso_tilt_signed = math.degrees(math.atan2(torso_2d[0], torso_2d[1]))
    torso_tilt        = abs(torso_tilt_signed)

    dir_thigh_L = normalize(w_l_knee  - w_l_hip)
    dir_thigh_R = normalize(w_r_knee  - w_r_hip)
    dir_shin_L  = normalize(w_l_ankle - w_l_knee)
    dir_shin_R  = normalize(w_r_ankle - w_r_knee)
    dir_foot_L  = normalize(w_l_toe   - w_l_ankle)
    dir_foot_R  = normalize(w_r_toe   - w_r_ankle)

    thighR_xyz = bone_euler_from_parent(dir_thigh_R, torso_dir, torso_right)
    thighL_xyz = bone_euler_from_parent(dir_thigh_L, torso_dir, torso_right)

    shinR_xyz = (total_angle(dir_thigh_R, dir_shin_R), 0.0, 0.0)
    shinL_xyz = (total_angle(dir_thigh_L, dir_shin_L), 0.0, 0.0)

    footR_xyz = (90.0 - total_angle(dir_shin_R, dir_foot_R), 0.0, 0.0)
    footL_xyz = (90.0 - total_angle(dir_shin_L, dir_foot_L), 0.0, 0.0)

    dir_torso_side_R = normalize(w_r_hip - w_r_shoulder)
    dir_torso_side_L = normalize(w_l_hip - w_l_shoulder)
    dir_upper_arm_R  = normalize(w_r_elbow - w_r_shoulder)
    dir_upper_arm_L  = normalize(w_l_elbow - w_l_shoulder)
    dir_forearm_R    = normalize(w_r_wrist - w_r_elbow)
    dir_forearm_L    = normalize(w_l_wrist - w_l_elbow)

    upper_armR_total = angle_2d(dir_torso_side_R, dir_upper_arm_R)
    upper_armL_total = angle_2d(dir_torso_side_L, dir_upper_arm_L)
    upper_armR_xyz = (0.0, upper_armR_total, 70.0)
    upper_armL_xyz = (0.0, 360.0 - upper_armL_total, 289.0)

    forearmR_total = angle_2d(dir_upper_arm_R, dir_forearm_R)
    forearmL_total = angle_2d(dir_upper_arm_L, dir_forearm_L)
    forearmR_xyz = (forearmR_total, 0.0, 0.0)
    forearmL_xyz = (forearmL_total, 0.0, 0.0)

    euler_degrees = {bone: [0, 0, 0] for bone in ALL_BONES}
    results = {
        "torso":      (torso_tilt, 0.0, 0.0),
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
    for name, (xd, yd, zd) in results.items():
        euler_degrees[name] = [round(xd, 1), round(yd, 1), round(zd, 1)]

    world = dict(
        torso_dir=torso_dir, torso_right=torso_right,
        torso_tilt=torso_tilt, torso_tilt_signed=torso_tilt_signed,
        hip_mid=hip_mid, sho_mid=sho_mid,
        dir_thigh_L=dir_thigh_L, dir_thigh_R=dir_thigh_R,
        dir_shin_L=dir_shin_L, dir_shin_R=dir_shin_R,
        dir_foot_L=dir_foot_L, dir_foot_R=dir_foot_R,
        dir_upper_arm_R=dir_upper_arm_R, dir_upper_arm_L=dir_upper_arm_L,
        dir_forearm_R=dir_forearm_R, dir_forearm_L=dir_forearm_L,
        upper_armR_total=upper_armR_total, upper_armL_total=upper_armL_total,
        forearmR_total=forearmR_total, forearmL_total=forearmL_total,
        results=results,
    )
    return euler_degrees, world


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_bone_angle(img, joint_px, parent_dir_px, child_px, label, color, font,
                    x_deg, z_deg, tot_deg):
    parent_norm = normalize(parent_dir_px.astype(float))
    for i in range(0, 80, 10):
        p1 = (joint_px.astype(float) + parent_norm * i).astype(int)
        p2 = (joint_px.astype(float) + parent_norm * min(i + 6, 80)).astype(int)
        cv2.line(img, tuple(p1), tuple(p2), GREEN, 1, cv2.LINE_AA)
    cv2.line(img, tuple(joint_px), tuple(child_px), color, 2, cv2.LINE_AA)
    arc_r    = 30
    rest_ang = np.degrees(np.arctan2(parent_norm[1], parent_norm[0]))
    child_vec = child_px.astype(float) - joint_px.astype(float)
    child_ang = np.degrees(np.arctan2(child_vec[1], child_vec[0]))
    sa, ea = min(rest_ang, child_ang), max(rest_ang, child_ang)
    if ea - sa > 180: sa, ea = ea, sa + 360
    cv2.ellipse(img, tuple(joint_px), (arc_r, arc_r), 0, sa, ea, color, 1, cv2.LINE_AA)
    pos  = (joint_px[0] + 12, joint_px[1] - 8)
    text = f"{label} {tot_deg:.0f}d X:{x_deg:.0f} Z:{z_deg:.0f}"
    cv2.putText(img, text, pos, font, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, 0.35, color,     1, cv2.LINE_AA)


def annotate_frame(bgr, lm, world, euler_degrees):
    img  = bgr.copy()
    h_img, w_img = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    def to_px(landmark):
        return np.array([landmark.x * w_img, landmark.y * h_img])

    hip_mid = world['hip_mid']
    sho_mid = world['sho_mid']

    px_l_hip      = to_px(lm[23]).astype(int)
    px_r_hip      = to_px(lm[24]).astype(int)
    px_l_knee     = to_px(lm[25]).astype(int)
    px_r_knee     = to_px(lm[26]).astype(int)
    px_l_ankle    = to_px(lm[27]).astype(int)
    px_r_ankle    = to_px(lm[28]).astype(int)
    px_l_toe      = to_px(lm[31]).astype(int)
    px_r_toe      = to_px(lm[32]).astype(int)
    px_torso_top  = to_px(sho_mid).astype(int)
    px_torso_bot  = to_px(hip_mid).astype(int)
    px_r_shoulder = to_px(lm[12]).astype(int)
    px_r_elbow    = to_px(lm[14]).astype(int)
    px_l_shoulder = to_px(lm[11]).astype(int)
    px_l_elbow    = to_px(lm[13]).astype(int)
    px_r_wrist    = to_px(lm[16]).astype(int)
    px_l_wrist    = to_px(lm[15]).astype(int)

    # Skeleton
    pts_2d = [(int(lm[i].x * w_img), int(lm[i].y * h_img)) for i in range(len(lm))]
    for s, e in POSE_CONNECTIONS:
        if s < len(pts_2d) and e < len(pts_2d):
            cv2.line(img, pts_2d[s], pts_2d[e], WHITE, 2, cv2.LINE_AA)
    for idx in DRAWN_LANDMARKS:
        if idx < len(pts_2d):
            cv2.circle(img, pts_2d[idx], 5, WHITE, 2, cv2.LINE_AA)

    cv2.circle(img, tuple(px_torso_top), 6, CYAN, -1, cv2.LINE_AA)
    cv2.circle(img, tuple(px_torso_bot), 6, CYAN, -1, cv2.LINE_AA)
    cv2.line(img, tuple(px_torso_bot), tuple(px_torso_top), CYAN, 2, cv2.LINE_AA)

    torso_px_dir = normalize((px_torso_bot - px_torso_top).astype(float))
    torso_tilt = world['torso_tilt']
    vertical_px_dir = np.array([0.0, -1.0])

    for i in range(0, 110, 12):
        p1 = (px_torso_bot.astype(float) + vertical_px_dir * i).astype(int)
        p2 = (px_torso_bot.astype(float) + vertical_px_dir * min(i + 7, 110)).astype(int)
        cv2.line(img, tuple(p1), tuple(p2), GREEN, 2, cv2.LINE_AA)
    cv2.line(img, tuple(px_torso_bot), tuple(px_torso_top), (255, 200, 0), 3, cv2.LINE_AA)

    arc_r = 45
    rest_ang = np.degrees(np.arctan2(vertical_px_dir[1], vertical_px_dir[0]))
    torso_vec_px = px_torso_top - px_torso_bot
    torso_ang = np.degrees(np.arctan2(torso_vec_px[1], torso_vec_px[0]))
    sa, ea = min(rest_ang, torso_ang), max(rest_ang, torso_ang)
    if ea - sa > 180: sa, ea = ea, sa + 360
    cv2.ellipse(img, tuple(px_torso_bot), (arc_r, arc_r), 0, sa, ea, (255, 200, 0), 2, cv2.LINE_AA)
    tpos = (px_torso_bot[0] + 15, px_torso_bot[1] - 50)
    cv2.putText(img, f"torso {torso_tilt:.0f}d", tpos, font, 0.5, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, f"torso {torso_tilt:.0f}d", tpos, font, 0.5, (255,200,0), 1, cv2.LINE_AA)

    # Upper arms
    for side, px_sho, px_elb, px_hip_side, arm_total, label in [
        ("R", px_r_shoulder, px_r_elbow, px_r_hip, world['upper_armR_total'], "uaR"),
        ("L", px_l_shoulder, px_l_elbow, px_l_hip, world['upper_armL_total'], "uaL"),
    ]:
        torso_side_px = normalize((px_hip_side - px_sho).astype(float))
        for i in range(0, 80, 10):
            p1 = (px_sho.astype(float) + torso_side_px * i).astype(int)
            p2 = (px_sho.astype(float) + torso_side_px * min(i + 6, 80)).astype(int)
            cv2.line(img, tuple(p1), tuple(p2), GREEN, 1, cv2.LINE_AA)
        cv2.line(img, tuple(px_sho), tuple(px_elb), LIGHT_BLUE, 3, cv2.LINE_AA)
        arc_r = 35
        rest_ang = np.degrees(np.arctan2(torso_side_px[1], torso_side_px[0]))
        arm_vec_px = px_elb - px_sho
        arm_ang = np.degrees(np.arctan2(arm_vec_px[1], arm_vec_px[0]))
        sa, ea = min(rest_ang, arm_ang), max(rest_ang, arm_ang)
        if ea - sa > 180: sa, ea = ea, sa + 360
        cv2.ellipse(img, tuple(px_sho), (arc_r, arc_r), 0, sa, ea, LIGHT_BLUE, 2, cv2.LINE_AA)
        apos = (px_sho[0] + 12, px_sho[1] - 12)
        cv2.putText(img, f"{label} {arm_total:.0f}d", apos, font, 0.45, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(img, f"{label} {arm_total:.0f}d", apos, font, 0.45, LIGHT_BLUE, 1, cv2.LINE_AA)

    # Forearms
    for side, px_elb, px_wri, px_sho, fa_total, label in [
        ("R", px_r_elbow, px_r_wrist, px_r_shoulder, world['forearmR_total'], "faR"),
        ("L", px_l_elbow, px_l_wrist, px_l_shoulder, world['forearmL_total'], "faL"),
    ]:
        ua_px_dir = normalize((px_elb - px_sho).astype(float))
        for i in range(0, 70, 10):
            p1 = (px_elb.astype(float) + ua_px_dir * i).astype(int)
            p2 = (px_elb.astype(float) + ua_px_dir * min(i + 6, 70)).astype(int)
            cv2.line(img, tuple(p1), tuple(p2), GREEN, 1, cv2.LINE_AA)
        cv2.line(img, tuple(px_elb), tuple(px_wri), PURPLE, 2, cv2.LINE_AA)
        arc_r = 30
        rest_ang = np.degrees(np.arctan2(ua_px_dir[1], ua_px_dir[0]))
        fa_vec_px = px_wri - px_elb
        fa_ang = np.degrees(np.arctan2(fa_vec_px[1], fa_vec_px[0]))
        sa, ea = min(rest_ang, fa_ang), max(rest_ang, fa_ang)
        if ea - sa > 180: sa, ea = ea, sa + 360
        cv2.ellipse(img, tuple(px_elb), (arc_r, arc_r), 0, sa, ea, PURPLE, 2, cv2.LINE_AA)
        fpos = (px_elb[0] + 10, px_elb[1] - 10)
        cv2.putText(img, f"{label} {fa_total:.0f}d", fpos, font, 0.4, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(img, f"{label} {fa_total:.0f}d", fpos, font, 0.4, PURPLE, 1, cv2.LINE_AA)

    # Legs
    results = world['results']
    torso_dir = world['torso_dir']
    YELLOW = (0, 255, 255)
    annotations = [
        (px_r_hip,   torso_px_dir, px_r_knee,  "thR", YELLOW, results["thighR"]),
        (px_l_hip,   torso_px_dir, px_l_knee,  "thL", YELLOW, results["thighL"]),
        (px_r_knee,  normalize((px_r_knee - px_r_hip).astype(float)), px_r_ankle, "shR", ORANGE, results["shinR"]),
        (px_l_knee,  normalize((px_l_knee - px_l_hip).astype(float)), px_l_ankle, "shL", ORANGE, results["shinL"]),
        (px_r_ankle, normalize((px_r_ankle - px_r_knee).astype(float)), px_r_toe, "ftR", PINK, results["footR"]),
        (px_l_ankle, normalize((px_l_ankle - px_l_knee).astype(float)), px_l_toe, "ftL", PINK, results["footL"]),
    ]
    par_dir_3d_map = {
        "thR": -torso_dir, "thL": -torso_dir,
        "shR": world['dir_thigh_R'], "shL": world['dir_thigh_L'],
        "ftR": world['dir_shin_R'],  "ftL": world['dir_shin_L'],
    }
    child_dir_3d_map = {
        "thR": world['dir_thigh_R'], "thL": world['dir_thigh_L'],
        "shR": world['dir_shin_R'],  "shL": world['dir_shin_L'],
        "ftR": world['dir_foot_R'],  "ftL": world['dir_foot_L'],
    }
    for joint_px, par_dir_px, child_px, label, color, (xd, yd, zd) in annotations:
        td = total_angle(par_dir_3d_map[label], child_dir_3d_map[label])
        draw_bone_angle(img, joint_px, par_dir_px, child_px, label, color, font, xd, zd, td)

    return img


# ── Image mode ─────────────────────────────────────────────────────────────────

def process_image(image_path, detector):
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

    lm = result.pose_landmarks[0]
    euler_degrees, world = compute_euler_from_landmarks(lm)

    bgr_frame     = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
    annotated_bgr = annotate_frame(bgr_frame, lm, world, euler_degrees)
    cv2.imwrite(OUTPUT_ANNOTATED, annotated_bgr)
    print(f"[INFO] Saved annotated image -> '{OUTPUT_ANNOTATED}'")

    pose = {
        "image":         os.path.basename(image_path),
        "euler_degrees": euler_degrees,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(pose, f, indent=2)
    print(f"[INFO] Saved pose -> '{OUTPUT_JSON}'")

    viewer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manequinn")
    os.makedirs(viewer_dir, exist_ok=True)
    for src, dst_name in [
        (image_path, os.path.basename(image_path)),
        (OUTPUT_ANNOTATED, OUTPUT_ANNOTATED),
        (OUTPUT_JSON, OUTPUT_JSON),
    ]:
        shutil.copy2(src, os.path.join(viewer_dir, dst_name))
        print(f"[INFO] Copied '{src}' -> manequinn/{dst_name}")


# ── Video mode ─────────────────────────────────────────────────────────────────

def process_video(video_path, detector):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video: {video_path}  FPS:{video_fps:.2f}  Frames:{total_frames}  Size:{frame_w}x{frame_h}")

    TEMP_AVI = OUTPUT_ANNOTATED_VIDEO.replace('.mp4', '_tmp.avi')
    fourcc      = cv2.VideoWriter_fourcc(*'MJPG')
    anno_writer = cv2.VideoWriter(TEMP_AVI, fourcc, video_fps, (frame_w, frame_h))
    if not anno_writer.isOpened():
        print("[WARN] MJPG writer failed, falling back to mp4v")
        fourcc      = cv2.VideoWriter_fourcc(*'mp4v')
        anno_writer = cv2.VideoWriter(OUTPUT_ANNOTATED_VIDEO, fourcc, video_fps, (frame_w, frame_h))
        TEMP_AVI    = None
        if not anno_writer.isOpened():
            print("[WARN] Could not open VideoWriter")
            anno_writer = None

    frames_data         = []
    frame_idx           = 0
    annotated_jpg_saved = False

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        rgb      = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect(mp_image)

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            euler_degrees, world = compute_euler_from_landmarks(lm)
            annotated_bgr = annotate_frame(bgr, lm, world, euler_degrees)
            if anno_writer:
                anno_writer.write(annotated_bgr)
            if not annotated_jpg_saved:
                cv2.imwrite(OUTPUT_ANNOTATED, annotated_bgr)
                annotated_jpg_saved = True
            frames_data.append({
                "frame":         frame_idx,
                "euler_degrees": euler_degrees,
            })
        else:
            if anno_writer:
                anno_writer.write(bgr)
            frames_data.append({
                "frame":         frame_idx,
                "euler_degrees": {bone: [0, 0, 0] for bone in ALL_BONES},
            })

        frame_idx += 1
        if frame_idx % 10 == 0:
            pct = frame_idx / max(total_frames, 1) * 100
            print(f"[INFO] Frame {frame_idx}/{total_frames} ({pct:.0f}%)")

    cap.release()
    if anno_writer:
        anno_writer.release()
        if TEMP_AVI and os.path.exists(TEMP_AVI):
            import subprocess
            ffmpeg_bin = shutil.which("ffmpeg")
            if not ffmpeg_bin:
                for c in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"]:
                    if os.path.isfile(c): ffmpeg_bin = c; break
            if ffmpeg_bin:
                print(f"[INFO] Re-encoding to H.264...")
                ret_code = subprocess.call([
                    ffmpeg_bin, "-y", "-i", TEMP_AVI,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-movflags", "+faststart", "-pix_fmt", "yuv420p",
                    OUTPUT_ANNOTATED_VIDEO
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if ret_code == 0:
                    os.remove(TEMP_AVI)
                    print(f"[INFO] Saved '{OUTPUT_ANNOTATED_VIDEO}'")
                else:
                    os.replace(TEMP_AVI, OUTPUT_ANNOTATED_VIDEO.replace('.mp4', '.avi'))
                    print(f"[WARN] ffmpeg failed; saved .avi instead")
            else:
                avi_out = OUTPUT_ANNOTATED_VIDEO.replace('.mp4', '.avi')
                os.replace(TEMP_AVI, avi_out)
                print(f"[WARN] ffmpeg not found. Saved '{avi_out}'")

    print(f"[INFO] Processed {frame_idx} frames")

    pose = {
        "source":      os.path.basename(video_path),
        "video_fps":   round(video_fps, 3),
        "frame_count": len(frames_data),
        "frames":      frames_data,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(pose, f)
    print(f"[INFO] Saved pose.json — {len(frames_data)} frames")

    viewer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manequinn")
    os.makedirs(viewer_dir, exist_ok=True)

    copy_list = [
        (video_path, os.path.basename(video_path)),
        (OUTPUT_JSON, OUTPUT_JSON),
    ]
    avi_fallback = OUTPUT_ANNOTATED_VIDEO.replace('.mp4', '.avi')
    if os.path.exists(OUTPUT_ANNOTATED_VIDEO):
        copy_list.append((OUTPUT_ANNOTATED_VIDEO, OUTPUT_ANNOTATED_VIDEO))
    elif os.path.exists(avi_fallback):
        copy_list.append((avi_fallback, avi_fallback))
    if annotated_jpg_saved:
        copy_list.append((OUTPUT_ANNOTATED, OUTPUT_ANNOTATED))

    for src, dst_name in copy_list:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(viewer_dir, dst_name))
            print(f"[INFO] Copied '{src}' -> manequinn/{dst_name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(input_path=None):
    download_file(MODEL_URL, MODEL_PATH)

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        output_segmentation_masks=False,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    if input_path is None:
        if os.path.exists(DEFAULT_VIDEO_PATH):
            input_path = DEFAULT_VIDEO_PATH
        else:
            input_path = DEFAULT_IMAGE_PATH

    ext = os.path.splitext(input_path)[1].lower()
    if ext in {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}:
        process_video(input_path, detector)
    else:
        process_image(input_path, detector)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
#!/usr/bin/env python3
"""
Generate imu_body_anim.gif — stick figure with 7 IMU sensor positions animating.
Tries to pull real quaternion data from HuggingFace eidon-ai/tracker-pov,
falls back to synthesized sinusoidal motion if unreachable.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
import io
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# ─── Constants ────────────────────────────────────────────────────────────────
OUTPUT_PATH = "/Users/visheshng/school/CMSC838/Diff-Kinem-WM/proposal/assets/imu_body_anim.gif"
N_FRAMES = 30
FRAME_DELAY_MS = 80   # ~12 fps
W, H = 500, 400

# Colors
BG_COLOR      = "#0a0a0a"
SKELETON_COLOR= "#444444"
CHEST_COLOR   = "#ffffff"
LEFT_COLOR    = "#7c6fcd"   # purple
RIGHT_COLOR   = "#d4a853"   # amber

DOT_RADIUS_PX = 8           # in data-units we'll scale
FONT_SIZE     = 7

# Segment lengths (in arbitrary units, figure lives in a ~[-1,1]×[-1,1.5] box)
SEG = dict(
    torso       = 0.5,
    upper_arm   = 0.38,
    forearm     = 0.34,
    hand        = 0.16,
)

# ─── Quaternion → Euler (ZYX) ─────────────────────────────────────────────────

def quat_to_euler(q):
    """q = [w, x, y, z] → (roll, pitch, yaw) in radians"""
    w, x, y, z = q
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


# ─── Try fetching real data from HuggingFace ─────────────────────────────────

def fetch_hf_data():
    """
    Returns a dict keyed by sensor name → list of (roll, pitch, yaw) tuples
    (one per frame, N_FRAMES frames), or None on failure.
    """
    try:
        import signal

        def _timeout_handler(signum, frame):
            raise TimeoutError("HF fetch timed out")

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(25)   # 25-second hard timeout

        from datasets import load_dataset
        print("Attempting HuggingFace data fetch …")
        ds = load_dataset(
            "eidon-ai/tracker-pov",
            split="train",
            streaming=True,
        )
        rows = []
        for i, row in enumerate(ds):
            rows.append(row)
            if i >= 59:
                break

        signal.alarm(0)  # cancel alarm

        if not rows:
            return None

        print(f"  Fetched {len(rows)} rows")

        # Detect quaternion column names
        sample = rows[0]
        print("  Columns:", list(sample.keys())[:20])

        sensors = ["chest", "left_shoulder", "left_forearm", "left_hand",
                   "right_shoulder", "right_forearm", "right_hand"]

        angles = {s: [] for s in sensors}

        for row in rows:
            for s in sensors:
                # Try common naming conventions
                found = False
                for fmt in [
                    f"{s}_quat", f"{s}_quaternion",
                    f"imu_{s}_quat", f"imu_{s}",
                    s,
                ]:
                    if fmt in row:
                        val = row[fmt]
                        if isinstance(val, (list, np.ndarray)) and len(val) >= 4:
                            q = np.array(val[:4], dtype=float)
                            norm = np.linalg.norm(q)
                            if norm > 1e-6:
                                q /= norm
                            angles[s].append(quat_to_euler(q))
                            found = True
                            break
                if not found:
                    angles[s].append((0.0, 0.0, 0.0))

        # Interpolate / subsample to exactly N_FRAMES
        result = {}
        for s in sensors:
            arr = np.array(angles[s])   # (N_rows, 3)
            xs = np.linspace(0, len(arr)-1, N_FRAMES)
            interp = np.zeros((N_FRAMES, 3))
            for ch in range(3):
                interp[:, ch] = np.interp(xs, np.arange(len(arr)), arr[:, ch])
            result[s] = interp          # (N_FRAMES, 3)
        return result

    except (Exception, TimeoutError) as e:
        import signal
        signal.alarm(0)
        print(f"  HuggingFace fetch failed: {e}")
        return None


# ─── Synthetic motion ─────────────────────────────────────────────────────────

def synthesize_motion():
    """
    Returns same dict structure as fetch_hf_data() but with sinusoidal motion.
    All angles in radians. Columns = (roll, pitch, yaw).
    """
    t = np.linspace(0, 2 * np.pi, N_FRAMES, endpoint=False)

    def arm_swing(phase, amp_pitch=0.45, amp_roll=0.15):
        pitch = amp_pitch * np.sin(t + phase)
        roll  = amp_roll  * np.sin(2*(t + phase) + 0.5)
        yaw   = np.zeros_like(t)
        return np.stack([roll, pitch, yaw], axis=1)

    def forearm_swing(phase, amp=0.3):
        # secondary pendulum — slightly faster frequency
        pitch = amp * np.sin(1.7*t + phase + 0.8)
        roll  = 0.08 * np.sin(t + phase)
        yaw   = np.zeros_like(t)
        return np.stack([roll, pitch, yaw], axis=1)

    def hand_wiggle(phase, amp=0.12):
        pitch = amp * np.sin(2.1*t + phase)
        roll  = 0.05 * np.sin(t + phase + 1.2)
        yaw   = np.zeros_like(t)
        return np.stack([roll, pitch, yaw], axis=1)

    def chest_sway(amp=0.06):
        pitch = amp * np.sin(t * 0.5)
        roll  = 0.03 * np.sin(t * 0.7 + 0.3)
        yaw   = np.zeros_like(t)
        return np.stack([roll, pitch, yaw], axis=1)

    return {
        "chest":           chest_sway(),
        "left_shoulder":   arm_swing(phase=0.0),
        "left_forearm":    forearm_swing(phase=0.0),
        "left_hand":       hand_wiggle(phase=0.0),
        "right_shoulder":  arm_swing(phase=np.pi),          # opposite phase
        "right_forearm":   forearm_swing(phase=np.pi),
        "right_hand":      hand_wiggle(phase=np.pi),
    }


# ─── Forward kinematics (2-D front view) ─────────────────────────────────────

def fk_frame(angles_dict, frame_idx):
    """
    Given per-sensor Euler angles at a specific frame, compute 2-D (x,y)
    positions for every joint using simple planar FK.

    Convention (front view):
      - y axis points UP
      - x axis points RIGHT
      - chest is anchor at (0, 0)
      - shoulders branch left/right
      - angles: pitch rotates the segment "forward/backward" in the sagittal
        plane (changes y component), roll tips sideways
    """

    def a(sensor, col):
        """Get angle col (0=roll,1=pitch,2=yaw) for sensor at frame_idx."""
        return float(angles_dict[sensor][frame_idx, col])

    # ── chest ─────────────────────────────────────────────
    chest_roll  = a("chest", 0)
    chest_pitch = a("chest", 1)

    # Chest center (anchor). Slight sway.
    chest_pos = np.array([chest_pitch * 0.3, 0.0])

    # Shoulder attachment points (fixed relative offset from chest)
    shoulder_offset_y = SEG["torso"] * 0.5   # shoulders at mid-torso height
    left_shoulder_attach  = chest_pos + np.array([-0.30 + chest_roll*0.2,  shoulder_offset_y])
    right_shoulder_attach = chest_pos + np.array([ 0.30 + chest_roll*0.2, shoulder_offset_y])

    # Head position (not drawn but used for visual anchor)
    head_pos = chest_pos + np.array([0, SEG["torso"] * 0.5 + 0.18])

    # Hip position
    hip_pos = chest_pos + np.array([0, -SEG["torso"] * 0.5])

    # ── left arm ──────────────────────────────────────────
    def segment_end(start, length, pitch, roll, side=-1):
        """
        Compute end-point of a segment.
        Pitch tilts up/down (affects y), roll swings in/out (affects x).
        side: -1 for left arm, +1 for right.
        """
        dx = side * length * np.sin(roll) * 0.5
        dy = -length * np.cos(pitch)           # hangs downward by default
        return start + np.array([dx, dy])

    # Left shoulder joint (dot position = top of upper arm)
    ls_pitch = a("left_shoulder", 1)
    ls_roll  = a("left_shoulder", 0)
    ls_pos   = left_shoulder_attach  # dot at shoulder attachment

    le_pos   = segment_end(ls_pos, SEG["upper_arm"], ls_pitch, ls_roll, side=-1)

    lf_pitch = a("left_forearm", 1)
    lf_roll  = a("left_forearm", 0)
    # forearm hangs from elbow, inherits shoulder orientation + local bend
    lf_pos   = le_pos   # elbow joint
    lh_pos   = segment_end(lf_pos, SEG["forearm"], ls_pitch + lf_pitch, ls_roll + lf_roll, side=-1)

    lwrist_pos = lh_pos
    lhand_pitch = a("left_hand", 1)
    lhand_roll  = a("left_hand", 0)
    lhand_end   = segment_end(lwrist_pos, SEG["hand"],
                               ls_pitch + lf_pitch + lhand_pitch,
                               ls_roll  + lf_roll  + lhand_roll, side=-1)

    # ── right arm ─────────────────────────────────────────
    rs_pitch = a("right_shoulder", 1)
    rs_roll  = a("right_shoulder", 0)
    rs_pos   = right_shoulder_attach

    re_pos   = segment_end(rs_pos, SEG["upper_arm"], rs_pitch, rs_roll, side=+1)

    rf_pitch = a("right_forearm", 1)
    rf_roll  = a("right_forearm", 0)
    rf_pos   = re_pos
    rh_pos   = segment_end(rf_pos, SEG["forearm"], rs_pitch + rf_pitch, rs_roll + rf_roll, side=+1)

    rwrist_pos = rh_pos
    rhand_pitch = a("right_hand", 1)
    rhand_roll  = a("right_hand", 0)
    rhand_end   = segment_end(rwrist_pos, SEG["hand"],
                               rs_pitch + rf_pitch + rhand_pitch,
                               rs_roll  + rf_roll  + rhand_roll, side=+1)

    # ── assemble ──────────────────────────────────────────
    joints = {
        "head":            head_pos,
        "chest":           chest_pos,
        "hip":             hip_pos,
        "left_shoulder":   ls_pos,
        "left_elbow":      le_pos,
        "left_forearm":    lf_pos,
        "left_wrist":      lh_pos,
        "left_hand":       lhand_end,
        "right_shoulder":  rs_pos,
        "right_elbow":     re_pos,
        "right_forearm":   rf_pos,
        "right_wrist":     rh_pos,
        "right_hand":      rhand_end,
    }
    return joints


# ─── Draw one frame ───────────────────────────────────────────────────────────

# Map sensor name → dot position key and color
SENSOR_DOT = {
    "chest":           ("chest",          CHEST_COLOR,  "chest"),
    "left_shoulder":   ("left_shoulder",  LEFT_COLOR,   "L.shoulder"),
    "left_forearm":    ("left_forearm",   LEFT_COLOR,   "L.forearm"),
    "left_hand":       ("left_hand",      LEFT_COLOR,   "L.hand"),
    "right_shoulder":  ("right_shoulder", RIGHT_COLOR,  "R.shoulder"),
    "right_forearm":   ("right_forearm",  RIGHT_COLOR,  "R.forearm"),
    "right_hand":      ("right_hand",     RIGHT_COLOR,  "R.hand"),
}

# Skeleton edges: (joint_a, joint_b)
SKELETON = [
    ("chest",         "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow",    "left_wrist"),
    ("left_wrist",    "left_hand"),
    ("chest",         "right_shoulder"),
    ("right_shoulder","right_elbow"),
    ("right_elbow",   "right_wrist"),
    ("right_wrist",   "right_hand"),
    ("chest",         "hip"),
    # simple legs (static)
    ("hip",           "left_knee_static"),
    ("hip",           "right_knee_static"),
    ("left_knee_static",  "left_foot_static"),
    ("right_knee_static", "right_foot_static"),
    # neck
    ("chest",         "head"),
]

def draw_frame(angles_dict, frame_idx, static_joints):
    """Render one frame, return PIL Image."""
    joints = fk_frame(angles_dict, frame_idx)
    joints.update(static_joints)   # add static leg joints

    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw skeleton lines
    for (a_name, b_name) in SKELETON:
        if a_name in joints and b_name in joints:
            pa = joints[a_name]
            pb = joints[b_name]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                    color=SKELETON_COLOR, lw=2.0, solid_capstyle="round", zorder=1)

    # Draw sensor dots + labels
    dot_r = 0.055   # data-unit radius (roughly 8px at our scale)
    for sensor, (joint_key, color, label) in SENSOR_DOT.items():
        if joint_key not in joints:
            continue
        pos = joints[joint_key]
        circ = Circle(pos, radius=dot_r, color=color, zorder=3)
        ax.add_patch(circ)

        # Label offset: left-side sensors → label to the left, right → right, chest → above
        if "left" in sensor:
            lx, ly, ha = pos[0] - dot_r*1.5, pos[1], "right"
        elif "right" in sensor:
            lx, ly, ha = pos[0] + dot_r*1.5, pos[1], "left"
        else:
            lx, ly, ha = pos[0], pos[1] + dot_r*2.2, "center"

        ax.text(lx, ly, label,
                color=color, fontsize=FONT_SIZE, ha=ha, va="center",
                fontfamily="monospace", fontweight="bold", zorder=4)

    # Subtle title
    ax.text(0, 1.2, "IMU Sensor Positions  •  Body Motion",
            color="#666666", fontsize=7, ha="center", va="top",
            fontfamily="sans-serif", zorder=5)

    # Convert to PIL image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                pad_inches=0, dpi=100, facecolor=BG_COLOR)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    # Crop/resize to exact size
    img = img.resize((W, H), Image.LANCZOS)
    return img


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # 1. Get angle data
    # HF dataset (eidon-ai/tracker-pov) streams MP4 video files, not tabular
    # quaternion rows — skip HF and use high-quality synthetic motion directly.
    print("Using synthesized motion (HF dataset streams video, not tabular IMU rows).")
    angles = synthesize_motion()

    # 2. Static joints for legs
    static_joints = {
        "left_knee_static":  np.array([-0.22, -0.85]),
        "right_knee_static": np.array([ 0.22, -0.85]),
        "left_foot_static":  np.array([-0.28, -1.20]),
        "right_foot_static": np.array([ 0.28, -1.20]),
    }

    # 3. Render frames
    print(f"Rendering {N_FRAMES} frames …")
    frames = []
    for fi in range(N_FRAMES):
        img = draw_frame(angles, fi, static_joints)
        frames.append(img)
        if (fi+1) % 10 == 0:
            print(f"  Frame {fi+1}/{N_FRAMES}")

    # 4. Save GIF
    print(f"Saving GIF → {OUTPUT_PATH}")
    # Use Pillow to save animated GIF
    frames[0].save(
        OUTPUT_PATH,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=FRAME_DELAY_MS,
        disposal=2,
    )
    print("Done.")


if __name__ == "__main__":
    main()

import csv
import os
from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque

WEIGHTS_PATH = '/home/lionheart/tennis-analytics/runs/phase2_unfreeze_combined_l4/weights/best.pt'
VIDEO_PATH   = '/home/lionheart/tennis-analytics/processed_videos/W2019_Djokovic_v_Federer_part_002.mp4'
OUTPUT_PATH  = '/home/lionheart/tennis-analytics/runs/detect/runs/feature_demo2.mp4'
CSV_PATH     = '/home/lionheart/tennis-analytics/runs/detect/runs/feature_demo2.csv'

model = YOLO(WEIGHTS_PATH)

# Threshold above which a mean frame difference is classified as a camera cut
CUT_THRESH = 0.25

FEATURES_TO_SHOW = [
    # label                  key                    color            (vmin, vmax)
    # -- Frame-level --
    ('Frame diff',           'frame_diff',           (200, 200, 200), (0,    1)),
    ('Camera cut',           'camera_cut',           (255,   0,   0), (0,    1)),
    ('Both players vis.',    'both_players_visible', (255, 255,   0), (0,    1)),
    ('Zoom level',           'zoom_level',           (200, 160, 255), (0,    1)),
    # -- Ball --
    ('Ball detected',        'ball_detected',        (0,   140, 255), (0,    1)),
    ('Ball confidence',      'ball_conf',            (0,   200, 255), (0,    1)),
    ('Ball pos X',           'ball_x',               (100, 200, 255), (0,    1)),
    ('Ball pos Y',           'ball_y',               (100, 200, 255), (0,    1)),
    ('Ball speed px/s',      'ball_speed',           (0,   255, 200), (0, 3000)),
    ('Ball freq (2s)',        'ball_freq_2s',         (0,   100, 255), (0,    1)),
    ('Ball freq (5s)',        'ball_freq_5s',         (50,  120, 255), (0,    1)),
    ('Ball streak (norm)',   'ball_streak_norm',     (0,    80, 200), (0,    1)),
    ('No-ball time (s)',     'time_since_ball_s',    (180, 100,   0), (0,   10)),
    # -- Players --
    ('P1 speed px/s',        'p1_speed',             (0,   255, 100), (0,  2000)),
    ('P2 speed px/s',        'p2_speed',             (100, 255, 150), (0,  2000)),
    ('Combined speed',       'combined_speed',       (0,   220, 120), (0,  2000)),
    ('P1 acceleration',      'p1_accel',             (200, 255, 100), (0,  2000)),
    ('P2 acceleration',      'p2_accel',             (180, 255, 120), (0,  2000)),
    ('P1 move intensity',    'p1_move_intensity',    (150, 255,  50), (0, 2000)),
    ('P2 move intensity',    'p2_move_intensity',    (130, 255,  70), (0, 2000)),
    ('P1 avg spd (3s)',      'p1_avg_speed_3s',      (0,   200,  80), (0,  2000)),
    ('P2 avg spd (3s)',      'p2_avg_speed_3s',      (50,  200, 100), (0,  2000)),
    ('Player separation',    'separation',           (255, 180,   0), (0,    1)),
    ('Since big move (s)',   'time_since_move_s',    (255, 120,   0), (0,   30)),
]


def draw_feature_panel(frame, feature_history, current_features):
    h, w = frame.shape[:2]
    panel_w = 300
    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    canvas[:, :w] = frame
    canvas[:, w:] = (18, 18, 18)
    cv2.line(canvas, (w, 0), (w, h), (60, 60, 60), 1)

    n = len(FEATURES_TO_SHOW)
    title_h = 28
    available_h = h - title_h
    row_h   = max(24, available_h // n)
    graph_h = max(12, row_h - 13)
    graph_w = panel_w - 22

    cv2.putText(canvas, "RALLY FEATURES", (w + 8, title_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

    for i, (label, key, color, (vmin, vmax)) in enumerate(FEATURES_TO_SHOW):
        y   = title_h + i * row_h
        val = current_features.get(key, 0.0)

        cv2.putText(canvas, label, (w + 8, y + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (160, 160, 160), 1)
        cv2.putText(canvas, f"{val:.2f}", (w + panel_w - 50, y + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1)

        gx, gy = w + 8, y + 12
        cv2.rectangle(canvas, (gx, gy), (gx + graph_w, gy + graph_h), (35, 35, 35), -1)

        history = list(feature_history.get(key, []))
        if len(history) > 1:
            pts = []
            for j, v in enumerate(history):
                px   = gx + int(j / (len(history) - 1) * graph_w)
                norm = np.clip((v - vmin) / (vmax - vmin + 1e-9), 0, 1)
                py   = gy + graph_h - int(norm * graph_h)
                pts.append((px, py))
            for j in range(1, len(pts)):
                cv2.line(canvas, pts[j-1], pts[j], color, 1)

        norm_val = np.clip((val - vmin) / (vmax - vmin + 1e-9), 0, 1)
        bar_h = int(norm_val * graph_h)
        if bar_h > 0:
            cv2.rectangle(canvas,
                          (gx + graph_w - 5, gy + graph_h - bar_h),
                          (gx + graph_w,     gy + graph_h),
                          color, -1)

    return canvas



def run_feature_visualisation(video_path, csv_path, output_path=None, duration_mins=None):
    cap      = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames   = int(duration_mins * 60 * fps) if duration_mins else float('inf')
    sample_every = max(1, round(fps / 25))
    EFF_FPS      = fps / sample_every

    win_2s = max(1, int(EFF_FPS * 2))
    win_3s = max(1, int(EFF_FPS * 3))
    win_5s = max(1, int(EFF_FPS * 5))

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out = cv2.VideoWriter(output_path,
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              EFF_FPS,
                              (w + 300, h))
    else:
        out = None

    history_len  = 150
    panel_keys   = [t[1] for t in FEATURES_TO_SHOW]
    feature_history = {k: deque([0.0] * history_len, maxlen=history_len) for k in panel_keys}

    # All keys written to CSV (superset of panel keys)
    csv_keys = panel_keys + [
        'timestamp_s', 'frame_idx',
        'p1_cx', 'p1_cy', 'p2_cx', 'p2_cy',
        'p1_vx', 'p1_vy', 'p2_vx', 'p2_vy',
        'p1_ax', 'p1_ay', 'p2_ax', 'p2_ay',
        'p1_max_speed_3s', 'p2_max_speed_3s',
        'p1_std_speed_3s', 'p2_std_speed_3s',
        'ball_streak', 'frames_since_ball',
    ]

    # Rolling windows
    ball_win_2s  = deque(maxlen=win_2s)
    ball_win_5s  = deque(maxlen=win_5s)
    # Per-track-ID state (ByteTrack IDs, so velocity history survives player swaps)
    prev_by_id      = {}  # track_id -> last detection dict
    vel_by_id       = {}  # track_id -> (vx, vy)
    speed_win_by_id = {}  # track_id -> deque(maxlen=win_3s)
    disp_win_by_id  = {}  # track_id -> deque(maxlen=win_3s)
    prev_ball       = None          # (cx, cy) of last detected ball
    prev_gray       = None          # grayscale frame for diff computation
    ball_streak         = 0
    frames_since_ball   = 0
    frames_since_move   = 0
    LARGE_MOVE_PX_S     = 50.0

    frame_idx = 0

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_keys, extrasaction='ignore')
        writer.writeheader()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx > max_frames:
                break

            if frame_idx % sample_every != 0:
                frame_idx += 1
                continue

            # ── Frame-level features ───────────────────────────────────────────
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                diff       = cv2.absdiff(gray, prev_gray)
                frame_diff = float(diff.mean()) / 255.0
                camera_cut = 1.0 if frame_diff > CUT_THRESH else 0.0
            else:
                frame_diff = 0.0
                camera_cut = 0.0

            prev_gray = gray

            # ── YOLO inference (ByteTrack for players, plain detect for ball) ───
            results = model.track(frame, tracker="bytetrack.yaml",
                                  persist=True, verbose=False)[0]

            tracked_persons, balls = [], []
            for box in results.boxes:
                cls, conf = int(box.cls), float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                entry = dict(cx=cx, cy=cy, bw=x2-x1, bh=y2-y1, conf=conf)
                if cls == 0 and conf > 0.4:
                    entry['id'] = int(box.id) if box.id is not None else -1
                    tracked_persons.append(entry)
                elif cls == 1 and conf > 0.25:
                    balls.append(entry)

            # Top 2 by area, then sorted by track ID → stable P1/P2 assignment
            tracked_persons = sorted(tracked_persons, key=lambda p: p['bw'] * p['bh'], reverse=True)[:2]
            persons = sorted(tracked_persons, key=lambda p: p['id'])
            while len(persons) < 2:
                persons.append(None)

            # Court visibility: both players detected + zoom estimate
            both_players_visible = 1.0 if (persons[0] and persons[1]) else 0.0

            # Zoom level: mean player bbox area relative to frame area (higher = more zoomed in)
            bbox_areas = [p['bw'] * p['bh'] for p in persons if p]
            if bbox_areas:
                zoom_level = float(np.mean(bbox_areas)) / (w * h)
                zoom_level = min(zoom_level * 20, 1.0)  # scale so ~5% fill = 1.0
            else:
                zoom_level = 0.0

            # ── Ball features ──────────────────────────────────────────────────
            ball_detected = 1.0 if balls else 0.0
            ball_conf     = balls[0]['conf'] if balls else 0.0

            if balls:
                ball_x = balls[0]['cx'] / w
                ball_y = balls[0]['cy'] / h
                if prev_ball is not None:
                    ball_speed = float(np.hypot(
                        (balls[0]['cx'] - prev_ball[0]) * EFF_FPS,
                        (balls[0]['cy'] - prev_ball[1]) * EFF_FPS,
                    ))
                else:
                    ball_speed = 0.0
                prev_ball = (balls[0]['cx'], balls[0]['cy'])
            else:
                ball_x = ball_y = ball_speed = 0.0
                prev_ball = None

            ball_win_2s.append(ball_detected)
            ball_win_5s.append(ball_detected)
            ball_freq_2s = sum(ball_win_2s) / len(ball_win_2s)
            ball_freq_5s = sum(ball_win_5s) / len(ball_win_5s)

            if ball_detected:
                ball_streak += 1
                frames_since_ball = 0
            else:
                ball_streak = 0
                frames_since_ball += 1

            ball_streak_norm  = min(ball_streak / max(win_2s, 1), 1.0)
            time_since_ball_s = frames_since_ball / EFF_FPS

            # ── Player features ────────────────────────────────────────────────
            speeds, accels, accel_xs, accel_ys, disps = [], [], [], [], []
            velocities = []

            for p in persons:
                if p:
                    tid = p['id']
                    if tid not in speed_win_by_id:
                        speed_win_by_id[tid] = deque(maxlen=win_3s)
                        disp_win_by_id[tid]  = deque(maxlen=win_3s)
                    if tid in prev_by_id:
                        prev = prev_by_id[tid]
                        vx   = (p['cx'] - prev['cx']) * EFF_FPS
                        vy   = (p['cy'] - prev['cy']) * EFF_FPS
                        speed = float(np.hypot(vx, vy))
                        disp  = float(np.hypot(p['cx'] - prev['cx'], p['cy'] - prev['cy']))
                        if tid in vel_by_id:
                            ax = vx - vel_by_id[tid][0]
                            ay = vy - vel_by_id[tid][1]
                            accel = float(np.hypot(ax, ay))
                        else:
                            ax = ay = accel = 0.0
                        vel_by_id[tid] = (vx, vy)
                        velocities.append((vx, vy))
                    else:
                        speed = accel = ax = ay = disp = 0.0
                        velocities.append((0.0, 0.0))
                    prev_by_id[tid] = p
                    speed_win_by_id[tid].append(speed)
                    disp_win_by_id[tid].append(disp)
                else:
                    speed = accel = ax = ay = disp = 0.0
                    velocities.append((0.0, 0.0))

                speeds.append(speed)
                accels.append(accel)
                accel_xs.append(ax)
                accel_ys.append(ay)
                disps.append(disp)

            def _ws(tid, wins, fn):
                w = wins.get(tid)
                return float(fn(w)) if w else 0.0

            p0id = persons[0]['id'] if persons[0] else None
            p1id = persons[1]['id'] if persons[1] else None

            p1_avg_speed_3s  = _ws(p0id, speed_win_by_id, np.mean)
            p2_avg_speed_3s  = _ws(p1id, speed_win_by_id, np.mean)
            p1_max_speed_3s  = _ws(p0id, speed_win_by_id, np.max)
            p2_max_speed_3s  = _ws(p1id, speed_win_by_id, np.max)
            p1_std_speed_3s  = _ws(p0id, speed_win_by_id, np.std)
            p2_std_speed_3s  = _ws(p1id, speed_win_by_id, np.std)
            p1_move_intensity = float(sum(disp_win_by_id.get(p0id, [])))
            p2_move_intensity = float(sum(disp_win_by_id.get(p1id, [])))
            combined_speed    = (speeds[0] + speeds[1]) / 2.0

            if persons[0] and persons[1]:
                sep = float(np.hypot(
                    (persons[0]['cx'] - persons[1]['cx']) / w,
                    (persons[0]['cy'] - persons[1]['cy']) / h,
                ))
            else:
                sep = 0.0

            if speeds[0] > LARGE_MOVE_PX_S or speeds[1] > LARGE_MOVE_PX_S:
                frames_since_move = 0
            else:
                frames_since_move += 1
            time_since_move_s = frames_since_move / EFF_FPS

            # ── Assemble feature dict ──────────────────────────────────────────
            current_features = {
                # frame-level
                'frame_diff':           frame_diff,
                'camera_cut':           camera_cut,
                'both_players_visible': both_players_visible,
                'zoom_level':           zoom_level,
                # ball
                'ball_detected':        ball_detected,
                'ball_conf':            ball_conf,
                'ball_x':               ball_x,
                'ball_y':               ball_y,
                'ball_speed':           ball_speed,
                'ball_freq_2s':         ball_freq_2s,
                'ball_freq_5s':         ball_freq_5s,
                'ball_streak_norm':     ball_streak_norm,
                'time_since_ball_s':    time_since_ball_s,
                # players
                'p1_speed':             speeds[0],
                'p2_speed':             speeds[1],
                'combined_speed':       combined_speed,
                'p1_accel':             accels[0],
                'p2_accel':             accels[1],
                'p1_move_intensity':    p1_move_intensity,
                'p2_move_intensity':    p2_move_intensity,
                'p1_avg_speed_3s':      p1_avg_speed_3s,
                'p2_avg_speed_3s':      p2_avg_speed_3s,
                'separation':           sep,
                'time_since_move_s':    time_since_move_s,
                # CSV-only
                'timestamp_s':          round(frame_idx / fps, 3),
                'frame_idx':            frame_idx,
                'p1_cx':                round(persons[0]['cx'] / w, 4) if persons[0] else 0.0,
                'p1_cy':                round(persons[0]['cy'] / h, 4) if persons[0] else 0.0,
                'p2_cx':                round(persons[1]['cx'] / w, 4) if persons[1] else 0.0,
                'p2_cy':                round(persons[1]['cy'] / h, 4) if persons[1] else 0.0,
                'p1_vx':                round(velocities[0][0], 3),
                'p1_vy':                round(velocities[0][1], 3),
                'p2_vx':                round(velocities[1][0], 3),
                'p2_vy':                round(velocities[1][1], 3),
                'p1_ax':                round(accel_xs[0], 3),
                'p1_ay':                round(accel_ys[0], 3),
                'p2_ax':                round(accel_xs[1], 3),
                'p2_ay':                round(accel_ys[1], 3),
                'p1_max_speed_3s':      p1_max_speed_3s,
                'p2_max_speed_3s':      p2_max_speed_3s,
                'p1_std_speed_3s':      p1_std_speed_3s,
                'p2_std_speed_3s':      p2_std_speed_3s,
                'ball_streak':          float(ball_streak),
                'frames_since_ball':    float(frames_since_ball),
            }

            for k in panel_keys:
                feature_history[k].append(current_features.get(k, 0.0))

            writer.writerow(current_features)

            # ── Draw detections on frame ───────────────────────────────────────
            for i, p in enumerate(persons):
                if p:
                    x1 = int(p['cx'] - p['bw'] / 2)
                    y1 = int(p['cy'] - p['bh'] / 2)
                    x2 = int(p['cx'] + p['bw'] / 2)
                    y2 = int(p['cy'] + p['bh'] / 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
                    cv2.putText(frame, f"P{i+1} {speeds[i]:.0f}px/s",
                                (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1)

            if balls:
                b   = balls[0]
                bx1 = int(b['cx'] - b['bw'] / 2)
                by1 = int(b['cy'] - b['bh'] / 2)
                bx2 = int(b['cx'] + b['bw'] / 2)
                by2 = int(b['cy'] + b['bh'] / 2)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 140, 255), 2)

            cv2.putText(frame, f"{frame_idx / fps:.1f}s",
                        (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            if out:
                canvas = draw_feature_panel(frame, feature_history, current_features)
                out.write(canvas)

            frame_idx += 1

    cap.release()
    if out:
        out.release()
        print(f"Video saved to  {output_path}")
    print(f"CSV saved to    {csv_path}")


if __name__ == '__main__':
    run_feature_visualisation(
        video_path=VIDEO_PATH,
        csv_path=CSV_PATH,
        output_path=OUTPUT_PATH,
        duration_mins=5,
    )

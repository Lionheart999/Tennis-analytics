"""
Segment-based rally labeller.

Controls:
  P           — pause / unpause
  SPACE       — mark rally start / end (toggles open/close a segment)
  X           — mark current segment as excluded (not rally, not deadtime — ambiguous)
  BACKSPACE   — delete last completed segment
  A / D       — seek ±0.5 seconds
  W / S       — seek ±5 seconds
  [ / ]       — slow down / speed up playback
  ENTER       — save segments to CSV and quit
  Q           — quit without saving

Output: CSV with columns start_s, end_s, label.
Use expand_labels() to convert to a per-frame numpy array for training.
"""

import csv
import cv2
import numpy as np
from pathlib import Path

RALLY    = 'rally'
EXCLUDED = 'excluded'


def label_segments(video_path, output_path=None):
    video_path  = Path(video_path)
    output_path = Path(output_path) if output_path else \
                  video_path.parent.parent / 'labels' / (video_path.stem + '_segments.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap         = cv2.VideoCapture(str(video_path))
    fps         = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s  = total_frames / fps
    frame_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    segments       = []      # completed {start_s, end_s, label}
    open_start_s   = None    # timestamp of in-progress segment start
    open_label     = RALLY   # label for the in-progress segment
    delay          = max(1, int(1000 / fps))
    paused         = False

    def current_time():
        return cap.get(cv2.CAP_PROP_POS_FRAMES) / fps

    def seek(delta_s):
        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        target = int(np.clip(pos + delta_s * fps, 0, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)

    def draw_timeline(frame, t):
        tl_y  = frame_h - 18
        tl_h  = 10
        tl_x0, tl_x1 = 0, frame_w

        # Background
        cv2.rectangle(frame, (tl_x0, tl_y), (tl_x1, tl_y + tl_h), (40, 40, 40), -1)

        # Completed segments
        for seg in segments:
            x0 = int(seg['start_s'] / duration_s * frame_w)
            x1 = int(seg['end_s']   / duration_s * frame_w)
            color = (0, 200, 80) if seg['label'] == RALLY else (120, 120, 120)
            cv2.rectangle(frame, (x0, tl_y), (x1, tl_y + tl_h), color, -1)

        # Open (in-progress) segment
        if open_start_s is not None:
            x0 = int(open_start_s / duration_s * frame_w)
            x1 = int(t            / duration_s * frame_w)
            cv2.rectangle(frame, (x0, tl_y), (x1, tl_y + tl_h), (0, 220, 255), -1)

        # Playhead
        px = int(t / duration_s * frame_w)
        cv2.line(frame, (px, tl_y - 2), (px, tl_y + tl_h + 2), (255, 255, 255), 2)

    print(f"\nVideo: {video_path.name}  ({duration_s/60:.1f} min)")
    print("P=pause  SPACE=start/end  X=excluded  BACKSPACE=undo  A/D=±1s  W/S=±5s  ENTER=save  Q=quit\n")

    window_title = 'Labeller - ' + video_path.name
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, min(frame_w, 1280), min(frame_h, 720))

    frame = None
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                _save(segments, output_path)
                break
        elif frame is None:
            _save(segments, output_path)
            break

        t = current_time()

        # ── State banner ─────────────────────────────────────────────────────
        if open_start_s is not None:
            banner_color = (0, 180, 255) if open_label == RALLY else (80, 80, 80)
            label_text   = f"RECORDING {open_label.upper()}  started {open_start_s:.1f}s"
        else:
            banner_color = (30, 30, 30)
            label_text   = f"IDLE — {len(segments)} segments"

        cv2.rectangle(frame, (0, 0), (frame_w, 36), banner_color, -1)
        cv2.putText(frame, label_text, (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Time + speed overlay
        mins, secs = divmod(t, 60)
        pause_str = "PAUSED  |  " if paused else ""
        cv2.putText(frame, f"{pause_str}{int(mins):02d}:{secs:05.2f}  |  {1000/delay:.1f}fps",
                    (frame_w - 200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

        # Key reference — bottom-right
        keys_ref = [
            "SPACE  rally start/end",
            "X      excluded start/end",
            "BKSP   undo last segment",
            "P      pause / unpause",
            "A / D  seek -/+0.5s",
            "W / S  seek +/-5s",
            "[ / ]  slower / faster",
            "ENTER  save & quit",
            "Q      quit no save",
        ]
        for i, line in enumerate(reversed(keys_ref)):
            cv2.putText(frame, line,
                        (frame_w - 230, frame_h - 36 - i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

        # Last few segments listed bottom-left
        for i, seg in enumerate(segments[-4:]):
            txt = f"{seg['start_s']:.1f}s – {seg['end_s']:.1f}s  [{seg['label']}]"
            cv2.putText(frame, txt, (12, frame_h - 36 - i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 200, 80) if seg['label'] == RALLY else (150, 150, 150), 1)

        draw_timeline(frame, t)

        cv2.imshow(window_title, frame)
        key = cv2.waitKey(0 if paused else delay) & 0xFF

        if key == ord(' '):
            if open_start_s is None:
                open_start_s = t
                open_label   = RALLY
            else:
                if t > open_start_s:
                    segments.append({'start_s': round(open_start_s, 3),
                                     'end_s':   round(t, 3),
                                     'label':   open_label})
                open_start_s = None

        elif key == ord('x'):
            if open_start_s is None:
                open_start_s = t
                open_label   = EXCLUDED
            else:
                if t > open_start_s:
                    segments.append({'start_s': round(open_start_s, 3),
                                     'end_s':   round(t, 3),
                                     'label':   EXCLUDED})
                open_start_s = None

        elif key == 8:   # BACKSPACE
            if segments:
                removed = segments.pop()
                print(f"Removed: {removed}")

        elif key == ord('a'):
            seek(-0.5)
            if paused: ret, frame = cap.read()
        elif key == ord('d'):
            seek(0.5)
            if paused: ret, frame = cap.read()
        elif key == ord('w'):
            seek(5)
            if paused: ret, frame = cap.read()
        elif key == ord('s'):
            seek(-5)
            if paused: ret, frame = cap.read()

        elif key == ord('p'):
            paused = not paused

        elif key == ord(']'):
            step  = 50 if delay > 100 else 10
            delay = max(5, delay - step)
        elif key == ord('['):
            step  = 50 if delay >= 100 else 10
            delay = min(1000, delay + step)

        elif key == 13:              # ENTER
            _save(segments, output_path)
            break
        elif key == ord('q'):
            print("Quit without saving.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return segments


def _save(segments, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['start_s', 'end_s', 'label'])
        writer.writeheader()
        writer.writerows(segments)
    rally_segs = [s for s in segments if s['label'] == RALLY]
    excl_segs  = [s for s in segments if s['label'] == EXCLUDED]
    rally_dur  = sum(s['end_s'] - s['start_s'] for s in rally_segs)
    print(f"\nSaved {len(segments)} segments → {output_path}")
    print(f"  Rally:    {len(rally_segs)} segments  ({rally_dur:.1f}s total)")
    print(f"  Excluded: {len(excl_segs)} segments")


def expand_labels(segments_path, video_path, sample_every=1):
    """
    Convert a segments CSV into a per-effective-frame label array.

    Returns
    -------
    labels : np.ndarray, shape (n_effective_frames,), dtype int8
        1 = rally, 0 = non-rally, -1 = excluded
    timestamps : np.ndarray of float
        Timestamp in seconds for each label entry.
    """
    with open(segments_path, newline='') as f:
        segments = list(csv.DictReader(f))
    for seg in segments:
        seg['start_s'] = float(seg['start_s'])
        seg['end_s']   = float(seg['end_s'])

    cap          = cv2.VideoCapture(str(video_path))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    eff_frames   = total_frames // sample_every
    labels       = np.zeros(eff_frames, dtype=np.int8)
    timestamps   = np.array([i * sample_every / fps for i in range(eff_frames)])

    for seg in segments:
        mask = (timestamps >= seg['start_s']) & (timestamps < seg['end_s'])
        if seg['label'] == RALLY:
            labels[mask] = 1
        elif seg['label'] == EXCLUDED:
            labels[mask] = -1

    return labels, timestamps


if __name__ == '__main__':
    VIDEO = '/home/lionheart/tennis-analytics/raw_videos/A2022_Nadal_v_Medvedev_h264.mp4'
    segs  = label_segments(VIDEO)

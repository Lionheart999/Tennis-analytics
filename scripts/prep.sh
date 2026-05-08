#!/usr/bin/env bash
set -euo pipefail

mkdir -p processed_videos

for infile in ../raw_videos/U2024_Sinner_v_Fritz.mp4; do
  [[ -e "$infile" ]] || continue
  base=$(basename "$infile" .mp4)

  echo ">>> Processing $infile"

  # 1) Re-encode to constant 25 fps (CFR) + sensible keyframe interval
  ffmpeg -y -i "$infile" \
    -vf "fps=25" \
    -c:v libx264 -preset fast -crf 23 \
    -x264-params "keyint=50:min-keyint=50:scenecut=0" \
    -c:a aac \
    "../processed_videos/${base}_25fps.mp4"

  # 2) Split into exact 10-min chunks (still keyframe-aligned, but now predictable)
  ffmpeg -y -i "../processed_videos/${base}_25fps.mp4" \
    -map 0 -c copy \
    -f segment -segment_time 900 -reset_timestamps 1 \
    "../processed_videos/${base}_part_%03d.mp4"
done
#!/usr/bin/env bash
set -e

for infile in raw_videos/*.mp4; do
  [ -e "$infile" ] || continue   # skip if no files
  base=$(basename "$infile" .mp4)

  echo ">>> Processing $infile"

  # 1) CFR at 25 fps
  ffmpeg -y -i "$infile" \
    -c:v libx264 -preset fast -crf 23 -c:a aac -r 25 \
    processed_videos/${base}_25fps.mp4

  # 2) Split into 10-min chunks
  ffmpeg -y -i processed_videos/${base}_25fps.mp4 \
    -c copy -map 0 -segment_time 600 -f segment \
    processed_videos/${base}_part_%03d.mp4

  # 3) Extract 5 fps frames
  mkdir -p frames/$base
  ffmpeg -y -i processed_videos/${base}_25fps.mp4 \
    -vf fps=5 frames/$base/frame_%06d.jpg
done

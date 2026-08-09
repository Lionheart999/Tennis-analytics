#!/bin/bash
# Brev instance setup script for SimCLR pre-training.
# Run manually after SSH-ing in: bash brev_setup.sh

set -e

export PATH="$HOME/.local/bin:$PATH"

# ── Python deps ────────────────────────────────────────────────────────────────
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics numpy opencv-python matplotlib pandas scikit-learn \
            scikit-image shapely yt-dlp

# ── Clone repo ─────────────────────────────────────────────────────────────────
cd /home/ubuntu
git clone https://github.com/Lionheart999/Tennis-analytics.git
cd Tennis-analytics
mkdir -p raw_videos models runs/eval

# ── Download training videos (h264, skips validation matches) ─────────────────
# Uncomment and fill in YouTube URLs for each match
MATCHES=(
    # "W2019_Djokovic_v_Federer  <youtube-url>"
    # "R2025_Sinner_v_Alcaraz    <youtube-url>"
    # "U2024_Sinner_v_Fritz      <youtube-url>"
    # "W2025_Sinner_v_Alcaraz    <youtube-url>"
    # "A2025_Sinner_v_Zverev     <youtube-url>"
    # "R2019_Federer_v_Wawrinka  <youtube-url>"
    # "W2024_Fritz_v_Zverev      <youtube-url>"
    # "W2024_Alcaraz_v_Djokovic  <youtube-url>"
    # "R2025_Sinner_v_Djokovic   <youtube-url>"
    # "A2022_Nadal_v_Medvedev    <youtube-url>"
)

for entry in "${MATCHES[@]}"; do
    name=$(echo "$entry" | awk '{print $1}')
    url=$(echo "$entry"  | awk '{print $2}')
    echo "Downloading $name ..."
    yt-dlp -f "bestvideo[vcodec^=avc1]+bestaudio/best" \
           --merge-output-format mp4 \
           -o "raw_videos/${name}.mp4" "$url"
done

echo ""
echo "Setup complete. To start training:"
echo "  cd /home/ubuntu/Tennis-analytics"
echo "  python scripts/simclr_pretrain.py"

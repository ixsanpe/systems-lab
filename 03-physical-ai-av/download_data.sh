#!/usr/bin/env bash
# Recreates ./data: the Voxel51 100-clip sample (FiftyOneDataset format) of
# nvidia/PhysicalAI-Autonomous-Vehicles, published on the Hugging Face hub.
set -euo pipefail

REPO_ID="dgural/PhysicalAI-Autonomous-Vehicles-Sample"
TARGET_DIR="$(cd "$(dirname "$0")" && pwd)/data"

uv run --with huggingface_hub python - "$REPO_ID" "$TARGET_DIR" <<'EOF'
import sys
from huggingface_hub import snapshot_download

repo_id, target_dir = sys.argv[1:3]
snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=target_dir)
print(f"Downloaded {repo_id} -> {target_dir}")
EOF

# Optional: load it into FiftyOne and open the app
#   uv run --with fiftyone python -c "
#   import fiftyone as fo
#   ds = fo.Dataset.from_dir('$TARGET_DIR', fo.types.FiftyOneDataset, name='nvidia-physical-ai')
#   fo.launch_app(ds).wait()"

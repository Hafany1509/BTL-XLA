#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
cd "$JPEG_ROOT"
test -f "$JPEG_BASELINE" || { echo "Missing baseline: $JPEG_BASELINE"; exit 1; }
python -c 'import torch, torchvision; print("Torch",torch.__version__,"Torchvision",torchvision.__version__); assert torch.cuda.is_available(), "CUDA GPU required"; print(torch.cuda.get_device_name(0))'
python -m venv --system-site-packages "$JPEG_ROOT/.venv"
"$JPEG_ROOT/.venv/bin/python" -m pip install numpy==1.26.4 Pillow==10.3.0 scipy==1.15.3 scikit-learn==1.7.2 opencv-python-headless==4.10.0.84 tensorboardX==2.6.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
"$JPEG_ROOT/.venv/bin/python" -c 'import torch,torchvision,numpy,cv2,PIL,tensorboardX; print("Environment ready",torch.__version__,torchvision.__version__,numpy.__version__,PIL.__version__,cv2.__version__)'
"$JPEG_ROOT/.venv/bin/python" -m pip freeze > "$JPEG_ROOT/environment.txt"

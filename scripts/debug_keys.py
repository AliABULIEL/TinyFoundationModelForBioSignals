#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

# Load SSL checkpoint
ckpt = torch.load('artifacts/butppg_ssl/best_model.pt', map_location='cpu', weights_only=False)
ssl_state = ckpt['encoder_state_dict']

print("SSL checkpoint keys (first 5 with encoder.backbone):")
for k in list(ssl_state.keys())[:5]:
    if 'encoder.backbone' in k:
        print(f"  Original: {k}")
        print(f"  Stripped: {k.replace('encoder.', '')}")
        print()

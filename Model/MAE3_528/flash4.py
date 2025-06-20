import torch

print("PyTorch version:", torch.__version__)
print("Flash SDP enabled:", torch.backends.cuda.flash_sdp_enabled())
print("Math mode default:", torch.backends.cuda.matmul.allow_tf32)  # 推奨: True

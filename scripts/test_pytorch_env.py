import torch
import numpy as np
import time

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Benchmark tensor operations
x = torch.randn(64, 120, 64, device=device)
t0 = time.time()
for _ in range(100):
    y = torch.fft.rfft(x, dim=1)
    m = torch.abs(y)
    loss = (m ** 2).sum()
print("FFT Benchmark 100 runs completed in:", round(time.time() - t0, 4), "seconds")

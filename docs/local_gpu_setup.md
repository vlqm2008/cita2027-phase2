# Local RTX 5060 / Blackwell support notes

1. Before installing torch, run `nvidia-smi` and `python scripts/inspect_environment.py`.
   If torch is absent, the script exits 2 after reporting environment metadata.
2. NVIDIA RTX 5060 is Blackwell compute capability 12.0 (`sm_120`). A CUDA build
   supporting Blackwell is required. NVIDIA driver compatibility must be checked
   against the CUDA runtime bundled in the chosen wheel.
3. PyTorch 2.7+ introduced Blackwell support with CUDA 12.8 wheels. In 2026, newer
   releases may use CUDA 13; check the official PyTorch installer for *your OS*.
4. Do **not** force-reinstall torch blindly. Use a separate inference environment
   after checking existing `torch.__version__`, `torch.version.cuda`, and driver.
   Older `cu121/cu124/cu126` builds may report CUDA availability but cannot run
   kernels on `sm_120`.
5. Even for a build with `sm_120` advertised, run actual matmul and elementwise
   kernel operations. If Windows DLL/CUBIN issues arise, preserve logs and compare
   another official stable release with matching driver; do not assert GPU support
   based on `torch.cuda.is_available()` alone.
6. Phase 2 CPU data operations do not need PyTorch. Do not install Moirai, Chronos,
   flash-attn, or auxiliary GPU libraries until Phase 3 compatibility testing.

Primary sources:
- https://developer.nvidia.com/cuda/gpus
- https://pytorch.org/blog/pytorch-2-7/
- https://pytorch.org/get-started/locally/

"""Report CUDA runtime and execute real GPU operations; never auto-install packages."""

from __future__ import annotations

import json
import platform
import subprocess
import sys


def main() -> None:
    result: dict = {
        "os": platform.platform(), "python": sys.version,
        "executable": sys.executable, "status": "NOT_TESTED",
    }
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
             "--format=csv,noheader"], text=True, capture_output=True,
            timeout=15, check=False,
        )
        result["nvidia_smi"] = completed.stdout.strip() if completed.returncode == 0 else completed.stderr.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        result["nvidia_smi"] = f"UNAVAILABLE: {exc}"
    try:
        import torch
    except ImportError:
        result["torch"] = "NOT_INSTALLED"
        print(json.dumps(result, indent=2))
        return
    result.update({
        "torch": torch.__version__, "torch_compiled_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "compiled_arch_list": torch.cuda.get_arch_list() if torch.version.cuda else [],
    })
    if not torch.cuda.is_available():
        result["status"] = "FAIL_CUDA_UNAVAILABLE"
        print(json.dumps(result, indent=2))
        return
    try:
        device = torch.device("cuda:0")
        result["gpu_name"] = torch.cuda.get_device_name(0)
        result["compute_capability"] = list(torch.cuda.get_device_capability(0))
        result["total_vram_mib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**20)
        torch.cuda.reset_peak_memory_stats()
        a = torch.arange(1024 * 1024, device=device, dtype=torch.float32).reshape(1024, 1024) / 1024
        b = torch.eye(1024, device=device)
        with torch.inference_mode():
            c = torch.mm(a, b)
            d = torch.relu(c.square() + 1)
        torch.cuda.synchronize()
        torch.testing.assert_close(c.cpu(), a.cpu(), rtol=0, atol=1e-5)
        torch.testing.assert_close(d.cpu(), (a.cpu().square() + 1), rtol=0, atol=1e-4)
        result["matrix_multiply"] = "PASS"
        result["elementwise_inference_mode"] = "PASS"
        result["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 2)
        result["status"] = "PASS" if tuple(result["compute_capability"]) == (12, 0) else "PASS_DIFFERENT_ARCH"
    except Exception as exc:
        result["status"] = "FAIL_KERNEL"
        result["kernel_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

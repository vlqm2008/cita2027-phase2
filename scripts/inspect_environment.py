"""Read-only environment inspection; executes GPU kernels when CUDA exists."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys


def main() -> int:
    result = {"os": platform.platform(), "python": sys.version,
              "executable": sys.executable, "nvidia_smi_available": bool(shutil.which("nvidia-smi"))}
    if shutil.which("nvidia-smi"):
        proc = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
                               "--format=csv,noheader"], capture_output=True, text=True, check=False)
        result["nvidia_smi_result"] = proc.stdout.strip() or proc.stderr.strip()
        result["nvidia_smi_return_code"] = proc.returncode
    try:
        import torch
    except ImportError:
        result["torch"] = "NOT_INSTALLED"
        print(json.dumps(result, indent=2))
        return 2
    result.update(torch=str(torch.__version__), torch_cuda_build=torch.version.cuda,
                  cuda_available=torch.cuda.is_available())
    if not torch.cuda.is_available():
        print(json.dumps(result, indent=2))
        return 3
    try:
        result["supported_arches"] = torch.cuda.get_arch_list()
        result["devices"] = []
        for index in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(index)
            info = {"name": p.name, "capability": list(torch.cuda.get_device_capability(index)),
                    "total_vram_bytes": p.total_memory}
            with torch.cuda.device(index):
                a = torch.randn(256, 256, device="cuda")
                b = torch.randn(256, 256, device="cuda")
                c = a @ b
                d = c.square().mean()
                torch.cuda.synchronize()
                info["actual_kernel_test"] = "PASS" if bool(torch.isfinite(d).item()) else "FAIL_NONFINITE"
                info["kernel_result"] = float(d.item())
            result["devices"].append(info)
        result["gpu_test_status"] = "PASS" if all(x["actual_kernel_test"] == "PASS" for x in result["devices"]) else "FAIL"
    except Exception as exc:
        result["gpu_test_status"] = "FAIL"
        result["error"] = repr(exc)
    print(json.dumps(result, indent=2))
    return 0 if result["gpu_test_status"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())

"""Read-only environment diagnostic for the Isaac Lab / ProtoMotions setup."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from typing import Any


_DISTRIBUTIONS = {
    "torch": "torch",
    "isaaclab": "isaaclab",
    "isaacsim": "isaacsim",
    "isaaclab_tasks": "isaaclab_tasks",
    "protomotions": "protomotions",
}


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvidia_smi() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "devices": [], "error": "nvidia-smi not found"}
    try:
        result = subprocess.run(
            [executable, "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "devices": [], "error": str(exc)}
    devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    error = result.stderr.strip() or None
    return {
        "available": result.returncode == 0 and bool(devices),
        "devices": devices,
        "error": error if result.returncode else None,
    }


def _torch_cuda() -> dict[str, Any]:
    if _version("torch") is None:
        return {"available": None, "runtime": None, "devices": [], "error": "PyTorch not installed"}
    try:
        import torch

        available = bool(torch.cuda.is_available())
        devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())] if available else []
        return {
            "available": available,
            "runtime": torch.version.cuda,
            "devices": devices,
            "error": None,
        }
    except Exception as exc:  # A broken/incompatible torch install is diagnostic output, not a crash.
        return {"available": False, "runtime": None, "devices": [], "error": f"{type(exc).__name__}: {exc}"}


def build_report() -> dict[str, Any]:
    versions = {key: _version(distribution) for key, distribution in _DISTRIBUTIONS.items()}
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
        "nvidia_smi": _nvidia_smi(),
        "torch_cuda": _torch_cuda(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--require-cuda", action="store_true", help="exit with an error if CUDA is unavailable")
    parser.add_argument("--require-sim", action="store_true", help="exit with an error if Isaac Lab/Sim are missing")
    args = parser.parse_args(argv)

    report = build_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Python: {report['python']} ({report['machine']})")
        print(f"Platform: {report['platform']}")
        print("Packages:")
        for name, version in report["packages"].items():
            print(f"  {name}: {version or 'not installed'}")
        print(f"nvidia-smi: {', '.join(report['nvidia_smi']['devices']) or report['nvidia_smi']['error']}")
        cuda = report["torch_cuda"]
        devices = ", ".join(cuda["devices"]) or cuda["error"] or "no CUDA device detected"
        print(f"PyTorch CUDA: {cuda['available']} (runtime={cuda['runtime']}; {devices})")

    failed = False
    if args.require_cuda and report["torch_cuda"]["available"] is not True:
        failed = True
    if args.require_sim and not (report["packages"]["isaaclab"] and report["packages"]["isaacsim"]):
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

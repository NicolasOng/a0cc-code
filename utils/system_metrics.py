"""
Background system-metrics logger. Writes JSON-lines entries (one JSON object per
line) at a fixed interval. Every collector is best-effort: if a backend isn't
available the corresponding field is omitted rather than failing.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from typing import Any, Optional


def _try_jax_gpu_memory() -> Optional[list[dict[str, Any]]]:
    """Per-device memory stats via JAX. Works on CUDA; None on CPU-only builds."""
    try:
        import jax
        devices = [d for d in jax.devices() if d.platform in ("gpu", "cuda", "rocm")]
        if not devices:
            return None
        out = []
        for d in devices:
            try:
                stats = d.memory_stats()
            except Exception:
                continue
            if stats is None:
                continue
            out.append({
                "device": str(d),
                "bytes_in_use": stats.get("bytes_in_use"),
                "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
                "bytes_limit": stats.get("bytes_limit"),
            })
        return out or None
    except Exception:
        return None


def _try_nvidia_smi() -> Optional[list[dict[str, Any]]]:
    """nvidia-smi subprocess fallback. Gives util and temperature on top of memory."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        out = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            out.append({
                "index": int(parts[0]),
                "memory_used_mb": int(parts[1]),
                "memory_total_mb": int(parts[2]),
                "utilization_pct": int(parts[3]),
                "temperature_c": int(parts[4]),
            })
        return out or None
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None


def _try_psutil_system() -> Optional[dict[str, Any]]:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            "cpu_pct": psutil.cpu_percent(interval=None),
            "ram_used_gb": round(vm.used / (1024 ** 3), 3),
            "ram_total_gb": round(vm.total / (1024 ** 3), 3),
            "ram_pct": vm.percent,
        }
    except Exception:
        return None


def _try_proc_meminfo() -> Optional[dict[str, Any]]:
    """Linux-only fallback for RAM when psutil isn't installed."""
    try:
        with open("/proc/meminfo") as f:
            info: dict[str, int] = {}
            for line in f:
                k, _, rest = line.partition(":")
                val_kb = int(rest.strip().split()[0])
                info[k] = val_kb
        total = info.get("MemTotal", 0)
        available = info.get("MemAvailable", info.get("MemFree", 0))
        used = total - available
        return {
            "ram_used_gb": round(used / (1024 ** 2), 3),
            "ram_total_gb": round(total / (1024 ** 2), 3),
            "ram_pct": round(100.0 * used / total, 1) if total else None,
        }
    except Exception:
        return None


def _collect_snapshot(start_time: float) -> dict[str, Any]:
    gpu = _try_nvidia_smi() or _try_jax_gpu_memory()
    system = _try_psutil_system() or _try_proc_meminfo()
    return {
        "timestamp": time.time(),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "gpu": gpu,
        "system": system,
    }


class SystemMetricsLogger:
    """
    Writes a JSON-lines snapshot to `output_path` every `interval_seconds` on a
    daemon thread. Use as a context manager or call start()/stop() manually.
    The metrics it collects are system-wide, so they include every process on
    the machine (main thread, self-play workers, anything else running).
    """

    def __init__(self, output_path: str, interval_seconds: float = 60.0):
        self.output_path = output_path
        self.interval_seconds = float(interval_seconds)
        self._start_time = time.time()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="SystemMetricsLogger")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    def start(self) -> "SystemMetricsLogger":
        self._write(_collect_snapshot(self._start_time))
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval_seconds + 5.0)

    def __enter__(self) -> "SystemMetricsLogger":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._write(_collect_snapshot(self._start_time))

    def _write(self, entry: dict[str, Any]) -> None:
        try:
            with open(self.output_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass  # never let logging errors kill the thread

"""Hardware measurement for every model call (O2): wall time, TTFT, GPU-seconds, energy.

GpuMeter samples NVML in a background thread and reads each GPU's energy counter
before and after a call. Two GPU-second figures are recorded per call:

  gpu_seconds       device-occupancy seconds = wall time x number of GPUs holding the
                    model. This is what the GPU time costs, and is what the database stores.
  gpu_busy_seconds  sum over those GPUs of wall time x mean utilisation. Shows how busy the
                    GPUs actually were (decode at batch 1 is memory-bound, so this is lower).

energy_j comes from nvmlDeviceGetTotalEnergyConsumption (Volta and newer, including
T4), so it does not depend on how a model is split across GPUs.

Without NVML (no NVIDIA GPU, e.g. a CPU smoke test) only wall time is measured and
nvml=False is recorded; GPU figures are then 0 and must not be reported.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field


@dataclass
class Measurement:
    devices: list[int] = field(default_factory=list)
    wall_s: float = 0.0
    gpu_seconds: float = 0.0
    gpu_busy_seconds: float = 0.0
    gpu_util_pct: float | None = None
    gpu_mem_mb: float | None = None  # peak allocated by torch on these devices during the call
    energy_j: float | None = None
    nvml: bool = False
    t_start: float = 0.0  # perf_counter at start, for TTFT

    def to_dict(self) -> dict:
        return asdict(self)


def _cuda_sync(devices):
    try:
        import torch

        if torch.cuda.is_available():
            for d in devices:
                torch.cuda.synchronize(d)
    except ImportError:
        pass


def _reset_peak(devices):
    try:
        import torch

        if torch.cuda.is_available():
            for d in devices:
                torch.cuda.reset_peak_memory_stats(d)
    except ImportError:
        pass


def _peak_mb(devices) -> float | None:
    try:
        import torch

        if torch.cuda.is_available() and devices:
            return sum(torch.cuda.max_memory_allocated(d) for d in devices) / 2**20
    except ImportError:
        pass
    return None


class GpuMeter:
    """One per process. `with meter.measure([0, 1]) as m: ...` fills m on exit."""

    def __init__(self, interval_s: float = 0.02, nvml=None):
        self.interval_s = interval_s
        self._nv = nvml
        self._handles = []
        self._samples: list[tuple[float, int, float]] = []  # (t, device, util%)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        if self._nv is None:
            try:
                import pynvml  # provided by the nvidia-ml-py package

                pynvml.nvmlInit()
                self._nv = pynvml
            except Exception:  # noqa: BLE001  (no driver, no package, CPU machine)
                self._nv = None
        if self._nv is not None:
            n = self._nv.nvmlDeviceGetCount()
            self._handles = [self._nv.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()

    @property
    def available(self) -> bool:
        return self._nv is not None

    def _sample_loop(self):
        while not self._stop.is_set():
            t = time.perf_counter()
            for i, h in enumerate(self._handles):
                try:
                    u = float(self._nv.nvmlDeviceGetUtilizationRates(h).gpu)
                except Exception:  # noqa: BLE001
                    continue
                with self._lock:
                    self._samples.append((t, i, u))
            with self._lock:  # keep about the last minute
                if len(self._samples) > 20000:
                    del self._samples[:10000]
            self._stop.wait(self.interval_s)

    def _energy_mj(self, d: int) -> float | None:
        try:
            return float(self._nv.nvmlDeviceGetTotalEnergyConsumption(self._handles[d]))
        except Exception:  # noqa: BLE001
            return None

    def close(self):
        self._stop.set()

    def hardware(self) -> dict:
        """GPU names / driver for the results manifest."""
        if not self.available:
            return {"nvml": False, "gpus": []}

        def s(x):
            return x.decode() if isinstance(x, bytes) else str(x)

        out = {"nvml": True, "driver": s(self._nv.nvmlSystemGetDriverVersion()), "gpus": []}
        for h in self._handles:
            mem = self._nv.nvmlDeviceGetMemoryInfo(h)
            out["gpus"].append({"name": s(self._nv.nvmlDeviceGetName(h)),
                                "memory_mb": round(mem.total / 2**20)})
        return out

    @contextmanager
    def measure(self, devices: list[int] | None = None):
        devices = sorted(set(devices or []))
        m = Measurement(devices=devices, nvml=self.available and bool(devices))
        _cuda_sync(devices)
        _reset_peak(devices)
        e0 = {d: self._energy_mj(d) for d in devices} if m.nvml else {}
        t0 = m.t_start = time.perf_counter()
        try:
            yield m
        finally:
            _cuda_sync(devices)
            t1 = time.perf_counter()
            m.wall_s = t1 - t0
            m.gpu_mem_mb = _peak_mb(devices)
            if m.nvml:
                e1 = {d: self._energy_mj(d) for d in devices}
                if all(e0[d] is not None and e1[d] is not None for d in devices):
                    m.energy_j = sum(e1[d] - e0[d] for d in devices) / 1000.0
                with self._lock:
                    win = [(d, u) for t, d, u in self._samples if t0 <= t <= t1 and d in devices]
                utils = []
                for d in devices:
                    du = [u for dd, u in win if dd == d]
                    utils.append(sum(du) / len(du) if du else None)
                known = [u for u in utils if u is not None]
                m.gpu_util_pct = sum(known) / len(known) if known else None
                m.gpu_seconds = m.wall_s * len(devices)
                m.gpu_busy_seconds = sum(m.wall_s * (u or 0) / 100.0 for u in utils)

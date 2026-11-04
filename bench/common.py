"""Power rails and runner processes, shared by sweep.py and dash.py (Orin, Python 3.10)."""

import glob
import json
import os
import signal
import subprocess
import threading
import time

import engines


def _find(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


# the hwmon index is assigned at boot, so look it up instead of hardcoding hwmon1
HWMON = _find("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*")
GPU_FREQ = _find("/sys/class/devfreq/17000000.gpu/cur_freq")
GPU_LOAD = _find("/sys/devices/platform/*/17000000.gpu/load") or _find("/sys/devices/platform/17000000.gpu/load")

RAILS = ("VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC")


def _read(path):
    with open(path) as f:
        return f.read().strip()


def read_rails():
    """Watts on the three INA3221 channels: board input, CPU+GPU+CV, SoC."""
    if HWMON is None:
        raise RuntimeError("no INA3221 hwmon found")
    out = []
    for ch in (1, 2, 3):
        mv = int(_read(f"{HWMON}/in{ch}_input"))
        ma = int(_read(f"{HWMON}/curr{ch}_input"))
        out.append(mv * ma / 1e6)
    return tuple(out)


def gpu_mhz():
    try:
        return int(_read(GPU_FREQ)) / 1e6
    except Exception:
        return None


def gpu_load():
    """GPU busy fraction, 0..1 (the sysfs value is per mille)."""
    try:
        return int(_read(GPU_LOAD)) / 1000.0
    except Exception:
        return None


def power_mode():
    try:
        out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            if "Power Mode" in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unknown"


class Sampler:
    """Polls the rails in a background thread. Rows are
    (t, vdd_in, vdd_cpu_gpu_cv, vdd_soc, gpu_mhz, gpu_load)."""

    def __init__(self, hz=50, keep=None):
        self.period = 1.0 / hz
        self.rows = []
        self.keep = keep
        self.lock = threading.Lock()
        self.error = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self):
        nxt = time.monotonic()
        while not self._stop.is_set():
            try:
                vin, cgc, soc = read_rails()
                row = (time.monotonic(), vin, cgc, soc, gpu_mhz(), gpu_load())
                with self.lock:
                    self.rows.append(row)
                    if self.keep and len(self.rows) > self.keep:
                        del self.rows[: len(self.rows) - self.keep]
                self.error = ""
            except Exception as exc:
                self.error = f"rail read: {exc}"
            nxt += self.period
            time.sleep(max(0.0, nxt - time.monotonic()))

    def since(self, t0):
        with self.lock:
            return [r for r in self.rows if r[0] >= t0]

    def snapshot(self):
        with self.lock:
            return list(self.rows)


class Runner:
    """One engine process. Collects the runner's per-second JSON reports."""

    def __init__(self, engine, fps_cap):
        self.engine = engine
        self.fps_cap = fps_cap
        self.reports = []          # (t, fps, ms, duty)
        self.stderr_tail = []
        self.lock = threading.Lock()
        self.proc = subprocess.Popen(
            engines.command(engine, fps_cap),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=engines.environment(engine), preexec_fn=os.setsid)
        threading.Thread(target=self._out, daemon=True).start()
        threading.Thread(target=self._err, daemon=True).start()

    def _out(self):
        for line in iter(self.proc.stdout.readline, b""):
            try:
                d = json.loads(line.decode())
            except Exception:
                continue
            with self.lock:
                self.reports.append((time.monotonic(), float(d.get("fps", 0)),
                                     float(d.get("ms", 0)), float(d.get("duty", 0))))

    def _err(self):
        for line in iter(self.proc.stderr.readline, b""):
            with self.lock:
                self.stderr_tail.append(line.decode(errors="replace").rstrip())
                del self.stderr_tail[:-20]

    def alive(self):
        return self.proc.poll() is None

    def since(self, t0):
        with self.lock:
            return [r for r in self.reports if r[0] >= t0]

    def latest(self):
        with self.lock:
            return self.reports[-1] if self.reports else None

    def failure(self):
        code = self.proc.poll()
        if code is None or code in (0, -signal.SIGTERM, -signal.SIGKILL):
            return ""
        with self.lock:
            tail = self.stderr_tail[-1] if self.stderr_tail else "no stderr"
        return f"{self.engine['label']} exited {code}: {tail}"

    def stop(self):
        if self.proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            self.proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                self.proc.wait(timeout=5)
            except Exception:
                pass


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")

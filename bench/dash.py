#!/usr/bin/env python3
"""Live power view (runs on the Orin): pick a framework and a precision, watch
the INA3221 rails and energy per frame update in real time.

  dash.py [--port 8080]      then open http://192.168.55.1:8080 from the PC
"""

import argparse
import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402
import engines  # noqa: E402

HZ = 20
WINDOW_S = 90
AVG_S = 5.0          # tiles average over this many seconds
IDLE_AFTER_S = 5.0   # idle baseline needs this long with nothing running

sampler = common.Sampler(hz=HZ, keep=HZ * WINDOW_S)
lock = threading.Lock()
state = {"engine": "idle", "fps_cap": 30, "since": time.monotonic(), "idle_w": None,
         "error": "", "switching": False}
runner = None
snapshots = []


def select(engine_id, fps_cap):
    global runner
    with lock:
        state["switching"] = True
    try:
        if runner is not None:
            runner.stop()
            runner = None
        eng = engines.BY_ID.get(engine_id)
        err = ""
        if engine_id != "idle":
            if eng is None:
                err = f"unknown engine {engine_id}"
            elif not engines.available(eng):
                err = f"{eng['label']} is not built"
            else:
                runner = common.Runner(eng, fps_cap)
        with lock:
            state.update(engine=engine_id if not err else "idle", fps_cap=fps_cap,
                         since=time.monotonic(), error=err)
    finally:
        with lock:
            state["switching"] = False


def summary():
    now = time.monotonic()
    rows = sampler.snapshot()
    with lock:
        st = dict(state)
    run = runner
    recent = [r for r in rows if r[0] >= now - AVG_S]
    out = {
        "engine": st["engine"], "fps_cap": st["fps_cap"], "switching": st["switching"],
        "error": st["error"] or sampler.error, "running_s": round(now - st["since"], 1),
        "power_mode": POWER_MODE,
    }
    if recent:
        out["now_w"] = recent[-1][1]
        out["mean_w"] = common.mean(r[1] for r in recent)
        out["rail_w"] = common.mean(r[2] for r in recent)
        out["soc_w"] = common.mean(r[3] for r in recent)
        out["gpu_mhz"] = common.mean(r[4] for r in recent)
        out["gpu_load"] = common.mean(r[5] for r in recent)

    # the idle floor is only trusted once nothing has run for a while
    if st["engine"] == "idle" and now - st["since"] >= IDLE_AFTER_S and recent:
        with lock:
            state["idle_w"] = out["mean_w"]
    out["idle_w"] = state["idle_w"]

    if run is not None:
        if not run.alive():
            out["error"] = run.failure() or "runner stopped"
        reps = run.since(now - AVG_S)
        if reps:
            fps = common.mean(r[1] for r in reps)
            out["fps"] = fps
            out["ms"] = common.mean(r[2] for r in reps)
            out["duty"] = common.mean(r[3] for r in reps)
            if fps > 0 and "mean_w" in out:
                out["j_per_frame"] = out["mean_w"] / fps
                if out["idle_w"]:
                    out["j_per_frame_marginal"] = (out["mean_w"] - out["idle_w"]) / fps
        elif run.alive():
            out["warming"] = True

    t_end = rows[-1][0] if rows else now
    step = max(1, HZ // 10)  # ship 10 Hz to the browser
    series = [[round(r[0] - t_end, 2), round(r[1], 3), round(r[2], 3), round(r[3], 3)]
              for r in rows[::step]]
    return out, series


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode())

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/api/engines":
            self._json({"frameworks": engines.FRAMEWORKS, "engines": [
                {"id": e["id"], "framework": e["framework"], "precision": e["precision"],
                 "label": e["label"], "available": engines.available(e),
                 "caveat": e.get("caveat", "")}
                for e in engines.ENGINES]})
        elif self.path == "/api/state":
            st, series = summary()
            self._json({"state": st, "series": series, "snapshots": snapshots})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            d = json.loads(self.rfile.read(n).decode() or "{}")
        except Exception:
            self._send(400, b"bad json", "text/plain")
            return
        if self.path == "/api/select":
            try:
                fps = max(0, min(1000, int(d.get("fps", 30))))
            except Exception:
                fps = 30
            threading.Thread(target=select, args=(str(d.get("engine", "idle")), fps),
                             daemon=True).start()
            self._json({"ok": True})
        elif self.path == "/api/snapshot":
            st, _ = summary()
            if st["engine"] == "idle" or "fps" not in st:
                self._json({"ok": False, "error": "nothing running yet"}, 409)
                return
            eng = engines.BY_ID[st["engine"]]
            snapshots.append({k: st.get(k) for k in
                              ("fps_cap", "fps", "ms", "mean_w", "rail_w", "j_per_frame",
                               "j_per_frame_marginal", "gpu_mhz")} | {"label": eng["label"]})
            self._json({"ok": True})
        elif self.path == "/api/snapshot/clear":
            snapshots.clear()
            self._json({"ok": True})
        else:
            self._send(404, b"not found", "text/plain")


POWER_MODE = "unknown"


def main():
    global POWER_MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    POWER_MODE = common.power_mode()

    # runners live in their own session, so a dropped ssh would otherwise leave
    # one spinning on the GPU; treat hangup/terminate like Ctrl+C
    def _bail(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGHUP, _bail)
    signal.signal(signal.SIGTERM, _bail)

    sampler.start()
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    print(f"live view on http://0.0.0.0:{a.port}  (power mode {POWER_MODE})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if runner is not None:
            runner.stop()


if __name__ == "__main__":
    main()

import csv
import re
import threading
import time

import numpy as np
import serial

PORT = "/dev/ttyACM0"

# ---------------------------------------------------------------
# ANCHORS: table corners, in METRES. The table is 120 x 75 cm.
# !!! CHECK that each ID is really at the corner written next to it !!!
# ---------------------------------------------------------------
ANCHORS = {
    "8D13": (0.00, 0.00),   # bottom-left   (0, 0) cm
    "DD0E": (1.20, 0.00),   # bottom-right  (120, 0) cm
    "9B9E": (0.00, 0.75),   # top-left      (0, 75) cm
    "54A9": (1.20, 0.75),   # top-right     (120, 75) cm
}
START_TRUE = (0.60, 0.30)   # where the tag starts, in metres (put it EXACTLY here, use tape)
STILL_SECONDS = 10          # keep the tag still this long at the start (used for CALIBRATION)
CAL_MIN_SAMPLES = 20        # need at least this many readings before calibrating
DT = 0.2                    # seconds between two readings (tag set to 200 ms = 5 Hz)
OUT_FILE = "run_" + time.strftime("%H%M%S") + ".csv"

PATTERN = re.compile(r"([0-9A-Fa-f]{4})\[[^\]]*\]=([\d.]+)")


def read_distances(port=PORT):
    """Yields {anchor_id: distance_in_m} for every line the tag prints."""
    ser = serial.Serial(port, 115200, timeout=0.5)
    ser.write(b"\r")
    time.sleep(0.1)
    ser.write(b"\r")
    time.sleep(1.5)
    ser.reset_input_buffer()
    ser.write(b"les\r")
    try:
        while True:
            line = ser.readline().decode(errors="ignore")
            found = {a.upper(): float(d) for a, d in PATTERN.findall(line)}
            if found:
                yield found
    finally:
        ser.write(b"\r")
        time.sleep(0.2)
        ser.close()


def correct(dists, bias):
    """Subtract the calibration bias from every valid distance (0 = invalid, stays 0)."""
    return {a: (max(d - bias, 0.01) if d > 0 else 0) for a, d in dists.items()}


def calibrate(samples):
    """Tag is still at START_TRUE. Every anchor's distance should equal the true
    distance from START_TRUE to that anchor. The average amount by which the
    readings are too long (same for all anchors) is the bias to subtract."""
    per_anchor = {}
    for a, pos in ANCHORS.items():
        vals = [s[a] for s in samples if s.get(a, 0) > 0]
        if len(vals) >= 5:
            expected = float(np.hypot(START_TRUE[0] - pos[0], START_TRUE[1] - pos[1]))
            per_anchor[a] = float(np.median(vals)) - expected
    if len(per_anchor) < 3:
        return None, per_anchor
    return float(np.mean(list(per_anchor.values()))), per_anchor


def solve_ls(dists, guess):
    """2D least squares. Returns (position, fit_error) or None."""
    ids = [a for a in ANCHORS if dists.get(a, 0) > 0]
    if len(ids) < 3:
        return None
    A = np.array([ANCHORS[a] for a in ids])
    d = np.array([dists[a] for a in ids])
    p = np.array(guess, dtype=float)
    for _ in range(20):
        diff = p - A
        r = np.maximum(np.linalg.norm(diff, axis=1), 1e-6)
        J = diff / r[:, None]
        step = np.linalg.lstsq(J, r - d, rcond=None)[0]
        p = p - step
        if np.linalg.norm(step) < 1e-4:
            break
    r = np.linalg.norm(p - A, axis=1)
    return p, float(np.sqrt(np.mean((r - d) ** 2)))


class EKF2D:
    """State: [x, y, vx, vy]. Measurements: distances to the anchors."""

    def __init__(self, p0, p_std, q=0.2, r=0.10):
        self.x = np.array([p0[0], p0[1], 0.0, 0.0])
        # starting uncertainty comes from how much the LS fixes moved while still
        self.P = np.diag([p_std[0] ** 2, p_std[1] ** 2, 0.25, 0.25])
        self.q = q   # how fast the tag can speed up (m/s^2)
        self.r = r   # noise of one distance reading (m)
        self.t = None

    def predict(self, t):
        if self.t is None:
            self.t = t
            return
        dt = DT   # fixed step: the tag reports every 200 ms
        self.t = t
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]])
        Q = G @ G.T * self.q ** 2
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, dists, gate=5.0):
        used = 0
        for a, pos in ANCHORS.items():
            d = dists.get(a, 0)
            if d <= 0:
                continue
            diff = self.x[:2] - np.array(pos)
            rho = np.linalg.norm(diff)
            if rho < 1e-6:
                continue
            H = np.zeros((1, 4))
            H[0, :2] = diff / rho
            innov = d - rho
            S = (H @ self.P @ H.T)[0, 0] + self.r ** 2
            if abs(innov) / np.sqrt(S) > gate:   # outlier, skip it
                continue
            K = (self.P @ H.T) / S
            self.x = self.x + (K * innov).flatten()
            self.P = (np.eye(4) - K @ H) @ self.P
            used += 1
        self.x[2:] = np.clip(self.x[2:], -0.5, 0.5)   # cap the speed (m/s)
        return used


def input_loop(true):
    """Type the true position in cm, e.g.  90 30  and press Enter."""
    while True:
        s = input()
        try:
            xs, ys = s.replace(",", " ").split()
            true["p"] = (float(xs) / 100, float(ys) / 100)
            print(f"--> true position set to ({xs}, {ys}) cm")
        except Exception:
            print("type the true position in cm like:  90 30")


def main():
    true = {"p": START_TRUE}
    threading.Thread(target=input_loop, args=(true,), daemon=True).start()

    ids = list(ANCHORS)
    f = open(OUT_FILE, "w", newline="")
    w = csv.writer(f)
    # d_<ID> columns are the RAW distances from the board; "bias" is what was subtracted
    w.writerow(["t", "true_x", "true_y", "ls_x", "ls_y", "ekf_x", "ekf_y",
                "ekf_sx", "ekf_sy", "fit_err", "used"] + [f"d_{a}" for a in ids] + ["bias"])

    print(f"CALIBRATION: put the tag EXACTLY at {START_TRUE[0]*100:.0f},{START_TRUE[1]*100:.0f} cm "
          f"and keep it STILL for {STILL_SECONDS} s (hands away)...")
    print(f"Saving to {OUT_FILE}. While running, type the true position (cm) "
          f"each time you move the tag, e.g.  90 30")

    t0 = time.time()
    guess = [0.6, 0.35]
    ekf = None
    misses = 0
    n = 0
    bias = 0.0
    cal_samples = []   # raw distances collected while the tag sits at START_TRUE

    try:
        for raw in read_distances():
            now = time.time()

            # ---- start-up: calibrate, then start the EKF from the LS fixes ----
            if ekf is None:
                cal_samples.append(raw)
                if now - t0 < STILL_SECONDS or len(cal_samples) < CAL_MIN_SAMPLES:
                    continue
                b, per_anchor = calibrate(cal_samples)
                if b is None:
                    continue   # not enough anchors heard yet, keep collecting
                bias = b
                print("Calibration: reading too long by "
                      + ", ".join(f"{a} {v*100:+.0f}" for a, v in per_anchor.items())
                      + f" cm  ->  subtracting {bias*100:.1f} cm from every distance")
                if not (0.0 <= bias <= 0.40):
                    print("!!! WARNING: bias looks odd - is the tag exactly at the start "
                          "mark and are anchor IDs/corners right?")
                fixes = []
                g = list(START_TRUE)
                for s in cal_samples:
                    r = solve_ls(correct(s, bias), g)
                    if r is not None and r[1] < 0.15:
                        fixes.append(r[0])
                        g = r[0]
                if len(fixes) < 10:
                    continue   # keep collecting more samples
                arr = np.array(fixes)
                p0 = np.median(arr, axis=0)
                std = np.maximum(arr.std(axis=0), 0.03)
                ekf = EKF2D(p0, std)
                ekf.t = now
                guess = list(p0)
                print(f"EKF starts at x={p0[0]*100:.0f} y={p0[1]*100:.0f} cm "
                      f"(median of {len(arr)} LS fixes, spread "
                      f"{std[0]*100:.0f}/{std[1]*100:.0f} cm)")
                continue

            # ---- normal running: use the corrected distances ----
            dist = correct(raw, bias)
            res = solve_ls(dist, guess)
            if res is None:
                continue
            ls, err = res
            guess = ls

            # ---- normal running ----
            ekf.predict(now)
            used = ekf.update(dist)
            misses = misses + 1 if used < 2 else 0
            if misses >= 3:   # lost track: restart from the LS fix
                ekf = EKF2D(ls, [0.10, 0.10])
                ekf.t = now
                misses = 0
                print("EKF lost track, restarted from LS")

            ex, ey = ekf.x[0], ekf.x[1]
            sx, sy = np.sqrt(ekf.P[0, 0]), np.sqrt(ekf.P[1, 1])
            tx, ty = true["p"]
            w.writerow([f"{now-t0:.2f}", tx, ty, f"{ls[0]:.3f}", f"{ls[1]:.3f}",
                        f"{ex:.3f}", f"{ey:.3f}", f"{sx:.3f}", f"{sy:.3f}",
                        f"{err:.3f}", used] + [raw.get(a, 0) for a in ids] + [f"{bias:.4f}"])
            f.flush()

            n += 1
            if n % 10 == 0:
                e_ls = np.hypot(ls[0] - tx, ls[1] - ty) * 100
                e_ekf = np.hypot(ex - tx, ey - ty) * 100
                print(f"true({tx*100:.0f},{ty*100:.0f})  LS({ls[0]*100:.0f},{ls[1]*100:.0f}) "
                      f"err {e_ls:.0f} cm | EKF({ex*100:.0f},{ey*100:.0f}) "
                      f"err {e_ekf:.0f} cm | fit {err*100:.0f} cm | used {used}/4")
    except KeyboardInterrupt:
        pass
    finally:
        f.close()
        print(f"\nSaved {OUT_FILE}")


if __name__ == "__main__":
    main()
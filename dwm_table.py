"""
dwm_table.py  -  UWB tag -> distances -> calibration -> 3D trilateration (LS) -> 3D EKF -> CSV

THIS FILE IS THE "CORE". dwm_ros_node.py re-uses it, so keep both in the same folder.

WHAT YOU MUST EDIT (top of the file only):
   1. ANCHORS      : the ID and (x, y, z) of each of the 5 anchors, in METRES
   2. START_TRUE   : the exact point where you put the tag at the start for calibration (x, y, z) in metres
   3. PORT         : the tag's USB port, if it is not /dev/ttyACM0

Run:  python3 dwm_table.py     (stop with Ctrl+C, the CSV is saved as run_HHMMSS.csv)
"""
import csv
import re
import time

import numpy as np
import serial

PORT = "/dev/ttyACM0"

# ---------------------------------------------------------------
# ANCHORS: ID -> (x, y, z) in METRES.
# !!! Put the real numbers. Replace "XXXX" with the 5th anchor's ID. !!!
# !!! If all anchors have the same z, the 3D height (z) will NOT be reliable. !!!
# ---------------------------------------------------------------
ANCHORS = {
    "8D13": (5.177,-0.257,0.836),
    "DD0E": (3.632,3.380,0.114),
    "9B9E": (-3.523,-0.865,0.881),
    "54A9": (-0.649,3.549,1.820),
    "C79A": (-1.066, -3.782,0.847),   # <- 5th anchor: real ID + real (x, y, z)
}
START_TRUE = (0.00, 0.00, 0.00)     # exact tag position (x, y, z) in metres during the start calibration

STILL_SECONDS = 10          # keep the tag still this long at the start (used for calibration)
CAL_MIN_SAMPLES = 20        # need at least this many readings before calibrating
DT = 0.1                    # seconds between readings (tag set to 100 ms = 10 Hz)

# ---- EKF settings (change these if the track is too jumpy or too slow) ----
Q_ACCEL = 1.0               # how fast the tag can speed up (m/s^2). Bigger = trusts new readings more
R_RANGE = 0.10              # noise of one distance reading (m)
GATE = 5.0                  # reject a distance if it is this many sigmas away from the prediction
MAX_SPEED = 3.0             # cap on speed (m/s)
LOST_CYCLES = 5             # restart from LS after this many cycles with < 2 usable distances

OUT_FILE = "run_" + time.strftime("%H%M%S") + ".csv"
PATTERN = re.compile(r"([0-9A-Fa-f]{4})\[[^\]]*\]=([\d.]+)")


def read_distances(port=PORT):
    """Yields {anchor_id: distance_in_m} for every line the tag prints. 0 = no reading."""
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


def check_anchors():
    """Stop if the anchors were not filled in; warn if they are all at the same height."""
    if "XXXX" in ANCHORS:
        raise SystemExit('Fill in the 5th anchor: replace "XXXX" in ANCHORS (top of dwm_table.py) '
                         "with its real ID and (x, y, z) in metres.")
    zs = {round(p[2], 3) for p in ANCHORS.values()}
    if len(zs) == 1:
        print("!!! WARNING: all anchors are at the same height. x and y are fine, "
              "but z (height) will be unreliable.")


def correct(dists, bias):
    """Subtract the calibration bias from every valid distance (0 = invalid, stays 0)."""
    return {a: (max(d - bias, 0.01) if d > 0 else 0) for a, d in dists.items()}


def calibrate(samples):
    """Tag is still at START_TRUE. Each anchor's distance should equal the true distance
    from START_TRUE to that anchor. The average amount the readings are too long is the bias."""
    per_anchor = {}
    for a, pos in ANCHORS.items():
        vals = [s[a] for s in samples if s.get(a, 0) > 0]
        if len(vals) >= 5:
            expected = float(np.linalg.norm(np.array(START_TRUE) - np.array(pos)))
            per_anchor[a] = float(np.median(vals)) - expected
    if len(per_anchor) < 3:
        return None, per_anchor
    return float(np.mean(list(per_anchor.values()))), per_anchor


def solve_ls(dists, guess):
    """3D least squares (trilateration). Needs 4+ distances. Returns (position, fit_error) or None."""
    ids = [a for a in ANCHORS if dists.get(a, 0) > 0]
    if len(ids) < 4:
        return None
    A = np.array([ANCHORS[a] for a in ids], dtype=float)
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


class EKF3D:
    """State: [x, y, z, vx, vy, vz]. Measurements: distances to the anchors."""

    def __init__(self, p0, p_std, q=Q_ACCEL, r=R_RANGE):
        self.x = np.array([p0[0], p0[1], p0[2], 0.0, 0.0, 0.0])
        # starting uncertainty comes from how much the LS fixes moved while still
        self.P = np.diag([p_std[0] ** 2, p_std[1] ** 2, p_std[2] ** 2, 0.25, 0.25, 0.25])
        self.q = q
        self.r = r
        self.t = None

    def predict(self, t):
        if self.t is None:
            self.t = t
            return
        dt = DT   # fixed step: the tag reports every 100 ms
        self.t = t
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        G = np.vstack([0.5 * dt * dt * np.eye(3), dt * np.eye(3)])
        Q = G @ G.T * self.q ** 2
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, dists, gate=GATE):
        used = 0
        for a, pos in ANCHORS.items():
            d = dists.get(a, 0)
            if d <= 0:
                continue
            diff = self.x[:3] - np.array(pos)
            rho = np.linalg.norm(diff)
            if rho < 1e-6:
                continue
            H = np.zeros((1, 6))
            H[0, :3] = diff / rho
            innov = d - rho
            S = (H @ self.P @ H.T)[0, 0] + self.r ** 2
            if abs(innov) / np.sqrt(S) > gate:   # outlier, skip it
                continue
            K = (self.P @ H.T) / S
            self.x = self.x + (K * innov).flatten()
            self.P = (np.eye(6) - K @ H) @ self.P
            used += 1
        self.x[3:] = np.clip(self.x[3:], -MAX_SPEED, MAX_SPEED)
        return used


def main():
    check_anchors()
    ids = list(ANCHORS)
    f = open(OUT_FILE, "w", newline="")
    w = csv.writer(f)
    # d_<ID> columns are the RAW distances from the board; "bias" is what was subtracted
    w.writerow(["t", "ls_x", "ls_y", "ls_z",
                "ekf_x", "ekf_y", "ekf_z", "ekf_sx", "ekf_sy", "ekf_sz", "fit_err", "used"]
               + [f"d_{a}" for a in ids] + ["bias"])

    print(f"CALIBRATION: put the tag EXACTLY at {START_TRUE[0]*100:.0f},{START_TRUE[1]*100:.0f},"
          f"{START_TRUE[2]*100:.0f} cm and keep it STILL for {STILL_SECONDS} s (hands away)...")
    print(f"Saving to {OUT_FILE}")

    t0 = time.time()
    guess = [START_TRUE[0], START_TRUE[1], START_TRUE[2] + 0.05]
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
                    print("!!! WARNING: bias looks odd - is the tag exactly at START_TRUE "
                          "and are the anchor IDs / positions right?")
                fixes = []
                g = list(guess)
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
                ekf = EKF3D(p0, std)
                ekf.t = now
                guess = list(p0)
                print(f"EKF starts at x={p0[0]*100:.0f} y={p0[1]*100:.0f} z={p0[2]*100:.0f} cm "
                      f"(median of {len(arr)} LS fixes)")
                continue

            # ---- normal running: use the corrected distances ----
            dist = correct(raw, bias)
            res = solve_ls(dist, guess)
            if res is None:
                continue
            ls, err = res
            guess = ls

            ekf.predict(now)
            used = ekf.update(dist)
            misses = misses + 1 if used < 2 else 0
            if misses >= LOST_CYCLES:   # lost track: restart from the LS fix
                ekf = EKF3D(ls, [0.10, 0.10, 0.10])
                ekf.t = now
                misses = 0
                print("EKF lost track, restarted from LS")

            e = ekf.x[:3]
            s = np.sqrt(np.diag(ekf.P)[:3])
            w.writerow([f"{now-t0:.2f}"]
                       + [f"{v:.3f}" for v in ls] + [f"{v:.3f}" for v in e]
                       + [f"{v:.3f}" for v in s] + [f"{err:.3f}", used]
                       + [raw.get(a, 0) for a in ids] + [f"{bias:.4f}"])
            f.flush()

            n += 1
            if n % 10 == 0:
                print(f"LS({ls[0]*100:.0f},{ls[1]*100:.0f},{ls[2]*100:.0f}) | "
                      f"EKF({e[0]*100:.0f},{e[1]*100:.0f},{e[2]*100:.0f}) cm | "
                      f"fit {err*100:.0f} cm | used {used}/{len(ids)}")
    except KeyboardInterrupt:
        pass
    finally:
        f.close()
        print(f"\nSaved {OUT_FILE}")


if __name__ == "__main__":
    main()
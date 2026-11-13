import time
import numpy as np
from dwm_ls import ANCHORS, read_distances, solve_position


class RangeEKF:
    """State: [x, y, z, vx, vy, vz]. Measurements: distances to the anchors."""

    def __init__(self, x0, q=0.5, r=0.15):
        self.x = np.array([*x0, 0.0, 0.0, 0.0])
        self.P = np.eye(6)
        self.q = q   # how much the tag can speed up or slow down (m/s^2)
        self.r = r   # noise of one distance reading (m)
        self.t = None

    def predict(self, t):
        if self.t is None:
            self.t = t
            return
        dt = t - self.t
        self.t = t
        if dt <= 0:
            return
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        G = np.zeros((6, 3))
        G[:3] = 0.5 * dt**2 * np.eye(3)
        G[3:] = dt * np.eye(3)
        Q = G @ G.T * self.q**2
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, dists, gate=3.0):
        """Use each valid distance one by one. Returns how many were used."""
        used = 0
        for a, pos in ANCHORS.items():
            d = dists.get(a, 0)
            if d <= 0:
                continue
            diff = self.x[:3] - np.array(pos)
            rng = np.linalg.norm(diff)
            if rng < 1e-6:
                continue
            H = np.zeros((1, 6))
            H[0, :3] = diff / rng
            innov = d - rng
            S = (H @ self.P @ H.T)[0, 0] + self.r**2
            if abs(innov) / np.sqrt(S) > gate:   # outlier, skip it
                continue
            K = (self.P @ H.T) / S
            self.x = self.x + (K * innov).flatten()
            self.P = (np.eye(6) - K @ H) @ self.P
            used += 1
        return used


if __name__ == "__main__":
    guess = [0.6, 0.3, 0.0]
    ekf = None
    for d in read_distances():
        now = time.time()

        if ekf is None:                 # start from the first LS position
            res = solve_position(d, guess)
            if res is None:
                continue
            ekf = RangeEKF(res[0])
            ekf.t = now
            continue

        ekf.predict(now)
        used = ekf.update(d)
        x, y, z = ekf.x[:3]
        ls = solve_position(d, ekf.x[:3])
        ls_txt = f"LS: {ls[0][0]:.2f} {ls[0][1]:.2f} {ls[0][2]:.2f}" if ls else "LS: --"
        print(f"EKF: x={x:.2f} y={y:.2f} z={z:.2f}   ({ls_txt})   used {used}/4")
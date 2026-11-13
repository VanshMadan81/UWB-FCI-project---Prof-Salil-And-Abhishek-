import re
import time
import numpy as np
import serial

PORT = "/dev/ttyACM0"

# Anchor positions in metres (from our ASL). Edit here if you redo ASL.
ANCHORS = {
    "8D13": (0.00, 0.00, 0.00),
    "54A9": (1.20, 0.75, 0.00),
    "DD0E": (1.20, 0.00, 0.00),
    "9B9E": (0.00, 0.75, 0.00),
}

PATTERN = re.compile(r"([0-9A-Fa-f]{4})\[[^\]]*\]=([\d.]+)")


def read_distances(port=PORT):
    """Yields {anchor_id: distance} for every line the tag prints."""
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


def solve_position(distances, guess):
    """Least squares: find the point whose distances to the anchors
    best match the measured distances. Returns (position, fit_error)."""
    ids = [a for a in ANCHORS if distances.get(a, 0) > 0]
    if len(ids) < 4:  # need all 4 valid distances
        return None
    A = np.array([ANCHORS[a] for a in ids])
    d = np.array([distances[a] for a in ids])

    p = np.array(guess, dtype=float)
    for _ in range(20):
        diff = p - A
        r = np.linalg.norm(diff, axis=1)
        r[r < 1e-6] = 1e-6
        J = diff / r[:, None]
        step = np.linalg.lstsq(J, r - d, rcond=None)[0]
        p = p - step
        if np.linalg.norm(step) < 1e-4:
            break

    r = np.linalg.norm(p - A, axis=1)
    return p, float(np.sqrt(np.mean((r - d) ** 2)))


if __name__ == "__main__":
    guess = [0.5, 1.0, 1.0]
    for dist in read_distances():
        result = solve_position(dist, guess)
        if result is None:
            continue
        pos, err = result
        guess = pos  # start the next solve from the last position
        print(f"x={pos[0]:.2f}  y={pos[1]:.2f}  z={pos[2]:.2f}   (fit error {err:.2f} m)")
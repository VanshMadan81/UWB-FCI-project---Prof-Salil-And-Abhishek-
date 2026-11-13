#!/usr/bin/env python3
"""
dwm_ros_node.py  -  publishes the UWB tag data as ROS 2 topics (3D), for ros2 bag record.

Keep it in the SAME folder as dwm_table.py (it re-uses its calibration, LS and EKF code).
Edit the anchors / START_TRUE at the top of dwm_table.py, not here.

Run:
   source /opt/ros/<your_distro>/setup.bash
   python3 dwm_ros_node.py
Put the tag exactly at START_TRUE and keep it still for STILL_SECONDS. It then starts publishing.

Record (second terminal):
   ros2 bag record -o uwb_test1 /uwb/ranges /uwb/ls_position /uwb/ekf_position /uwb/bias

Topics:
   /uwb/ranges        std_msgs/Float64MultiArray   RAW distances (m), in the ANCHORS order printed at start (0 = no reading)
   /uwb/ls_position   geometry_msgs/PointStamped   trilateration (LS) position x, y, z (m)
   /uwb/ekf_position  geometry_msgs/PointStamped   EKF position x, y, z (m)
   /uwb/bias          std_msgs/Float64             calibration bias that was subtracted (m)
"""
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64, Float64MultiArray

import dwm_table as D

FRAME_ID = "odom"


def make_point(node, p):
    m = PointStamped()
    m.header.stamp = node.get_clock().now().to_msg()
    m.header.frame_id = FRAME_ID
    m.point.x, m.point.y, m.point.z = float(p[0]), float(p[1]), float(p[2])
    return m


def main():
    D.check_anchors()
    rclpy.init()
    node = rclpy.create_node("uwb_tag_estimator")
    pub_rng = node.create_publisher(Float64MultiArray, "/uwb/ranges", 10)
    pub_ls = node.create_publisher(PointStamped, "/uwb/ls_position", 10)
    pub_ekf = node.create_publisher(PointStamped, "/uwb/ekf_position", 10)
    pub_bias = node.create_publisher(Float64, "/uwb/bias", 10)
    ids = list(D.ANCHORS)
    print("/uwb/ranges order:", ", ".join(ids), "(0 = no reading)")
    print(f"CALIBRATION: put the tag EXACTLY at {D.START_TRUE[0]*100:.0f},{D.START_TRUE[1]*100:.0f},"
          f"{D.START_TRUE[2]*100:.0f} cm and keep it STILL for {D.STILL_SECONDS} s...")

    t0 = time.time()
    guess = [D.START_TRUE[0], D.START_TRUE[1], D.START_TRUE[2] + 0.05]
    ekf, misses, bias, cal_samples = None, 0, 0.0, []

    try:
        for raw in D.read_distances():
            now = time.time()

            # ---- start-up: calibrate, then start the EKF from the LS fixes ----
            if ekf is None:
                cal_samples.append(raw)
                if now - t0 < D.STILL_SECONDS or len(cal_samples) < D.CAL_MIN_SAMPLES:
                    continue
                b, per_anchor = D.calibrate(cal_samples)
                if b is None:
                    continue
                bias = b
                print("Calibration: reading too long by "
                      + ", ".join(f"{a} {v*100:+.0f}" for a, v in per_anchor.items())
                      + f" cm -> subtracting {bias*100:.1f} cm")
                if not (0.0 <= bias <= 0.40):
                    print("!!! WARNING: bias looks odd - tag exactly at START_TRUE? anchors right?")
                fixes, g = [], list(guess)
                for s in cal_samples:
                    r = D.solve_ls(D.correct(s, bias), g)
                    if r is not None and r[1] < 0.15:
                        fixes.append(r[0])
                        g = r[0]
                if len(fixes) < 10:
                    continue
                arr = np.array(fixes)
                p0 = np.median(arr, axis=0)
                ekf = D.EKF3D(p0, np.maximum(arr.std(axis=0), 0.03))
                ekf.t = now
                guess = list(p0)
                print(f"EKF starts at x={p0[0]*100:.0f} y={p0[1]*100:.0f} z={p0[2]*100:.0f} cm. "
                      f"Publishing now - start ros2 bag record.")
                continue

            # ---- normal running ----
            dist = D.correct(raw, bias)
            res = D.solve_ls(dist, guess)
            if res is None:
                continue
            ls, _err = res
            guess = ls

            ekf.predict(now)
            used = ekf.update(dist)
            misses = misses + 1 if used < 2 else 0
            if misses >= D.LOST_CYCLES:
                ekf = D.EKF3D(ls, [0.10, 0.10, 0.10])
                ekf.t = now
                misses = 0
                print("EKF lost track, restarted from LS")

            rng = Float64MultiArray()
            rng.data = [float(raw.get(a, 0.0)) for a in ids]
            pub_rng.publish(rng)
            pub_ls.publish(make_point(node, ls))
            pub_ekf.publish(make_point(node, ekf.x[:3]))
            pub_bias.publish(Float64(data=float(bias)))
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
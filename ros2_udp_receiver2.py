#!/usr/bin/env python3
"""ROS 2 UDP receiver: receives the pose sent by ros1_udp_sender.py and publishes it as
geometry_msgs/PoseStamped on /qualisys/uwb_tag/pose.

   python3 ros2_udp_receiver.py
   python3 ros2_udp_receiver.py --ros-args -p port:=5555 -p frame_id:=map
"""

import socket
import struct

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

# must match the sender: seq (uint32) + 8 doubles: stamp, px, py, pz, qx, qy, qz, qw
POSE_FMT = struct.Struct("!I8d")


class Bridge(Node):
    def __init__(self) -> None:
        super().__init__("ros2_udp_receiver")

        bind_ip = self.declare_parameter("bind_ip", "0.0.0.0").value
        port = int(self.declare_parameter("port", 5555).value)
        topic = self.declare_parameter("topic", "/qualisys/uwb_tag/pose").value
        self.frame_id = self.declare_parameter("frame_id", "map").value
        poll_period = float(self.declare_parameter("poll_period_s", 0.001).value)

        self.pub = self.create_publisher(PoseStamped, topic, 10)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((bind_ip, port))
        self.sock.setblocking(False)

        self.last_seq = None
        self.last_warn_ns = 0
        self.create_timer(poll_period, self.poll)
        self.get_logger().info(f"Listening on udp://{bind_ip}:{port}, publishing {topic}")

    def poll(self) -> None:
        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
            except BlockingIOError:
                break

            if len(data) != POSE_FMT.size:
                self._warn(f"Unexpected packet size {len(data)} bytes (expected {POSE_FMT.size})")
                continue

            seq, stamp, px, py, pz, qx, qy, qz, qw = POSE_FMT.unpack(data)

            # old / duplicated packet? (UDP can reorder) -> skip it
            if self.last_seq is not None and 0 < (self.last_seq - seq) % 4294967296 < 2147483648:
                continue
            self.last_seq = seq

            msg = PoseStamped()
            msg.header.stamp.sec = int(stamp)
            msg.header.stamp.nanosec = int((stamp - int(stamp)) * 1e9)
            msg.header.frame_id = self.frame_id
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = px, py, pz
            msg.pose.orientation.x, msg.pose.orientation.y = qx, qy
            msg.pose.orientation.z, msg.pose.orientation.w = qz, qw
            self.pub.publish(msg)

    def _warn(self, text: str) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self.last_warn_ns > 1_000_000_000:   # at most 1 warning per second
            self.get_logger().warning(text)
            self.last_warn_ns = now_ns

    def destroy_node(self) -> bool:
        try:
            self.sock.close()
        finally:
            return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Bridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

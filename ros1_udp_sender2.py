#!/usr/bin/env python
"""ROS1 UDP sender: forwards ONE topic (geometry_msgs/PoseStamped) over UDP.

Runs on the PC that has ROS 1.  Default topic: /qualisys/uwb_tag/pose

   rosrun <pkg> ros1_udp_sender.py _target_ip:=192.168.1.85
or
   python ros1_udp_sender.py _target_ip:=192.168.1.85 _port:=5555
"""

import socket
import struct

import rospy
from geometry_msgs.msg import PoseStamped

# packet = seq (uint32) + 8 doubles: stamp, px, py, pz, qx, qy, qz, qw   (network byte order)
POSE_FMT = struct.Struct("!I8d")


class Ros1UdpSender(object):
    def __init__(self):
        self.target_ip = rospy.get_param("~target_ip", "192.168.1.85")
        self.port = int(rospy.get_param("~port", 5555))
        self.topic = rospy.get_param("~topic", "/qualisys/uwb_tag/pose")

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target = (self.target_ip, self.port)
        self.seq = 0

        rospy.Subscriber(self.topic, PoseStamped, self._cb, queue_size=10)
        rospy.on_shutdown(self.sock.close)
        rospy.loginfo("ros1_udp_sender: %s -> udp://%s:%d", self.topic, self.target_ip, self.port)

    def _cb(self, msg):
        p, q = msg.pose.position, msg.pose.orientation
        packet = POSE_FMT.pack(
            self.seq,
            msg.header.stamp.to_sec(),
            p.x, p.y, p.z,
            q.x, q.y, q.z, q.w,
        )
        self.seq = (self.seq + 1) % 4294967296
        try:
            self.sock.sendto(packet, self.target)
        except socket.error as exc:
            rospy.logerr_throttle(1.0, "UDP send failed: %s", str(exc))


def main():
    rospy.init_node("ros1_udp_sender")
    Ros1UdpSender()
    rospy.spin()


if __name__ == "__main__":
    main()

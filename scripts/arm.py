#!/usr/bin/env python3
"""CRANE+ V2 の腕とグリッパを動かす。

MoveIt は使わず、URDF の寸法から解析的に逆運動学を解き、
FollowJointTrajectory アクションへ関節角を直接送る。
"""
import math
import time

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

# ---- ロボットの寸法（URDF より）[m] ----
H_SHOULDER = 0.0454 + 0.026  # base_link から joint2 までの高さ
L2 = 0.083                   # joint2 → joint3
L3 = 0.0935                  # joint3 → joint4
GRASP_OFFSET = 0.09          # joint4 → 指先でつかむ点

ARM_JOINTS = ["crane_plus_joint1", "crane_plus_joint2", "crane_plus_joint3", "crane_plus_joint4"]
HAND_JOINT = "crane_plus_joint_hand"
ARM_LIMITS = [(-2.618, 2.618), (-1.825, 1.784), (-2.557, 2.454), (-1.840, 1.768)]

GRIPPER_OPEN = -0.65   # 下限 -0.709。5cm のブロックが入るよう大きく開く
GRIPPER_CLOSE = 0.50   # 上限 0.668。物体に当たって止まり、目標との差の分だけ強く挟む
GRIPPER_RELEASE = 0.25  # 放すときは、挟んでいる角度からこれだけ開く（大きく開くとブロックをはじく）
GRIPPER_NEUTRAL = 0.0  # 放して持ち上げた後に戻す角度
GRASP_EMPTY_MARGIN = 0.08  # 閉じた後の角度が GRIPPER_CLOSE にこれ以上近ければ「何もつかめていない」

HOME = [0.0, 0.0, 0.0, 0.0]

TRAVEL_Z = 0.15      # 運ぶときの指先の高さ（届かない場所では届く高さまで自動で下げる）
DESCEND_SECONDS = 3.0  # 真上から下ろす・持ち上げる時間（遅いほどずれにくい）
# つかむときは手先を傾けない（傾くとつかむ位置がずれる）。その分、届く高さは指先 8.5cm までになる
MAX_TILT_DEG = 0
# 30° 傾ければ 13〜14cm まで上がる
CARRY_TILT_DEG = 30


def solve_ik(x, y, z, max_tilt=None):
    """base_link 基準の点 (x, y, z) に指先を合わせる関節角 [q1..q4] を返す。"""
    if max_tilt is None:
        max_tilt = MAX_TILT_DEG
    r = math.hypot(x, y)
    if r < 1e-6:
        raise ValueError(f"too close to base: ({x:.3f}, {y:.3f})")
    q1 = math.atan2(y, x)

    for tilt_deg in range(0, max_tilt + 1, 2):
        phi4 = math.pi - math.radians(tilt_deg)
        wx = r - GRASP_OFFSET * math.sin(phi4)
        wz = z - GRASP_OFFSET * math.cos(phi4) - H_SHOULDER
        d2 = wx ** 2 + wz ** 2
        c = (d2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
        if abs(c) > 1.0:
            continue
        delta = math.acos(c)
        phi2 = math.atan2(wx, wz) - math.atan2(L3 * math.sin(delta), L2 + L3 * math.cos(delta))
        phi3 = phi2 + delta
        # joint2 は +y 軸, joint3/4 は -y 軸まわり
        q = [q1, phi2, -delta, phi3 - phi4]
        if all(lo <= v <= hi for v, (lo, hi) in zip(q, ARM_LIMITS)):
            return q
    raise ValueError(f"unreachable: ({x:.3f}, {y:.3f}, {z:.3f})")


def reachable_z(x, y, z_want, step=0.005, max_tilt=None):
    """(x, y) の真上で、z_want 以下で届く一番高い z を返す。"""
    z = z_want
    while z > 0:
        try:
            solve_ik(x, y, z, max_tilt)
            return z
        except ValueError:
            z -= step
    raise ValueError(f"unreachable at any height: ({x:.3f}, {y:.3f})")


class Arm(Node):
    def __init__(self):
        super().__init__("arm")
        self.arm = ActionClient(self, FollowJointTrajectory, "/crane_plus_arm_controller/follow_joint_trajectory")
        self.hand = ActionClient(self, FollowJointTrajectory, "/crane_plus_gripper_controller/follow_joint_trajectory")
        for client in (self.arm, self.hand):
            if not client.wait_for_server(timeout_sec=10.0):
                raise RuntimeError(f"action server not available: {client._action_name}")
        self.tip = None  # 指先の現在位置 (x, y, z)。関節指定で動かした後は不明として None
        self.hand_position = None
        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 10)

    def _on_joint_states(self, msg):
        if HAND_JOINT in msg.name:
            self.hand_position = msg.position[msg.name.index(HAND_JOINT)]

    def holding_object(self):
        """グリッパが閉じ切らずに止まっていれば、何かを挟んでいると判断する。"""
        self.hand_position = None
        while self.hand_position is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.hand_position < GRIPPER_CLOSE - GRASP_EMPTY_MARGIN

    def _send(self, client, joint_names, waypoints, seconds):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = joint_names
        n = len(waypoints)
        for i, positions in enumerate(waypoints, start=1):
            t = seconds * i / n
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in positions]
            point.time_from_start = Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
            goal.trajectory.points.append(point)
        future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        if not handle.accepted:
            raise RuntimeError("trajectory goal rejected")
        result = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result)

    def move_joints(self, q, seconds=2.0):
        self._send(self.arm, ARM_JOINTS, [q], seconds)
        self.tip = None

    def move_line(self, x, y, z_from, z_to, seconds=1.5, steps=10, max_tilt=None):
        """指先を真上下方向に直線的に動かす。"""
        waypoints = [solve_ik(x, y, z_from + (z_to - z_from) * i / steps, max_tilt)
                     for i in range(1, steps + 1)]
        self._send(self.arm, ARM_JOINTS, waypoints, seconds)
        self.tip = (x, y, z_to)

    def move_arc(self, x, y, z=None, seconds=4.0, steps=12, max_tilt=None):
        """根元まわりに回り込んで (x, y) の真上へ移動する。"""
        x0, y0, _ = self.tip
        r0, t0 = math.hypot(x0, y0), math.atan2(y0, x0)
        r1, t1 = math.hypot(x, y), math.atan2(y, x)
        waypoints = []
        for i in range(1, steps + 1):
            s = i / steps
            r, t = r0 + (r1 - r0) * s, t0 + (t1 - t0) * s
            px, py = r * math.cos(t), r * math.sin(t)
            pz = z if z is not None else reachable_z(px, py, TRAVEL_Z, max_tilt=max_tilt)
            waypoints.append(solve_ik(px, py, pz, max_tilt))
        self._send(self.arm, ARM_JOINTS, waypoints, seconds)
        self.tip = (x, y, pz)

    def travel_height(self, x, y):
        """今の位置から (x, y) まで、同じ高さで運べる一番高い z（TRAVEL_Z 以下）。"""
        points = [(x, y)]
        if self.tip is not None:
            x0, y0, _ = self.tip
            r0, t0 = math.hypot(x0, y0), math.atan2(y0, x0)
            r1, t1 = math.hypot(x, y), math.atan2(y, x)
            points += [((r0 + (r1 - r0) * s) * math.cos(t0 + (t1 - t0) * s),
                        (r0 + (r1 - r0) * s) * math.sin(t0 + (t1 - t0) * s)) for s in (0, 0.25, 0.5, 0.75)]
        return min(reachable_z(px, py, TRAVEL_Z) for px, py in points)

    def transfer(self, x, y, z):
        """真上に持ち上げ → 高い位置を保って移動 → 真下へゆっくり下ろす。"""
        zt = self.travel_height(x, y)
        if self.tip is None:
            self.move_joints(solve_ik(x, y, zt))
            self.tip = (x, y, zt)
        else:
            x0, y0, z0 = self.tip
            if abs(z0 - zt) > 1e-3:
                self.move_line(x0, y0, z0, zt, seconds=DESCEND_SECONDS)
            self.move_arc(x, y, zt)
        self.move_line(x, y, zt, z, seconds=DESCEND_SECONDS)

    def lift(self, seconds=DESCEND_SECONDS):
        x, y, z = self.tip
        self.move_line(x, y, z, reachable_z(x, y, TRAVEL_Z), seconds=seconds)

    def gripper(self, value, seconds=1.0):
        self._send(self.hand, [HAND_JOINT], [[value]], seconds)

    def pick(self, x, y, z):
        """つかんで運ぶ高さまで持ち上げる。持ち上げた後もつかめていれば True。"""
        self.gripper(GRIPPER_OPEN)
        self.transfer(x, y, z)
        self.gripper(GRIPPER_CLOSE, seconds=1.5)
        time.sleep(0.5)
        self.lift()
        held = self.holding_object()
        self.get_logger().info(f"gripper={self.hand_position:.3f} -> {'holding' if held else 'EMPTY'}")
        return held


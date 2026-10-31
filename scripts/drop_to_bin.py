#!/usr/bin/env python3
"""指定した色のブロックを、カメラ + AI で見つけて指定順にゴミ箱へ片付ける。

使い方（Gazebo・spawn_blocks.sh・ros_gz_bridge を起動した状態で）:
    python3 drop_to_bin.py red blue green
"""
import sys
import time

import rclpy

from arm import CARRY_TILT_DEG, GRIPPER_NEUTRAL, GRIPPER_OPEN, GRIPPER_RELEASE, HOME, TRAVEL_Z, Arm, reachable_z
from detect_blocks import CATALOG, detect_scene, draw, grab_image, make_detector

ALIASES = {"赤": "red", "青": "blue", "緑": "green", "黒": "black"}
MAX_TRIES = 3

BIN_XY = (0.11, 0.0)      # ゴミ箱の中心（spawn_blocks.sh と同じ。グレーなのでカメラでは探さない）
BIN_INNER_HALF = 0.06     # ゴミ箱の内側の半分の幅。中心からこれ以内なら「箱の中」


def parse_args(argv):
    detector = "yolo"
    colors = []
    args = iter(argv)
    for a in args:
        if a == "--detector":
            detector = next(args)
        else:
            colors.append(ALIASES.get(a, a.lower()))
    bad = [c for c in colors if c not in CATALOG]
    if not colors or bad:
        sys.exit(f"usage: drop_to_bin.py <color> ... ({'/'.join(CATALOG)})  invalid: {bad}")
    return colors, detector


def reachable(x, y, z):
    """真下向きでつかめる位置か（腕の届く範囲は根元から約 8〜16cm）。"""
    try:
        return reachable_z(x, y, z) >= z - 1e-6
    except ValueError:
        return False


def in_bin(d):
    return abs(d["x"] - BIN_XY[0]) < BIN_INNER_HALF and abs(d["y"] - BIN_XY[1]) < BIN_INNER_HALF


def observe(node, detector, path="detections.png"):
    """アームを退避して撮影し、検出結果を返す。"""
    node.move_joints(HOME)
    time.sleep(1.0)
    img = grab_image(node)
    scene = detect_scene(img, detector)
    node.get_logger().info(f"detections saved to {draw(img, scene, path)}")
    for color, d in scene.items():
        where = "in bin" if in_bin(d) else "on table"
        node.get_logger().info(f"  {color}: x={d['x']:.3f} y={d['y']:.3f} conf={d['conf']:.2f} ({where})")
    return scene


def pick(node, detector, color, scene):
    """つかめるまで最大 MAX_TRIES 回試す。つかめたら True。"""
    for attempt in range(1, MAX_TRIES + 1):
        if attempt > 1:
            node.get_logger().warn(f"{color}: pick retry {attempt}/{MAX_TRIES}")
            scene = observe(node, detector, f"detections_retry_{color}_{attempt}.png")
            if color not in scene:
                node.get_logger().error(f"{color}: not found on retry")
                return False
        d = scene[color]
        if not reachable(d["x"], d["y"], CATALOG[color] / 2):
            node.get_logger().error(f"{color}: out of reach at ({d['x']:.3f}, {d['y']:.3f})")
            return False
        node.get_logger().info(f"{color}: pick ({d['x']:.3f}, {d['y']:.3f})")
        if node.pick(d["x"], d["y"], CATALOG[color] / 2):
            return True
        node.gripper(GRIPPER_OPEN)
    return False


def drop(node):
    """ゴミ箱の真上へ運び、静かに放す。

    真下向きのままだと低い位置しか通れず箱のふちに当たるので、運ぶ間だけ手先を
    CARRY_TILT_DEG まで傾けて高く持ち上げ、箱の真上で真下向きに戻しながら下ろしてから放す。
    """
    x, y = BIN_XY
    x0, y0, z0 = node.tip
    node.move_line(x0, y0, z0, reachable_z(x0, y0, TRAVEL_Z, max_tilt=CARRY_TILT_DEG),
                   seconds=2.0, max_tilt=CARRY_TILT_DEG)
    node.move_arc(x, y, seconds=5.0, max_tilt=CARRY_TILT_DEG)
    # 真下向きで届く一番高い位置まで下ろす（傾きは下りながら自然に 0 に戻る）
    node.move_line(x, y, node.tip[2], reachable_z(x, y, TRAVEL_Z), seconds=2.0, max_tilt=CARRY_TILT_DEG)
    time.sleep(0.5)  # 揺れが収まるのを待つ
    # 大きく開くと指がブロックをはじくので、挟んでいる角度から少しだけ開く
    node.holding_object()
    node.gripper(node.hand_position - GRIPPER_RELEASE, seconds=1.0)
    time.sleep(1.0)  # 落ちきるまで待つ
    node.gripper(GRIPPER_NEUTRAL)


def tidy(node, detector, color, scene):
    """1 色を片付ける。(成功したか, 最新の検出結果) を返す。"""
    log = node.get_logger()
    for attempt in range(1, MAX_TRIES + 1):
        if color not in scene:
            log.error(f"{color}: not found")
            return False, scene
        if in_bin(scene[color]):
            log.info(f"{color}: already in the bin")
            return True, scene
        if attempt > 1:
            log.warn(f"{color}: fell outside the bin, retry {attempt}/{MAX_TRIES}")
        if not pick(node, detector, color, scene):
            log.error(f"{color}: could not pick")
            return False, scene
        drop(node)
        scene = observe(node, detector, f"after_{color}_{attempt}.png")  # 入ったか確かめる
        if color not in scene or in_bin(scene[color]):
            log.info(f"{color}: in the bin")
            return True, scene
    return False, scene


def main():
    colors, detector_name = parse_args(sys.argv[1:])
    detector = make_detector(detector_name)
    rclpy.init()
    node = Arm()
    try:
        scene = observe(node, detector)
        results = {}
        for color in colors:
            results[color], scene = tidy(node, detector, color, scene)
        node.move_joints(HOME)
        node.get_logger().info("result: " + ", ".join(f"{c}={'OK' if ok else 'NG'}" for c, ok in results.items()))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


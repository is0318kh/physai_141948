#!/usr/bin/env python3
"""真上カメラの画像から、AI（YOLO-World）でブロックを検出して色と位置を求める。

    python3 detect_blocks.py --detector color   # 比較・デバッグ用: AI を使わず色だけで検出
    python3 detect_blocks.py --debug            # AI が出した全ての枠と、捨てた理由を表示
"""
import math
import sys
from pathlib import Path

# YOLO-World が使う CLIP は pip から入れても認識されなかったので、直接 src/CLIP を sys.path に追加する
CLIP_DIR = Path(__file__).resolve().parent.parent / "src" / "CLIP"
if CLIP_DIR.is_dir():
    sys.path.insert(0, str(CLIP_DIR))

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

IMAGE_TOPIC = "/overhead_camera/image"

# ---- カメラ（spawn_blocks.sh の配置と合わせる。base_link 基準）----
CAM_X, CAM_Y = 0.10, 0.0
CAM_HEIGHT = 0.45     # 机の天板からの高さ
HFOV = 1.047

# 色ごとのブロック: 上面の高さ [m]（どれも 5cm 立方体）
# ゴミ箱はグレーで、どの色にも当てはまらないので検出結果には入らない
CATALOG = {
    "red": 0.05,     # red_cube
    "blue": 0.05,    # blue_cube
    "green": 0.05,   # green_cube
    "black": 0.05,   # wood_cube_5cm（元のワールドにある黒い箱）
}

# 真上から見るとブロックは「色付きの四角」に見えるので、色と形の言葉で指定する
PROMPTS = ["red square", "green square", "blue square", "black square", "cube", "toy block", "box"]
MIN_CONF = 0.01
WEIGHTS = "yolov8s-worldv2.pt"
ROBOT_RADIUS = 0.06   # ロボットの根元付近の検出は無視する
BOX_SIZE_PX = (12, 180)


def grab_image(node, topic=IMAGE_TOPIC, timeout=10.0):
    """トピックから 1 枚受け取り BGR の画像を返す。"""
    holder = {}

    def callback(msg):
        holder["msg"] = msg

    sub = node.create_subscription(Image, topic, callback, 1)
    end = node.get_clock().now().nanoseconds + timeout * 1e9
    while "msg" not in holder:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.get_clock().now().nanoseconds > end:
            raise RuntimeError(f"no image on {topic}. ros_gz_bridge は起動していますか？")
    node.destroy_subscription(sub)
    msg = holder["msg"]
    img = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)
    img = img[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == "rgb8" else img.copy()


def pixel_to_base(u, v, width, height, object_height):
    """画像上の点 (u, v) を、高さ object_height の物体上面として base_link の (x, y) に変換する。

    カメラは真下向きで、画像の上が +x、右が -y になる。
    """
    f = (width / 2) / math.tan(HFOV / 2)
    depth = CAM_HEIGHT - object_height
    x = CAM_X + (height / 2 - v) * depth / f
    y = CAM_Y - (u - width / 2) * depth / f
    return x, y


def classify_color(img, box):
    """枠の中央部分の色から black / blue / red / green のどれかを返す（該当なしは None）。"""
    x1, y1, x2, y2 = box
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 4, (y2 - y1) / 4
    crop = img[int(cy - h):int(cy + h) + 1, int(cx - w):int(cx + w) + 1]
    if crop.size == 0:
        return None
    hue, sat, val = np.median(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3), axis=0)
    if val < 70:
        return "black"
    if sat < 80:
        return None
    if hue < 8 or hue > 170:
        return "red"
    if 35 <= hue <= 85:
        return "green"
    if 95 <= hue <= 135:
        return "blue"
    return None


class YoloWorldDetector:
    def __init__(self):
        # ultralytics は CLIP の import に失敗すると理由を隠して pip で入れ直そうとするので、先に自分で import する
        if not CLIP_DIR.is_dir():
            raise RuntimeError(f"CLIP が見つからない: {CLIP_DIR}")
        import clip 
        from ultralytics import YOLO
        self.model = YOLO(WEIGHTS)
        self.model.set_classes(PROMPTS)

    def __call__(self, img):
        result = self.model.predict(img, conf=MIN_CONF, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        return [(tuple(b), float(c)) for b, c in zip(boxes, confs)]


class ColorDetector:
    """比較・デバッグ用。AI を使わず色のしきい値だけで物体を探す。"""

    def __call__(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        masks = [
            cv2.inRange(hsv, (0, 0, 0), (180, 255, 69)),
            cv2.inRange(hsv, (0, 80, 30), (7, 255, 255)) | cv2.inRange(hsv, (171, 80, 30), (180, 255, 255)),
            cv2.inRange(hsv, (35, 80, 30), (85, 255, 255)),
            cv2.inRange(hsv, (95, 80, 30), (135, 255, 255)),
        ]
        found = []
        for mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                found.append(((x, y, x + w, y + h), 1.0))
        return found


def detect_scene(img, detector, debug=False):
    """色ごとに最も確からしい検出を 1 つ選び {color: {x, y, box, conf}} を返す。

    debug=True のときは、AI が出した全ての枠と、採用しなかった理由を表示する。
    """
    height, width = img.shape[:2]
    scene = {}
    for box, conf in detector(img):
        x1, y1, x2, y2 = box
        size = max(x2 - x1, y2 - y1)
        tag = f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) size={size:.0f}px conf={conf:.2f}"
        if not BOX_SIZE_PX[0] <= size <= BOX_SIZE_PX[1]:
            if debug:
                print(f"  skip {tag}: size out of range {BOX_SIZE_PX}")
            continue
        color = classify_color(img, box)
        if color is None:
            if debug:
                print(f"  skip {tag}: color unknown")
            continue
        x, y = pixel_to_base((x1 + x2) / 2, (y1 + y2) / 2, width, height, CATALOG[color])
        if math.hypot(x, y) < ROBOT_RADIUS:
            if debug:
                print(f"  skip {tag}: {color} near robot base ({x:.3f}, {y:.3f})")
            continue
        if debug:
            print(f"  keep {tag}: {color} ({x:.3f}, {y:.3f})")
        if color not in scene or conf > scene[color]["conf"]:
            scene[color] = {"x": x, "y": y, "box": box, "conf": conf}
    return scene


BGR = {"black": (60, 60, 60), "blue": (255, 80, 0), "red": (0, 0, 255), "green": (0, 200, 0)}


def draw(img, scene, path="detections.png"):
    out = img.copy()
    for color, d in scene.items():
        x1, y1, x2, y2 = map(int, d["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), BGR[color], 2)
        label = f"{color} {d['conf']:.2f} ({d['x']:.3f}, {d['y']:.3f})"
        cv2.putText(out, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, BGR[color], 1)
    cv2.imwrite(path, out)
    return path


def make_detector(name):
    return ColorDetector() if name == "color" else YoloWorldDetector()


def main():
    detector = sys.argv[sys.argv.index("--detector") + 1] if "--detector" in sys.argv else "yolo"
    rclpy.init()
    node = Node("block_detector")
    try:
        img = grab_image(node)
        scene = detect_scene(img, make_detector(detector), debug="--debug" in sys.argv)
        for color, d in scene.items():
            print(f"{color:6s} x={d['x']:.3f} y={d['y']:.3f} conf={d['conf']:.2f}")
        cv2.imwrite("camera_raw.png", img)
        print("saved:", draw(img, scene), "camera_raw.png")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


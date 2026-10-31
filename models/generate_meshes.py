#!/usr/bin/env python3
import struct
from pathlib import Path

BLOCK = 0.03
BIN_W, BIN_H, BIN_T = 0.10, 0.04, 0.005
BIN_PARTS = [
    ((BIN_W, BIN_W, BIN_T), (0, 0, BIN_T / 2)),                                 # 底
    ((BIN_W, BIN_T, BIN_H), (0, (BIN_W - BIN_T) / 2, BIN_H / 2)),               # 壁 +y
    ((BIN_W, BIN_T, BIN_H), (0, -(BIN_W - BIN_T) / 2, BIN_H / 2)),              # 壁 -y
    ((BIN_T, BIN_W - 2 * BIN_T, BIN_H), ((BIN_W - BIN_T) / 2, 0, BIN_H / 2)),   # 壁 +x
    ((BIN_T, BIN_W - 2 * BIN_T, BIN_H), (-(BIN_W - BIN_T) / 2, 0, BIN_H / 2)),  # 壁 -x
]

# name: [(大きさ (x, y, z), 中心 (x, y, z)), ...]
SHAPES = {
    "red_cube": ("cube/red_cube", [((BLOCK, BLOCK, BLOCK), (0, 0, 0))]),
    "blue_cube": ("cube/blue_cube", [((BLOCK, BLOCK, BLOCK), (0, 0, 0))]),
    "green_cube": ("cube/green_cube", [((BLOCK, BLOCK, BLOCK), (0, 0, 0))]),
    "trash_bin": ("trash_bin", BIN_PARTS),  # 原点は底面の中心
}

MODELS_DIR = Path(__file__).resolve().parent


def box_triangles(size, center=(0, 0, 0)):
    """center を中心とする直方体を 12 枚の三角形（外向き法線付き）で返す。"""
    (sx, sy, sz), (cx, cy, cz) = size, center
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    v = [
        (cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz),
    ]
    # 各面: (法線, 外側から見て反時計回りの頂点 4 つ)
    faces = [
        ((0, 0, -1), (0, 3, 2, 1)),
        ((0, 0, 1), (4, 5, 6, 7)),
        ((0, -1, 0), (0, 1, 5, 4)),
        ((0, 1, 0), (2, 3, 7, 6)),
        ((-1, 0, 0), (0, 4, 7, 3)),
        ((1, 0, 0), (1, 2, 6, 5)),
    ]
    tris = []
    for normal, (a, b, c, d) in faces:
        tris.append((normal, v[a], v[b], v[c]))
        tris.append((normal, v[a], v[c], v[d]))
    return tris


def write_binary_stl(path, name, tris):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(name.encode("ascii").ljust(80, b"\0")[:80])
        f.write(struct.pack("<I", len(tris)))
        for normal, p1, p2, p3 in tris:
            f.write(struct.pack("<12fH", *normal, *p1, *p2, *p3, 0))


def main():
    for name, (folder, parts) in SHAPES.items():
        tris = [t for size, center in parts for t in box_triangles(size, center)]
        out = MODELS_DIR / folder / "meshes" / f"{name}.stl"
        write_binary_stl(out, name, tris)
        print(f"{name}: {len(parts)} box(es), {len(tris)} triangles -> {out}")


if __name__ == "__main__":
    main()


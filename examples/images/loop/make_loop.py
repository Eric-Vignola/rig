"""Writes image_loop.py's demo frames beside this file: loop.0001.png ... loop.0024.png.

A ball bouncing in place for one second at 24 fps, on a transparent background, with a soft shadow
that fades as it rises: the alpha remap of image_loop has something to show (opaque, soft-edged and
transparent pixels). 128 x 128 RGBA, the same bytes on every run (no timestamp).

    python make_loop.py

Plain Python (math, struct, zlib): no numpy, no Maya.
"""
import math
import os
import struct
import zlib

FRAMES = 24
SIZE = 128
GROUND = 112.0              # the row the ball lands on
RADIUS = 17.0
RISE = 72.0                 # how high the ball goes
COLOR = (240, 124, 46)      # the ball's orange
LIGHT = (-0.45, -0.55, 0.70)
HERE = os.path.dirname(os.path.abspath(__file__))


def _png(path, width, height, rgba):
    """A PNG of 8-bit RGBA rows (bytes), with no ancillary chunk."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + rgba[y * width * 4:(y + 1) * width * 4] for y in range(height))
    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        fh.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)))
        fh.write(chunk(b"IDAT", zlib.compress(raw, 9)))
        fh.write(chunk(b"IEND", b""))


def _clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def frame(i):
    """Frame i (0-based) as RGBA bytes."""
    t = i / FRAMES                          # 0 at contact, 0.5 at the top
    height = 1.0 - (2.0 * t - 1.0) ** 2     # one bounce a loop
    if i == 0:                              # contact: squash
        sx, sy = 1.30, 0.74
    else:                                   # in the air: stretched along the motion, by its speed
        sy = 1.0 + 0.18 * (2.0 * t - 1.0) ** 2
        sx = 1.0 / sy
    rx, ry = RADIUS * sx, RADIUS * sy
    cx, cy = SIZE / 2.0, GROUND - ry - height * RISE
    srx, sry = 22.0 * (1.1 - 0.5 * height), 5.0 * (1.1 - 0.5 * height)     # the shadow
    shadow_alpha = 0.45 * (1.0 - 0.6 * height)
    norm = math.sqrt(sum(c * c for c in LIGHT))
    light = tuple(c / norm for c in LIGHT)

    out = bytearray(SIZE * SIZE * 4)
    for y in range(SIZE):
        for x in range(SIZE):
            px, py = x + 0.5, y + 0.5
            # the shadow: black, feathered over its outer 40%
            f = math.hypot((px - cx) / srx, (py - (GROUND + 2.0)) / sry)
            a_s = shadow_alpha * _clamp((1.0 - f) / 0.4)
            # the ball: coverage from the distance to its edge, in pixels
            u, v = (px - cx) / rx, (py - cy) / ry
            f = math.hypot(u, v)
            cover = _clamp((1.0 - f) * min(rx, ry) + 0.5)
            if cover > 0.0:
                if f > 1.0:
                    u, v = u / f, v / f
                n = (u, v, math.sqrt(max(0.0, 1.0 - u * u - v * v)))
                diffuse = max(0.0, n[0] * light[0] + n[1] * light[1] + n[2] * light[2])
                spec = diffuse ** 24 * 0.55
                ball = [_clamp(c / 255.0 * (0.35 + 0.65 * diffuse) + spec) for c in COLOR]
            else:
                ball = [0.0, 0.0, 0.0]
            alpha = cover + a_s * (1.0 - cover)                 # the ball over the shadow
            rgb = [b * cover / alpha if alpha > 0.0 else 0.0 for b in ball]
            k = (y * SIZE + x) * 4
            out[k:k + 4] = bytes([round(c * 255) for c in rgb] + [round(alpha * 255)])
    return bytes(out)


if __name__ == "__main__":
    for i in range(FRAMES):
        _png(os.path.join(HERE, "loop.%04d.png" % (i + 1)), SIZE, SIZE, frame(i))
    print(f"wrote {FRAMES} frames in {HERE}")

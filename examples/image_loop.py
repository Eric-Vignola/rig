"""Simple animated image-sequence plane built with ``rig``.

Creates one poly plane that:

  * Auto-detects a numbered image sequence in a directory.
  * Loops the sequence forever via Maya's file-node ``useFrameExtension``.
  * Shapes itself to the image's exact aspect ratio (largest side ==
    ``target_size``; the other side scales down).
  * Honors per-frame transparency via a ``colorCorrect`` remap, gated by
    two channel-box dials (``alpha`` + ``opacity``).

Companion to ``perspective_image_planes.py`` -- same primitives, simpler
setup. No camera math, no perspective scaling.

Showcases:
  * Math primitives: ``functions.frame``, ``%`` (modulo), ``condition``
  * Operators: ``<<`` inject, ``>>`` read, comparison-builds-network
  * Spec objects: ``Float``, ``Int``, ``lock`` for per-shape attrs
  * Bridges: ``commands as rc`` for cmds wrappers, ``nodes as rn`` for
    nodetype factories with one-line attr-init

Run from inside Maya / mayapy::

    from rig.examples import image_loop
    setup = image_loop.create_plane("/Users/me/Downloads/frames_alpha")
"""

from __future__ import annotations

import glob
import os
import re
from collections import namedtuple

from rig import condition, container, functions as rf, Plug
from rig.bridges import commands as rc, nodes as rn
from rig.spec import Float, Int, lock, String


# ---------------------------------------------------------------------------
# Sequence discovery
# ---------------------------------------------------------------------------


def _find_sequence(image_dir):
    """Scan ``image_dir`` for a numbered image sequence.

    Args:
        image_dir: directory containing image files (any common format).

    Returns:
        A 3-tuple ``(first_frame_path, start_frame, end_frame)``.

    Raises:
        ValueError: if no image files are found.
    """
    exts  = ("png", "jpg", "jpeg", "tif", "tiff", "exr", "tga")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(image_dir, "*.{}".format(ext))))
        files.extend(glob.glob(os.path.join(image_dir, "*.{}".format(ext.upper()))))
    if not files:
        raise ValueError("No image files found in {}".format(image_dir))

    # Trailing digits before the final extension: 'frame.0001.png' -> 0001
    digits_re = re.compile(r"(\d+)(?=\.[^.]+$)")
    numbered  = []
    for path in files:
        m = digits_re.search(os.path.basename(path))
        if m:
            numbered.append((int(m.group(1)), path))

    if not numbered:
        # No frame numbers in filenames -- treat as a single still.
        return sorted(files)[0], 1, 1

    numbered.sort(key=lambda t: t[0])
    return numbered[0][1], numbered[0][0], numbered[-1][0]


# ---------------------------------------------------------------------------
# Rig builder
# ---------------------------------------------------------------------------


def create_plane(image_dir, name="image_loop", target_size=10.0):
    """Create one looping image-sequence plane shaped to the image aspect.

    Args:
        image_dir:   directory containing a numbered image sequence.
        name:        base name for the plane / material / container.
        target_size: largest plane dimension in scene units.

    Returns:
        A namedtuple ``(transform, shape, material, texture)`` containing
        the plane transform Node, the plane shape Node (with per-shape
        ``sequenceStart`` / ``sequenceEnd`` / ``sequenceOffset`` /
        ``alpha`` / ``opacity`` attrs), the lambert material Node, and
        the animated file-texture Node.
    """
    # ``colorCorrect`` lives in the lookdevKit plugin; load it lazily and
    # refresh the bridges-nodes type cache so ``rn.colorCorrect()`` resolves.
    from maya import cmds as _mc

    _mc.loadPlugin("lookdevKit", quiet=True)
    rn._refresh_node_types()

    first_frame, start, end = _find_sequence(image_dir)
    Output = namedtuple("Output", "transform shape material texture")

    with container("{}_container".format(name)):
        # ---- Plane geometry --------------------------------------------------
        transform, plane = rc.polyPlane(
            sx        = 1,
            sy        = 1,
            ax        = [0, 0, 1],  # face +Z (camera-friendly)
            n         = "mesh_{}".format(name),
            container = False,      # transform stays at scene root
        )
        container.add(plane)  # but the polyPlane node is rig-owned
        shape = rc.listRelatives(transform, type="mesh")[0]

        # ---- Material + shading group ---------------------------------------
        material = rc.shadingNode("lambert", asShader=True, name=name)
        material.diffuse      << 0
        material.ambientColor << 1  # full-bright in viewport

        sg = rc.sets(
            name            = "{}SG".format(name),
            empty           = True,
            renderable      = True,
            noSurfaceShader = True,
        )
        sg.surfaceShader             << material.outColor
        Plug("defaultShaderList1.s") << material.msg
        rc.sets(shape, e=True, forceElement=sg)

        # ---- Animated file texture ------------------------------------------
        texture = rn.file()
        texture.fileTextureName   << first_frame
        texture.useFrameExtension << True
        texture.frameOffset       << lock  # we drive frameExtension directly
        material.color            << texture.outColor
        material.ambientColor     << texture.outColor
        material.diffuse          << 1

        # ---- User-facing attrs on the shape ---------------------------------
        shape << Int("sequenceStart", min=0, dv=start)
        shape << Int("sequenceEnd", min=0, dv=end)
        shape << Int("sequenceOffset", dv=0)
        shape << Float("alpha", dv=1, min=0, max=1)  # honor PNG alpha by default
        shape << Float("opacity", dv=1, min=0, max=1)
        shape << String("sequenceName") << first_frame
        texture.fileTextureName << shape.sequenceName

        # ---- Loop math -------------------------------------------------------
        #   cycle_length = end - start + 1
        #   looped       = start + ((scene_frame + offset - start) mod cycle_length)
        cycle = shape.sequenceEnd - shape.sequenceStart + 1
        looped = (
            rf.frame() + shape.sequenceOffset - shape.sequenceStart
        ) % cycle + shape.sequenceStart
        texture.frameExtension << looped

        # ---- Transparency remap (alpha * opacity gate) ----------------------
        # alpha   = how much PNG transparency to honor   (0=ignore, 1=full)
        # opacity = global multiplier on visibility      (1=opaque, 0=hidden)
        remap = rn.colorCorrect()
        remap.inColor         << texture.outTransparency
        remap.colGain         << rf.rev(rf.rev(shape.alpha) * shape.opacity)
        remap.colOffset       << rf.rev(shape.opacity)
        material.transparency << remap.outColor

        # ---- Auto-shape the plane to image aspect ---------------------------
        # Probe file node (separate from the animated one) so the plane
        # size doesn't wobble per-frame if individual frames happen to
        # have differing resolutions.
        probe = rn.file()
        probe.fileTextureName << first_frame
        w = condition(probe.outSizeX > 0, probe.outSizeX, 1)
        h = condition(probe.outSizeY > 0, probe.outSizeY, 1)

        # Largest dimension == target_size; the other shrinks proportionally.
        ratio_w = condition(w > h, 1.0, w / h)
        ratio_h = condition(h > w, 1.0, h / w)
        plane.width  << ratio_w * target_size
        plane.height << ratio_h * target_size

        return Output(transform, shape, material, texture)


if __name__ == "__main__":
    # ---- EXAMPLE WORKFLOW ----
    import time

    image_path = "/Users/ericvignola/Downloads/frames_alpha"

    t0      = time.perf_counter()
    setup   = create_plane(image_path, name="run_cycle")
    elapsed = time.perf_counter() - t0
    print(f"\n=== rig image_loop build time: {elapsed * 1000:.1f} ms ===\n")
    print(
        "Created {}  (frames {} -> {})".format(
            setup.transform,
            setup.shape.sequenceStart.get(),
            setup.shape.sequenceEnd.get(),
        )
    )
"""Port of Eric Vignola's BSD-3 ``perspective_image_planes.py`` example
from ``third_party.rig`` to ``rig``.

Builds N camera-aligned image planes shaped to each image's exact aspect
ratio, scaled by distance to the camera via the camera's angle of view.
Supports static, sequence, looping, and ping-pong animation modes per
plane.

Showcases:
  * Math primitives: ``functions``, ``trigonometry``
  * Operators: ``<<`` inject, condition node via ``==`` / ``>``
  * Spec objects: ``Float``, ``Int``, ``Enum``, ``String`` for per-shape attrs
  * Bridges: ``commands as rc`` for cmds wrappers, ``nodes as rn`` for
    nodetype factories with one-line attr-init

Run from inside Maya / mayapy::

    from rig.examples import perspective_image_planes
    setup = perspective_image_planes.create_setup("camera1", 5)

Original BSD-3 license: Copyright (c) 2023, Eric Vignola.
"""

from __future__ import annotations

from collections import namedtuple

from rig import (
    condition,
    constant,
    container,
    dist,
    functions as rf,
    PlugList,
    trigonometry as trig,
)
from rig.bridges import commands as rc, nodes as rn
from rig.shade import Lambert
from rig.spec import Enum, Float, hide, Int, lock, String


def get_scale(aperture, focal_length, distance):
    """Compute the scale of an image plane relative to the camera, given
    its aperture (in inches), focal length (in mm), and distance to camera.

    Uses ``trig.atand(aperture / (2 * focal_length))`` for the angle of
    view, then trig identities to get plane size at the given distance. The
    degree-variant ``atand`` + ``sind`` feed the native atan/sin nodes
    angle->angle, avoiding the degrees<->radians conversion nodes that the
    radian variants ``atan`` + ``sin`` would insert.
    """
    focal_length = focal_length * 0.0393700787                # mm -> inches
    aov          = trig.atand(aperture / (2 * focal_length))  # angle of view (deg)

    x = trig.sind(aov)
    y = (1 - (x**2)) ** 0.5
    return x * 2 * distance / y


def create_setup(
    camera, count, name="STICKER_LAYER", default=None, parent=None, offset=10
):
    """Create a camera + N image planes setup.

    Args:
        camera: name of the camera transform to create.
        count: number of image planes to spawn.
        name: shared base name for the planes.
        default: default ``image`` value (for use as the file path).
        parent: optional parent transform for the planes (defaults to camera).
        offset: spacing along Z between consecutive planes.

    Returns:
        A namedtuple ``(camera, planes, shapes)`` containing the camera
        transform Node, a PlugList of plane transforms, and a PlugList of
        plane shapes (with per-shape ``image`` / ``sequenceType`` /
        ``sequenceStart`` / ``sequenceEnd`` / ``sequenceOffset`` /
        ``alpha`` / ``opacity`` attrs).
    """
    # ``colorCorrect`` lives in the lookdevKit plugin; load it lazily and
    # refresh the bridges-nodes type cache so ``rn.colorCorrect()`` resolves.
    from maya import cmds as _mc

    _mc.loadPlugin("lookdevKit", quiet=True)
    rn._refresh_node_types()

    output = namedtuple("output", "camera planes shapes")

    # Camera + camera shape.
    camera_xform, camera_shape = rc.camera(name=camera, container=False)
    output.camera = camera_xform
    output.planes = PlugList()
    output.shapes = PlugList()

    camera_shape.filmFit                << 3      # overscan
    camera_shape.overscan               << 1.1    # always-visible box
    camera_shape.displayResolution      << False  # hide resolution gate
    camera_shape.displayFilmGate        << True   # show film gate
    camera_shape.displayGateMask        << True   # show film mask
    camera_shape.displayGateMaskOpacity << 1      # opaque

    # Accumulators driving the camera's aspect ratio.
    horizontal_list = PlugList()
    vertical_list   = PlugList()

    with container("{}_container".format(name.lower())):
        # Add the camera shape into the container scope.
        container.add(camera_shape)

        for i in range(count):
            name_suffix = "{}_{}".format(name, i)

            # Poly plane (mesh transform + plane node).
            transform, plane = rc.polyPlane(
                sx        = 1,
                sy        = 1,
                ax        = [0, 0, 1],
                n         = "mesh_{}".format(name_suffix.upper()),
                container = False,
            )

            # Plane node joins the container; transform reparents under camera.
            container.add(plane)

            if parent:
                rc.parent(transform, parent)
            else:
                rc.parent(transform, camera_xform)

            # Offset the plane along -Z relative to the camera.
            transform.tz << (i + 1) * -offset

            # Lock + hide all but tz.
            transform.tx << lock << hide
            transform.ty << lock << hide
            transform.s  << lock << hide
            transform.r  << lock << hide
            transform.v  << lock << hide

            # Shading network for the plane. unique=True keeps today's
            # behaviour: a rebuild gets its OWN material.
            shape = rc.listRelatives(transform, type="mesh")[0]
            m     = Lambert(name_suffix.lower(), unique=True, diffuse=0, ambientColor=1)
            shape << m
            material = m.node

            # File texture for color.
            texture = rn.file()
            material.color        << texture.outColor
            material.ambientColor << texture.outColor
            material.diffuse      << 1

            # Per-shape exposed attrs (visible via select-then-channel-box).
            shape << String("image") << default
            shape << Enum("sequenceType", en="Static:Sequence:Looping:Ping Pong")
            shape << Int("sequenceStart", min=0)
            shape << Int("sequenceEnd", min=0, dv=100)
            shape << Int("sequenceOffset")

            # Animated background math.
            current  = rf.frame() - shape.sequenceStart + shape.sequenceOffset
            duration = shape.sequenceEnd - shape.sequenceStart
            static   = rf.clamp(current, shape.sequenceStart, shape.sequenceEnd)
            looping  = current % (duration + 1)

            # Ping-pong.
            length = 2 * duration
            state  = rf.abs(current - shape.sequenceStart) % length
            pingpong = condition(
                state > duration, length - state, state + shape.sequenceStart
            )

            texture.fileTextureName   << shape.image
            texture.useFrameExtension << (shape.sequenceType > 0)
            texture.frameOffset       << lock  # don't use this!
            texture.frameExtension << rf.choice(
                [constant(0), static, looping, pingpong],
                selector=shape.sequenceType,
            )

            # Remap transparency.
            remap = rn.colorCorrect()
            remap.inColor         << texture.outTransparency
            shape                 << Float("alpha", dv=0, min=0, max=1)
            shape                 << Float("opacity", dv=1, min=0, max=1)
            remap.colGain         << rf.rev(rf.rev(shape.alpha) * shape.opacity)
            remap.colOffset       << rf.rev(shape.opacity)
            material.transparency << remap.outColor

            # Math node setup driving plane scale.
            # Proxy file node to retrieve image size (otherwise circular).
            texture_ = rn.file()
            texture_.fileTextureName << shape.image

            distance   = dist(transform.wm, camera_xform.wm)
            width      = condition(texture_.outSizeX > 0, texture_.outSizeX, 1)
            height     = condition(texture_.outSizeY > 0, texture_.outSizeY, 1)
            horizontal = condition(height > width,        width / height,    1)
            vertical   = condition(width > height,        height / width,    1)

            X = get_scale(horizontal, camera_shape.fl, distance)
            Y = get_scale(vertical, camera_shape.fl, distance)

            # Orthographic camera special case (use ortho width directly).
            X  = condition(camera_shape.orthographic, (X / Y) * camera_shape.ow, X)
            Y  = condition(camera_shape.orthographic, camera_shape.ow,           Y)

            X_ = condition(camera_shape.orthographic, camera_shape.ow,           X)
            Y_ = condition(camera_shape.orthographic, (Y / X) * camera_shape.ow, Y)

            plane.width  << condition(X < Y, X, X_)
            plane.height << condition(X < Y, Y, Y_)

            # Aspect-ratio drive values for the camera.
            h = condition(vertical < horizontal, 1, X / Y)
            v = condition(horizontal < vertical, 1, Y / X)

            horizontal_list.append(h)
            vertical_list.append(v)

            # Bundle outputs.
            output.planes.append(transform)
            output.shapes.append(shape)

        # Plug the system back into the camera's aspect ratio.
        if count > 1:
            camera_shape.hfa << rf.max(horizontal_list)
            camera_shape.vfa << rf.max(vertical_list)
        else:
            camera_shape.hfa << horizontal_list
            camera_shape.vfa << vertical_list

        return output


if __name__ == "__main__":
    # ---- EXAMPLE WORKFLOW ----
    import glob
    import os
    import time

    t0      = time.perf_counter()
    setup   = create_setup("camera1", 5)
    elapsed = time.perf_counter() - t0
    print(
        f"\n=== rig perspective_image_planes build time: {elapsed * 1000:.1f} ms ===\n"
    )

    # Populate planes with texture files (if available locally).
    image_path = "/Users/ericvignola/Downloads/rig"
    images     = glob.glob(os.path.expandvars(rf"{image_path}/_examples/images/*.jpg"))
    if images:
        setup.shapes.image[:] << images
        print(f"Assigned {len(images)} images to {len(setup.shapes)} planes.")
    else:
        print("No images found at: {}/_examples/images/*.jpg".format(image_path))
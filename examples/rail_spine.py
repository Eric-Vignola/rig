"""Port of Eric Vignola's BSD-3 ``rail_spine.py`` example from
``third_party.rig`` to ``rig``.

Builds a spine rig that lays N joints on a curve, twisted and scaled
according to per-control modes along the curve. Showcases the FULL
rig API surface:

  * Math primitives: ``functions``, ``matrix``, ``interpolate``
  * Operators: ``<<`` inject, compound fan-out, ``cv[:]`` slice
  * Spec objects: ``Float``, ``Enum`` for parameterized rig knobs
  * Bridges: ``commands as rc`` for cmds wrappers, ``nodes as rn``
    for nodetype factories with one-line attr-init

Run from inside Maya / mayapy::

    from rig.examples import rail_spine
    rail = rail_spine.create_rail(...)

Or run the ``__main__`` block to spawn a 5-control demo rig.

Original BSD-3 license: Copyright (c) 2023, Eric Vignola.
"""

from __future__ import annotations

import numbers

from maya import cmds as mc
from rig import (
    condition,
    constant,
    container,
    dist,
    elerp,
    functions as rf,
    interpolate,
    matrix,
    Node,
    normalize,
    PlugList,
    set_options,
    slerp,
)
from rig._internal.types import _is_sequence
from rig.bridges import commands as rc, nodes as rn
from rig.spec import Enum, Float, hide, lock


def make_curve(cv, degree=3, periodic=False, name="railCurve1", k=None):
    """Build an open or closed (periodic) curve with the given CVs.

    A custom knot vector ``k`` may be supplied for advanced curve shapes.
    """
    count = len(cv)
    if periodic:
        if not k:
            k = list(range(0 - degree + 1, count + degree))
        curve = rc.curve(per=periodic, d=degree, p=cv + cv[:degree], k=k)
    else:
        if not k:
            k = [
                min(max(0, x - degree + 1), count - degree)
                for x in range(count + degree - 1)
            ]
        curve = rc.curve(per=periodic, d=degree, p=cv, k=k)

    # Rename and set up display.
    curve_name = mc.rename(str(curve), name)
    curve      = Node(curve_name)
    mc.displaySmoothness(curve_name, pointsWire=32)
    shape_name = mc.listRelatives(curve_name, type="nurbsCurve")[-1]
    return curve, Node(shape_name)


def create_rail(
    position_controls,
    u,
    orient_controls   = None,
    scale_controls    = None,
    rail_name         = "rail1",
    rider_name        = "rider1",
    degree            = 3,
    periodic          = False,
    aim_axis          = 1,
    up_axis           = 0,
    invert_aim        = False,
    invert_up         = False,
    control_up        = None,
    invert_up_control = False,
    debug             = True,
):
    """Ye olde rail rig.

    Args:
        position_controls: list of transforms whose positions become curve CVs.
        u: number of riders (int) OR a list of float positions in [0, 1].
        orient_controls: optional list of controls driving rider orientation.
        scale_controls: optional list of controls driving rider scale.
        rail_name: name of the rail transform.
        rider_name: base name for each rider joint.
        degree: curve degree (1, 3, 5, 7).
        periodic: True for a closed-loop rail.
        aim_axis / up_axis: per-rider axis enums (0=X, 1=Y, 2=Z).
        invert_aim / invert_up: flip the chosen axes.
        debug: when True, exposes useful intermediate attrs and draws
            visible cubes at each rider position.

    Returns:
        The :class:`Node` wrapping the rail transform.
    """

    def _order_controls(controls, position_controls):
        if controls is None:
            return None
        if not _is_sequence(controls):
            controls = [controls]
        if len(controls) > 1:
            pc_strs = [str(p) for p in position_controls]
            if not all([str(x) in pc_strs for x in controls]):
                raise Exception(
                    "When more than 1, controls must be members of position_controls."
                )
            ctrl_strs = [str(y) for y in controls]
            controls  = [x for x in position_controls if str(x) in ctrl_strs]
        return PlugList(controls)

    with container("railNode1"):
        if control_up is None:
            control_up = up_axis

        position_controls = PlugList(position_controls)
        orient_controls   = _order_controls(orient_controls, position_controls)
        scale_controls    = _order_controls(scale_controls, position_controls)

        # Build the spread of u values for the riders.
        if isinstance(u, numbers.Real):
            if u > 1:
                u = [x / (u - 1.0) for x in range(u)]
            elif u == 1:
                u = [0.0]
        else:
            u = list(u)
            for i, v in enumerate(u):
                if isinstance(v, str):
                    u[i] = Node(u[i])
                else:
                    u[i] = float(u[i])

        # Capture initial control positions.
        pos = [mc.xform(str(x), q=True, ws=True, t=True) for x in position_controls]

        # If periodic, roll positions so u=0 matches the first control.
        if periodic and degree > 1:
            if degree == 2:
                raise Exception("periodic rail spine does not support degree 2 curves")
            shift = {3: 2, 5: 3, 7: 4}[degree]
            pos   = pos[shift:] + pos[:shift]

        # Build the live rail and a frozen "proxy" copy used to compute
        # the original arc length (for stretch ratio).
        rail, rail_shape = make_curve(
            pos, degree=degree, periodic=periodic, name=rail_name
        )
        rail.s << hide
        rail.r << hide
        rail.t << hide

        proxy, proxy_shape = make_curve(
            pos, degree=degree, periodic=periodic, name="{}Proxy".format(rail)
        )
        proxy_shape.v << False
        # Reparent the proxy shape under the rail transform (Maya doesn't
        # support multi-shape via API, so we hack it via cmds.parent).
        mc.parent(str(proxy_shape), str(rail), r=True, s=True)
        mc.delete(str(proxy))

        # Connect controls to rail CVs via matrix shorthand.
        if not periodic or (periodic and degree == 1):
            rail_shape.cv[:] << position_controls.wm * rail.wim
        else:
            shift = {3: 1, 5: 2, 7: 3}[degree]
            cvs   = list(rail_shape.cv[:])
            cvs   = cvs[shift:] + cvs[:shift]
            cvs   = PlugList(cvs)
            cvs << position_controls.wm * rail.wim

        # If periodic, append the first control to the end so the
        # cumsum loop later wraps correctly.
        if periodic:
            position_controls.append(position_controls[0])

        # curveInfo nodes for current vs default arc length.
        ci_current = rn.curveInfo()
        ci_current.inputCurve << rail_shape.worldSpace[0]
        current_length = ci_current.arcLength

        ci_default = rn.curveInfo()
        ci_default.inputCurve << proxy_shape.worldSpace[0]
        default_length = ci_default.arcLength

        stretch_ratio  = default_length / current_length
        stretch_delta  = (current_length - default_length) / current_length

        # Debug-friendly attrs on the rail.
        rail << Float("defaultLength", k=False) << default_length
        rail << Float("currentLength", k=False) << current_length
        rail << Float("stretchRatio", k=False) << stretch_ratio
        rail << Float("stretchDelta", k=False) << stretch_delta

        # User-facing rig knobs.
        rail << Enum("aimAxis", en="X:Y:Z:", dv=aim_axis)
        rail << Enum("upAxis", en="X:Y:Z:", dv=up_axis)
        rail << Enum("invertAim")
        rail << Enum("invertUp")

        rail << Float("pivot", min=0, max=1)
        rail << Float("stretch", dv=1, min=0, max=1)
        rail << Float("Scale", dv=1)
        rail << Float("shift")

        rail << Enum("scaleProjection", en="Frozen:Infinite:Clamped")
        rail << Enum("rotateProjection", en="Frozen:Infinite:Clamped")

        # Prep orient/scale vector modulation.
        orient_vectors = None
        orient_weights = None
        scale_vectors  = None
        scale_weights  = None

        if orient_controls or scale_controls:
            # Cumulative arclength ratios between controls.
            deltas = dist(position_controls.wm[:-1], position_controls.wm[1:])
            if len(position_controls) > 2:
                cumsum = rf.cumsum(deltas)
                cumsum = cumsum / cumsum[-1]
                cumsum = [constant(0)] + list(cumsum)
            else:
                cumsum = [constant(0), constant(1)]

            if debug:
                position_controls << Float("u") << cumsum << lock

            # Orient driver setup.
            if orient_controls is not None:
                orient_controls = PlugList(orient_controls)
                orient_controls << Enum("upAxis", en="X:Y:Z:", dv=up_axis)
                orient_controls << Enum("invertUp")

                # Pick the up vector via a choice node (per-control axis enum).
                orient_vectors = rf.choice(
                    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    selector=orient_controls.upAxis,
                )
                orient_vectors = orient_vectors * condition(
                    orient_controls.invertUp, -1, 1
                )
                orient_vectors = normalize(
                    matrix.multiply(orient_vectors, orient_controls.wm, local=True)
                )

                if len(orient_controls) > 1:
                    orient_weights = PlugList(
                        [cumsum[position_controls.index(x)] for x in orient_controls]
                    )

                    if periodic:
                        if position_controls.index(orient_controls[0]) == 0:
                            orient_weights.append(constant(1))
                            orient_vectors.append(orient_vectors[0])
                        else:
                            orient_weights = PlugList(
                                [0 - (1 - orient_weights[-1])] + list(orient_weights)
                            )
                            orient_vectors = PlugList(
                                [orient_vectors[-1]] + list(orient_vectors)
                            )
                            orient_weights.append(1 - (0 - orient_weights[1]))
                            orient_vectors.append(orient_vectors[1])

                    # Frozen state: snap weights to their default values.
                    orient_weights = condition(
                        rail.rotateProjection == 0,
                        orient_weights.get(),
                        orient_weights,
                    )

            # Scale driver setup.
            if scale_controls is not None:
                scale_controls = PlugList(scale_controls)
                scale_vectors  = matrix.decompose(scale_controls.wm).outputScale

                if len(scale_controls) > 1:
                    scale_weights = PlugList(
                        [cumsum[position_controls.index(x)] for x in scale_controls]
                    )

                    if periodic:
                        if position_controls.index(scale_controls[0]) == 0:
                            scale_weights.append(constant(1))
                            scale_vectors.append(scale_vectors[0])
                        else:
                            scale_weights = PlugList(
                                [0 - (1 - scale_weights[-1])] + list(scale_weights)
                            )
                            scale_vectors = PlugList(
                                [scale_vectors[-1]] + list(scale_vectors)
                            )
                            scale_weights.append(1 - (0 - scale_weights[1]))
                            scale_vectors.append(scale_vectors[1])

                    scale_weights = condition(
                        rail.scaleProjection == 0,
                        scale_weights.get(),
                        scale_weights,
                    )

        # Tangent extrapolation at curve ends (open rails only).
        if not periodic:
            rail << Enum("translateProjection", en="Clamped:Infinite", dv=1)
            rail << Float("uTangentStart", dv=0.001, min=0, max=1) << hide
            rail << Float("uTangentEnd", dv=0.999, min=0, max=1) << hide

            tangent_aim = rf.choice(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                selector=rail.aimAxis,
            )

            tangent0 = rn.motionPath()
            tangent0.fractionMode << 1
            tangent0.uValue       << rail.uTangentStart
            tangent0.geometryPath << rail_shape.worldSpace[0]
            tangent0.frontAxis    << rail.aimAxis
            tangent0.upAxis       << rail.upAxis
            tangent0.inverseFront << rail.invertAim
            tangent0.inverseUp    << rail.invertUp
            edge0 = tangent0.allCoordinates
            tangent0 = (
                normalize(tangent_aim * tangent0.orientMatrix)
                * current_length
                * rail.translateProjection
            )

            tangent1 = rn.motionPath()
            tangent1.fractionMode << 1
            tangent1.uValue       << rail.uTangentEnd
            tangent1.geometryPath << rail_shape.worldSpace[0]
            tangent1.frontAxis    << rail.aimAxis
            tangent1.upAxis       << rail.upAxis
            tangent1.inverseFront << rail.invertAim
            tangent1.inverseUp    << rail.invertUp
            edge1 = tangent1.allCoordinates
            tangent1 = (
                normalize(tangent_aim * tangent1.orientMatrix)
                * current_length
                * rail.translateProjection
            )

        # ---- ADD RAIL RIDERS ---- #

        for u_default in u:
            # Joint sits OUTSIDE the container so its parent-rename doesn't
            # confuse container membership tracking. ``rc.parent`` keeps
            # the Node reference up-to-date when Maya renames on parent.
            rider = rn.joint(name=rider_name, container=False)
            rc.parent(rider, rail)

            if debug:
                debug_cube = rc.polyCube()[0]
                rc.parent(debug_cube, rider)

            rider << Float("uDefault") << u_default
            u_default  = rider.uDefault

            rider_path = rn.motionPath()
            rider_path.fractionMode << 1
            rider_path.geometryPath << rail_shape.worldSpace[0]
            rider_path.frontAxis    << rail.aimAxis
            rider_path.upAxis       << rail.upAxis
            rider_path.inverseFront << rail.invertAim
            rider_path.inverseUp    << rail.invertUp

            # Modulate the u translation: stretch + scale + shift + pivot.
            u_translate = (
                u_default * stretch_ratio + stretch_delta * rail.pivot
            ) * rf.rev(rail.stretch)
            u_translate = u_translate + u_default * rail.stretch
            u_translate = (
                rail.shift + rail.pivot + (u_translate - rail.pivot) * rail.Scale
            )

            if periodic:
                u_translate = u_translate % 1

            rider_path.uValue << u_translate

            # Final rider transform via composeMatrix.
            rider_matrix = rn.composeMatrix()

            # Plug scale.
            if scale_controls:
                if len(scale_controls) == 1:
                    rider_matrix.inputScale << scale_vectors
                else:
                    frozen = condition(
                        rail.scaleProjection == 0, u_default, u_translate
                    )
                    clamped = condition(
                        rail.scaleProjection == 2,
                        rf.clamp(frozen, scale_weights[0], scale_weights[-1]),
                        frozen,
                    )
                    rider_matrix.inputScale << interpolate.sequence(
                        clamped,
                        scale_weights,
                        scale_vectors,
                        method=elerp,
                    )

            # Plug rotation.
            if orient_controls:
                rider_matrix.inputRotate << rider_path.rotate
                if len(orient_controls) == 1:
                    rider_path.worldUpVector << orient_vectors
                else:
                    frozen = condition(
                        rail.rotateProjection == 0, u_default, u_translate
                    )
                    clamped = condition(
                        rail.rotateProjection == 2,
                        rf.clamp(frozen, orient_weights[0], orient_weights[-1]),
                        frozen,
                    )
                    rider_path.worldUpVector << interpolate.sequence(
                        clamped,
                        orient_weights,
                        orient_vectors,
                        method=slerp,
                    )

            # Plug translation (with end-tangent extrapolation for open rails).
            local_position = rider_path.allCoordinates
            if not periodic:
                local_position = condition(
                    u_translate < 0,
                    (tangent0 * u_translate) + edge0,
                    rider_path.allCoordinates,
                )
                local_position = condition(
                    u_translate > 1,
                    (tangent1 * (u_translate - 1)) + edge1,
                    local_position,
                )

            rider_matrix.inputTranslate << local_position
            rider_matrix_out = rider_matrix.outputMatrix * rail.wim
            rider << rider_matrix_out  # SRT+shear injection

    return rail


if __name__ == "__main__":
    # ---- EXAMPLE WORKFLOW ----
    import time

    # Toggle this to False to see the un-grouped node graph.
    set_options(create_containers=True)

    # 5 control locators stacked along Y.
    position_controls = PlugList()
    for i in range(5):
        loc_xform = mc.spaceLocator()[0]
        position_controls.append(Node(loc_xform))
        position_controls[i].ty << i * 5

    # 20 evenly-spaced riders.
    count = 20
    u     = [(x / (count - 1)) for x in range(count)]

    # Cubic open rail.
    degree   = 3
    periodic = False
    debug    = True

    t0 = time.perf_counter()
    rail = create_rail(
        position_controls,
        u,
        scale_controls  = position_controls,
        orient_controls = position_controls,
        degree          = degree,
        periodic        = periodic,
        aim_axis        = 1,
        up_axis         = 0,
        debug           = debug,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n=== rig rail_spine build time: {elapsed * 1000:.1f} ms ===\n")

    # Demonstrate the rig in action.
    rail.stretch         << 0
    rail.scaleProjection << 2

    position_controls[1].tx << 10
    position_controls[1].s  << [5, 0.1, 5]
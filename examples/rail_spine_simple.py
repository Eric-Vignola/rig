"""Simplified rail-spine tutorial for ``rig``.

A stripped-down sibling of :mod:`rail_spine` that walks through the
DSL one feature at a time. Around 100 lines of rig code, designed to
be read top-to-bottom in one sitting.

What this version does
----------------------

  * Build a cubic NURBS curve from N control transforms.
  * Drive the curve's CVs from the controls (live).
  * Lay M evenly-spaced rider joints along the curve via ``motionPath``.
  * Expose four interactive knobs on the rail:
      - ``pivot``   -- u value that stays put for stretch & Scale.
      - ``stretch`` -- 0 = riders glide with the curve, 1 = locked.
      - ``Scale``   -- spread (>1) or squeeze (<1) riders around pivot.
      - ``shift``   -- slide all riders along the curve.
  * Slerp each rider's up vector between the FIRST and LAST control's
    up axis -- its share of one quaternion turn from the first to the
    last -- frozen on the rider's default u (so the up axis doesn't
    drift as the spine deforms).
  * Infinite end-tangent projection -- riders whose computed u falls
    outside [0, 1] extrapolate along the curve's end tangents instead
    of piling up at the endpoints.

What :mod:`rail_spine` adds (and this version intentionally OMITS)
------------------------------------------------------------------

  * Periodic / closed-loop rails
  * Variable curve degree
  * Per-control scale / orient projection modes
    (Frozen / Infinite / Clamped)
  * Interactive aim/up-axis enums, debug-arc-length attrs
  * Clamped vs Infinite projection enum (this tutorial bakes Infinite)

DSL features showcased here
---------------------------

  * ``with container("..."):``                -- scoped utility graph
  * ``List(controls).wm * rail.wim``          -- broadcast matrix mul
  * ``rail_shape.cv[:] << matrices``          -- matrix -> CV shorthand
  * ``rn.curveInfo / pointOnCurveInfo / motionPath`` -- node-type bridges
  * ``Float("pivot", min=0, max=1)``          -- spec-object injection
  * ``plug.get()``                            -- read-value idiom
  * ``matrix.axis(wm, up_axis)``              -- extract basis vector
  * ``quaternion.from_axis_angle / pow``      -- a turn, and a share of it
  * ``matrix.compose(rotate=quat)``           -- quaternion -> rotation matrix
  * ``matrix.multiply(vec, m, local=True)``   -- vec x matrix (rotation only)
  * ``normalize / functions.condition``       -- dispatch verb + ternary
  * ``lerp``                                  -- top-level interpolation verb
  * ``rider << matrix_plug``                  -- matrix -> t/r/s decompose

Run from inside Maya / mayapy::

    from rig.examples import rail_spine_simple
    rail = rail_spine_simple.create_simple_rail(controls, riders=10)

Or run the ``__main__`` block to spawn a 4-control / 12-rider demo.
"""

from __future__ import annotations

from maya import cmds as mc
from rig import (
    container,
    functions as rf,
    lerp,
    List,
    matrix,
    Node,
    normalize,
    quaternion,
)
from rig.bridges import commands as rc, nodes as rn
from rig.spec import Float, hide


def create_simple_rail(
    position_controls,
    riders     = 10,
    rail_name  = "rail1",
    rider_name = "rider1",
    aim_axis   = 1,  # 0=X, 1=Y, 2=Z -- baked into motionPath at build time
    up_axis    = 0,  # 0=X, 1=Y, 2=Z -- baked into motionPath + up-vector blend
    debug      = True,
):
    """Build a minimal rail-spine rig.

    Parameters
    ----------
    position_controls
        Iterable of transforms (Node / str) whose world positions
        become the curve's CVs.
    riders
        Number of evenly-spaced rider joints.
    rail_name
        Name for the rail transform.
    rider_name
        Base name for the rider joints.
    aim_axis, up_axis
        ``motionPath`` axis indices (0 = X, 1 = Y, 2 = Z). Baked into
        the rig at build time -- not user-editable.
    debug
        When True, parents a polyCube under each rider for visibility.

    Returns
    -------
    Node
        The rail transform.
    """
    position_controls = List(position_controls)

    with container("simpleRail1"):
        # ---- 1. Build the curve from the current control positions. ----
        cv_positions = [
            mc.xform(str(c), q=True, ws=True, t=True) for c in position_controls
        ]
        rail_str   = mc.rename(str(rc.curve(d=3, p=cv_positions)), rail_name)
        rail       = Node(rail_str)
        rail_shape = Node(mc.listRelatives(rail_str, type="nurbsCurve")[-1])

        # The rail transform is just a parent -- hide its channels.
        rail.t << hide
        rail.r << hide
        rail.s << hide

        # ---- 2. Drive every CV from the controls' world matrices. ----
        # ``controls.wm`` is a List of worldMatrix plugs. Multiplying
        # by ``rail.wim`` demotes them into rail-local space. Injecting a
        # matrix into a CV (vector destination) auto-extracts translation
        # via the matrix-shorthand -- no manual decomposeMatrix nodes.
        rail_shape.cv[:] << position_controls.wm * rail.wim

        # ---- 3. Stretch math driven by current vs default arc length. ----
        ci = rn.curveInfo()
        ci.inputCurve << rail_shape.worldSpace[0]
        current_length = ci.arcLength

        # Snapshot the build-time arclength as a Python constant.
        # (``plug.get()`` is the rig idiom for ``cmds.getAttr``.)
        # For a "live" default that tracks rig-edit-time changes, see
        # rail_spine.py's proxy-curve trick.
        default_length = current_length.get()
        stretch_ratio  = default_length / current_length  # builds a Plug

        # User-facing knobs.
        rail << Float("pivot", min=0, max=1, dv=0)    # u-anchor for stretch & Scale
        rail << Float("stretch", min=0, max=1, dv=0)  # 0=glide, 1=locked
        rail << Float("Scale", dv=1)                  # spread (>1) or squeeze (<1) around pivot
        rail << Float("shift", dv=0)                  # slide all riders along the curve

        # ---- 4. Frozen up-vector endpoints (first and last controls). ----
        # ``matrix.axis(wm, up_axis)`` extracts the rotation-only basis
        # vector for the chosen axis (``vectorProduct`` op=3 pre-2024,
        # ``axisFromMatrix`` on 2024+). ``normalize`` normalises in case
        # controls have non-unit scale.
        first_up = normalize(matrix.axis(position_controls[0].wm, up_axis))
        last_up  = normalize(matrix.axis(position_controls[-1].wm, up_axis))

        # The turn from ``first_up`` to ``last_up`` as one quaternion: an
        # ``angleBetween`` gives the axis and the angle between them (the
        # short way, 0 to 180 degrees), ``quaternion.from_axis_angle``
        # makes the quaternion. Each rider takes its share in step 7.
        between = rn.angleBetween()
        between.vector1 << first_up
        between.vector2 << last_up
        turn = quaternion.from_axis_angle(between.axis, between.angle)

        # ---- 5. End-tangent setup (infinite projection). ----
        # A ``pointOnCurveInfo`` on each end of the curve: ``position`` is
        # the end point, ``normalizedTangent`` the unit tangent there.
        # ``turnOnPercentage`` reads ``parameter`` as 0..1 of the curve's
        # parameter range, so 0 and 1 are its two ends. Scale by
        # ``current_length`` so one unit of u-overshoot ~= one
        # curve-length of travel.
        curve_start = rn.pointOnCurveInfo()
        curve_start.turnOnPercentage << 1
        curve_start.parameter        << 0
        curve_start.inputCurve       << rail_shape.worldSpace[0]
        edge_start    = curve_start.position
        tangent_start = curve_start.normalizedTangent * current_length

        curve_end = rn.pointOnCurveInfo()
        curve_end.turnOnPercentage << 1
        curve_end.parameter        << 1
        curve_end.inputCurve       << rail_shape.worldSpace[0]
        edge_end    = curve_end.position
        tangent_end = curve_end.normalizedTangent * current_length

        # ---- 6. Build evenly-spaced riders. ----
        for i in range(riders):
            u_default = i / max(riders - 1, 1)

            # ``container=False`` keeps the joint OUT of the container
            # -- Maya renames a node when it's re-parented, which would
            # otherwise confuse container membership tracking.
            rider = rn.joint(name=rider_name, container=False)
            rc.parent(rider, rail)

            if debug:
                rc.parent(rc.polyCube()[0], rider)

            # Per-rider default-u attribute (handy for debugging and
            # also the slerp weight for the up-vector blend below).
            rider << Float("uDefault", k=False) << u_default

            # ``anchored`` is the glide-mode u: as the curve stretches,
            # the rider AT pivot stays put; others scale proportionally.
            # ``stretch`` then lerps between glide and locked-to-default.
            anchored  = rail.pivot + (rider.uDefault - rail.pivot) * stretch_ratio
            stretched = lerp(anchored, rider.uDefault, rail.stretch)
            # ``Scale`` spreads/squeezes around pivot, ``shift`` offsets all.
            u_value = rail.shift + rail.pivot + (stretched - rail.pivot) * rail.Scale

            # ---- 7. Sample the curve at u_value via motionPath. ----
            mp = rn.motionPath()
            mp.fractionMode << 1  # treat u as a 0..1 fraction
            mp.geometryPath << rail_shape.worldSpace[0]
            mp.frontAxis    << aim_axis
            mp.upAxis       << up_axis

            # Frozen up-vector blend -- slerps between the spine's start
            # and end up axes by the rider's DEFAULT u (not u_value), so
            # the rider's twist stays put as the spine stretches.
            # ``quaternion.pow(turn, w)`` is one ``quatSlerp`` from identity:
            # the rider's share of the turn, which carries ``first_up``
            # along the great circle to ``last_up`` as w goes 0 -> 1.
            twist = matrix.compose(rotate=quaternion.pow(turn, rider.uDefault))
            mp.worldUpType   << 3  # 3 = Vector (use the connected worldUpVector)
            mp.worldUpVector << matrix.multiply(first_up, twist, local=True)
            mp.uValue        << u_value

            # ---- 8. Compose translate+rotate, demote to local, inject. ----
            # Translation: motionPath sample inside [0, 1]; linear
            # extrapolation along the end tangents outside that range.
            local_position = mp.allCoordinates
            local_position = rf.condition(
                u_value < 0,
                tangent_start * u_value + edge_start,
                local_position,
            )
            local_position = rf.condition(
                u_value > 1,
                tangent_end * (u_value - 1) + edge_end,
                local_position,
            )

            rider_matrix = rn.composeMatrix()
            rider_matrix.inputTranslate << local_position
            rider_matrix.inputRotate    << mp.rotate

            # Matrix -> Node injection auto-decomposes into t/r/s (+shear).
            rider << rider_matrix.outputMatrix * rail.wim

    return rail


if __name__ == "__main__":
    # ---- Demo: 4 locators stacked along Y, 12 evenly-spaced riders. ----
    controls = List()
    for i in range(4):
        loc = mc.spaceLocator()[0]
        controls.append(Node(loc))
        controls[i].ty << i * 5

    rail = create_simple_rail(controls, riders=12)

    # Try it:
    #   * Move controls[1] in X -- riders glide along the curve.
    #   * Set rail.pivot to 0.5 and rail.stretch to 1 -- riders lock
    #     symmetrically around the middle of the curve.
    #   * Set rail.Scale to 2 (or 0.5) -- riders spread out / squeeze
    #     in around pivot.
    #   * Set rail.shift to 0.5 -- the whole spine slides halfway
    #     along the curve.
    #   * Rotate controls[0] or controls[-1] -- rider twist slerps
    #     smoothly between the two endpoints.
    #   * Push shift past 1 (or scale way up) -- riders past u=0 or
    #     u=1 extrapolate along the end tangents (infinite
    #     projection) rather than piling up at the endpoints.
"""Limb twist helper for ``rig``.

N twist nodes between two matrices, a start and an end (an upper arm and
its elbow, say), that share the limb's twist: every node aims down the
limb and turns by its own share of the twist between the two ends.

What this does
--------------

  * Pick a pole on each end: one of its axes (X, Y or Z), flipped or not.
    The pick is live: ``startAxis`` / ``startInvert`` and ``endAxis`` /
    ``endInvert`` on the root.
  * Build each end's frame once with ``matrix.aim`` (an ``aimMatrix``):
    aim down the limb, from start to end, up toward that end's pole.
  * Spread ``count`` twist nodes from the start to the end: transforms by
    default, or locators, joints (to skin to) or flat cubes (to see the
    twist). Each carries a ``u``, 0 on the start and 1 on the end: give
    them with ``weights=``, or change them in the channel box later; by
    default they are even.
  * Each node slerps the two end frames by its ``u`` and sits at
    ``lerp(start, end, u)`` on the line between them.

The twist between the two ends is taken the short way: past 180 degrees
it flips, as with any slerp.

DSL features showcased here
---------------------------

  * ``with container("..."):``                             -- scoped utility graph
  * ``root << Enum("startAxis", en="X:Y:Z:")``             -- spec-object injection
  * ``rf.choice(AXES, selector=...) * rf.condition(...)``  -- a live axis pick and flip
  * ``matrix.multiply(vec, m, local=True)``                -- vec x matrix (rotation only)
  * ``matrix.translation`` / ``matrix.aim``                -- a point and an aim frame from matrices
  * ``to_quaternion`` / ``slerp`` / ``lerp``               -- dispatch verbs: a quatSlerp, a vector lerp
  * ``matrix.compose(rotate=quat, translate=point)``       -- a matrix from a quaternion and a point
  * ``twist << matrix_plug * root.wim``                    -- matrix -> t/r/s decompose

Run from inside Maya / mayapy::

    from rig import Node
    from rig.examples import limb_twist
    twist = limb_twist.create_limb_twist(Node("upperarm").wm, Node("lowerarm").wm, count=5, node_type="joint")

Or run the ``__main__`` block to spawn an upper arm, a lower arm and five
twist cubes between them.
"""

from __future__ import annotations

import collections

import numpy as np
from rig import (
    container,
    functions as rf,
    lerp,
    List,
    matrix,
    normalize,
    slerp,
    to_quaternion,
)
from rig.bridges import commands as rc, nodes as rn
from rig.spec import Enum, Float

AXES       = ([1, 0, 0], [0, 1, 0], [0, 0, 1])  # a local axis by its index: 0 = X, 1 = Y, 2 = Z
NODE_TYPES = ("transform", "locator", "joint", "cube")

LimbTwist  = collections.namedtuple("LimbTwist", "root twists")


def create_limb_twist(
    start_matrix,
    end_matrix,
    count        = 5,
    node_type    = "transform",
    name         = "limbTwist1",
    twist_name   = "twist1",
    start_axis   = 1,      # 0=X, 1=Y, 2=Z -- the start's pole, live as root.startAxis
    start_invert = False,  # live as root.startInvert
    end_axis     = 1,      # the end's pole, live as root.endAxis
    end_invert   = False,  # live as root.endInvert
    aim_axis     = 0,      # the twist nodes' axis down the limb -- baked at build time
    up_axis      = 1,      # the twist nodes' axis toward the poles -- baked at build time
    weights      = None,
):
    """Build N twist nodes between two matrices.

    Parameters
    ----------
    start_matrix, end_matrix
        Matrix plugs, the two ends of the limb (``upperarm.wm``,
        ``lowerarm.wm``). Only their translation and rotation are read.
    count
        Number of twist nodes, at least one.
    node_type
        What each twist node is: ``"transform"`` (the default),
        ``"locator"``, ``"joint"`` (to skin to) or ``"cube"``, a flat cube
        standing along the node's up axis, so the twist shows.
    name
        Name for the root transform: the twist nodes' parent and the home
        of the four pole knobs.
    twist_name
        Base name for the twist nodes (``twist1``, ``twist2`` ...).
    start_axis, end_axis
        Each end's pole: the index of one of its axes (0 = X, 1 = Y,
        2 = Z). They are the defaults of the live ``startAxis`` and
        ``endAxis`` enums on the root.
    start_invert, end_invert
        Flip that axis. The defaults of the live ``startInvert`` and
        ``endInvert`` toggles.
    aim_axis, up_axis
        The twist nodes' own axes: the one down the limb, and the one that
        follows the poles. Baked at build time, and they must differ.
    weights
        One ``u`` per twist node, each from 0 (on the start) to 1 (on the
        end). None spreads the nodes evenly from start to end.

    Returns
    -------
    LimbTwist
        A namedtuple: ``root`` (the root transform) and ``twists`` (a
        ``List`` of the twist nodes).

    Raises
    ------
    ValueError
        Before anything is built: a count below one, a ``node_type`` that
        is not one of the four, weights that are not ``count`` numbers from
        0 to 1, an axis that is not 0, 1 or 2, or the same ``aim_axis`` and
        ``up_axis``.
    """
    if count < 1:
        raise ValueError(f"create_limb_twist: count is {count}; it takes at least one twist node")

    if node_type not in NODE_TYPES:
        raise ValueError(f"create_limb_twist: node_type is {node_type!r}; it takes one of {', '.join(NODE_TYPES)}")

    for label, axis in (("start_axis", start_axis), ("end_axis", end_axis), ("aim_axis", aim_axis),
                        ("up_axis", up_axis)):
        if axis not in (0, 1, 2):
            raise ValueError(f"create_limb_twist: {label} is {axis!r}; it takes 0 (X), 1 (Y) or 2 (Z)")

    if aim_axis == up_axis:
        raise ValueError(f"create_limb_twist: aim_axis and up_axis are both {aim_axis}; they must differ")

    if weights is None:
        weights = [i / max(count - 1, 1) for i in range(count)]

    weights = [float(weight) for weight in weights]
    if len(weights) != count or not all(0 <= weight <= 1 for weight in weights):
        raise ValueError(f"create_limb_twist: weights must be {count} numbers from 0 to 1, not {weights}")

    with container(f"{name}Setup"):
        # ---- 1. The root: the twist nodes' parent, and the live pole knobs. ----
        # ``container=False`` keeps it OUT of the container, as the rails
        # do with their riders: Maya renames a node it re-parents.
        root = rn.transform(name=name, container=False)
        root << Enum("startAxis", en="X:Y:Z:", dv=start_axis)
        root << Enum("startInvert", dv=int(start_invert))
        root << Enum("endAxis", en="X:Y:Z:", dv=end_axis)
        root << Enum("endInvert", dv=int(end_invert))

        # ---- 2. Each end's pole, in world space. ----
        # ``rf.choice`` picks the local axis the enum names; ``rf.condition``
        # turns the toggle into -1 or 1. ``matrix.multiply(..., local=True)``
        # turns that vector through the matrix without its translation, and
        # ``normalize`` drops any scale.
        start_axis_vector = rf.choice(AXES, selector=root.startAxis) * rf.condition(root.startInvert, -1, 1)
        end_axis_vector   = rf.choice(AXES, selector=root.endAxis) * rf.condition(root.endInvert, -1, 1)
        start_pole        = normalize(matrix.multiply(start_axis_vector, start_matrix, local=True))
        end_pole          = normalize(matrix.multiply(end_axis_vector, end_matrix, local=True))

        # ---- 3. The two end frames, built once. ----
        # Both aim down the limb, from start to end; each turns its up axis
        # toward its own pole. They differ only by a roll about the limb:
        # the twist the nodes share. As quaternions, they slerp.
        start_point = matrix.translation(start_matrix)
        end_point   = matrix.translation(end_matrix)
        limb        = end_point - start_point
        start_frame = to_quaternion(matrix.aim(limb, start_pole, aim_axis=AXES[aim_axis], up_axis=AXES[up_axis]))
        end_frame   = to_quaternion(matrix.aim(limb, end_pole, aim_axis=AXES[aim_axis], up_axis=AXES[up_axis]))

        # The build-time limb length, a Python float: it sizes the cubes and locators.
        length = float(np.linalg.norm(limb.get()))

        # ---- 4. The twist nodes, each with its own u. ----
        twists = List()
        for weight in weights:
            twist = _twist_node(node_type, twist_name, length, count, aim_axis, up_axis)
            rc.parent(twist, root)

            # 0 on the start, 1 on the end: where the node sits, and how much
            # of the twist it takes. Change it at any time.
            twist << Float("u", min=0, max=1) << weight

            # ---- 5. Slerp the frames by u, lerp the position, inject. ----
            # ``slerp`` of two quaternions is one ``quatSlerp`` node; the
            # matrix it composes with the lerped point lands in root space,
            # and the injection decomposes it into the node's channels.
            rotation = slerp(start_frame, end_frame, twist.u)
            position = lerp(start_point, end_point, twist.u)
            twist << matrix.compose(rotate=rotation, translate=position) * root.wim
            twists.append(twist)

    return LimbTwist(root, twists)


def _twist_node(node_type, name, length, count, aim_axis, up_axis):
    """A new twist node, kept out of the container: a transform, a locator
    sized to the limb, a joint, or a flat cube with no construction history,
    long along the up axis and thin across, so its twist reads at a glance."""
    if node_type == "joint":
        return rn.joint(name=name, container=False)
    if node_type == "locator":
        locator = rc.spaceLocator(name=name, container=False)[0]
        locator.get_shape().localScale << [length * 0.1] * 3
        return locator
    if node_type == "cube":
        size           = [length * 0.02] * 3
        size[aim_axis] = length / (2 * count)
        size[up_axis]  = length * 0.2
        return rc.polyCube(name=name, width=size[0], height=size[1], depth=size[2], constructionHistory=False,
                           container=False)[0]
    return rn.transform(name=name, container=False)


if __name__ == "__main__":
    # ---- Demo: an upper arm, a lower arm 10 units down X, five twist cubes. ----
    upperarm = rn.joint(name="upperarm")
    lowerarm = rn.joint(name="lowerarm")
    lowerarm.t << (10, 0, 0)
    lowerarm.r << (-90, 0, 0)  # its -Y lines up with the upper arm's Z: no twist at rest

    # The upper arm's Z axis and the lower arm's -Y axis are the two poles; cubes show the twist.
    twist = create_limb_twist(upperarm.wm, lowerarm.wm, count=5, node_type="cube", start_axis=2, end_axis=1,
                              end_invert=True)

    # Try it:
    #   * Rotate lowerarm in X -- the twist spreads over the five cubes,
    #     from none on the first to all of it on the last.
    #   * Set twist2's u to 0.5 -- it slides to the middle and takes half
    #     the twist, like twist3.
    #   * Set limbTwist1.endInvert to False -- the end's +Y becomes its
    #     pole: the twist flips sign.
    #   * Set limbTwist1.endAxis to Z -- the lower arm's Z is the pole now.
    #   * Move lowerarm -- the cubes follow the line and keep aiming down it.

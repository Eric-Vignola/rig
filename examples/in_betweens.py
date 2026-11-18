"""In-betweens for ``rig``: N nodes between two matrices.

The smallest rig of rows: every in-between is ``blend(start, end, weight)``,
a ``blendMatrix`` that slerps the rotation and lerps the translation and the
scale, with its own weight; and one call makes them all, because a list of
weights is one blend per weight.

What this does
--------------

  * Make ``count`` in-between nodes: transforms by default, or locators or
    joints.
  * Give each a live ``weight``, 0 on the start and 1 on the end: from
    ``weights=``, or evenly spaced strictly between the two (three nodes:
    0.25, 0.5 and 0.75).
  * Blend the two matrices by every node's weight, one ``blendMatrix`` per
    node, and inject each result into its node.

The nodes are made at the top of the scene, as the matrices are world ones
(``start.wm``): parent them and the injection would read the matrix in the
parent's space.

DSL features showcased here
---------------------------

  * ``nodes << Float("weight", min=0, max=1) << weights``  -- one value per node
  * ``blend(start, end, nodes.weight)``                    -- a List of weights: one blendMatrix per node
  * ``blend(start, end, [0.25, 0.5, 0.75])``               -- so is a plain list
  * ``nodes << matrices``                                  -- matrix -> t/r/s decompose, node by node

Run from inside Maya / mayapy::

    from rig import Node
    from rig.examples import in_betweens
    mids = in_betweens.create_in_betweens(Node("hip").wm, Node("knee").wm, count=3)

Or run the ``__main__`` block to spawn two poles and three locators between them.
"""

from __future__ import annotations

from rig import blend, container, List
from rig.bridges import commands as rc, nodes as rn
from rig.spec import Float

NODE_TYPES = ("transform", "locator", "joint")


def create_in_betweens(
    start_matrix,
    end_matrix,
    count     = 3,
    weights   = None,
    node_type = "transform",
    name      = "inBetween1",
):
    """Build N nodes between two matrices.

    Parameters
    ----------
    start_matrix, end_matrix
        Matrix plugs, the two poles (``hip.wm``, ``knee.wm``).
    count
        Number of in-between nodes, at least one.
    weights
        One weight per node, each from 0 (on the start) to 1 (on the end).
        None spreads the nodes evenly strictly between the poles.
    node_type
        ``"transform"`` (the default), ``"locator"`` or ``"joint"``.
    name
        Base name for the nodes (``inBetween1``, ``inBetween2`` ...).

    Returns
    -------
    List
        The in-between nodes, each with its live ``weight``.

    Raises
    ------
    ValueError
        Before anything is built: a count below one, a ``node_type`` that is
        not one of the three, or weights that are not ``count`` numbers from
        0 to 1.
    """
    if count < 1:
        raise ValueError(f"create_in_betweens: count is {count}; it takes at least one node")

    if node_type not in NODE_TYPES:
        raise ValueError(f"create_in_betweens: node_type is {node_type!r}; it takes one of {', '.join(NODE_TYPES)}")

    if weights is None:
        weights = [(i + 1) / (count + 1) for i in range(count)]

    weights = [float(weight) for weight in weights]
    if len(weights) != count or not all(0 <= weight <= 1 for weight in weights):
        raise ValueError(f"create_in_betweens: weights must be {count} numbers from 0 to 1, not {weights}")

    with container(f"{name}Setup"):
        # The nodes stay OUT of the container (``container=False``), as the
        # rails' riders do: they are the rig's output, not its plumbing.
        nodes = List([_node(node_type, name) for _ in weights])

        # One weight per node, from the list: change any of them at any time.
        nodes << Float("weight", min=0, max=1) << weights

        # ``nodes.weight`` is a List, so ``blend`` makes one blendMatrix per
        # node; the List of matrices it gives goes into the nodes one by one.
        nodes << blend(start_matrix, end_matrix, nodes.weight)

    return nodes


def _node(node_type, name):
    """A new in-between node, kept out of the container"""
    if node_type == "joint":
        return rn.joint(name=name, container=False)
    if node_type == "locator":
        return rc.spaceLocator(name=name, container=False)[0]
    return rn.transform(name=name, container=False)


if __name__ == "__main__":
    # ---- Demo: two poles, three locators between them. ----
    start = rn.transform(name="start")
    end   = rn.transform(name="end")
    end.t << (10, 4, 0)
    end.r << (0, 90, 0)
    end.s << (2, 2, 2)

    mids = create_in_betweens(start.wm, end.wm, count=3, node_type="locator")

    # Try it:
    #   * Move or rotate end -- the three locators follow, a quarter, a half
    #     and three quarters of the way there.
    #   * Set inBetween2.weight to 0.9 -- it slides most of the way to end.
    #   * Scale end -- the locators scale by their share of it.

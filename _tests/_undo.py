"""Undo test helpers shared by rig's undo tests (a helper module, not a test module).

:class:`UndoWalk` is a mixin for :class:`rig._tests._base.MayaTestCase` subclasses::

    class TestSomething(UndoWalk, MayaTestCase):
        TEST_START_NEW_SCENE = True

        def test_x(self):
            before = self.scene_state(meshes=["mShape"])
            ...                                   # the edit under test
            after = self.scene_state(meshes=["mShape"])
            self.undo_steps(1)
            self.assert_state_equal(self.scene_state(meshes=["mShape"]), before)
            self.redo_steps(1)
            self.assert_state_equal(self.scene_state(meshes=["mShape"]), after)

Undo tests check the **scene**, never the return value of ``cmds.undo()`` /
``cmds.redo()``: Maya swallows an exception raised inside an ``undoIt`` /
``redoIt``, prints it, consumes the step and returns normally.

* ``setUp`` turns the undo queue on with an infinite length and flushes it;
  ``tearDown`` flushes it and restores the state, infinity and length the test
  found. :meth:`UndoWalk.undo_queue_on` / :meth:`UndoWalk.undo_queue_restore`
  do the work and may be called directly.
* :meth:`UndoWalk.scene_state` is a comparable snapshot of the scene: every node
  (long name, type, UUID), every connection, and the data of the meshes and
  skin clusters named.
* :meth:`UndoWalk.assert_no_dangling` checks that both ends of every connection
  exist.
* :meth:`UndoWalk.undo_steps` / :meth:`UndoWalk.redo_steps` walk ``n`` steps
  with the dangling check after each one; :meth:`UndoWalk.undo_all` /
  :meth:`UndoWalk.redo_all` walk until the queue is empty. "Empty" is read from
  ``undoInfo(q=True, undoQueueEmpty=True)`` (``redoQueueEmpty``), not from an
  empty ``undoName``: in mayapy a Python ``cmds`` entry has an empty name, so
  an empty ``undoName`` does not mean the queue is empty.
"""

from maya import cmds
from maya.api import OpenMaya, OpenMayaAnim

# decimals kept for points, UVs, colours and weights
ROUND = 6
# the longest walk undo_all / redo_all take before failing the test
UNDO_WALK_LIMIT = 1000


def _r(x) -> float:
    return round(float(x), ROUND) + 0.0  # + 0.0 folds -0.0 into 0.0


def _dag_path(name):
    """The MDagPath of ``name`` (a shape, or a transform extended to its shape), or None."""
    sel = OpenMaya.MSelectionList()
    try:
        sel.add(str(name))
        path = sel.getDagPath(0)
    except (RuntimeError, TypeError):
        return None
    if path.apiType() == OpenMaya.MFn.kTransform:
        try:
            path.extendToShape()
        except RuntimeError:
            return None
    return path


def mesh_state(name):
    """The data of the mesh ``name`` (shape, or its transform), or None when there is none.

    Object-space points (rounded), ``getVertices``, ``getHoles``, the UV set names in
    order with each set's UVs and assignments, the current UV set, and the colour set
    names with each set's representation and per-face-vertex colours (an unset colour
    reads ``(-1, -1, -1, -1)``), and the current colour set.
    """
    path = _dag_path(name)
    if path is None or not path.hasFn(OpenMaya.MFn.kMesh):
        return None
    fn = OpenMaya.MFnMesh(path)
    counts, ids = fn.getVertices()
    uv_sets = list(fn.getUVSetNames())
    uvs = {}
    for uv_set in uv_sets:
        u, v = fn.getUVs(uv_set)
        uv_counts, uv_ids = fn.getAssignedUVs(uv_set)
        uvs[uv_set] = (
            tuple(_r(x) for x in u),
            tuple(_r(x) for x in v),
            tuple(uv_counts),
            tuple(uv_ids),
        )
    color_sets = list(fn.getColorSetNames())
    unset = OpenMaya.MColor((-1.0, -1.0, -1.0, -1.0))
    colors = {}
    for color_set in color_sets:
        values = fn.getFaceVertexColors(colorSet=color_set, defaultUnsetColor=unset)
        colors[color_set] = (
            int(fn.getColorRepresentation(color_set)),
            tuple((_r(c.r), _r(c.g), _r(c.b), _r(c.a)) for c in values),
        )
    return {
        "points": tuple(
            (_r(p.x), _r(p.y), _r(p.z)) for p in fn.getPoints(OpenMaya.MSpace.kObject)
        ),
        "vertices": (tuple(counts), tuple(ids)),
        "holes": tuple((int(face), tuple(verts)) for face, verts in fn.getHoles()),
        "uv_sets": uv_sets,
        "current_uv_set": fn.currentUVSetName() if uv_sets else "",
        "uvs": uvs,
        "color_sets": color_sets,
        "current_color_set": fn.currentColorSetName() if color_sets else "",
        "colors": colors,
    }


def skin_state(name):
    """The influences and weights (rounded) of the skin cluster ``name``, or None."""
    sel = OpenMaya.MSelectionList()
    try:
        sel.add(str(name))
        obj = sel.getDependNode(0)
    except (RuntimeError, TypeError):
        return None
    if not obj.hasFn(OpenMaya.MFn.kSkinClusterFilter):
        return None
    fn = OpenMayaAnim.MFnSkinCluster(obj)
    influences = [p.fullPathName() for p in fn.influenceObjects()]
    geometry = cmds.skinCluster(str(name), query=True, geometry=True) or []
    rows = None
    path = _dag_path(geometry[0]) if geometry else None
    if path is not None and path.hasFn(OpenMaya.MFn.kMesh):
        comp_fn = OpenMaya.MFnSingleIndexedComponent()
        comp = comp_fn.create(OpenMaya.MFn.kMeshVertComponent)
        comp_fn.setCompleteData(OpenMaya.MFnMesh(path).numVertices)
        weights, count = fn.getWeights(path, comp)
        flat = [_r(x) for x in weights]
        rows = tuple(tuple(flat[i : i + count]) for i in range(0, len(flat), count)) if count else ()
    return {"influences": influences, "geometry": geometry, "weights": rows}


def scene_nodes():
    """Sorted ``(long name, node type, UUID)`` of every node ``cmds.ls(long=True)`` lists."""
    out = []
    for name in cmds.ls(long=True) or []:
        uuid = cmds.ls(name, uuid=True) or [None]
        out.append((name, cmds.nodeType(name), uuid[0]))
    return sorted(out)


def scene_connections():
    """Sorted ``(source plug, destination plug)`` of every connection, each listed once
    (from its destination node's ``listConnections(c=True, p=True, s=True, d=False)``)."""
    pairs = set()
    for name in cmds.ls(long=True) or []:
        conns = cmds.listConnections(
            name, connections=True, plugs=True, source=True, destination=False
        ) or []
        for dst, src in zip(conns[::2], conns[1::2]):
            pairs.add((src, dst))
    return sorted(pairs)


def dangling_connections():
    """Connection endpoints that do not exist (``cmds.objExists`` of the plug is False)."""
    bad = []
    for src, dst in scene_connections():
        for plug in (src, dst):
            if not cmds.objExists(plug):
                bad.append((src, dst, plug))
    return bad


class UndoWalk:
    """Mixin for undo tests on :class:`rig._tests._base.MayaTestCase` (list it first:
    ``class TestX(UndoWalk, MayaTestCase)``). See the module docstring."""

    _undo_prev = None

    # --- queue setup / restore

    def setUp(self):
        super().setUp()
        self.undo_queue_on()

    def tearDown(self):
        try:
            self.undo_queue_restore()
        finally:
            super().tearDown()

    def undo_queue_on(self) -> None:
        """Remember the queue's state, infinity and length, then turn it on with an
        infinite length and flush it."""
        if self._undo_prev is None:
            self._undo_prev = (
                cmds.undoInfo(query=True, state=True),
                cmds.undoInfo(query=True, infinity=True),
                cmds.undoInfo(query=True, length=True),
            )
        cmds.undoInfo(state=True, infinity=True)
        cmds.flushUndo()

    def undo_queue_restore(self) -> None:
        """Flush the queue (it references the test's scene) and put back the state,
        infinity and length :meth:`undo_queue_on` found. A no-op without it."""
        prev, self._undo_prev = self._undo_prev, None
        if prev is None:
            return
        state, infinity, length = prev
        cmds.flushUndo()
        if infinity:
            cmds.undoInfo(infinity=True)
        else:
            cmds.undoInfo(infinity=False, length=length)
        cmds.undoInfo(state=state)

    # --- snapshots

    @staticmethod
    def scene_state(meshes=(), skins=()) -> dict:
        """A comparable snapshot of the scene: ``nodes`` (:func:`scene_nodes`),
        ``connections`` (:func:`scene_connections`), and per name given, ``meshes``
        (:func:`mesh_state`) and ``skins`` (:func:`skin_state`); a missing mesh or skin
        cluster is None. Pass names, not node objects (a freed object has no name)."""
        return {
            "nodes": scene_nodes(),
            "connections": scene_connections(),
            "meshes": {str(m): mesh_state(m) for m in meshes},
            "skins": {str(s): skin_state(s) for s in skins},
        }

    def assert_state_equal(self, actual: dict, expected: dict, msg: str = "") -> None:
        """Compare two :meth:`scene_state` snapshots key by key, naming the first part
        that differs."""
        for key in ("nodes", "connections", "meshes", "skins"):
            if key == "meshes" or key == "skins":
                self.assertEqual(sorted(actual[key]), sorted(expected[key]), f"{msg} {key}: names")
                for name in expected[key]:
                    self.assertEqual(actual[key][name], expected[key][name], f"{msg} {key}[{name!r}]")
            else:
                self.assertEqual(actual[key], expected[key], f"{msg} {key}")

    def assert_no_dangling(self, msg: str = "") -> None:
        """Both ends of every connection exist."""
        bad = dangling_connections()
        self.assertEqual(bad, [], f"{msg} connections with an endpoint that does not exist")

    # --- walks

    def undo_steps(self, n: int, check: bool = True) -> None:
        """``cmds.undo()`` ``n`` times, failing if the queue runs out; with ``check``, the
        dangling check after every step."""
        for i in range(n):
            self.assertFalse(
                cmds.undoInfo(query=True, undoQueueEmpty=True),
                f"undo step {i + 1} of {n}: nothing left to undo",
            )
            cmds.undo()
            if check:
                self.assert_no_dangling(f"after undo step {i + 1} of {n}:")

    def redo_steps(self, n: int, check: bool = True) -> None:
        """``cmds.redo()`` ``n`` times, failing if the queue runs out; with ``check``, the
        dangling check after every step."""
        for i in range(n):
            self.assertFalse(
                cmds.undoInfo(query=True, redoQueueEmpty=True),
                f"redo step {i + 1} of {n}: nothing left to redo",
            )
            cmds.redo()
            if check:
                self.assert_no_dangling(f"after redo step {i + 1} of {n}:")

    def undo_all(self, check: bool = True) -> int:
        """Undo until the undo queue is empty; the number of steps taken."""
        n = 0
        while not cmds.undoInfo(query=True, undoQueueEmpty=True):
            self.assertLess(n, UNDO_WALK_LIMIT, "undo_all: the queue never empties")
            cmds.undo()
            n += 1
            if check:
                self.assert_no_dangling(f"after undo step {n}:")
        return n

    def redo_all(self, check: bool = True) -> int:
        """Redo until the redo queue is empty; the number of steps taken."""
        n = 0
        while not cmds.undoInfo(query=True, redoQueueEmpty=True):
            self.assertLess(n, UNDO_WALK_LIMIT, "redo_all: the queue never empties")
            cmds.redo()
            n += 1
            if check:
                self.assert_no_dangling(f"after redo step {n}:")
        return n

    @staticmethod
    def undo_name() -> str:
        """The name of the entry the next ``cmds.undo()`` reverts ('' when unnamed)."""
        return cmds.undoInfo(query=True, undoName=True)

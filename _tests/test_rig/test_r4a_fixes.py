"""Round 4a, step SF: small fixes independent of the node merge.

* A rotate order takes one of the six rotate-order names (decision S3 Q1):
  ``"xyz"`` .. ``"zyx"`` map to 0-5 before the function runs, and any other
  str raises TypeError before any node or container is created.
* ``functions.searchsorted`` checks ``side=`` before it creates its container
  (decision S3 Q5).
"""

from maya import cmds

from rig import (
    container,
    euler as E,
    force_nodes,
    functions as F,
    matrix as M,
    Node,
    PlugList,
    quaternion as Q,
    set_options,
    vector as V,
)
from rig import _dispatch as D
from rig._internal import operands as operands_module
from rig._tests._base import MayaTestCase


_FREED = r"already deleted!$"

_NAMES = ("xyz", "yzx", "zxy", "xzy", "yxz", "zyx")


def _scene():
    """Every node and container in the scene, by long name."""
    return set(cmds.ls(long=True))


class _SceneCase(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.loadPlugin("quatNodes", quiet=True)
        cmds.loadPlugin("matrixNodes", quiet=True)
        for name in ("t", "u"):
            cmds.createNode("transform", name=name)
        self.t, self.u = Node("t"), Node("u")

    def tearDown(self):
        set_options(constant_folding=True, flatten_containers=True)
        super().tearDown()


class TestRotateOrderNames(_SceneCase):
    """Every function that takes a rotate order takes its name."""

    def _calls(self):
        """label -> (the rotate-order parameter, call(rotate_order)), one per
        function that takes a rotate order (the two of euler.reorder each)."""
        t, u = self.t, self.u
        wm = t.worldMatrix[0]
        quat = E.to_quaternion(u.r)
        return {
            "rig.matrix.decompose": ("rotate_order", lambda ro: M.decompose(wm, rotate_order=ro)),
            "rig.matrix.to_euler": ("rotate_order", lambda ro: M.to_euler(wm, rotate_order=ro)),
            "rig.matrix.compose": ("rotate_order", lambda ro: M.compose(rotate=t.r, rotate_order=ro)),
            "rig.euler.to_matrix": ("rotate_order", lambda ro: E.to_matrix(t.r, rotate_order=ro)),
            "rig.euler.to_quaternion": ("rotate_order", lambda ro: E.to_quaternion(t.r, rotate_order=ro)),
            "rig.euler.reorder": ("rotate_order0", lambda ro: E.reorder(t.r, ro, 0)),
            "rig.euler.reorder ": ("rotate_order1", lambda ro: E.reorder(t.r, 0, ro)),
            "rig.quaternion.to_euler": ("rotate_order", lambda ro: Q.to_euler(quat, rotate_order=ro)),
            "rig.vector.rotate": ("rotate_order", lambda ro: V.rotate(t.t, t.r, rotate_order=ro)),
            "rig.to_euler": ("rotate_order", lambda ro: D.to_euler(quat, rotate_order=ro)),
            "rig.to_quaternion": ("rotate_order", lambda ro: D.to_quaternion(t.r, rotate_order=ro)),
            "rig.to_matrix": ("rotate_order", lambda ro: D.to_matrix(t.r, rotate_order=ro)),
        }

    def test_every_name_through_every_function(self):
        for label, (_, call) in self._calls().items():
            for order, name in enumerate(_NAMES):
                with self.subTest(function=label, name=name):
                    by_int = call(order)
                    before = _scene()
                    self.assertIs(call(name), by_int)
                    self.assertEqual(_scene(), before)

    def test_a_name_sets_its_rotate_order(self):
        t = self.t
        cases = {
            "decompose": (lambda ro: M.decompose(t.worldMatrix[0], rotate_order=ro), "inputRotateOrder"),
            "compose": (lambda ro: M.compose(rotate=t.r, rotate_order=ro), "inputRotateOrder"),
            "to_quaternion": (lambda ro: E.to_quaternion(t.r, rotate_order=ro), "inputRotateOrder"),
            "rotate": (lambda ro: V.rotate(t.t, t.r, rotate_order=ro), "rotateOrder"),
        }
        for label, (call, attr) in cases.items():
            for order, name in enumerate(_NAMES):
                with self.subTest(function=label, name=name):
                    plug = call(name)
                    node = plug.node
                    if not cmds.attributeQuery(attr, node=str(node), exists=True):
                        # compose publishes its matrix out of a container: read
                        # the composeMatrix that feeds it
                        node = cmds.listConnections(str(plug), source=True, destination=False)[0]
                    self.assertEqual(cmds.getAttr(f"{node}.{attr}"), order)

    def test_positional_rotate_orders(self):
        wm = self.t.worldMatrix[0]
        self.assertIs(M.decompose(wm, "yxz"), M.decompose(wm, 4))
        self.assertIs(E.reorder(self.t.r, "xyz", "zxy"), E.reorder(self.t.r, 0, 2))
        self.assertIs(E.reorder(self.t.r, rotate_order0="yzx", rotate_order1="zyx"),
                      E.reorder(self.t.r, rotate_order0=1, rotate_order1=5))

    def test_one_memo_entry_for_a_name_and_its_int(self):
        wm = self.t.worldMatrix[0]
        size = len(M.decompose._cache)
        first = M.decompose(wm, rotate_order="zxy")
        self.assertIs(M.decompose(wm, rotate_order=2), first)
        self.assertEqual(len(M.decompose._cache), size + 1)

    def test_ints_plugs_and_none_pass_unchanged(self):
        wm = self.t.worldMatrix[0]
        self.assertIs(M.decompose(wm, rotate_order=None), M.decompose(wm))
        self.assertIs(M.decompose(wm, rotate_order=3), M.decompose(wm, rotate_order=3.0))
        driven = M.decompose(wm, rotate_order=self.u.ro)
        self.assertEqual(
            cmds.listConnections(f"{driven.node}.inputRotateOrder", source=True, plugs=True),
            ["u.rotateOrder"],
        )
        for value in (0, 5, None, self.u.ro, [1, 2], (self.u.ro, 3), 2.0):
            with self.subTest(value=value):
                self.assertIs(operands_module._rotate_order("f", "rotate_order", value), value)
        self.assertEqual(operands_module._ROTATE_ORDERS, dict(zip(_NAMES, range(6))))

    def test_any_other_str_raises_before_any_node(self):
        bad = ("XYZ", "", "xy", "xyzz", " xyz", "Zyx", "0")
        scopes = {
            "top": lambda: _Nothing(),
            "container": lambda: container("box"),
            "force_nodes": force_nodes,
        }
        calls = self._calls()
        for flatten in (True, False):
            set_options(flatten_containers=flatten)
            for scope_name, scope in scopes.items():
                cmds.file(new=True, force=True)
                self.setUp()
                calls = self._calls()
                with scope():
                    for label, (parameter, call) in calls.items():
                        for name in bad:
                            with self.subTest(flatten=flatten, scope=scope_name, function=label, name=name):
                                before = _scene()
                                with self.assertRaises(TypeError) as ctx:
                                    call(name)
                                self.assertEqual(sorted(_scene() - before), [])
                                self.assertEqual(
                                    str(ctx.exception),
                                    f"{label.strip()}() argument {parameter!r}: {name!r} is not a "
                                    f"rotate order; use 'xyz', 'yzx', 'zxy', 'xzy', 'yxz' or 'zyx' "
                                    f"(0-5), an int 0-5 or a plug",
                                )
                if scope_name == "container" and cmds.objExists("box"):
                    self.assertEqual(cmds.container("box", query=True, nodeList=True) or [], [])

    def test_a_list_broadcast(self):
        mats = PlugList([self.t.worldMatrix[0], self.u.worldMatrix[0]])
        for orders in (["zxy", "yxz"], ("zxy", "yxz"), ["zxy", 4]):
            with self.subTest(orders=orders):
                out = M.decompose(mats, rotate_order=orders)
                self.assertIsInstance(out, PlugList)
                self.assertEqual(
                    [cmds.getAttr(f"{plug.node}.inputRotateOrder") for plug in out], [2, 4]
                )
                self.assertIs(out[0], M.decompose(self.t.worldMatrix[0], rotate_order=2))
        before = _scene()
        with self.assertRaisesRegex(TypeError, r"argument 'rotate_order\[1\]': 'bad' is not a rotate order"):
            M.decompose(mats, rotate_order=["xyz", "bad"])
        self.assertEqual(_scene(), before)

    def test_a_freed_plug_raises_first(self):
        cmds.createNode("transform", name="gone")
        held = Node("gone").worldMatrix[0]
        held_order = Node("gone").ro
        cmds.file(new=True, force=True)
        cmds.createNode("transform", name="t")
        before = _scene()
        for name in ("xyz", "bad"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeError, _FREED):
                    M.decompose(held, rotate_order=name)
                with self.assertRaisesRegex(RuntimeError, _FREED):
                    E.reorder(held, name, "zyx")
        with self.assertRaisesRegex(RuntimeError, _FREED):
            M.decompose(Node("t").worldMatrix[0], rotate_order=held_order)
        self.assertEqual(_scene(), before)


class TestSearchsortedSide(_SceneCase):
    """``side=`` is checked before the searchsorted container is created."""

    def test_both_sides_and_both_return_kinds(self):
        # the values v2.0.0a2 and round 3 give (tokens 0, 1, 2)
        expected = {
            -1:  (0, 0, 0, 0),
            0.5: (0, 0, 1, 1),
            1:   (1, 1, 1, 1),
            1.5: (1, 1, 2, 2),
            3:   (2, 2, 2, 2),
        }
        outs = [
            F.searchsorted([0, 1, 2], self.t.tx, return_index=index, side=side)
            for side in ("left", "right")
            for index in (True, False)
        ]
        for query, values in expected.items():
            with self.subTest(query=query):
                cmds.setAttr("t.tx", query)
                self.assertEqual(tuple(cmds.getAttr(str(out)) for out in outs), values)

    def test_a_bad_side_raises_before_the_container(self):
        scopes = {
            "top": lambda: _Nothing(),
            "container": lambda: container("box"),
            "force_nodes": force_nodes,
        }
        for flatten in (True, False):
            set_options(flatten_containers=flatten)
            for scope_name, scope in scopes.items():
                with scope():
                    for side in ("middle", "", "LEFT", None, self.u.tx):
                        for index in (True, False):
                            with self.subTest(flatten=flatten, scope=scope_name, side=side, index=index):
                                before = cmds.ls()
                                size = len(F.searchsorted._cache)
                                with self.assertRaises(ValueError) as ctx:
                                    F.searchsorted([0, 1, 2], self.t.tx, return_index=index, side=side)
                                self.assertEqual(
                                    str(ctx.exception), f"side must be 'left' or 'right'; got {side!r}"
                                )
                                self.assertEqual(cmds.ls(), before)
                                self.assertEqual(len(F.searchsorted._cache), size)
                if cmds.objExists("box"):
                    self.assertEqual(cmds.container("box", query=True, nodeList=True) or [], [])
        self.assertFalse(cmds.ls("searchsorted*"))


class _Nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

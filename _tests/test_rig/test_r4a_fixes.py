"""Round 4a, step SF: small fixes independent of the node merge.

* A rotate order takes one of the six rotate-order names (decision S3 Q1):
  ``"xyz"`` .. ``"zyx"`` map to 0-5 before the function runs, and any other
  str raises TypeError before any node or container is created.
* ``functions.searchsorted`` checks ``side=`` before it creates its container
  (decision S3 Q5).
* Operand shapes (F14): a set, frozenset or dict operand raises TypeError before
  anything is built, and an iterator operand (generator, map, zip, iter, an
  itertools object) is read into a list first, so it memoizes like the list.
* A memo entry that returned a typed attribute (``PyNode("a").find_attr("tx")``)
  keeps its node's handle, like a Plug's, so a delete, a new scene or a
  reference unload drops it (decision S4 Q5).
* Every Plug operator is checked by one frame (``_checking_operands``).

Round 4a, step X1 (decision X1): ``p == q`` / ``p != q`` of two objects of one
Maya plug fold to ``True`` / ``False`` and build no node, unless constant
folding is off (``force_nodes()``); different plugs build the condition node.
A dict / set lookup through a second object of a plug builds nothing (round-3
F10), and a plain list scan still compares every different plug it passes.

Round 4a, step M6: ``PlugList`` is now ``List`` (``PlugList is List``, both
exported, repr ``List([...])``), and ``in`` / ``index`` / ``count`` /
``remove`` read a plain str probe as the Maya plug it names (decision S3 Q4).
"""

import itertools
import os
import shutil
import tempfile

from maya import cmds

from rig import (
    condition,
    container,
    euler as E,
    force_nodes,
    functions as F,
    InjectionError,
    matrix as M,
    Node,
    PlugList,
    quaternion as Q,
    set_options,
    vector as V,
)
from rig import _dispatch as D
from rig._internal import memoize as memoize_module
from rig._internal import operands as operands_module
from rig._internal.memoize import memoize
from rig._internal.plug import Plug
from rig.nodetypes import PyNode
from rig.nodetypes._base import Attribute
from rig.spec import Float
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


class TestOperandShapes(_SceneCase):
    """Sets, frozensets and dicts are refused; iterators are read into lists."""

    def test_iterators_are_read_into_a_list(self):
        t, u = self.t, self.u
        cases = {
            # label: (the iterator form, the list form)
            "sum generator": (lambda: F.sum(x.tx for x in (t, u)), lambda: F.sum([t.tx, u.tx])),
            "sum map": (lambda: F.sum(map(lambda n: n.tx, (t, u))), lambda: F.sum([t.tx, u.tx])),
            "sum iter": (lambda: F.sum(iter([t.tx, u.tx])), lambda: F.sum([t.tx, u.tx])),
            "sum zip": (
                lambda: F.sum(zip([t.tx, u.tx], [t.ty, u.ty])),
                lambda: F.sum([(t.tx, t.ty), (u.tx, u.ty)]),
            ),
            "sum chain": (lambda: F.sum(itertools.chain([t.tx], [u.tx])), lambda: F.sum([t.tx, u.tx])),
            "sum keyword": (lambda: F.sum(tokens=(x.ty for x in (t, u))), lambda: F.sum(tokens=[t.ty, u.ty])),
            "lerp generator": (
                lambda: V.lerp((float(i) for i in (1, 2, 3)), u.t, 0.5),
                lambda: V.lerp([1.0, 2.0, 3.0], u.t, 0.5),
            ),
            "lerp map keyword": (
                lambda: V.lerp(input1=map(float, (4, 5, 6)), input2=u.t),
                lambda: V.lerp(input1=[4.0, 5.0, 6.0], input2=u.t),
            ),
            "plug operator generator": (lambda: t.t + (i for i in (1, 2, 3)), lambda: t.t + [1, 2, 3]),
            "plug operator map": (lambda: t.t * map(float, (1, 2, 3)), lambda: t.t * [1.0, 2.0, 3.0]),
            "plug operator zip": (lambda: t.t - zip([1, 2, 3]), lambda: t.t - [(1,), (2,), (3,)]),
            "reflected plug operator": (lambda: iter([1, 2, 3]) + t.t, lambda: [1, 2, 3] + t.t),
        }
        for label, (iterated, listed) in cases.items():
            with self.subTest(label):
                by_list = listed()
                before = _scene()
                self.assertIs(iterated(), by_list)
                self.assertEqual(_scene(), before)
        pairs = PlugList([t.tx, u.tx])
        by_list = pairs + [1, 2]
        reflected_by_list = [3, 4] - pairs
        before = _scene()
        for label, operand in (
            ("generator", (i for i in (1, 2))),
            ("map", map(int, "12")),
            ("iter", iter([1, 2])),
        ):
            with self.subTest("PlugList operator", kind=label):
                out = pairs + operand
                self.assertIsInstance(out, PlugList)
                self.assertEqual(len(out), 2)
                self.assertTrue(all(a is b for a, b in zip(out, by_list)))
        reflected = iter([3, 4]) - pairs
        self.assertEqual(len(reflected), 2)
        self.assertTrue(all(a is b for a, b in zip(reflected, reflected_by_list)))
        self.assertEqual(_scene(), before)

    def test_one_memo_entry_for_an_iterator_and_its_list(self):
        t, u = self.t, self.u
        size = len(F.sum._cache)
        first = F.sum(x.tx for x in (t, u))
        self.assertIs(F.sum(x.tx for x in (t, u)), first)
        self.assertIs(F.sum([t.tx, u.tx]), first)
        self.assertEqual(len(F.sum._cache), size + 1)

    def _refused(self, test):
        t, u = self.t, self.u
        return {
            # entry point: [(label, call, the message's text)]
            "function": [
                ("sum set", lambda: F.sum({t.tx, u.tx}), "rig.functions.sum() argument 'tokens': a set is unordered"),
                ("sum frozenset", lambda: F.sum(frozenset([t.tx])), "a frozenset is unordered"),
                ("sum dict", lambda: F.sum({t.tx: 1}), "argument 'tokens': a dict is a mapping, not a sequence"),
                ("sum keyword set", lambda: F.sum(tokens={1, 2}), "argument 'tokens': a set is unordered"),
                ("lerp set", lambda: V.lerp({1, 2, 3}, u.t), "rig.vector.lerp() argument 'input1': a set"),
                ("lerp weight dict", lambda: V.lerp(t.t, u.t, weight={"w": 0.5}), "argument 'weight': a dict"),
                ("condition branch", lambda: condition(test, {1}, 0), "argument 'if_true': a set"),
                ("decompose", lambda: M.decompose({t.worldMatrix[0]}), "argument 'token': a set"),
            ],
            "plug operator": [
                ("+ set", lambda: t.tx + {1, 2}, "t.translateX + {1, 2}: a set is unordered"),
                ("reflected + set", lambda: {1, 2} + t.tx, "{1, 2} + t.translateX: a set is unordered"),
                ("| set", lambda: {1, 2} | t.tx, "{1, 2} | t.translateX: a set"),
                ("+ dict", lambda: t.t + {"x": 1}, "t.translate + {'x': 1}: a dict is a mapping"),
                ("== frozenset", lambda: t.tx == frozenset(), "t.translateX == frozenset(): a frozenset"),
                ("reflected ==", lambda: {1} == t.tx, "t.translateX == {1}: a set"),
                ("< set", lambda: t.tx < {1}, "t.translateX < {1}: a set"),
                ("matrix * set", lambda: t.worldMatrix[0] * {1}, "a set is unordered"),
            ],
            "PlugList operator": [
                ("+ set", lambda: PlugList([t.tx, u.tx]) + {1, 2}, "+ {1, 2}: a set is unordered"),
                ("reflected + set", lambda: {1, 2} + PlugList([t.tx]), "{1, 2} + List(["),
                ("== dict", lambda: PlugList([t.tx]) == {"a": 1}, "== {'a': 1}: a dict is a mapping"),
                ("* frozenset", lambda: PlugList([t.tx]) * frozenset([2]), "a frozenset is unordered"),
            ],
        }

    def test_sets_frozensets_and_dicts_are_refused_before_any_node(self):
        test = self.t.tx > 0  # a condition's test, built before the scopes
        scopes = {
            "top": lambda: _Nothing(),
            "container": lambda: container("box"),
            "force_nodes": force_nodes,
        }
        for scope_name, scope in scopes.items():
            set_options(flatten_containers=scope_name != "container")
            with scope():
                for entry, cases in self._refused(test).items():
                    for label, call, text in cases:
                        with self.subTest(scope=scope_name, entry=entry, case=label):
                            before = _scene()
                            with self.assertRaises(TypeError) as ctx:
                                call()
                            self.assertNotIsInstance(ctx.exception, InjectionError)
                            self.assertIn(text, str(ctx.exception))
                            self.assertIn(
                                "and a DSL operand is a Plug, a number or a sequence of them; "
                                "pass a list (",
                                str(ctx.exception),
                            )
                            self.assertEqual(sorted(_scene() - before), [])
            if scope_name == "container":
                self.assertEqual(cmds.container("box", query=True, nodeList=True) or [], [])

    def test_the_messages(self):
        t = self.t
        with self.assertRaises(TypeError) as ctx:
            t.tx + {1, 2}
        self.assertEqual(
            str(ctx.exception),
            "t.translateX + {1, 2}: a set is unordered, and a DSL operand is a Plug, a "
            "number or a sequence of them; pass a list (sorted(...) for a set)",
        )
        with self.assertRaises(TypeError) as ctx:
            F.sum({"a": t.tx})
        self.assertEqual(
            str(ctx.exception),
            "rig.functions.sum() argument 'tokens': a dict is a mapping, not a sequence, and "
            "a DSL operand is a Plug, a number or a sequence of them; pass a list "
            "(list(d.values()) for its values)",
        )

    def test_a_plain_str_inside_an_iterator_is_rejected(self):
        t = self.t
        cmds.createNode("transform", name="cube")
        before = _scene()
        for label, call in (
            ("function", lambda: F.sum(x for x in [t.tx, "cube.ty"])),
            ("plug operator", lambda: t.t + iter([1, "cube.ty", 2])),
            ("PlugList operator", lambda: PlugList([t.tx]) + (x for x in ["cube.ty"])),
        ):
            with self.subTest(label):
                with self.assertRaises(TypeError) as ctx:
                    call()
                self.assertNotIsInstance(ctx.exception, InjectionError)
                self.assertIn("'cube.ty' is a plain str", str(ctx.exception))
        self.assertEqual(_scene(), before)

    def test_a_generator_holding_a_freed_plug(self):
        cmds.createNode("transform", name="gone")
        held = Node("gone").tx
        cmds.file(new=True, force=True)
        cmds.createNode("transform", name="t")
        t = Node("t")
        before = _scene()
        for label, call in (
            ("function", lambda: F.sum(x for x in (held, t.tx))),
            ("function keyword", lambda: F.sum(tokens=iter([t.tx, held]))),
            ("lerp", lambda: V.lerp(t.tx, map(lambda p: p, [held]))),
            ("plug operator", lambda: t.t + (x for x in (held, 1, 2))),
            ("PlugList operator", lambda: PlugList([t.tx]) + iter([held])),
        ):
            with self.subTest(label):
                with self.assertRaisesRegex(RuntimeError, _FREED):
                    call()
        self.assertEqual(_scene(), before)

    def test_a_deleted_node_in_an_iterator_after_undo(self):
        cmds.undoInfo(state=True, infinity=True)
        try:
            cmds.createNode("transform", name="gone")
            held = Node("gone").tx
            cmds.delete("gone")
            before = _scene()
            with self.assertRaisesRegex(RuntimeError, r"^gone already deleted!$"):
                F.sum(x for x in (held, self.t.tx))
            with self.assertRaisesRegex(RuntimeError, r"^gone already deleted!$"):
                self.t.tx + (x for x in [held])
            self.assertEqual(_scene(), before)
            cmds.undo()
            self.assertIs(F.sum(x for x in (held, self.t.tx)), F.sum([held, self.t.tx]))
        finally:
            cmds.undoInfo(state=False)

    def test_a_python_picked_condition(self):
        # condition() picks its branch in Python for a number test: the
        # iterator is read into a list first, and a set is still refused
        self.assertEqual(condition(1, (i for i in (1, 2)), 0), [1, 2])
        self.assertEqual(condition(0, 1, iter([3, 4])), [3, 4])
        before = _scene()
        with self.assertRaisesRegex(TypeError, "argument 'if_true': a set is unordered"):
            condition(1, {1, 2}, 0)
        self.assertEqual(_scene(), before)

    def test_other_iterables_pass_as_before(self):
        t = self.t
        self.assertEqual(F.sum(range(3)), 3)
        self.assertEqual(cmds.nodeType(F.sum({"a": t.tx, "b": 1}.values()).node), "sum")
        self.assertEqual(cmds.nodeType((t.t + [1, 2, 3]).node), "plusMinusAverage")
        self.assertEqual(F.sum([1, 2, 3]), 6)
        self.assertEqual(F.sum(i for i in (1, 2, 3)), 6)  # folded, as the list is
        plug = t.tx
        self.assertIs(operands_module._prepared_operand(plug, "x"), plug)
        self.assertEqual(operands_module._prepared_operand(iter((1, 2)), "x"), [1, 2])


class _TypedAttrMemo:
    """A user @memoize function returning ``PyNode(name).find_attr("tx")``,
    taken out of the memo registry after the test."""

    def __init__(self, case, name):
        self.calls = 0
        count = len(memoize_module._ALL_MEMOIZED)

        @memoize
        def typed_tx(i):
            self.calls += 1
            return PyNode(name).find_attr("tx")

        self.function = typed_tx
        added = memoize_module._ALL_MEMOIZED[count:]
        case.addCleanup(
            lambda: [memoize_module._ALL_MEMOIZED.remove(w) for w in added
                     if w in memoize_module._ALL_MEMOIZED]
        )

    def __call__(self, i=0):
        return self.function(i)

    def holds(self, value):
        return any(entry.value is value for entry in self.function._cache.values())


class TestMemoHandleOfTypedAttr(MayaTestCase):
    """A memo entry that returned a typed Attribute keeps its node's handle."""

    TEST_START_NEW_SCENE = True

    def test_the_handle_is_collected(self):
        cmds.createNode("transform", name="a")
        cmds.createNode("transform", name="b")
        typed = PyNode("a").find_attr("tx")
        self.assertIs(type(typed), Attribute)
        for label, value, count in (
            ("typed attr", typed, 1),
            ("plug", Plug("a.tx"), 1),
            ("typed attrs in a list", [typed, PyNode("b").find_attr("ty")], 2),
            ("typed attr in a PlugList", PlugList([typed]), 1),
            ("typed shape attr", PyNode(cmds.createNode("locator", name="aShape", parent="a")).find_attr("localPositionX"), 1),
        ):
            with self.subTest(label):
                out = []
                memoize_module._collect_handles(value, out)
                self.assertEqual(len(out), count)
                self.assertTrue(all(h.isAlive() and h.isValid() for h in out))
        out = []
        memoize_module._collect_handles(typed, out)
        cmds.delete("a")  # the undo queue is off: freed
        self.assertFalse(out[0].isAlive())
        # a freed typed attr adds nothing and does not raise
        out = []
        memoize_module._collect_handles(typed, out)
        memoize_module._collect_handles([typed, "text", 1.0, None], out)
        self.assertEqual(out, [])

    def test_a_delete_recomputes(self):
        cmds.createNode("transform", name="a")
        memo = _TypedAttrMemo(self, "a")
        first = memo()
        self.assertIs(memo(), first)
        self.assertEqual(memo.calls, 1)
        cmds.delete("a")  # freed
        with self.assertRaisesRegex(TypeError, "No object matches name: a"):
            memo()
        self.assertEqual(memo.calls, 2)
        self.assertFalse(memo.holds(first))
        # same-name reuse: the new node's attr, never the freed one
        cmds.createNode("transform", name="a")
        again = memo()
        self.assertEqual(memo.calls, 3)
        self.assertIsNot(again, first)
        self.assertEqual(str(again), "a.translateX")
        cmds.setAttr("a.tx", 4.0)
        self.assertEqual(again.get(), 4.0)

    def test_a_rename_keeps_the_entry(self):
        cmds.createNode("transform", name="a")
        memo = _TypedAttrMemo(self, "a")
        first = memo()
        cmds.rename("a", "renamed")
        self.assertIs(memo(), first)
        self.assertEqual(memo.calls, 1)
        self.assertEqual(str(first), "renamed.translateX")

    def test_an_undone_delete(self):
        cmds.undoInfo(state=True, infinity=True)
        try:
            cmds.createNode("transform", name="a")
            memo = _TypedAttrMemo(self, "a")
            first = memo()
            cmds.delete("a")
            cmds.undo()
            # the node is back: the entry is valid again, and its attr is live
            self.assertIs(memo(), first)
            self.assertEqual(memo.calls, 1)
            cmds.setAttr("a.tx", 2.0)
            self.assertEqual(first.get(), 2.0)
            # deleted to the undo queue: the entry is stale, recomputed
            cmds.delete("a")
            with self.assertRaisesRegex(TypeError, "No object matches name: a"):
                memo()
            self.assertEqual(memo.calls, 2)
            cmds.undo()
            again = memo()
            self.assertEqual(memo.calls, 3)
            self.assertEqual(again.get(), 2.0)
        finally:
            cmds.undoInfo(state=False)

    def test_a_new_scene_and_a_file_open(self):
        folder = tempfile.mkdtemp(prefix="rig_r4a_memo_")
        path = os.path.join(folder, "typed_attr.ma").replace("\\", "/")
        try:
            cmds.createNode("transform", name="a")
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
            memo = _TypedAttrMemo(self, "a")
            first = memo()
            cmds.file(new=True, force=True)
            self.assertFalse(memo.holds(first))
            with self.assertRaisesRegex(TypeError, "No object matches name: a"):
                memo()
            cmds.file(path, open=True, force=True)
            opened = memo()
            self.assertEqual(memo.calls, 3)
            self.assertEqual(opened.get(), 0.0)
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)

    def test_a_reference_unload_prunes_the_entry(self):
        folder = tempfile.mkdtemp(prefix="rig_r4a_unload_")
        path = os.path.join(folder, "typed_ref.ma").replace("\\", "/")
        try:
            cmds.file(new=True, force=True)
            cmds.createNode("transform", name="refT")
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
            cmds.file(new=True, force=True)
            cmds.file(path, reference=True, namespace="ref")
            memo = _TypedAttrMemo(self, "ref:refT")
            first = memo()
            self.assertTrue(memo.holds(first))
            cmds.file(unloadReference="refRN")
            # pruned by the unload callback: the stale typed attr is never returned
            self.assertFalse(memo.holds(first))
            with self.assertRaisesRegex(TypeError, "No object matches name: ref:refT"):
                memo()
            self.assertEqual(memo.calls, 2)
            cmds.file(loadReference="refRN")
            again = memo()
            self.assertEqual(str(again), "ref:refT.translateX")
            self.assertEqual(again.get(), 0.0)
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)


# the Plug operators wrapped at the end of rig._internal.plug
_WRAPPED_OPERATORS = (
    "__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__",
    "__truediv__", "__rtruediv__", "__pow__", "__rpow__", "__floordiv__",
    "__rfloordiv__", "__mod__", "__rmod__", "__and__", "__rand__", "__or__",
    "__ror__", "__xor__", "__rxor__", "__neg__", "__invert__",
    "__eq__", "__ne__", "__ge__", "__le__", "__gt__", "__lt__",
)


class TestOneOperandCheckFrame(MayaTestCase):
    """Every Plug operator is wrapped by `_checking_operands` once, and nothing
    else wraps it: one check frame per operator call."""

    def test_every_operator_has_one_check_frame(self):
        self.assertEqual(len(_WRAPPED_OPERATORS), 28)
        code = None
        for name in _WRAPPED_OPERATORS:
            with self.subTest(name):
                operator = Plug.__dict__[name]
                self.assertTrue(hasattr(operator, "__wrapped__"))
                self.assertFalse(hasattr(operator.__wrapped__, "__wrapped__"))
                self.assertEqual(operator.__name__, name)
                self.assertEqual(operator.__code__.co_qualname, "_checking_operands.<locals>.checked")
                code = code or operator.__code__
                self.assertIs(operator.__code__, code)


class _Nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# --------------------------------------------------------------------- #
#  Step X1: same-plug == / != fold (decision X1; resolves round-3 F10)
# --------------------------------------------------------------------- #


def _new(before):
    """The nodes made since `before` (a `_scene()`), sorted."""
    return sorted(_scene() - before)


def _instanced():
    """`|T1|S` and `|T2|S`: one locator shape under two transforms."""
    cmds.createNode("transform", name="T1")
    cmds.createNode("locator", name="S", parent="T1")
    cmds.createNode("transform", name="T2")
    cmds.parent("|T1|S", "T2", addObject=True, shape=True)
    return Node("|T1|S"), Node("|T2|S")


class _FoldingOff:
    """`set_options(constant_folding=False)` for a `with` block."""

    def __enter__(self):
        set_options(constant_folding=False)
        return self

    def __exit__(self, *exc):
        set_options(constant_folding=True)
        return False


class TestSamePlugFold(_SceneCase):
    """``p == q`` / ``p != q`` of two objects of one Maya plug are ``True`` /
    ``False`` and build nothing (unless constant folding is off); different
    plugs build the condition node as before."""

    def _assert_folds(self, p, q):
        self.assertIsNot(p, q)
        before = _scene()
        self.assertIs(p == q, True)
        self.assertIs(p != q, False)
        self.assertIs(q == p, True)
        self.assertIs(q != p, False)
        self.assertEqual(_new(before), [])

    def test_two_objects_of_one_plug(self):
        t = self.t
        cmds.createNode("multiplyDivide", name="md")
        cmds.addAttr("t", ln="knob", at="double")
        cmds.namespace(add="ns")
        cmds.createNode("transform", name="ns:a")
        cases = {
            "DG": lambda: (Node("md").input1X, Node("md").input1X),
            "DAG": lambda: (t.tx, t.tx),
            "compound, short and long name": lambda: (t.t, t.translate),
            "a name-built plug": lambda: (Plug("t.tx"), t.translateX),
            "a matrix element": lambda: (t.worldMatrix[0], t.worldMatrix[0]),
            "an unindexed matrix and its element": lambda: (t.worldMatrix, t.worldMatrix[0]),
            "a dynamic attr": lambda: (t.knob, Node("t").knob),
            "a namespaced node": lambda: (Node("ns:a").tx, Plug("ns:a.translateX")),
            "a typed attr on the right": lambda: (t.tx, PyNode("t").find_attr("tx")),
            "a typed attr on the left": lambda: (PyNode("t").find_attr("tx"), t.tx),
        }
        for label, pair in cases.items():
            with self.subTest(label):
                self._assert_folds(*pair())

    def test_through_two_instance_paths(self):
        n1, n2 = _instanced()
        self.assertNotEqual(str(n1.v), str(n2.v))
        self._assert_folds(n1.v, n2.v)
        self._assert_folds(n1.worldMatrix[1], n2.worldMatrix)
        # worldMatrix[0] and [1] are two plugs: the node is built (and a matrix
        # cannot feed an equal node), as for any two plugs
        before = _scene()
        with self.assertRaises(InjectionError):
            n1.worldMatrix[0] == n1.worldMatrix[1]
        self.assertIsNone({n1.worldMatrix[0]: 1}.get(n2.worldMatrix))
        with self.assertRaises(InjectionError):
            n1.worldMatrix == n2.worldMatrix
        self.assertEqual([cmds.nodeType(x) for x in _new(before)], ["equal", "equal"])

    def test_a_component_element_and_its_storage(self):
        cmds.nurbsPlane(name="plane", u=3, v=3, ch=False)
        surface = Node("planeShape")
        element = surface.cv[1, 2]
        index   = element.__dict__["_mplug"].logicalIndex()
        self._assert_folds(element, surface.controlPoints[index])
        self._assert_folds(surface.cv[1, 2], surface.cv[1, 2])
        self._assert_folds(surface.cv, surface.cv)
        lattice = cmds.lattice(cmds.polyCube(name="box")[0], divisions=(2, 3, 2))[1]
        shape   = Node(cmds.listRelatives(lattice, shapes=True)[0])
        element = shape.pt[1, 2, 0]
        index   = element.__dict__["_mplug"].logicalIndex()
        self._assert_folds(shape.controlPoints[index], element)
        self._assert_folds(Node("boxShape").vtx[3], Node("boxShape").pnts[3])
        before = _scene()
        self.assertFalse(surface.cv[1, 2] == surface.cv[2, 1])
        self.assertNotEqual(_new(before), [])

    def test_a_container_published_plug(self):
        set_options(flatten_containers=False)
        with container("outer") as box:
            n = Node.create("transform", name="holder")
            n << Float("weight", dv=0.5)
            published = container.publish_input(n.weight, "weight")
            m = Node.create("multiplyDivide", name="mul")
            m.input1X << n.weight
            result = container.publish_output(m.outputX, "result")
        for label, pair in {
            "the input and the container's": (published, box.weight),
            "the container's through two nodes": (box.weight, Node("outer").weight),
            "a name-built plug of the container's": (Plug("outer.weight"), n.weight),
            "the output and its source": (result, Node("mul").outputX),
            "the output and the container's": (Node("outer").result, box.result),
        }.items():
            with self.subTest(label):
                self._assert_folds(*pair)

    def test_a_renamed_node(self):
        held = self.t.tx
        cmds.rename("t", "t_renamed")
        self._assert_folds(held, Node("t_renamed").tx)

    def test_different_plugs_build_the_node(self):
        t, u = self.t, self.u
        for label, (p, q) in {
            "two nodes": (t.tx, u.tx),
            "two attrs": (t.tx, t.ty),
            "a typed attr": (t.tz, PyNode("u").find_attr("tz")),
        }.items():
            with self.subTest(label):
                before = _scene()
                eq = p == q
                self.assertIsInstance(eq, Plug)
                self.assertIs(bool(eq), False)
                ne = p != q
                self.assertIsInstance(ne, Plug)
                self.assertIs(bool(ne), True)
                self.assertEqual(
                    sorted(cmds.nodeType(x) for x in _new(before)), ["condition", "equal"]
                )
                self.assertEqual(cmds.nodeType(eq.node), "equal")
                self.assertEqual(
                    cmds.listConnections(str(eq.node.input1), plugs=True), [str(p)]
                )

    def test_force_nodes_and_constant_folding_off_build_the_node(self):
        for label, scope in {
            "force_nodes": force_nodes,
            "constant_folding=False": _FoldingOff,
        }.items():
            with self.subTest(label):
                cmds.file(new=True, force=True)
                cmds.createNode("transform", name="t")
                t = Node("t")
                self.assertIs(t.tx == t.tx, True)  # folded: nothing memoized
                before = _scene()
                with scope():
                    eq = t.tx == t.tx
                    ne = t.tx != t.tx
                    typed = PyNode("t").find_attr("tx") == t.tx  # reflected: Plug.__eq__
                self.assertIsInstance(eq, Plug)
                self.assertIs(bool(eq), True)
                self.assertIsInstance(ne, Plug)
                self.assertIs(bool(ne), False)
                self.assertIs(typed, eq)  # the same inputs: one memoized node
                self.assertEqual(
                    sorted(cmds.nodeType(x) for x in _new(before)), ["condition", "equal"]
                )
                self.assertEqual(
                    cmds.listConnections(str(eq.node.input1), plugs=True), ["t.translateX"]
                )
                # folding on again: a bool; the memoized node is not looked up
                before = _scene()
                self.assertIs(t.tx == t.tx, True)
                self.assertIs(t.tx != t.tx, False)
                self.assertEqual(_new(before), [])

    def test_dict_set_and_list_lookups_build_nothing(self):
        # round-3 F10: a lookup through a second Plug object of one plug
        t = self.t
        before = _scene()
        self.assertEqual({t.tx: 1}[t.tx], 1)
        self.assertIs(t.tx in {t.tx, t.ty}, True)
        self.assertEqual(len({t.tx, t.tx}), 1)
        self.assertEqual({t.worldMatrix[0]: 1}[t.worldMatrix[0]], 1)  # no InjectionError
        self.assertIs(t.worldMatrix[0] in {t.worldMatrix[0]}, True)
        self.assertIs(t.tx in [t.tx], True)
        self.assertEqual([t.tx].index(t.tx), 0)
        self.assertEqual(_new(before), [])

    def test_a_plain_list_scan_still_compares_different_plugs(self):
        # the residual: each DIFFERENT plug a list scan passes is compared with
        # the DSL ==, which builds an equal node; a PlugList builds nothing
        t, u = self.t, self.u
        before = _scene()
        self.assertIs(t.tx in [u.tx, u.ty, t.tx], True)
        self.assertEqual([cmds.nodeType(x) for x in _new(before)], ["equal", "equal"])
        before = _scene()
        self.assertEqual([u.tx, u.ty, t.tx].index(t.tx), 2)
        self.assertEqual(_new(before), [])  # the same two comparisons, memoized
        before = _scene()
        self.assertIs(t.tx in PlugList([u.tx, u.ty, t.tx]), True)
        self.assertEqual(PlugList([u.tx, u.ty, t.tx]).index(t.tx), 2)
        self.assertIs(
            t.worldMatrix[0] in PlugList([u.worldMatrix[0], t.worldMatrix[0]]), True
        )
        self.assertEqual(_new(before), [])
        with self.assertRaises(InjectionError):
            t.worldMatrix[0] in [u.worldMatrix[0], t.worldMatrix[0]]

    def test_a_condition_of_one_plug_picks_in_python(self):
        t, u = self.t, self.u
        a, b = u.tx, u.ty
        before = _scene()
        self.assertIs(condition(t.tx == t.tx, a, b), a)
        self.assertIs(condition(t.tx != t.tx, a, b), b)
        self.assertEqual(_new(before), [])
        self.assertEqual(cmds.nodeType(condition(t.tx == u.tx, a, b).node), "condition")

    def test_the_result_type_depends_on_the_data(self):
        # CR-17, written into decision X1: one plug gives a bool, two a Plug
        t, u = self.t, self.u
        same, other = t.tx == t.tx, t.tx == u.tx
        self.assertIs(same, True)
        self.assertEqual(cmds.nodeType(other.node), "equal")
        with self.assertRaises(AttributeError):
            same.node
        # `>>` connects neither (a plug is connected with `destination << source`)
        with self.assertRaisesRegex(TypeError, r"unsupported operand type\(s\) for >>: 'bool'"):
            same >> u.v
        with self.assertRaisesRegex(TypeError, r"'>>' does not connect plugs"):
            other >> u.v
        # `<<` takes both: the bool is set, the Plug connected
        u.v << same
        self.assertIs(cmds.getAttr("u.v"), True)
        self.assertEqual(cmds.listConnections("u.v", source=True, destination=False) or [], [])
        u.v << other
        self.assertEqual(cmds.listConnections("u.v", plugs=True), [str(other)])

    def test_a_node_deleted_to_the_undo_queue(self):
        cmds.undoInfo(state=True, infinity=True)
        try:
            cmds.createNode("transform", name="gone")
            p, q = Node("gone").tx, Node("gone").tx
            keyed = {p: 1}
            cmds.delete("gone")
            before = _scene()
            for label, call in (("==", lambda: p == q), ("!=", lambda: p != q)):
                with self.subTest(label):
                    with self.assertRaisesRegex(RuntimeError, r"^gone already deleted!$"):
                        call()
            self.assertEqual(_new(before), [])
            cmds.undo()
            self._assert_folds(p, q)
            before = _scene()
            self.assertEqual(keyed[q], 1)
            self.assertEqual(_new(before), [])
        finally:
            cmds.undoInfo(state=False)

    def test_a_deleted_dynamic_attr(self):
        cmds.addAttr("t", ln="knob", at="double")
        p, q = self.t.knob, self.t.knob
        self._assert_folds(p, q)
        cmds.deleteAttr("t.knob")
        cmds.flushUndo()
        cmds.addAttr("t", ln="knob", at="long")
        before = _scene()
        with self.assertRaisesRegex(RuntimeError, r"^t\.knob already deleted!$"):
            p == q
        with self.assertRaisesRegex(RuntimeError, r"^t\.knob already deleted!$"):
            p != self.t.knob
        self.assertEqual(_new(before), [])
        self._assert_folds(self.t.knob, self.t.knob)

    def test_a_freed_node(self):
        p, q = self.t.tx, self.t.tx
        cmds.file(new=True, force=True)
        before = _scene()
        for label, call in (("==", lambda: p == q), ("!=", lambda: p != q)):
            with self.subTest(label):
                with self.assertRaisesRegex(RuntimeError, _FREED):
                    call()
        self.assertEqual(_new(before), [])


# --------------------------------------------------------------------- #
#  Step M6: List, with PlugList as its former name (the same class)
# --------------------------------------------------------------------- #


class TestListName(_SceneCase):
    """``PlugList`` is now ``List``; ``PlugList`` stays as the same class."""

    def test_pluglist_is_list(self):
        import rig
        from rig import List as exported, PlugList as former
        from rig._internal import list as list_module

        self.assertIs(former, exported)
        self.assertIs(list_module.PlugList, list_module.List)
        self.assertIs(exported, list_module.List)
        self.assertIs(rig.List, rig.PlugList)
        self.assertEqual(exported.__name__, "List")
        # what the package builds is a List under either name
        for built in (self.t.t[:], self.t.tx.get_inputs(), PlugList([self.t.tx]) + 1):
            self.assertIs(type(built), exported)
            self.assertIsInstance(built, former)

    def test_both_names_are_exported(self):
        import rig

        self.assertIn("List", rig.__all__)
        self.assertIn("PlugList", rig.__all__)
        namespace = {}
        exec("from rig import *", namespace)
        self.assertIs(namespace["List"], namespace["PlugList"])

    def test_repr(self):
        from rig import List

        cmds.createNode("transform", name="a")
        cmds.createNode("transform", name="b")
        self.assertEqual(
            repr(List([Node("a"), Node("b")])), 'List([Transform("a"), Transform("b")])'
        )
        self.assertEqual(repr(List([Plug("a.translateX")])), 'List([Plug("a.translateX")])')
        self.assertEqual(repr(PlugList(["a.tx", 1.5])), 'List([Plug("a.translateX"), 1.5])')
        self.assertEqual(repr(List()), "List([])")
        self.assertEqual(repr(Node("a").tx.get_inputs()), "List([])")

    def test_the_retired_sentinels_name_list(self):
        from rig import List

        plug, items = self.t.tx, List([self.t.tx])
        before = _scene()
        for label, call, arrow in (
            ("plug << List", lambda: plug << List, "<<"),
            ("plug >> List", lambda: plug >> List, ">>"),
            ("plug << PlugList", lambda: plug << PlugList, "<<"),
            ("plug << Plug", lambda: plug << Plug, "<<"),
            ("plug >> Plug", lambda: plug >> Plug, ">>"),
            ("list << List", lambda: items << List, "<<"),
            ("list >> List", lambda: items >> List, ">>"),
            ("list >> PlugList", lambda: items >> PlugList, ">>"),
        ):
            with self.subTest(label):
                with self.assertRaisesRegex(
                    TypeError, r"^'<?\w+>? %s List' \(formerly PlugList\) has been replaced" % arrow
                ):
                    call()
        self.assertEqual(_new(before), [])
        # an INSTANCE still connects
        self.u.tz << List([self.t.ty])
        self.assertEqual(cmds.listConnections("u.tz", plugs=True), ["t.translateY"])

    def test_a_missing_value_names_list(self):
        from rig import List

        with self.assertRaisesRegex(ValueError, r"^Plug\(\"u\.translateX\"\) is not in List$"):
            List([self.t.tx]).index(self.u.tx)
        with self.assertRaisesRegex(ValueError, r"^7 is not in List$"):
            List([1, 2]).remove(7)

    def test_the_generic_alias(self):
        import types
        from rig import List

        alias = List[str]
        self.assertIsInstance(alias, types.GenericAlias)
        self.assertIs(alias.__origin__, List)
        self.assertEqual(alias.__args__, (str,))
        self.assertIs(PlugList[int].__origin__, List)


class TestListStrProbe(_SceneCase):
    """``"a.tx" in List([...])`` reads the str as the Maya plug it names and
    compares identity (decision S3 Q4); ``index`` / ``count`` / ``remove``
    agree; nothing is built."""

    def _assert_finds(self, probe, element, others=()):
        """`probe` finds `element` in a List of `others` + [element], by
        every method, and builds nothing."""
        from rig import List

        before = _scene()
        items = List(list(others) + [element])
        at = len(items) - 1
        self.assertIs(probe in items, True)
        self.assertIs(probe not in items, False)
        self.assertEqual(items.index(probe), at)
        self.assertEqual(items.count(probe), 1)
        items.remove(probe)
        self.assertEqual(len(items), at)
        self.assertIs(probe in items, False)
        self.assertEqual(_new(before), [])

    def _assert_misses(self, probe, items):
        before = _scene()
        self.assertIs(probe in items, False)
        self.assertEqual(items.count(probe), 0)
        with self.assertRaisesRegex(ValueError, r" is not in List$"):
            items.index(probe)
        with self.assertRaisesRegex(ValueError, r" is not in List$"):
            items.remove(probe)
        self.assertEqual(_new(before), [])

    def test_names_of_one_plug(self):
        cmds.createNode("transform", name="a")
        cmds.addAttr("a", ln="knob", at="double")
        cmds.aliasAttr("slide", "a.tz")
        cmds.namespace(add="ns")
        cmds.createNode("transform", name="ns:a")
        a, u = Node("a"), self.u
        for label, probe, element in (
            ("the short name", "a.tx", a.tx),
            ("the long name", "a.translateX", a.tx),
            ("a long-name element for a short-name probe", "a.tx", a.translateX),
            ("the full path", "|a.tx", a.tx),
            ("an alias", "a.slide", a.tz),
            ("the attr of an alias", "a.translateZ", a.slide),
            ("a compound", "a.t", a.translate),
            ("a compound child", "a.translate.translateX", a.tx),
            ("a namespaced node (E4)", "ns:a.tx", Node("ns:a").tx),
            ("a dynamic attr (E7)", "a.knob", a.knob),
            ("a matrix element", "a.worldMatrix[0]", a.worldMatrix[0]),
            ("an unindexed world-space array", "a.worldMatrix", a.worldMatrix[0]),
            ("a typed attr element", "a.tx", PyNode("a").find_attr("translateX")),
            ("a name-built element", "a.translateX", Plug("a.tx")),
        ):
            with self.subTest(label):
                self._assert_finds(probe, element, others=[u.tx, u.ty])

    def test_through_another_instance_path(self):
        # E3: one Maya plug, whatever path names it
        from rig import List

        n1, n2 = _instanced()
        for label, probe, element in (
            ("|T1|S.v for |T2|S", "|T1|S.v", n2.v),
            ("T2|S.visibility for |T1|S", "T2|S.visibility", n1.v),
            ("the name of the shape", "S.v", n2.v),
            ("the world matrix of the instance", "T2|S.worldMatrix", n2.worldMatrix),
            ("its element by index", "S.worldMatrix[1]", n2.worldMatrix),
        ):
            with self.subTest(label):
                self._assert_finds(probe, element, others=[self.u.v])
        # world space elements of two instances are two plugs
        self._assert_misses("T1|S.worldMatrix", List([n2.worldMatrix]))
        self._assert_misses("S.worldMatrix[0]", List([n1.worldMatrix[1]]))

    def test_components(self):
        # E6: a component name is the plug of its storage
        from rig import List

        plane = cmds.nurbsPlane(name="np", degree=3, patchesU=1, patchesV=1, ch=False)[0]
        shape = cmds.listRelatives(plane, shapes=True)[0]
        cmds.polyCube(name="box", ch=False)
        for label, probe, element in (
            ("a surface cv for its storage", "np.cv[1][2]", Plug(f"{shape}.controlPoints[6]")),
            ("a surface cv for the ComponentPlug", "np.cv[1][2]", Node(shape).cv[1, 2]),
            ("the storage for the ComponentPlug", f"{shape}.controlPoints[6]", Node(shape).cv[1, 2]),
            ("a mesh vertex", "box.vtx[3]", Node("box").vtx[3]),
            ("a mesh pnts element", "boxShape.pnts[3]", Node("box").vtx[3]),
            ("a one-vertex range", "box.vtx[3:3]", Node("box").vtx[3]),
            ("a uv", "box.map[1]", Node("box").map[1]),
        ):
            with self.subTest(label):
                self._assert_finds(probe, element, others=[self.u.tx])
        for label, probe, element in (
            ("another cv", "np.cv[1][3]", Node(shape).cv[1, 2]),
            ("a range of cvs", "np.cv[0:1][2]", Node(shape).cv[1, 2]),
            ("a range of vertices", "box.vtx[3:4]", Node("box").vtx[3]),
            ("every vertex", "box.vtx[*]", Node("box").vtx[3]),
        ):
            with self.subTest(label):
                self._assert_misses(probe, List([element]))

    def test_a_container_published_plug(self):
        # E5: the published name on the container is the plug it publishes
        set_options(flatten_containers=False)
        with container("outer"):
            n = Node.create("transform", name="holder")
            n << Float("weight", dv=0.5)
            container.publish_input(n.weight, "weight")
        self._assert_finds("outer.weight", n.weight)
        self._assert_finds("holder.weight", Node("outer").weight)

    def test_strs_that_name_no_single_plug(self):
        from rig import List

        cmds.createNode("transform", name="a")
        for parent in ("P1", "P2"):
            cmds.createNode("transform", name=parent)
            cmds.createNode("transform", name="X", parent=parent)
        a = Node("a")
        items = List([a.tx, self.t.tx, Node("|P1|X").tx])
        cmds.select("a")
        for label, probe in (
            ("no such node", "nope.tx"),
            ("no such attr", "a.nope"),
            ("a node", "a"),
            ("a node with a trailing dot", "a."),
            ("the empty str", ""),
            ("a name read against the selection", ".tx"),
            ("a pattern", "a*.tx"),
            ("a one-character pattern", "?.tx"),
            ("a name two nodes have", "X.tx"),
            ("two names", "a.tx t.tx"),
            ("another attr", "a.ty"),
        ):
            with self.subTest(label):
                self._assert_misses(probe, items)
        self._assert_finds("P1|X.tx", Node("|P1|X").tx)

    def test_nodes_and_other_elements_keep_their_rule(self):
        from rig import List

        cmds.createNode("transform", name="a")
        nodes = List([Node("a"), self.t])
        before = _scene()
        self.assertIn("a", nodes)
        self.assertEqual(nodes.index("t"), 1)
        self.assertEqual(nodes.count("a"), 1)
        self.assertNotIn("a.tx", nodes)
        self.assertNotIn("|a", nodes)
        mixed = List([Node("a"), Node("a").tx])
        self.assertEqual(mixed.index("a"), 0)
        self.assertEqual(mixed.index("a.tx"), 1)
        values = List([1, 2.5])
        list.append(values, "a.tx")  # a str element (List() lifts a str)
        self.assertIn("a.tx", values)
        self.assertEqual(values.index("a.tx"), 2)
        self.assertNotIn("a.translateX", values)
        self.assertNotIn("x", List([1, 2]))
        self.assertEqual(_new(before), [])

    def test_the_str_is_resolved_once_and_only_for_a_plug_element(self):
        from unittest import mock

        from rig import List
        from rig._internal import list as list_module

        cmds.createNode("transform", name="a")
        a, u = Node("a"), self.u
        with mock.patch.object(
            list_module, "_plug_named", wraps=list_module._plug_named
        ) as resolve:
            for label, call, calls in (
                ("in", lambda: "a.tx" in List([u.tx, u.ty, u.tz, a.tx]), 1),
                ("a miss", lambda: "a.nope" in List([u.tx, u.ty, a.tx]), 1),
                ("index", lambda: List([u.tx, a.tx, a.tx]).index("a.tx", 2), 1),
                ("count", lambda: List([a.tx, a.tx, u.tx]).count("a.translateX"), 1),
                ("remove", lambda: List([u.tx, a.tx]).remove("a.tx"), 1),
                ("a node list", lambda: "a" in List([a, u]), 0),
                ("a node found first", lambda: "a" in List([a, u.tx]), 0),
                ("numbers", lambda: "a.tx" in List([1, 2]), 0),
                ("an empty list", lambda: "a.tx" in List(), 0),
                ("a Plug probe", lambda: Plug("a.tx") in List([u.tx, a.tx]), 0),
                ("index outside the plugs", lambda: List([a, u.tx]).index("a", 0, 1), 0),
            ):
                with self.subTest(label):
                    resolve.reset_mock()
                    call()
                    self.assertEqual(resolve.call_count, calls)

    def test_a_renamed_node(self):
        from rig import List

        cmds.createNode("transform", name="a")
        items = List([Node("a").tx])
        cmds.rename("a", "b")
        self._assert_finds("b.tx", items[0])
        self._assert_misses("a.tx", items)

    def test_a_deleted_element(self):
        # E1: a plug of a node deleted to the undo queue raises, as its name
        # does; the undo brings it back; a new node of its name is another plug
        from rig import List

        cmds.undoInfo(state=True, infinity=True)
        try:
            cmds.createNode("transform", name="gone")
            items = List([Node("gone").tx])
            live = List([self.t.tx])
            cmds.delete("gone")
            before = _scene()
            for label, call in (
                ("in", lambda: "gone.tx" in items),
                ("index", lambda: items.index("t.tx")),
                ("count", lambda: items.count("gone.tx")),
                ("remove", lambda: items.remove("gone.tx")),
            ):
                with self.subTest(label):
                    with self.assertRaisesRegex(RuntimeError, r"^gone already deleted!$"):
                        call()
            # a str that names the plug of the deleted node names no plug
            self._assert_misses("gone.tx", live)
            self.assertEqual(_new(before), [])
            cmds.undo()
            self._assert_finds("gone.tx", items[0])
            cmds.delete("gone")
            cmds.createNode("transform", name="gone")
            with self.assertRaisesRegex(RuntimeError, r"^gone already deleted!$"):
                "gone.tx" in items
            self._assert_misses("gone.tx", List([self.u.tx]))
        finally:
            cmds.undoInfo(state=False)

    def test_a_deleted_dynamic_attr(self):
        # E7: the attribute of the held element is freed
        from rig import List

        cmds.addAttr("t", ln="knob", at="double")
        items = List([self.t.knob])
        cmds.deleteAttr("t.knob")
        cmds.flushUndo()
        cmds.addAttr("t", ln="knob", at="long")
        with self.assertRaisesRegex(RuntimeError, r"^t\.knob already deleted!$"):
            "t.knob" in items
        self._assert_finds("t.knob", self.t.knob)

    def test_a_freed_element(self):
        # E2: a new scene or a file open frees the node of the held element
        from rig import List

        folder = tempfile.mkdtemp(prefix="rig_r4a_list_")
        path = os.path.join(folder, "held.ma").replace("\\", "/")
        try:
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
            for label, free in (
                ("a new scene", lambda: cmds.file(new=True, force=True)),
                ("a file open that reuses the names", lambda: cmds.file(path, open=True, force=True)),
            ):
                with self.subTest(label):
                    cmds.file(path, open=True, force=True)
                    items = List([Node("t").tx, Node("u").tx])
                    free()
                    before = _scene()
                    for probe in ("t.tx", "nope.tx", "u.translateX"):
                        with self.assertRaisesRegex(RuntimeError, _FREED):
                            probe in items
                        with self.assertRaisesRegex(RuntimeError, _FREED):
                            items.index(probe)
                    self.assertEqual(_new(before), [])
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)

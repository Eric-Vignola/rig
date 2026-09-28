"""Round 4a, step SF: small fixes independent of the node merge.

* A rotate order takes one of the six rotate-order names (decision S3 Q1):
  ``"xyz"`` .. ``"zyx"`` map to 0-5 before the function runs, and any other
  str raises TypeError before any node or container is created.
* ``functions.searchsorted`` checks ``side=`` before it creates its container
  (decision S3 Q5).
* Operand shapes (F14): a set, frozenset or dict operand raises TypeError before
  anything is built, and an iterator operand (generator, map, zip, iter, an
  itertools object) is read into a list first, so it memoizes like the list.
* A memo entry that returned a typed attribute (``Node("a").find_attr("tx")``)
  keeps its node's handle, like a Plug's, so a delete, a new scene or a
  reference unload drops it (decision S4 Q5).
* Every Plug operator is checked by one frame (``_checking_operands``).

Round 4a, step X1 (decision X1): ``p == q`` / ``p != q`` of two objects of one
Maya plug fold to ``True`` / ``False`` and build no node, unless constant
folding is off (``force_nodes()``); different plugs build the condition node.
A dict / set lookup through a second object of a plug builds nothing (round-3
F10), and a plain list scan still compares every different plug it passes.

Round 4a, step M6: the list class is ``List`` (repr ``List([...])``), and
``in`` / ``index`` / ``count`` / ``remove`` read a plain str probe as the Maya
plug it names (decision S3 Q4). Step R1 removed its former name ``PlugList``:
``from rig import PlugList`` is an ImportError, and no package module,
export or message names it.

Round 4a, step M10 (spec S5): a named spec's plug is owned by the node object
it was applied to, ``(node << Float("x")).node is node``, and by the node object
a Plug target holds (``plug << Float("x")``), named for cmds through the path
that node holds.

Round 4a, step R2: ``Node`` is the only node factory; ``PyNode`` was removed
(``from rig.nodetypes import PyNode`` is an ImportError and no package module
names it). ``Node(x)``'s full input table, the private cast core, and
``Node.create`` (a registered type's typed create, with D13's opt-outs; any
other type ``container.createNode``) and ``Node.find_all`` are pinned in
``TestNodeOnly``.
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
    List,
    matrix as M,
    Node,
    quaternion as Q,
    set_options,
    vector as V,
)
from rig import _dispatch as D
from rig._internal import memoize as memoize_module
from rig._internal import operands as operands_module
from rig._internal.memoize import memoize
from rig._internal.plug import Plug
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

    def test_names_match_like_an_enum_plug(self):
        # one loose rule (user, 2026-09-28): a function takes every spelling an
        # enum plug takes, and both give the same rotate order
        wm = self.t.worldMatrix[0]
        for spelling, order in (("XYZ", 0), (" xyz", 0), ("Zyx", 5), ("ZXY ", 2), ("x_z_y", 3), ("Y-X-Z", 4)):
            with self.subTest(spelling=spelling):
                self.assertIs(M.decompose(wm, rotate_order=spelling), M.decompose(wm, rotate_order=order))
                self.assertEqual(operands_module._rotate_order("f", "rotate_order", spelling), order)
                self.u.ro << spelling
                self.assertEqual(cmds.getAttr(f"{self.u}.rotateOrder"), order)

    def test_any_other_str_raises_before_any_node(self):
        bad = ("", "xy", "xyzz", "0", "abc", "x y")
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
        mats = List([self.t.worldMatrix[0], self.u.worldMatrix[0]])
        for orders in (["zxy", "yxz"], ("zxy", "yxz"), ["zxy", 4]):
            with self.subTest(orders=orders):
                out = M.decompose(mats, rotate_order=orders)
                self.assertIsInstance(out, List)
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
        pairs = List([t.tx, u.tx])
        by_list = pairs + [1, 2]
        reflected_by_list = [3, 4] - pairs
        before = _scene()
        for label, operand in (
            ("generator", (i for i in (1, 2))),
            ("map", map(int, "12")),
            ("iter", iter([1, 2])),
        ):
            with self.subTest("List operator", kind=label):
                out = pairs + operand
                self.assertIsInstance(out, List)
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
            "List operator": [
                ("+ set", lambda: List([t.tx, u.tx]) + {1, 2}, "+ {1, 2}: a set is unordered"),
                ("reflected + set", lambda: {1, 2} + List([t.tx]), "{1, 2} + List(["),
                ("== dict", lambda: List([t.tx]) == {"a": 1}, "== {'a': 1}: a dict is a mapping"),
                ("* frozenset", lambda: List([t.tx]) * frozenset([2]), "a frozenset is unordered"),
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
            ("List operator", lambda: List([t.tx]) + (x for x in ["cube.ty"])),
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
            ("List operator", lambda: List([t.tx]) + iter([held])),
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
    """A user @memoize function returning ``Node(name).find_attr("tx")``,
    taken out of the memo registry after the test."""

    def __init__(self, case, name):
        self.calls = 0
        count = len(memoize_module._ALL_MEMOIZED)

        @memoize
        def typed_tx(i):
            self.calls += 1
            return Node(name).find_attr("tx")

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
        typed = Node("a").find_attr("tx")
        self.assertIs(type(typed), Attribute)
        for label, value, count in (
            ("typed attr", typed, 1),
            ("plug", Plug("a.tx"), 1),
            ("typed attrs in a list", [typed, Node("b").find_attr("ty")], 2),
            ("typed attr in a List", List([typed]), 1),
            ("typed shape attr", Node(cmds.createNode("locator", name="aShape", parent="a")).find_attr("localPositionX"), 1),
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
            "a typed attr on the right": lambda: (t.tx, Node("t").find_attr("tx")),
            "a typed attr on the left": lambda: (Node("t").find_attr("tx"), t.tx),
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
            "a typed attr": (t.tz, Node("u").find_attr("tz")),
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
                    typed = Node("t").find_attr("tx") == t.tx  # reflected: Plug.__eq__
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
        # the DSL ==, which builds an equal node; a List builds nothing
        t, u = self.t, self.u
        before = _scene()
        self.assertIs(t.tx in [u.tx, u.ty, t.tx], True)
        self.assertEqual([cmds.nodeType(x) for x in _new(before)], ["equal", "equal"])
        before = _scene()
        self.assertEqual([u.tx, u.ty, t.tx].index(t.tx), 2)
        self.assertEqual(_new(before), [])  # the same two comparisons, memoized
        before = _scene()
        self.assertIs(t.tx in List([u.tx, u.ty, t.tx]), True)
        self.assertEqual(List([u.tx, u.ty, t.tx]).index(t.tx), 2)
        self.assertIs(
            t.worldMatrix[0] in List([u.worldMatrix[0], t.worldMatrix[0]]), True
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
#  Step M6: List (step R1: its former name PlugList is gone)
# --------------------------------------------------------------------- #


def _names_pluglist(path):
    """The (line, token) pairs of a .py file whose token mentions PlugList."""
    import io
    import tokenize

    with open(path, encoding="utf-8") as fh:
        source = fh.read()
    return [
        (tok.start[0], tok.string)
        for tok in tokenize.generate_tokens(io.StringIO(source).readline)
        if "PlugList" in tok.string
    ]


class TestListName(_SceneCase):
    """``List`` is the class's only name: ``PlugList`` was removed (step R1)."""

    def test_pluglist_is_gone(self):
        import importlib
        import rig
        import rig.nodetypes
        from rig._internal import list as list_module

        for module in ("rig", "rig.nodetypes", "rig._internal.list"):
            with self.subTest(module=module):
                with self.assertRaises(ImportError):
                    exec(f"from {module} import PlugList", {})
                self.assertFalse(hasattr(importlib.import_module(module), "PlugList"))
        with self.assertRaises(AttributeError):
            rig.PlugList
        with self.assertRaises(AttributeError):
            rig.nodetypes.PlugList
        with self.assertRaises(AttributeError):
            list_module.PlugList
        # no loaded rig module holds the name
        import sys

        holders = sorted(
            name
            for name, module in list(sys.modules.items())
            if (name == "rig" or name.startswith("rig.")) and module is not None
            and "PlugList" in vars(module)
        )
        self.assertEqual(holders, [])
        # what the package builds is a List
        self.assertIs(List, list_module.List)
        self.assertEqual(List.__name__, "List")
        for built in (self.t.t[:], self.t.tx.get_inputs(), List([self.t.tx]) + 1):
            self.assertIs(type(built), List)

    def test_only_list_is_exported(self):
        import rig

        self.assertIn("List", rig.__all__)
        self.assertNotIn("PlugList", rig.__all__)
        namespace = {}
        exec("from rig import *", namespace)
        self.assertIs(namespace["List"], List)
        self.assertNotIn("PlugList", namespace)

    def test_no_package_module_names_pluglist(self):
        """No .py file of the package outside the tests mentions PlugList: no
        name, string, docstring or comment (the .md docs are the docs step's)."""
        import rig

        root = os.path.dirname(os.path.abspath(rig.__file__))
        found = []
        for folder, subfolders, files in os.walk(root):
            subfolders[:] = [d for d in subfolders if d not in ("_tests", "__pycache__")]
            for file in files:
                if file.endswith(".py"):
                    path = os.path.join(folder, file)
                    found += [
                        (os.path.relpath(path, root), line, text)
                        for line, text in _names_pluglist(path)
                    ]
        self.assertEqual(found, [])

    def test_no_message_names_pluglist(self):
        plug, items = self.t.tx, List([self.t.tx, self.t.ty])
        before = _scene()
        for label, call, error in (
            ("plug << List", lambda: plug << List, TypeError),
            ("plug >> List", lambda: plug >> List, TypeError),
            ("list << List", lambda: items << List, TypeError),
            ("list >> List", lambda: items >> List, TypeError),
            ("row error", lambda: items + [1, "u.ty"], TypeError),
            ("index", lambda: items.index(self.u.tx), ValueError),
        ):
            with self.subTest(label):
                with self.assertRaises(error) as caught:
                    call()
                self.assertNotIn("PlugList", str(caught.exception))
                self.assertIn("List", str(caught.exception))
        self.assertEqual(_new(before), [])

    def test_repr(self):
        cmds.createNode("transform", name="a")
        cmds.createNode("transform", name="b")
        self.assertEqual(
            repr(List([Node("a"), Node("b")])), 'List([Transform("a"), Transform("b")])'
        )
        self.assertEqual(repr(List([Plug("a.translateX")])), 'List([Plug("a.translateX")])')
        self.assertEqual(repr(List(["a.tx", 1.5])), 'List([Plug("a.translateX"), 1.5])')
        self.assertEqual(repr(List()), "List([])")
        self.assertEqual(repr(Node("a").tx.get_inputs()), "List([])")

    def test_the_retired_sentinels_name_list(self):
        plug, items = self.t.tx, List([self.t.tx])
        before = _scene()
        for label, call, arrow in (
            ("plug << List", lambda: plug << List, "<<"),
            ("plug >> List", lambda: plug >> List, ">>"),
            ("plug << Plug", lambda: plug << Plug, "<<"),
            ("plug >> Plug", lambda: plug >> Plug, ">>"),
            ("list << List", lambda: items << List, "<<"),
            ("list >> List", lambda: items >> List, ">>"),
        ):
            with self.subTest(label):
                with self.assertRaisesRegex(
                    TypeError, r"^'<?\w+>? %s List' has been replaced" % arrow
                ):
                    call()
        self.assertEqual(_new(before), [])
        # an INSTANCE still connects
        self.u.tz << List([self.t.ty])
        self.assertEqual(cmds.listConnections("u.tz", plugs=True), ["t.translateY"])

    def test_a_missing_value_names_list(self):
        with self.assertRaisesRegex(ValueError, r"^Plug\(\"u\.translateX\"\) is not in List$"):
            List([self.t.tx]).index(self.u.tx)
        with self.assertRaisesRegex(ValueError, r"^7 is not in List$"):
            List([1, 2]).remove(7)

    def test_the_generic_alias(self):
        import types

        alias = List[str]
        self.assertIsInstance(alias, types.GenericAlias)
        self.assertIs(alias.__origin__, List)
        self.assertEqual(alias.__args__, (str,))


class TestListStrProbe(_SceneCase):
    """``"a.tx" in List([...])`` reads the str as the Maya plug it names and
    compares identity (decision S3 Q4); ``index`` / ``count`` / ``remove``
    agree; nothing is built."""

    def _assert_finds(self, probe, element, others=()):
        """`probe` finds `element` in a List of `others` + [element], by
        every method, and builds nothing."""
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
            ("a name-built element", "a.translateX", Plug("a.tx")),
        ):
            with self.subTest(label):
                self._assert_finds(probe, element, others=[u.tx, u.ty])
        # a typed Attribute element (List() would lift it to a Plug)
        items = List([u.tx])
        list.append(items, Node("a").find_attr("translateX"))
        self.assertIs(type(items[1]), Attribute)
        before = _scene()
        self.assertIn("a.tx", items)
        self.assertEqual(items.index("a.translateX"), 1)
        self.assertEqual(items.count("a.tx"), 1)
        self.assertNotIn("a.ty", items)
        self.assertEqual(_new(before), [])

    def test_through_another_instance_path(self):
        # E3: one Maya plug, whatever path names it
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

    def test_node_elements_match_what_names_them_and_others_keep_their_rule(self):
        cmds.createNode("transform", name="a")
        nodes = List([Node("a"), self.t])
        before = _scene()
        self.assertIn("a", nodes)
        self.assertEqual(nodes.index("t"), 1)
        self.assertEqual(nodes.count("a"), 1)
        self.assertNotIn("a.tx", nodes)
        # a str that names the node finds it (as Node(text) == element)
        self.assertIn("|a", nodes)
        self.assertNotIn("|nosuch", nodes)
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
        cmds.createNode("transform", name="a")
        items = List([Node("a").tx])
        cmds.rename("a", "b")
        self._assert_finds("b.tx", items[0])
        self._assert_misses("a.tx", items)

    def test_a_deleted_element(self):
        # E1: a plug of a node deleted to the undo queue raises, as its name
        # does; the undo brings it back; a new node of its name is another plug
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


def _owned_spec_scene():
    """``held`` (a transform with the locator shape ``heldShape``) and ``other``
    (a transform), as node objects."""
    cmds.createNode("transform", name="held")
    cmds.createNode("locator", name="heldShape", parent="held")
    cmds.createNode("transform", name="other")
    return Node("held"), Node("other")


def _same_plug(a, b):
    """`nodetypes._base._same_plug`: one Maya plug (the one-key identity)."""
    from rig.nodetypes._base import _same_plug as same

    return same(a, b)


class TestOwnerBoundSpecApply(MayaTestCase):
    """M10 (spec S5): a named spec's plug is owned by the node object it was
    applied to (``(node << Float("x")).node is node``, as ``node.x.node`` is),
    and, for a Plug target, by the node object the plug holds. Its str buffer
    names the node as ``str(node)`` does (through the path it holds)."""

    TEST_START_NEW_SCENE = True

    def _assert_owned(self, plug, node, name):
        """`plug` is a Plug of `node`'s attr `name`, owned by the node object, in
        the state the node's own attribute access (``node.<name>``) builds."""
        self.assertIs(type(plug), Plug)
        self.assertIs(plug.node, node)
        self.assertIs(vars(plug)["_node"], node)
        self.assertIsNone(vars(plug)["_handle1"])
        self.assertEqual(str.__str__(plug), f"{node}.{name}")
        same = getattr(node, name)
        self.assertEqual(str(plug), str(same))
        self.assertTrue(_same_plug(plug, same))
        self.assertEqual(hash(plug), hash(same))
        self.assertEqual(tuple(vars(plug)), tuple(vars(same)))
        self.assertIs(vars(plug)["_attr1"], vars(same)["_attr1"])

    def test_every_spec_kind_is_owned(self):
        from rig.spec import Enum, Int, Matrix, Message, String, Vector

        node, _ = _owned_spec_scene()
        for label, spec, name in (
            ("Float", Float("k"), "k"),
            ("Float dv", Float("kd", dv=2.5, min=0, max=10), "kd"),
            ("Int", Int("ik", dv=3), "ik"),
            ("Enum", Enum("ek", en="a:b:c"), "ek"),
            ("String", String("sk"), "sk"),
            ("Message", Message("mk"), "mk"),
            ("Matrix", Matrix("xk"), "xk"),
            ("Vector", Vector("vk"), "vk"),
            ("multi", Float("arr", multi=True, size=3), "arr"),
            ("a Python member's name", Float("rename"), "rename"),
        ):
            with self.subTest(label):
                plug = node << spec
                if name == "rename":
                    # a method wins over the Maya attr in `node.rename`
                    self.assertIs(plug.node, node)
                    self.assertEqual(plug, node.find_attr("rename"))
                else:
                    self._assert_owned(plug, node, name)
                # a dynamic attr: the handle of its attribute
                self.assertTrue(vars(plug)["_attr1"].isAlive())
        self.assertEqual(node.kd.get(), 2.5)
        # a compound spec gives its parent plug, whose children are owned too
        vector = node.vk
        self.assertEqual(str(vector), "held.vk")
        for child in (vector.vkX, vector[1], vector.child(2)):
            self.assertIs(child.node, node)
        # a multi spec gives its array root, pre-sized
        array = node.arr
        self.assertEqual(list(array.get_logical_indices()), [0, 1, 2])
        self.assertIs(array[2].node, node)

    def test_an_existing_attr(self):
        node, _ = _owned_spec_scene()
        first = node << Float("k")
        first << 5.0
        # overwrite=True (the default): deleted and re-added, the value reset
        again = node << Float("k", overwrite=True)
        self._assert_owned(again, node, "k")
        self.assertEqual(again.get(), 0.0)
        again << 7.0
        # overwrite=False: the existing attr, its value kept
        kept = node << Float("k", overwrite=False)
        self._assert_owned(kept, node, "k")
        self.assertEqual(kept.get(), 7.0)
        # a static attr, not overwritten: named as the spec names it, with no
        # handle of its attribute
        static = node << Float("tx", overwrite=False)
        self.assertIs(static.node, node)
        self.assertEqual(str.__str__(static), "held.tx")
        self.assertEqual(str(static), "held.translateX")
        self.assertIsNone(vars(static)["_attr1"])
        self.assertTrue(_same_plug(static, node.tx))
        # a real attr named like a component alias: the attr the name names, as
        # its str buffer does (``node.pnts`` is the canonical controlPoints)
        cmds.polyCube(name="box", constructionHistory=False)
        shape = Node("boxShape")
        pnts = shape << Float("pnts", overwrite=False)
        self.assertIs(pnts.node, shape)
        self.assertEqual(str.__str__(pnts), "boxShape.pnts")
        self.assertEqual(str(pnts), "boxShape.pnts")

    def test_output_note_and_chain(self):
        from rig import lock
        from rig.spec import Note

        node, _ = _owned_spec_scene()
        out = node >> Float("outk")
        self._assert_owned(out, node, "outk")
        self.assertFalse(cmds.addAttr("held.outk", query=True, writable=True))
        notes = node << Note("hello")
        self.assertIs(notes.node, node)
        self.assertEqual(cmds.getAttr("held.notes"), "hello")
        chained = node << Float("ck") << 3.0 << lock
        self.assertIs(chained.node, node)
        self.assertEqual(cmds.getAttr("held.ck"), 3.0)
        self.assertTrue(cmds.getAttr("held.ck", lock=True))

    def test_a_plug_target(self):
        node, other = _owned_spec_scene()
        # a plug of a node object: owned by that node object
        self._assert_owned(node.tx << Float("pk"), node, "pk")
        # a plug built from a name casts its node once, which owns the new plug
        named = Plug("held.tx")
        added = named << Float("pk2")
        self.assertIs(added.node, named.node)
        self.assertIs(vars(added)["_node"], named.node)
        self.assertEqual(str.__str__(added), "held.pk2")
        # an attr of the shape read through its transform: the shape's
        shape_attr = node.localPositionX
        on_shape = shape_attr << Float("lk")
        self.assertIs(on_shape.node, shape_attr.node)
        self.assertEqual(str(on_shape), "heldShape.lk")
        # a typed Attribute target (spec.apply): its owner too
        typed = node.find_attr("ty")
        self.assertIs(Float("tk").apply(typed).node, node)
        # the clone of plug >> node is owned by the node
        clone = node.pk >> other
        self.assertIs(clone.node, other)
        self.assertEqual(str(clone), "other.pk")

    def test_a_list_target(self):
        node, other = _owned_spec_scene()
        added = List([node, other.tx]) << Float("lst")
        self.assertEqual([str(p) for p in added], ["held.lst", "other.lst"])
        self.assertIs(added[0].node, node)
        self.assertIs(added[1].node, other)

    def test_a_leading_underscore(self):
        node, _ = _owned_spec_scene()
        parked = node << Float("__parked__")
        self._assert_owned(parked, node, "__parked__")
        node.__parked__ << 4.0
        self.assertEqual(cmds.getAttr("held.__parked__"), 4.0)
        self.assertEqual(parked.get(), 4.0)
        self.assertIs(node.__parked__.node, node)

    def test_the_string_path(self):
        # a name findPlug does not resolve (an alias of an element, a component
        # name) and a node that is not a live node object take the string path:
        # a plug with no owner, checked through the node's handle, resolved as
        # Plug(name) resolves it
        from rig.spec import _base as spec_base

        cmds.createNode("plusMinusAverage", name="pma")
        cmds.setAttr("pma.input1D[0]", 1)
        cmds.aliasAttr("smile", "pma.input1D[0]")
        cmds.polyCube(name="box", constructionHistory=False)
        pma, shape = Node("pma"), Node("boxShape")
        for label, node, name, same in (
            ("an alias", pma, "smile", lambda: pma.input1D[0]),
            ("a component", shape, "vtx[1]", lambda: Plug("boxShape.vtx[1]")),
        ):
            with self.subTest(label):
                found = spec_base._plug_of(str(node), name, node)
                self.assertIs(type(found), Plug)
                self.assertIsNone(vars(found)["_node"])
                self.assertEqual(
                    vars(found)["_handle1"].hashCode(), vars(node)["_objhandle1"].hashCode()
                )
                self.assertTrue(_same_plug(found, same()))
        cmds.createNode("transform", name="held")
        cmds.addAttr("held", longName="k", attributeType="double")
        for node in (None, object.__new__(type(Node("held")))):
            with self.subTest(node=type(node).__name__):
                named = spec_base._plug_of("held", "k", node)
                self.assertIs(type(named), Plug)
                self.assertIsNone(vars(named)["_node"])
                self.assertEqual(str(named), "held.k")

    def test_a_node_whose_short_name_is_not_unique(self):
        # the str buffer cmds reads names the node's shortest unique path, not
        # the MPlug's name (``X.k``, which names both X)
        cmds.createNode("transform", name="A")
        cmds.createNode("transform", name="X", parent="A")
        cmds.createNode("transform", name="B")
        cmds.createNode("transform", name="X", parent="B")
        ax = Node("|A|X")
        plug = ax << Float("k")
        self.assertIs(plug.node, ax)
        self.assertEqual(str.__str__(plug), "A|X.k")
        self.assertEqual(str(plug), "A|X.k")
        self.assertEqual(cmds.getAttr(plug), 0.0)
        plug << 3.0
        self.assertEqual(cmds.getAttr("|A|X.k"), 3.0)
        self.assertFalse(cmds.attributeQuery("k", node="|B|X", exists=True))

    def test_a_namespace_and_a_container(self):
        cmds.namespace(add="ns")
        spaced = Node(cmds.createNode("transform", name="ns:held"))
        self._assert_owned(spaced << Float("nk"), spaced, "nk")
        with container("box") as box:
            inside = Node.create("transform", name="inside")
            self._assert_owned(inside << Float("ck"), inside, "ck")
        self.assertIs((box << Float("bk")).node, box)
        self.assertEqual(str(box.bk), "box.bk")

    def test_an_instanced_target(self):
        # E3: named through the path the node object holds
        from rig.spec import Vector

        s1, s2 = _instanced()
        plug = s2 << Float("k")
        self._assert_owned(plug, s2, "k")
        self.assertEqual(str.__str__(plug), "T2|S.k")
        self.assertEqual(str(plug), "T2|S.k")
        self.assertEqual(cmds.getAttr(plug), 0.0)
        self.assertEqual(str(s1 << Float("k1")), "T1|S.k1")
        on_plug = s2.v << Float("pk")
        self.assertIs(on_plug.node, s2)
        self.assertEqual(str(on_plug), "T2|S.pk")
        vector = s2 << Vector("iv")
        self.assertEqual(str(vector), "T2|S.iv")
        self.assertEqual(str(vector.ivX), "T2|S.ivX")
        # a removed instance (a stale path): the node's live path names it
        cmds.undoInfo(state=True, infinity=True)
        try:
            cmds.parent("|T2|S", removeObject=True, shape=True)
            again = s2 << Float("k2")
            self.assertIs(again.node, s2)
            self.assertEqual(str.__str__(again), f"{s2}.k2")
            self.assertEqual(cmds.getAttr(again), 0.0)
        finally:
            cmds.undoInfo(state=False)

    def test_a_deleted_dynamic_attr(self):
        # E7: the held spec plug raises once the delete leaves the undo queue
        node, _ = _owned_spec_scene()
        cmds.undoInfo(state=True, infinity=True)
        try:
            plug = node << Float("gone")
            cmds.deleteAttr("held.gone")
            cmds.flushUndo()
            for label, op in (
                ("str", str),
                ("get", lambda p: p.get()),
                ("<<", lambda p: p << 1.0),
                ("+", lambda p: p + 1),
            ):
                with self.subTest(label):
                    with self.assertRaisesRegex(RuntimeError, r"^held\.gone already deleted!$"):
                        op(plug)
            # re-added (another type): a new plug, the held one still raises
            readded = node << Float("gone", at="long")
            self._assert_owned(readded, node, "gone")
            with self.assertRaisesRegex(RuntimeError, r"^held\.gone already deleted!$"):
                plug.get()
        finally:
            cmds.undoInfo(state=False)

    def test_an_undone_add_and_a_deleted_node(self):
        # E1
        node, _ = _owned_spec_scene()
        cmds.undoInfo(state=True, infinity=True)
        try:
            plug = node << Float("undone")
            cmds.undo()
            self.assertFalse(cmds.objExists("held.undone"))
            with self.assertRaisesRegex(ValueError, r"held\.undone"):
                plug.get()
            cmds.redo()
            self.assertIs(plug.node, node)
            self.assertEqual(plug.get(), 0.0)
            plug << 2.0
            self.assertEqual(cmds.getAttr("held.undone"), 2.0)
            # the node deleted to the undo queue, then undone
            kept = node << Float("kept")
            cmds.delete("held")
            with self.assertRaisesRegex(RuntimeError, r"^held already deleted!$"):
                kept.get()
            with self.assertRaisesRegex(RuntimeError, r"^held already deleted!$"):
                node << Float("more")
            # a node that took the name is never reached
            cmds.createNode("transform", name="held")
            with self.assertRaisesRegex(RuntimeError, r"^held already deleted!$"):
                str(kept)
            cmds.undo()
            cmds.undo()
            self.assertEqual(kept.get(), 0.0)
            self._assert_owned(node << Float("more"), node, "more")
            # a rename: the plug follows its node
            cmds.rename("held", "renamed")
            self.assertEqual(str(kept), "renamed.kept")
            kept << 5.0
            self.assertEqual(cmds.getAttr("renamed.kept"), 5.0)
        finally:
            cmds.undoInfo(state=False)

    def test_a_freed_node(self):
        # E2: a new scene, a file open that reuses the names, a reference unload
        folder = tempfile.mkdtemp(prefix="rig_r4a_spec_")
        path = os.path.join(folder, "held.ma").replace("\\", "/")
        try:
            _owned_spec_scene()
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
            for label, free in (
                ("a new scene", lambda: cmds.file(new=True, force=True)),
                ("a file open that reuses the names", lambda: cmds.file(path, open=True, force=True)),
            ):
                with self.subTest(label):
                    cmds.file(path, open=True, force=True)
                    node = Node("held")
                    plug = node << Float("k")
                    free()
                    if not cmds.objExists("held"):
                        cmds.createNode("transform", name="held")  # same-name reuse
                    before = _scene()
                    with self.assertRaisesRegex(RuntimeError, _FREED):
                        node << Float("k2")
                    for op in (str, lambda p: p.get(), lambda p: p << 1.0):
                        with self.assertRaisesRegex(RuntimeError, _FREED):
                            op(plug)
                    self.assertIs(plug.node, node)
                    self.assertFalse(cmds.attributeQuery("k2", node="held", exists=True))
                    self.assertEqual(_new(before), [])
            # a reference unload
            cmds.file(new=True, force=True)
            cmds.file(path, reference=True, namespace="ref")
            node = Node("ref:held")
            plug = node << Float("k")
            self.assertEqual(str(plug), "ref:held.k")
            cmds.file(unloadReference=cmds.referenceQuery(path, referenceNode=True))
            with self.assertRaisesRegex(RuntimeError, _FREED):
                node << Float("k2")
            with self.assertRaisesRegex(RuntimeError, _FREED):
                plug.get()
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)

    def test_a_memoized_spec_plug(self):
        # E9: a memo entry that returned a spec's plug keeps its node's handle
        _owned_spec_scene()
        calls = []
        count = len(memoize_module._ALL_MEMOIZED)

        @memoize
        def spec_plug(i):
            calls.append(i)
            return Node("held") << Float("memo", overwrite=False)

        added = memoize_module._ALL_MEMOIZED[count:]
        self.addCleanup(
            lambda: [memoize_module._ALL_MEMOIZED.remove(w) for w in added
                     if w in memoize_module._ALL_MEMOIZED]
        )
        first = spec_plug(0)
        handles = []
        memoize_module._collect_handles(first, handles)
        self.assertEqual(len(handles), 1)
        self.assertEqual(handles[0].hashCode(), vars(first.node)["_objhandle1"].hashCode())
        self.assertIs(spec_plug(0), first)
        self.assertEqual(len(calls), 1)
        # a new scene frees the node: the entry is dropped, never returned
        cmds.file(new=True, force=True)
        cmds.createNode("transform", name="held")
        again = spec_plug(0)
        self.assertEqual(len(calls), 2)
        self.assertIsNot(again, first)
        self.assertEqual(again.get(), 0.0)


def _names_pynode(path):
    """The (line, token) pairs of a .py file whose token mentions PyNode (any case)."""
    import io
    import tokenize

    with open(path, encoding="utf-8") as fh:
        source = fh.read()
    return [
        (tok.start[0], tok.string)
        for tok in tokenize.generate_tokens(io.StringIO(source).readline)
        if "pynode" in tok.string.lower()
    ]


def _mplug(name):
    from maya.api import OpenMaya

    sel = OpenMaya.MSelectionList()
    sel.add(name)
    return sel.getPlug(0)


class TestNodeOnly(_SceneCase):
    """``Node`` is the only node factory: ``PyNode`` was removed (step R2). The
    typed cast is the private ``nodetypes._base._cast``; ``Node.create`` runs a
    registered type's typed create (D13 opt-outs kept) and ``container.createNode``
    for any other type; ``Node.find_all`` is the former ``PyNode.find_all``."""

    def setUp(self):
        super().setUp()
        from rig.nodetypes import _base

        self._registered = dict(_base._NODE_CLASS_DICT)
        cmds.select(clear=True)

    def tearDown(self):
        from rig.nodetypes import _base

        _base._NODE_CLASS_DICT.clear()
        _base._NODE_CLASS_DICT.update(self._registered)
        _base._CLASS_BY_TYPE.clear()
        _base._CASTABLE_TYPES.clear()
        set_options(skip_selection=True)
        super().tearDown()

    # -- the name is gone -- #

    def test_pynode_is_gone(self):
        import importlib
        import sys

        import rig
        import rig.nodetypes
        from rig._internal import container as container_module
        from rig.nodetypes import _base

        for module in ("rig", "rig.nodetypes", "rig.nodetypes._base"):
            with self.subTest(module=module):
                with self.assertRaises(ImportError):
                    exec(f"from {module} import PyNode", {})
                self.assertFalse(hasattr(importlib.import_module(module), "PyNode"))
        with self.assertRaises(AttributeError):
            rig.PyNode
        with self.assertRaises(AttributeError):
            rig.nodetypes.PyNode
        with self.assertRaises(AttributeError):
            _base.PyNode
        # no loaded rig module holds the name, nor the private leftovers of it
        leftovers = ("PyNode", "_PYNODE_CLASS", "_PYNODE_CREATE_HOOK", "_pynode_legacy_tail",
                     "_pynode_create")
        holders = sorted(
            (name, left)
            for name, module in list(sys.modules.items())
            if (name == "rig" or name.startswith("rig.")) and module is not None
            and not name.startswith("rig._tests")
            for left in leftovers
            if left in vars(module)
        )
        self.assertEqual(holders, [])
        self.assertFalse(hasattr(container_module, "_pynode_create"))
        # the cast core is private: no public module exports it
        self.assertFalse(hasattr(rig, "_cast"))
        self.assertFalse(hasattr(rig.nodetypes, "_cast"))
        self.assertTrue(callable(_base._cast))

    def test_only_node_is_exported(self):
        import rig
        import rig.nodetypes

        self.assertIn("Node", rig.__all__)
        self.assertNotIn("PyNode", rig.__all__)
        for module in ("rig", "rig.nodetypes"):
            with self.subTest(module=module):
                namespace = {}
                exec(f"from {module} import *", namespace)
                self.assertIs(namespace["Node"], rig.Node)
                self.assertNotIn("PyNode", namespace)
                self.assertNotIn("_cast", namespace)
        self.assertIs(rig.nodetypes.Node, rig.Node)

    def test_no_package_module_names_pynode(self):
        """No .py file of the package outside the tests mentions PyNode: no
        name, string, docstring or comment (the .md docs are the docs step's)."""
        import rig

        root = os.path.dirname(os.path.abspath(rig.__file__))
        found = []
        for folder, subfolders, files in os.walk(root):
            subfolders[:] = [d for d in subfolders if d not in ("_tests", "__pycache__")]
            for file in files:
                if file.endswith(".py"):
                    path = os.path.join(folder, file)
                    found += [
                        (os.path.relpath(path, root), line, text)
                        for line, text in _names_pynode(path)
                    ]
        self.assertEqual(found, [])

    # -- Node(x): the full input table -- #

    def test_node_input_table(self):
        from maya.api import OpenMaya
        from rig.nodetypes import DGNode, Geometry, Joint, Mesh, Transform

        cmds.addAttr("t", longName="myDyn", attributeType="double")
        cube = cmds.polyCube(name="cube", ch=False)[0]
        cmds.createNode("transform", name="T1")
        cmds.createNode("transform", name="T2")
        cmds.createNode("locator", name="S", parent="T1")
        cmds.parent("|T1|S", "|T2", shape=True, addObject=True)
        cmds.namespace(add="ns")
        cmds.createNode("transform", name="ns:n")
        cmds.createNode("joint", name="jnt")
        cmds.createNode("multiplyDivide", name="md")
        node  = self.t
        uuid  = cmds.ls("t", uuid=True)[0]
        sel   = OpenMaya.MSelectionList()
        sel.add("t")
        sel.add("|T2|S")
        mobj  = sel.getDependNode(0)
        mdag  = sel.getDagPath(1)
        mplug = _mplug("t.tx")
        with container("box") as ctn:
            Node.create("transform", name="inside")
        # a node object is itself; an attribute or plug the node object it holds
        for label, value in (
            ("node object", node),
            ("plug", node.tx),
            ("compound child plug", node.t[1]),
            ("dynamic attr plug", node.myDyn),
            ("typed attribute", node.find_attr("tx")),
            ("typed child", node.find_attr("t").child(0)),
        ):
            with self.subTest(value=label):
                self.assertIs(Node(value), node)
        self.assertIs(Node(ctn), ctn)
        md = Node("md")
        self.assertIs(Node(md), md)
        # everything else is the typed node it names (a new node object)
        for label, value, cls, name in (
            ("name-built Plug", Plug("t.ty"), Transform, "t"),
            ("Attribute(str)", Attribute("t.tz"), Transform, "t"),
            ("component plug", Node(cube).vtx[1], Mesh, "cubeShape"),
            ("MPlug", mplug, Transform, "t"),
            ("MObject", mobj, Transform, "t"),
            ("MDagPath of an instance", mdag, Geometry, "T2|S"),
            ("name", "t", Transform, "t"),
            ("long name", "|t", Transform, "t"),
            ("uuid", uuid, Transform, "t"),
            ("joint", "jnt", Joint, "jnt"),
            ("DG node", "md", DGNode, "md"),
            ("instance path", "|T1|S", Geometry, "T1|S"),
            ("dotted", "t.tx", Transform, "t"),
            ("dotted long name", "t.translateX", Transform, "t"),
            ("dotted compound", "t.translate", Transform, "t"),
            ("dotted instance path", "|T2|S.v", Geometry, "T2|S"),
            ("dotted namespace", "ns:n.tx", Transform, "ns:n"),
            ("dotted component", "cube.vtx[1]", Transform, "cube"),
            ("dotted, empty attr", "t.", Transform, "t"),
            ("keyword", None, Transform, "t"),
        ):
            with self.subTest(value=label):
                result = Node(obj="t") if value is None else Node(value)
                self.assertIs(type(result), cls)
                self.assertEqual(str(result), name)
                self.assertIsInstance(result, Node)
        # what names no node raises the cast's TypeError; a non-name its ValueError
        for label, value in (("missing", "nope"), ("missing, dotted", "nope.tx"),
                             ("empty", ""), ("no node before the dot", ".tx")):
            with self.subTest(error=label):
                with self.assertRaisesRegex(TypeError, "No object matches name"):
                    Node(value)
        for label, value in (("int", 3), ("None", None), ("float", 1.5), ("list", [node]),
                             ("bytes", b"t"), ("faces", Node(cube).f[0:2])):
            with self.subTest(error=label):
                with self.assertRaisesRegex(ValueError, "is not a str, MObject, MDagPath, or MPlug"):
                    Node(value)
        for label, call in (("no argument", lambda: Node()), ("two", lambda: Node("t", "u")),
                            ("keyword", lambda: Node("t", k=1))):
            with self.subTest(error=label):
                with self.assertRaises(TypeError):
                    call()
        # the attribute a dotted name names is Attribute(...) (typed) or Plug(...) (DSL)
        self.assertIs(type(Attribute("t.tx")), Attribute)
        self.assertEqual(str(Attribute("t.tx")), "t.translateX")
        self.assertIs(type(Plug("t.tx")), Plug)
        self.assertEqual(str(Plug("t.tx")), "t.translateX")

    def test_node_and_the_cast_core_agree_on_nodes(self):
        from rig.nodetypes import _base

        cube = cmds.polyCube(name="cube", ch=False)[0]
        for value in ("t", "|t", cube, "cubeShape", cmds.ls("t", uuid=True)[0], _mplug("t.tx").node()):
            with self.subTest(value=str(value)):
                self.assertIs(type(Node(value)), type(_base._cast(value)))
                self.assertEqual(str(Node(value)), str(_base._cast(value)))
        # the core keeps the attribute branch the package relies on
        for value in ("t.tx", _mplug("t.tx")):
            with self.subTest(attr=str(value)):
                self.assertIs(type(_base._cast(value)), Attribute)
                self.assertIs(type(Node(value)), type(self.t))
        # a plugs=True connection list still gives attributes
        self.u.tx << self.t.tx
        self.assertEqual([type(x) for x in self.u.find_attr("tx").list_connections(plugs=True)],
                         [type(Attribute("t.tx"))])
        self.assertEqual([str(x) for x in self.u.list_connections(plugs=True)], ["t.translateX"])

    # -- Node.create -- #

    def test_node_create_runs_the_typed_create_of_a_registered_type(self):
        from rig.nodetypes import (
            BlendShape, DGNode, DisplayLayer, Joint, ObjectSet, ShadingEngine,
            SkinCluster, Transform,
        )

        joint = Node.create("joint", name="j")
        self.assertIs(type(joint), Joint)
        # a scene registry made by its own command, not a bare createNode
        sg = Node.create("shadingEngine", name="sg")
        self.assertIs(type(sg), ShadingEngine)
        self.assertIn("renderPartition", cmds.listConnections("sg.partition") or [])
        layer = Node.create("displayLayer", name="L")
        self.assertIs(type(layer), DisplayLayer)
        self.assertIn("layerManager", cmds.listConnections("L.identification") or [])
        self.assertIs(type(Node.create("objectSet", name="S")), ObjectSet)
        # positional arguments reach the typed create
        base   = cmds.polyCube(name="base", ch=False)[0]
        target = cmds.polyCube(name="target", ch=False)[0]
        self.assertIs(type(Node.create("blendShape", target, base)), BlendShape)
        skinned = cmds.polyCube(name="skinned", ch=False)[0]
        self.assertIs(type(Node.create("skinCluster", skinned, joint)), SkinCluster)

        # a user class: its own create, its custom type
        class _Meta(DGNode):
            NATIVE_NODE_TYPE = "network"
            CUSTOM_NODE_TYPE = "r2NodeCreateMeta"

        meta = Node.create("r2NodeCreateMeta", name="meta")
        self.assertIs(type(meta), _Meta)
        self.assertEqual(cmds.nodeType("meta"), "network")
        self.assertIs(type(Node("meta")), _Meta)
        self.assertIs(type(Node.create("transform", name="x")), Transform)

    def test_node_create_of_another_type_is_create_node(self):
        from rig.nodetypes import DGNode

        md = Node.create("multiplyDivide", name="md")
        self.assertIs(type(md), DGNode)
        self.assertEqual(str(md), "md")
        # the GC tag of a utility type, as container.createNode gives it
        self.assertTrue(cmds.attributeQuery("__rig__", node="md", exists=True))
        # no positional argument after a type with no class, and nothing is made
        before = _scene()
        with self.assertRaisesRegex(TypeError, "takes keyword arguments only"):
            Node.create("multiplyDivide", "x")
        self.assertEqual(_new(before), [])

    def test_node_create_joins_a_scope_with_the_d13_opt_outs(self):
        with container("box") as box:
            with container("inner"):
                t = Node.create("transform", name="t")
                j = Node.create("joint", name="j")
                m = Node.create("multiplyDivide", name="m")
                s = Node.create("objectSet", name="S")
                layer = Node.create("displayLayer", name="L")
                kept = Node.create("objectSet", name="K", container=True)
                loose = Node.create("transform", name="loose", container=False)
        self.assertEqual([str(x) for x in (t, j, m, s, layer, kept, loose)],
                         ["inner_t", "inner_j", "inner_m", "S", "L", "K", "inner_loose"])
        members = sorted(cmds.container(str(box), query=True, nodeList=True) or [])
        self.assertEqual(members, ["K", "inner_j", "inner_m", "inner_t"])
        self.assertEqual(cmds.ls(selection=True), [])

    def test_node_create_does_not_select(self):
        from rig.nodetypes import Transform

        for node_type in ("transform", "joint", "choice", "objectSet", "multiplyDivide"):
            with self.subTest(node_type=node_type):
                cmds.select(clear=True)
                Node.create(node_type, name=f"n_{node_type}")
                self.assertEqual(cmds.ls(selection=True), [])
        # the caller's flag wins, and the option is read
        Node.create("transform", name="chosen", skipSelect=False)
        self.assertEqual(cmds.ls(selection=True), ["chosen"])
        set_options(skip_selection=False)
        Node.create("transform", name="selected")
        self.assertEqual(cmds.ls(selection=True), ["selected"])
        set_options(skip_selection=True)
        # a typed create outside a scope still selects, as before (D13)
        Transform.create(name="typed")
        self.assertEqual(cmds.ls(selection=True), ["typed"])

    def test_node_create_on_a_container_is_the_root_factory(self):
        from rig import Container

        with container("box") as box:
            made = Container.create("transform", name="c")
        self.assertIs(type(made), type(self.t))
        self.assertIn(str(made), cmds.container(str(box), query=True, nodeList=True))

    # -- Node.find_all -- #

    def test_node_find_all(self):
        from rig.nodetypes import DGNode, Joint, Transform

        cmds.createNode("joint", name="j1")
        cmds.createNode("joint", name="j2")
        joints = Node.find_all("joint")
        self.assertEqual(sorted(str(x) for x in joints), ["j1", "j2"])
        self.assertTrue(all(type(x) is Joint for x in joints))
        exact = {str(x) for x in Node.find_all("transform")}
        self.assertIn("t", exact)
        self.assertNotIn("j1", exact)
        wide = {str(x) for x in Node.find_all("transform", exact_type=False)}
        self.assertTrue({"t", "u", "j1", "j2"} <= wide)

        class _Tagged(DGNode):
            NATIVE_NODE_TYPE = "network"
            CUSTOM_NODE_TYPE = "r2FindAllMeta"

        made = [_Tagged.create(name=f"meta{i}") for i in range(2)]
        cmds.createNode("network", name="plainNet")
        self.assertEqual(set(Node.find_all("r2FindAllMeta")), set(made))
        # a type no class is registered for lists the casts of its nodes
        cmds.createNode("multiplyDivide", name="md")
        found = Node.find_all("multiplyDivide")
        self.assertEqual([str(x) for x in found], ["md"])
        self.assertIs(type(found[0]), DGNode)
        self.assertEqual(Node.find_all("plusMinusAverage"), [])
        with self.assertRaisesRegex(ValueError, "'nosuchType' is not a Maya node type"):
            Node.find_all("nosuchType")
        # a node class keeps its own typed find_all
        self.assertEqual({str(x) for x in Joint.find_all()}, {"j1", "j2"})
        self.assertIs(Transform.find_all.__func__, DGNode.find_all.__func__)

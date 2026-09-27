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
                ("reflected + set", lambda: {1, 2} + PlugList([t.tx]), "{1, 2} + PlugList(["),
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

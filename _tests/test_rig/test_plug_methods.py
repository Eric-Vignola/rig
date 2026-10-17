"""Name clashes: plug methods (user decision 2026-10-16).

THE RULE, on a node or a plug: a bare name is the Maya attribute if one exists,
otherwise the method; with brackets it is the method. ``node.serialize`` is the
plug when the node has a Maya attr ``serialize`` (a ``_MethodPlug``: a real Plug
whose call runs the method), else the bound method; ``node.serialize()`` is the
method either way. A compound plug's child named like a plug method follows the
same rule (``plug.pair.get``). Assignment is strict: with the attr, ``x.m = v``
is ``x.m << v``; without it, it raises (no instance monkeypatching).

THE CALL-SITE CHECK (user decision): an access that is the callee of a call (the
caller's frame on a method load: 3.11 LOAD_METHOD; 3.12+ LOAD_ATTR with its
method bit, LOAD_SUPER_ATTR) returns the method without asking Maya; any other
access runs the full check (``hasAttribute``).

* `TestNodeMembers`: a hit and a miss for each member kind (a function, a
  classmethod, a staticmethod, a ``creates_or_updates``), class access, a short
  name, the attr's add / delete / rename and undo / redo, freed, deleted and
  half-built nodes, the multi-compound child that cannot be built (the bare name
  stays the method, the call works), the native clashes, subclass overrides and
  ``super()``.
* `TestAssignment`: strict assignment on nodes, plugs and bare Attributes.
* `TestPlugMembers`: a compound child named like a plug method; a scalar plug and
  an array root keep their methods; the API-edit class registration.
* `TestIntrospection`: what a method plug forwards of its method.
* `TestLists`: a broadcast where every element has the attr, where only some do
  (refused), and a bracketed call over a List.
* `TestAliases`: an alias counts as an attribute for the rule (user decision
  2026-10-16): an ``aliasAttr`` name and a blendShape target named like a method
  are the plug bare and the method with brackets, as are the alias's add, rename,
  removal and their undo / redo; a compound child's alias on a plug.
* `TestTheUnderscoreProbe`: a ``_`` probe on a plug of a node deleted to the undo
  queue raises AttributeError.
* `TestCallSiteForms`: the 18 call forms of the reference (``props\\critic\\
  tools\\sniff_port.py``) on the real descriptors, counting the full checks.
* `TestInternalAccesses`: rig's own code makes almost no instance access of a
  public node / plug method during a rail_spine build, and no full check.
"""

import functools
import inspect
import os
import sys
from collections import Counter
from unittest import mock

from maya import cmds

from rig import List, Node, get_options, set_options
from rig._internal import api_edits
from rig._internal import plug as _plug_module
from rig._internal.plug import Plug
from rig.nodetypes import _base
from rig.nodetypes._base import Attribute
from rig.nodetypes.transform import Transform
from rig._tests._base import MayaTestCase

# read by name, so that on a tree without plug methods each test fails on its
# own (the negative control) instead of the module failing to import
_ABSENT           = type("_Absent", (), {})
_MethodPlug       = getattr(_plug_module, "_MethodPlug", _ABSENT)
_MethodAttribute  = getattr(_base, "_MethodAttribute", _ABSENT)

def _is_plug_method(cls, name):
    """True if ``name`` on node class or plug class `cls` is a plug method. Re-pinned
    (user decision 2026-10-16, class data: ``rig.nodetypes._base.is_plug_method`` was
    deleted with ``spec._warn_shadowed``, its one user): read off the class dicts."""
    return isinstance(
        _base._class_attr(cls, name), _base._PLUG_DESCRIPTORS + (_base._ChildOrMethod,)
    )



def _is_plug(x):
    return isinstance(x, Attribute)


class _Case(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.t = Node.create("transform", name="T")


class TestNodeMembers(_Case):
    def test_a_function(self):
        t = self.t
        self.assertIs(type(t.serialize), type(t.get_parent))  # a bound method
        self.assertEqual(type(t.serialize()).__name__, "TransformData")
        cmds.addAttr("T", ln="serialize", at="bool")
        self.assertIs(type(t.serialize), _MethodPlug)
        self.assertEqual(str(t.serialize), "T.serialize")
        self.assertEqual(type(t.serialize()).__name__, "TransformData")
        t.serialize << True
        self.assertIs(t.serialize >> None, True)
        self.assertTrue(cmds.getAttr(t.serialize))

    def test_a_classmethod(self):
        t = self.t
        self.assertTrue(inspect.ismethod(t.exists))
        self.assertIs(t.exists.__self__, Transform)
        cmds.addAttr("T", ln="exists", at="double")
        self.assertIs(type(t.exists), _MethodPlug)
        self.assertTrue(t.exists("T"))  # the classmethod
        self.assertIsInstance(_base._class_attr(Transform, "create"), classmethod)

    def test_a_staticmethod(self):
        mesh = Node(cmds.polyCube(name="c", ch=False)[0]).get_children(type="mesh")[0]
        func = type(mesh).clear_all_paintable_attrs
        self.assertIs(mesh.clear_all_paintable_attrs, func)
        cmds.addAttr(str(mesh), ln="clear_all_paintable_attrs", at="double")
        self.assertIs(type(mesh.clear_all_paintable_attrs), _MethodPlug)
        self.assertIs(mesh.clear_all_paintable_attrs.__wrapped__, func)

    def test_a_creates_or_updates(self):
        t = self.t
        self.assertTrue(inspect.ismethod(t.load_hierarchy))
        cmds.addAttr("T", ln="load_hierarchy", at="double")
        self.assertIs(type(t.load_hierarchy), _MethodPlug)
        with self.assertRaises(TypeError):
            t.load_hierarchy("x.npz")  # the node form: refused, as before
        mesh = Node(cmds.polyCube(name="c", ch=False)[0]).get_children(type="mesh")[0]
        self.assertTrue(inspect.ismethod(mesh.load))
        cmds.addAttr(str(mesh), ln="load", at="double")
        self.assertIs(type(mesh.load), _MethodPlug)
        self.assertIs(mesh.load.__self__, mesh)

    def test_class_access_is_the_plain_member(self):
        cmds.addAttr("T", ln="serialize", at="bool")
        self.assertTrue(inspect.isfunction(Transform.serialize))
        self.assertEqual(Transform.serialize.__qualname__, "Transform.serialize")
        self.assertTrue(inspect.ismethod(Transform.create))
        self.assertIs(Transform.create.__self__, Transform)
        self.assertTrue(inspect.isfunction(type(self.t).ensure_valid))
        # the class dict entry still runs as the function
        self.assertEqual(type(Transform.__dict__["serialize"](self.t)).__name__, "TransformData")
        self.assertTrue(_is_plug_method(Transform, "serialize"))
        self.assertFalse(_is_plug_method(Transform, "name"))  # a property

    def test_a_short_name(self):
        cmds.addAttr("T", ln="renameWeight", sn="rename", at="double")
        self.assertIs(type(self.t.rename), _MethodPlug)
        self.assertEqual(str(self.t.rename), "T.renameWeight")
        self.t.rename("T2")
        self.assertEqual(str(self.t), "T2")

    def test_add_delete_rename_undo_redo(self):
        cmds.undoInfo(state=True, infinity=True)
        cmds.flushUndo()
        t = self.t
        cmds.addAttr("T", ln="serialize", at="bool")
        self.assertTrue(_is_plug(t.serialize))
        cmds.undo()
        self.assertFalse(_is_plug(t.serialize))
        cmds.redo()
        self.assertTrue(_is_plug(t.serialize))
        cmds.renameAttr("T.serialize", "label")
        self.assertFalse(_is_plug(t.serialize))
        cmds.renameAttr("T.label", "duplicate")
        self.assertTrue(_is_plug(t.duplicate))
        cmds.deleteAttr("T.duplicate")
        self.assertFalse(_is_plug(t.duplicate))

    def test_freed_deleted_and_half_built_nodes(self):
        """guard: a node that cannot answer gives the method, as before."""
        cmds.addAttr("T", ln="serialize", at="bool")
        a = Node.create("transform", name="A")
        cmds.addAttr("A", ln="serialize", at="bool")
        cmds.delete("A")
        self.assertTrue(inspect.ismethod(a.serialize))
        with self.assertRaises(RuntimeError):
            a.serialize()  # its own "already deleted!"
        half = object.__new__(Transform)
        self.assertTrue(inspect.ismethod(half.serialize))
        vars(half).update(vars(self.t))
        del vars(half)["_attr_dict"]
        self.assertTrue(inspect.ismethod(half.serialize))
        t = self.t
        cmds.file(new=True, force=True)
        self.assertTrue(inspect.ismethod(t.serialize))  # never asked: freed
        with self.assertRaises(RuntimeError):
            t.serialize()

    def test_a_multi_compound_child_named_like_a_method(self):
        # the critic's blocker: no plug can be built for the bare name (an
        # unindexed multi's child), so it is the method, and calls work
        cmds.addAttr("T", ln="items", at="compound", nc=2, multi=True)
        cmds.addAttr("T", ln="unlock", at="double", p="items")
        cmds.addAttr("T", ln="rename", at="double", p="items")
        t = self.t
        self.assertTrue(inspect.ismethod(t.unlock))
        self.assertTrue(inspect.ismethod(getattr(t, "rename")))
        self.assertIs(t.unlock(), t)
        t.rename("T3")
        self.assertEqual(str(t), "T3")
        self.assertEqual(str(t.items[0].unlock), "T3.items[0].unlock")

    def test_native_clashes(self):
        layer = Node.create("animLayer", name="AL")
        self.assertIs(type(layer.lock), _MethodPlug)
        layer.lock << True
        self.assertTrue(cmds.getAttr("AL.lock"))
        self.assertIs(layer.lock(), layer)
        layer.unlock()
        curve = Node(cmds.listRelatives(cmds.curve(p=[(0, 0, 0), (1, 0, 0)], d=1), shapes=True)[0])
        self.assertIs(type(curve.create), _MethodPlug)
        self.assertEqual(str(curve.create), f"{curve}.create")
        for node_type in ("subdiv", "meshVarGroup"):
            with self.subTest(node_type):
                node = Node(cmds.createNode(node_type))
                self.assertTrue(cmds.attributeQuery("create", node=str(node), exists=True))
                self.assertIs(type(node.create), _MethodPlug)
                self.assertTrue(inspect.ismethod(type(node).create))

    def test_subclass_overrides_and_super(self):
        calls = []

        class _MyTransform(Transform):
            def serialize(self, *args, **kwargs):
                calls.append("mine")
                return super().serialize(*args, **kwargs)

            def mine(self):
                return super().get_parent()

        self.assertTrue(_is_plug_method(_MyTransform, "serialize"))
        self.assertTrue(_is_plug_method(_MyTransform, "mine"))
        node = _MyTransform._wrap("T")
        self.assertEqual(type(node.serialize()).__name__, "TransformData")
        self.assertEqual(calls, ["mine"])
        cmds.addAttr("T", ln="serialize", at="bool")
        cmds.addAttr("T", ln="mine", at="bool")
        self.assertIs(type(node.serialize), _MethodPlug)
        self.assertIs(type(super(_MyTransform, node).serialize), _MethodPlug)
        self.assertIs(type(node.mine), _MethodPlug)
        self.assertIsNone(node.mine())
        node.serialize()
        self.assertEqual(calls, ["mine", "mine"])


class TestAssignment(_Case):
    def test_a_node_with_the_attr_connects_or_sets(self):
        t = self.t
        cmds.addAttr("T", ln="lock", at="bool")
        t.lock = True
        self.assertTrue(cmds.getAttr("T.lock"))
        src = Node.create("transform", name="S")
        cmds.addAttr("S", ln="lock", at="bool")
        t.lock = src.lock
        self.assertEqual(cmds.listConnections("T.lock", source=True, plugs=True), ["S.lock"])

    def test_a_node_without_the_attr_raises(self):
        t = self.t
        for value in (3, lambda: None, Node.create("transform", name="S").tx):
            with self.subTest(kind=type(value).__name__), self.assertRaisesRegex(
                AttributeError, "Transform.get_parent is a method and T has no Maya attr"
            ):
                t.get_parent = value
        self.assertNotIn("get_parent", vars(t))
        with self.assertRaises(AttributeError):
            with mock.patch.object(t, "get_parent", return_value=1):
                pass
        cmds.delete("T")
        with self.assertRaisesRegex(RuntimeError, "already deleted"):
            t.get_parent = 3

    def test_tests_patch_the_class_or_vars(self):
        t = self.t
        with mock.patch.object(Transform, "get_parent", return_value="P"):
            self.assertEqual(t.get_parent(), "P")
        self.assertIsNone(t.get_parent())
        self.assertTrue(_is_plug_method(Transform, "get_parent"))
        with mock.patch.dict(vars(t), get_parent=lambda: "Q"):
            self.assertEqual(t.get_parent(), "Q")
        self.assertIsNone(t.get_parent())

    def test_a_plug(self):
        t = self.t
        cmds.addAttr("T", ln="pair", at="compound", nc=2)
        cmds.addAttr("T", ln="get", at="double", p="pair")
        cmds.addAttr("T", ln="other", at="double", p="pair")
        t.pair.get = 4.0
        self.assertEqual(cmds.getAttr("T.pair.get"), 4.0)
        with self.assertRaisesRegex(AttributeError, "Plug.get is a method and 'T.translateX' has no child"):
            t.tx.get = 2
        # a sibling of that name is not reached either
        cmds.addAttr("T", ln="unlock", at="double")
        with self.assertRaisesRegex(AttributeError, "has no child attribute 'unlock'"):
            t.tx.unlock = 2
        self.assertEqual(cmds.getAttr("T.unlock"), 0.0)

    def test_a_bare_attribute(self):
        attr = self.t.find_attr("tx")
        with self.assertRaisesRegex(AttributeError, "Attribute.get is a method"):
            attr.get = 1
        self.assertNotIn("get", vars(attr))


class TestPlugMembers(_Case):
    def setUp(self):
        super().setUp()
        cmds.addAttr("T", ln="pair", at="compound", nc=2)
        cmds.addAttr("T", ln="get", at="double", p="pair")
        cmds.addAttr("T", ln="connect", at="double", p="pair")

    def test_a_compound_child_named_like_a_method(self):
        pair = self.t.pair
        self.assertIs(type(pair.get), _MethodPlug)
        self.assertEqual(str(pair.get), "T.get")
        self.assertEqual(list(pair.get()), [0.0, 0.0])  # the method of `pair`
        pair.get << 2.0
        self.assertEqual(pair.get >> None, 2.0)
        src = Node.create("transform", name="S")
        pair.connect << src.tx
        self.assertEqual(cmds.listConnections("T.connect", source=True, plugs=True), ["S.translateX"])
        self.assertTrue(inspect.ismethod(pair.child))
        self.assertTrue(inspect.ismethod(Attribute(pair).child))
        self.assertIs(type(self.t.find_attr("pair").get), _MethodAttribute)
        self.assertEqual(repr(self.t.find_attr("pair").get), 'Attribute("T.get")')

    def test_a_scalar_plug_and_an_array_root(self):
        self.assertTrue(inspect.ismethod(self.t.tx.get))
        cmds.addAttr("T", ln="items", at="compound", nc=1, multi=True)
        cmds.addAttr("T", ln="unlock", at="double", p="items")
        self.assertTrue(inspect.ismethod(self.t.items.unlock))  # the root: no child plug
        self.assertIs(type(self.t.items[0].unlock), _MethodPlug)  # an element: its child

    def test_class_access(self):
        self.assertTrue(inspect.isfunction(Plug.get))
        self.assertTrue(inspect.isfunction(Attribute.get_connected_attrs))
        self.assertTrue(_is_plug_method(Plug, "get"))

    def test_a_freed_node(self):
        """guard: a freed node's MPlug is never read."""
        pair = self.t.pair
        cmds.file(new=True, force=True)
        self.assertTrue(inspect.ismethod(pair.get))  # its MPlug is never read
        with self.assertRaisesRegex(RuntimeError, "already deleted"):
            pair.get()

    def test_the_api_edit_classes(self):
        api_edits._bind()
        self.assertIn(_MethodPlug, api_edits._API_CLASSES)
        self.assertIn(_MethodAttribute, api_edits._API_CLASSES)
        cmds.addAttr("T", ln="serialize", at="bool")
        self.assertIn(type(self.t.serialize), api_edits._API_CLASSES)
        self.assertIn(type(self.t.pair.get), api_edits._API_CLASSES)
        # the node identity constants hold the functions class access gives
        self.assertIs(api_edits._ENSURE, Transform.ensure_valid)


class TestIntrospection(_Case):
    def test_a_method_plug_forwards_its_method(self):
        cmds.addAttr("T", ln="serialize", at="bool")
        plug = self.t.serialize
        self.assertEqual(repr(plug), 'Plug("T.serialize")')
        self.assertEqual(plug.__name__, "serialize")
        self.assertEqual(plug.__qualname__, "Transform.serialize")
        self.assertIs(plug.__func__, Transform.serialize)
        self.assertIs(plug.__self__, self.t)
        self.assertEqual(plug.__doc__, Transform.serialize.__doc__)
        self.assertEqual(inspect.signature(plug), inspect.signature(Transform.serialize.__get__(self.t)))
        self.assertFalse(inspect.ismethod(plug))
        self.assertTrue(callable(plug))
        self.assertEqual(type(functools.partial(plug)()).__name__, "TransformData")
        self.assertEqual(plug, self.t.find_attr("serialize"))
        self.assertEqual(hash(plug), hash(Plug("T.serialize")))


class TestLists(_Case):
    def setUp(self):
        super().setUp()
        self.a = Node.create("transform", name="A")
        self.b = Node.create("transform", name="B")

    def test_every_element_has_the_attr(self):
        for name in ("A", "B"):
            cmds.addAttr(name, ln="serialize", at="bool")
        plugs = List([self.a, self.b]).serialize
        self.assertEqual([str(p) for p in plugs], ["A.serialize", "B.serialize"])
        plugs << True
        self.assertEqual(list(plugs >> None), [True, True])

    def test_only_some_elements_have_it(self):
        cmds.addAttr("A", ln="serialize", at="bool")
        with self.assertRaisesRegex(
            AttributeError,
            r"serialize is a method of each element, and a method does not broadcast over a "
            r"List .*element\(s\) \[0\] have a Maya attr 'serialize' \(their plugs\), \[1\] do not",
        ):
            List([self.a, self.b]).serialize

    def test_no_element_has_it_and_a_bracketed_call(self):
        """guard: a bracketed call over a List is refused with the loop, the
        attrs or not."""
        items = List([self.a, self.b])
        with self.assertRaisesRegex(TypeError, "a method does not broadcast over a List"):
            items.serialize()
        for name in ("A", "B"):
            cmds.addAttr(name, ln="serialize", at="bool")
        with self.assertRaisesRegex(TypeError, "a method does not broadcast over a List"):
            items.serialize()  # brackets: the method of each


class TestAliases(_Case):
    def test_an_alias_is_not_an_attribute_to_the_rule(self):
        """guard, re-pinned: an alias IS an attribute to the rule (user decision
        2026-10-16, "Aliases: ADOPT"); the id is kept."""
        cmds.addAttr("T", ln="w0", at="double")
        cmds.aliasAttr("rename", "T.w0")
        self.assertIs(type(self.t.rename), _MethodPlug)
        self.assertEqual(str(self.t.rename), "T.rename")
        self.t.rename << 3.0
        self.assertEqual(cmds.getAttr("T.w0"), 3.0)
        self.t.rename = 4.0  # '=' on the alias is '<<'
        self.assertEqual(cmds.getAttr("T.w0"), 4.0)
        self.t.rename("R")  # brackets: the method
        self.assertTrue(cmds.objExists("R"))
        self.assertEqual(str(Plug(self.t.find_attr("rename"))), "R.rename")

    def test_blendshape_targets_named_like_methods(self):
        base = cmds.polyCube(name="base", ch=False)[0]
        for name in ("delete", "duplicate"):
            cmds.polyCube(name=name, ch=False)
        bs = Node(cmds.blendShape("delete", "duplicate", base, name="bs")[0])
        self.assertIs(type(bs.delete), _MethodPlug)
        self.assertIs(type(bs.duplicate), _MethodPlug)
        self.assertEqual(str(bs.delete), "bs.delete")
        bs.delete << 0.5
        self.assertEqual(cmds.getAttr("bs.weight[0]"), 0.5)
        bs.duplicate = 0.25
        self.assertEqual(cmds.getAttr("bs.weight[1]"), 0.25)
        self.assertEqual(bs.delete >> None, 0.5)
        bs.delete()  # brackets: deletes the node
        self.assertFalse(cmds.objExists("bs"))

    def test_add_rename_remove_and_undo_redo(self):
        cmds.undoInfo(state=True, infinity=True)
        t = self.t
        cmds.addAttr("T", ln="w0", at="double")
        self.assertTrue(inspect.ismethod(t.lock))
        cmds.aliasAttr("lock", "T.w0")
        self.assertIs(type(t.lock), _MethodPlug)
        cmds.undo()  # the alias undone: the method again
        self.assertTrue(inspect.ismethod(t.lock))
        cmds.redo()
        self.assertIs(type(t.lock), _MethodPlug)
        cmds.aliasAttr("unlock", "T.lock")  # renamed: the new name is the plug
        self.assertTrue(inspect.ismethod(t.lock))
        self.assertIs(type(t.unlock), _MethodPlug)
        cmds.aliasAttr("T.unlock", remove=True)
        self.assertTrue(inspect.ismethod(t.unlock))
        self.assertIs(t.lock(), t)  # the methods still run
        self.assertIs(t.unlock(), t)

    def test_a_compound_childs_alias_on_a_plug(self):
        cmds.addAttr("T", ln="pair", at="compound", nc=2)
        cmds.addAttr("T", ln="pa", at="double", p="pair")
        cmds.addAttr("T", ln="pb", at="double", p="pair")
        cmds.aliasAttr("get", "T.pa")
        child = self.t.pair.get
        self.assertIs(type(child), _MethodPlug)
        self.assertEqual(child.get_name(), "pa")
        child << 2.0
        self.assertEqual(cmds.getAttr("T.pa"), 2.0)
        self.assertEqual(list(self.t.pair.get()), [2.0, 0.0])  # brackets: the method

    def test_a_node_without_aliases_reads_one_short_list(self):
        """the gate: a bare miss on a node with no alias reads the alias list
        once (empty), never findAlias."""
        calls = []

        class _Fn:
            def hasAttribute(self, name):  # noqa: N802
                calls.append("has")
                return False

            def getAliasList(self):  # noqa: N802
                calls.append("list")
                return ()

        node = _stand_in(Transform)
        vars(node)["_fn_set"] = _Fn()
        self.assertTrue(inspect.ismethod(node.serialize))
        self.assertEqual(calls, ["has", "list"])


class TestTheUnderscoreProbe(_Case):
    def test_a_node_deleted_to_the_undo_queue(self):
        cmds.undoInfo(state=True, infinity=True)
        owned   = self.t.tx
        unowned = Plug("T.ty")
        cmds.delete("T")
        for plug in (owned, unowned):
            with self.subTest(plug=str.__str__(plug)):
                self.assertFalse(hasattr(plug, "__array_interface__"))
                with self.assertRaises(AttributeError):
                    plug._probe


# --- the call-site check over the 18 call forms --------------------------------


class _Handle:
    def isValid(self):  # noqa: N802
        return True


class _CountingFn:
    """A stand-in fn set: every ``hasAttribute`` is a full check."""

    def __init__(self):
        self.checks = 0

    def hasAttribute(self, name):  # noqa: N802
        self.checks += 1
        return False

    def getAliasList(self):  # noqa: N802
        # a miss reads the aliases too (an alias counts as an attribute, user
        # decision 2026-10-16): none here
        return ()


def _stand_in(cls):
    node = object.__new__(cls)
    vars(node).update(_attr_dict={}, _objhandle=_Handle(), _fn_set=_CountingFn())
    return node


def _forms_classes():
    """A class whose method ``m`` is a plug method, and a subclass whose ``m``
    calls ``super().m()``."""

    class _N:
        def m(self, *args, **kwargs):
            return "ran"

        m = _base._PlugMethod(m, "m")

    class _Sub(_N):
        def m(self, *args, **kwargs):
            return super().m(*args, **kwargs) if args else super().m()

        m = _base._PlugMethod(m, "m")

    return _N, _Sub


def _call_forms(n, s):
    """(label, a call of the form, True if it must run the full check)"""
    ns = {}
    exec(
        "def big(n):\n"
        + "".join("    a%d = n.x%d if 0 else 0\n" % (i, i) for i in range(300))
        + "    return n.m()\n",
        ns,
    )
    exec(
        "def big2(n):\n"
        + "".join("    a%d = n.x%d if 0 else 0\n" % (i, i) for i in range(300))
        + "    return n.m\n",
        ns,
    )

    def gen():
        yield n.m()

    def traced():
        def tracer(frame, event, arg):
            return tracer

        old = sys.gettrace()
        sys.settrace(tracer)
        try:
            return n.m()
        finally:
            sys.settrace(old)

    def monitored():
        if sys.version_info < (3, 12):
            return n.m()
        mon  = sys.monitoring
        tool = 4
        mon.use_tool_id(tool, "rig-test")
        mon.register_callback(tool, mon.events.INSTRUCTION, lambda code, off: None)
        mon.set_events(tool, mon.events.INSTRUCTION)
        try:
            return n.m()
        finally:
            mon.set_events(tool, 0)
            mon.free_tool_id(tool)

    return [
        ("n.m()", lambda: n.m(), False),
        ("n.m(1, k=2)", lambda: n.m(1, k=2), False),
        ("(n.m)()", lambda: (n.m)(), False),
        ("n.m (bare)", lambda: n.m, True),
        ("f = n.m; f()", lambda: (lambda f: f())(n.m), True),
        ("getattr(n, 'm')()", lambda: getattr(n, "m")(), True),
        ("n.m(*a)", lambda: n.m(*(1,)), True),
        ("n.m(**kw)", lambda: n.m(**{"k": 1}), True),
        ("super().m() inside Sub.m", lambda: s.m(), False),
        ("eval('n.m()')", lambda: eval("n.m()", {"n": n}), False),
        ("generator: yield n.m()", lambda: next(gen()), False),
        ("EXTENDED_ARG: n.m()", lambda: ns["big"](n), False),
        ("EXTENDED_ARG: n.m (bare)", lambda: ns["big2"](n), True),
        ("under settrace: n.m()", traced, False),
        ("under sys.monitoring: n.m()", monitored, False),
        ("[x.m() for x in (n,)]", lambda: [x.m() for x in (n,)], False),
        ("functools.partial(n.m)()", lambda: functools.partial(n.m)(), True),
        ("n.m.__call__()", lambda: n.m.__call__(), True),
    ]


class TestCallSiteForms(MayaTestCase):
    def test_the_python_version_has_a_branch(self):
        table = _base._METHOD_LOAD
        if sys.version_info[:2] == (3, 11):
            import dis

            self.assertEqual(table[dis.opmap["LOAD_METHOD"]], 1)
            self.assertEqual(sum(1 for k in table if k), 1)

    def test_the_18_forms(self):
        n_cls, sub_cls = _forms_classes()
        n, s = _stand_in(n_cls), _stand_in(sub_cls)
        for label, form, want_checked in _call_forms(n, s):
            with self.subTest(label):
                n._fn_set.checks = s._fn_set.checks = 0
                form()
                checked = (n._fn_set.checks + s._fn_set.checks) > 0
                self.assertEqual(checked, want_checked)

    def test_each_descriptor_kind(self):
        class _K:
            def f(self):
                return "f"

            def c(cls):
                return cls

            def s():
                return "s"

            f = _base._PlugMethod(f, "f")
            c = _base._PlugClassMethod(c, "c")
            s = _base._PlugStaticMethod(s, "s")

        k = _stand_in(_K)
        for name, call, bare in (
            ("f", lambda: k.f(), lambda: k.f),
            ("c", lambda: k.c(), lambda: k.c),
            ("s", lambda: k.s(), lambda: k.s),
        ):
            with self.subTest(name):
                k._fn_set.checks = 0
                call()
                self.assertEqual(k._fn_set.checks, 0)
                bare()
                self.assertEqual(k._fn_set.checks, 1)

    def test_a_real_node(self):
        t = Node.create("transform", name="T")
        cmds.addAttr(str(t), ln="serialize", at="bool")
        hook = mock.Mock(wraps=_base._METHOD_PLUG_HOOK)
        with mock.patch.object(_base, "_METHOD_PLUG_HOOK", hook):
            t.serialize()
            [x.serialize() for x in (t,)]
            self.assertEqual(hook.call_count, 0)
            t.serialize
            getattr(t, "serialize")
            self.assertEqual(hook.call_count, 2)


# --- rig's own accesses during a build --------------------------------------------


def _count_accesses(test):
    """Count every instance access of a plug-method descriptor while the
    returned counter is on: {("call" | "check", name, "file:line"): n}."""
    import rig

    root    = os.path.dirname(rig.__file__)
    counts  = Counter()
    state   = {"on": False}
    kinds   = (
        (_base._PlugMethod, "name"),
        (_base._PlugDescriptor, "name"),
        (_base._PlugClassMethod, "_name"),
        (_base._PlugStaticMethod, "_name"),
        (_base._ChildOrMethod, "name"),
    )
    for cls, attr in kinds:
        original = cls.__dict__["__get__"]

        def counting(self, inst, owner=None, _original=original, _attr=attr):
            if inst is not None and state["on"]:
                frame = sys._getframe(1)
                path  = frame.f_code.co_filename
                kind  = "call" if _base._called(2) else "check"
                where = os.path.relpath(path, root) if path.startswith(root) else "user"
                counts[(kind, getattr(self, _attr), f"{where}:{frame.f_lineno}")] += 1
            return _original(self, inst, owner)

        type.__setattr__(cls, "__get__", counting)
        test.addCleanup(type.__setattr__, cls, "__get__", original)
    return counts, state


class TestInternalAccesses(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_a_rail_spine_build(self):
        # rig reaches its public methods by calls (the call-site path) or through
        # the class (``type(x).m(x)``) on its hot paths: a bare instance access
        # costs the full check (~1 us). Measured on 2026-10-16: 136 call-site
        # accesses and no full check per build
        from rig.examples import rail_spine as rs

        options = dict(get_options())
        self.addCleanup(lambda: set_options(**options))
        set_options(create_containers=True)
        pcs = List()
        for i in range(5):
            pcs.append(Node(cmds.spaceLocator()[0]))
            pcs[i].ty << i * 5
        counts, state = _count_accesses(self)
        u = [x / 19 for x in range(20)]
        state["on"] = True
        rs.create_rail(pcs, u, scale_controls=pcs, orient_controls=pcs, degree=3, periodic=False,
                       aim_axis=1, up_axis=0, debug=True)
        state["on"] = False
        internal = {k: v for k, v in counts.items() if not k[2].startswith("user") and not k[2].startswith("examples")}
        checks   = {k: v for k, v in internal.items() if k[0] == "check"}
        self.assertEqual(checks, {})
        self.assertLessEqual(sum(internal.values()), 300, sorted(internal.items(), key=lambda kv: -kv[1])[:10])

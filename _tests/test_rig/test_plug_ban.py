"""The property ban, plug side (user decision 2026-10-16).

Every public member of Attribute / Plug / ComponentPlug is a method, under the
rule: a bare name is the Maya attribute if one exists (a compound child of the
plug, else, on a Plug, a sibling or a container-published plug), otherwise the
method; with brackets it is the method. The 22 properties became methods
(``plug.get_node()``, ``plug.get_full_name()`` / ``str(plug)``,
``plug.get_mplug()``, ``plug.is_locked()`` ...); their setters became
``set_alias`` / ``set_keyable`` / ``set_channel_box`` / ``set_default_value``
and ``lock()`` / ``unlock()``; ``next_index`` is ``get_next_available_index()``.

* `TestRenamedReads`: each method answers what its property did.
* `TestNoPublicProperty`: no public property is left on the plug classes; the
  ``str`` method names are a non-data descriptor (`_StrRoute`).
* `TestTheClashesTheBanFixes`: names a property hid are plugs now
  (renderLayer.adjustments[0].plug, user compound children named alias / node /
  is_locked, a sibling named like an old property).
* `TestTheResidues`: the regressions the prototypes left (shorthand's
  quaternion routing, a Typed child of a compound in serialize, Node()'s
  defensive cast, the clone spec's type queries, an aliased channel's routing).
* `TestOldNames`: an old spelling raises naming the new one; a bare Attribute has
  no public state; ``hasattr`` answers on a bare Attribute.
* `TestBarePredicates`: a predicate named without brackets has no truth value
  (user decision: the guard), on a plug and over a List.
* `TestPlugClassEnforcement`: a plug class declares no public member but methods
  (and UPPER_CASE data), checked when the class is made.
* `TestRigSourceLint`: rig's own code reads no old name, names no predicate bare,
  and every class route it takes resolves.
"""

import ast
import inspect
import os
import re
from unittest import mock

from maya import cmds
from maya.api import OpenMaya

import rig
from rig import List, Node
from rig._internal import plug as _plug_module
from rig._internal.plug import ComponentPlug, Plug
from rig.nodetypes import _base
from rig.nodetypes._base import Attribute
from rig.nodetypes.transform import Transform
from rig._tests._base import MayaTestCase

# read by name, so that on a tree without the ban each test fails on its own
# (the negative control) instead of the module failing to import
_ABSENT        = type("_Absent", (), {})
_StrRoute      = getattr(_plug_module, "_StrRoute", _ABSENT)
_BarePredicate = getattr(_base, "_BarePredicate", _ABSENT)

# the 22 properties of the plug classes before the ban, and their methods
_RENAMED = {
    "plug": "get_mplug", "full_name": "get_full_name", "node": "get_node", "name": "get_name",
    "alias": "get_alias", "attribute_type": "get_attribute_type", "data_type": "get_data_type",
    "default_value": "get_default_value", "enums": "get_enums", "fn_set": "get_fn_set",
    "mobject": "get_mobject", "is_dynamic": "is_dynamic", "is_locked": "is_locked",
    "is_keyable": "is_keyable", "is_channel_box": "is_channel_box", "is_connected": "is_connected",
    "is_free_to_change": "is_free_to_change", "num_children": "get_num_children",
    "is_typed": "is_typed", "is_multi": "is_multi", "num_elements": "get_num_elements",
    "next_index": "get_next_available_index",
}
_SETTERS = ("set_alias", "set_keyable", "set_channel_box", "set_default_value")
_PREDICATES = ("is_dynamic", "is_locked", "is_keyable", "is_channel_box", "is_connected",
               "is_free_to_change", "is_typed", "is_multi")


def _plugins():
    for name in ("matrixNodes", "quatNodes"):
        if not cmds.pluginInfo(name, query=True, loaded=True):
            cmds.loadPlugin(name, quiet=True)


class TestRenamedReads(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.t = Node(cmds.createNode("transform", name="t"))
        cmds.addAttr("t", ln="knob", at="double", dv=2.5, keyable=True)
        cmds.addAttr("t", ln="mode", at="enum", en="a:b:c")
        cmds.addAttr("t", ln="w", at="double", multi=True)
        cmds.setAttr("t.w[0]", 1)
        cmds.setAttr("t.w[3]", 1)
        cmds.addAttr("t", ln="txt", dt="string")

    def test_every_method_answers_as_its_property_did(self):
        t, tx = self.t, self.t.tx
        self.assertIsInstance(tx.get_mplug(), OpenMaya.MPlug)
        self.assertEqual(tx.get_mplug().name(), "t.translateX")
        self.assertEqual(tx.get_full_name(), "t.translateX")
        self.assertEqual(str(tx), tx.get_full_name())
        self.assertIs(tx.get_node(), t)
        self.assertEqual(tx.get_name(), "translateX")
        self.assertEqual(t.t.get_name(), "translate")
        self.assertEqual(tx.get_alias(), "translateX")
        self.assertEqual(tx.get_attribute_type(), "doubleLinear")
        self.assertEqual(tx.get_data_type(), "doubleLinear")
        self.assertEqual(t.txt.get_data_type(), "string")
        self.assertEqual(t.knob.get_default_value(), 2.5)
        self.assertEqual(t.mode.get_enums(), ["a", "b", "c"])
        self.assertIsInstance(tx.get_fn_set(), OpenMaya.MFnAttribute)
        self.assertIsInstance(tx.get_mobject(), OpenMaya.MObject)
        self.assertFalse(tx.is_dynamic())
        self.assertTrue(t.knob.is_dynamic())
        self.assertFalse(tx.is_locked())
        self.assertTrue(tx.is_keyable())
        self.assertFalse(tx.is_channel_box())
        self.assertFalse(tx.is_connected())
        self.assertTrue(tx.is_free_to_change())
        self.assertEqual(t.t.get_num_children(), 3)
        self.assertTrue(t.txt.is_typed())
        self.assertFalse(tx.is_typed())
        self.assertTrue(t.w.is_multi())
        self.assertFalse(tx.is_multi())
        self.assertEqual(t.w.get_num_elements(), 2)
        self.assertEqual(t.w.get_next_available_index(), 1)

    def test_the_setters(self):
        t = self.t
        t.knob.set_alias("dial")
        self.assertEqual(t.knob.get_alias(), "dial")
        self.assertEqual(str(t.knob), "t.dial")
        t.knob.set_alias("dial")  # the same alias: left as it is
        self.assertEqual(cmds.aliasAttr("t", query=True), ["dial", "knob"])
        t.knob.set_alias(None)
        self.assertEqual(t.knob.get_alias(), "knob")
        # no alias to remove: nothing to do (the alias setter ran aliasAttr(None))
        t.knob.set_alias(None)
        t.knob.set_alias("")
        self.assertIsNone(cmds.aliasAttr("t", query=True))
        t.knob.set_keyable(False)
        self.assertFalse(cmds.getAttr("t.knob", keyable=True))
        t.knob.set_channel_box(True)
        self.assertTrue(t.knob.is_channel_box())
        t.knob.set_default_value(4.0)
        self.assertEqual(t.knob.get_default_value(), 4.0)
        with self.assertRaisesRegex(RuntimeError, "native attr"):
            t.tx.set_default_value(1.0)
        t.tx.lock()
        self.assertTrue(t.tx.is_locked())
        t.tx.unlock()
        self.assertFalse(t.tx.is_locked())

    def test_a_bare_attribute_answers_too(self):
        attr = self.t.find_attr("tx")
        self.assertIs(type(attr), Attribute)
        self.assertIs(attr.get_node(), self.t)
        self.assertEqual(attr.get_full_name(), "t.translateX")
        self.assertEqual(attr.get_name(), "translateX")
        self.assertFalse(attr.is_multi())

    def test_a_component_plug_names_itself(self):
        cmds.nurbsPlane(name="plane", ch=False)
        cv = Node("planeShape").cv[1, 1]
        self.assertIsInstance(cv, ComponentPlug)
        self.assertEqual(cv.get_full_name(), "planeShape.cv[1][1]")
        self.assertEqual(str(cv), "planeShape.cv[1][1]")
        self.assertEqual(repr(cv), 'ComponentPlug("planeShape.cv[1][1]")')
        self.assertEqual(Node("planeShape").cv.get_full_name(), "planeShape.controlPoints")


class TestNoPublicProperty(MayaTestCase):
    def test_no_public_property_on_the_plug_classes(self):
        for cls in (Attribute, Plug, ComponentPlug):
            for name in dir(cls):
                if name.startswith("_"):
                    continue
                with self.subTest(cls=cls.__name__, name=name):
                    member = inspect.getattr_static(cls, name)
                    self.assertNotIsInstance(member, property)

    def test_the_old_names_are_gone_and_the_new_ones_are_methods(self):
        for old, new in _RENAMED.items():
            with self.subTest(old=old):
                if old != new:
                    self.assertFalse(hasattr(Plug, old), old)
                self.assertTrue(inspect.isfunction(getattr(Plug, new)), new)
        for new in _SETTERS:
            self.assertTrue(inspect.isfunction(getattr(Plug, new)), new)

    def test_the_str_method_names_are_routes(self):
        for name in (n for n in vars(str) if not n.startswith("_")):
            with self.subTest(name=name):
                self.assertIsInstance(Plug.__dict__[name], _StrRoute)


class TestTheClashesTheBanFixes(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_a_render_layer_adjustment_plug(self):
        # renderLayer.adjustments[i] has a child named plug: the property hid it
        cmds.createNode("renderLayer", name="rl")
        child = Node("rl").adjustments[0].plug
        self.assertIsInstance(child, Plug)
        self.assertEqual(str(child), "rl.adjustments[0].plug")
        self.assertIsInstance(Node("rl").adjustments[0].get_mplug(), OpenMaya.MPlug)

    def test_user_compound_children_named_like_old_members(self):
        cmds.createNode("transform", name="t")
        cmds.addAttr("t", ln="info", at="compound", nc=3)
        for child in ("alias", "node", "is_locked"):
            cmds.addAttr("t", ln=child, at="double", p="info")
        info = Node("t").info
        self.assertEqual(str(info.alias), "t.alias")
        self.assertEqual(str(info.node), "t.node")
        self.assertIsInstance(info.is_locked, Plug)
        self.assertEqual(str(info.is_locked), "t.is_locked")
        self.assertFalse(info.is_locked())  # the method, with brackets
        # ``=`` reaches the child: before the ban, info.alias = ... ran the alias setter
        info.alias = 4.0
        info.node = 5.0
        info.is_locked = 6.0
        self.assertEqual(
            [cmds.getAttr(f"t.{c}") for c in ("alias", "node", "is_locked")], [4.0, 5.0, 6.0]
        )
        self.assertIsNone(cmds.aliasAttr("t", query=True))

    def test_a_sibling_named_like_an_old_member(self):
        # aiAOV.name / a user attr: the plug-level lookup reaches the sibling
        cmds.createNode("transform", name="t")
        cmds.addAttr("t", ln="data_type", at="double")
        self.assertEqual(str(Node("t").tx.data_type), "t.data_type")


class TestTheResidues(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_a_matrix_into_a_quaternion(self):
        # shorthand._matrix_to_quaternion read the decomposeMatrix plug's node
        _plugins()
        m = Node(cmds.createNode("transform", name="m"))
        q = Node(cmds.createNode("quatToEuler", name="q"))
        q.inputQuat << m.worldMatrix[0]
        self.assertEqual(cmds.listConnections("q.inputQuat", s=True, p=True), ["decomposeMatrix1.outputQuat"])

    def test_a_transform_into_a_quaternion(self):
        # shorthand._transform_to_quaternion: a bare transform decomposed (its
        # router, called as the shorthand table calls it)
        from rig._internal import shorthand

        _plugins()
        Node(cmds.createNode("transform", name="m"))
        q = Node(cmds.createNode("quatToEuler", name="q"))
        self.assertTrue(shorthand._transform_to_quaternion(Node("m"), q.inputQuat))
        self.assertEqual(cmds.listConnections("q.inputQuat", s=True, p=True), ["decomposeMatrix1.outputQuat"])

    def test_a_typed_child_of_a_compound_is_serialized(self):
        # Transform._serialize_user_attr read a compound parent's name; an
        # ``except Exception`` dropped the attr with a warning
        cmds.createNode("transform", name="t")
        cmds.addAttr("t", ln="box", at="compound", nc=1)
        cmds.addAttr("t", ln="slot", at="typed", p="box")
        with mock.patch.object(rig.nodetypes.transform.LOGGER, "warning") as warning:
            data = Transform("t").serialize()
        warning.assert_not_called()
        attrs = data.user_defined_attributes
        self.assertIn("slot", attrs)
        self.assertEqual(attrs["slot"]["parent"], "box")
        self.assertEqual(attrs["slot"]["attributeType"], "typed")

    def test_node_of_a_cast_that_gives_an_attribute(self):
        # Node()'s defensive path: an input the cast resolves to an attribute
        t = Node(cmds.createNode("transform", name="t"))
        mobject = t.get_mobject()
        with mock.patch.object(_base, "_cast", return_value=t.tx):
            self.assertIs(Node(mobject), t)

    def test_the_clone_spec_queries_the_types(self):
        # spec._safe_type_probe read type properties by name, inside an
        # ``except Exception``: a renamed one gave None (a wrong spec)
        from rig.spec._base import _spec_from_attribute

        cmds.createNode("transform", name="t")
        cmds.addAttr("t", ln="mtx", dt="matrix")
        cmds.addAttr("t", ln="txt", dt="string")
        for attr, spec in (("translate", "Vector"), ("rotate", "Euler"), ("mtx", "Matrix"),
                           ("txt", "String"), ("visibility", "Bool")):
            with self.subTest(attr=attr):
                self.assertEqual(type(_spec_from_attribute(Plug(f"t.{attr}"), "x", False)).__name__, spec)
        from rig.spec import _base as spec_base

        # a query name the plug class lacks raises, outside the handler

        with self.assertRaises(AttributeError):
            spec_base._safe_type_probe(Plug("t.txt"), "data_type")

    def test_an_aliased_channel_routes_a_matrix(self):
        # shorthand's routers read the attribute's name (isinstance Attribute,
        # rename- and alias-safe), where the str split read the alias
        _plugins()
        cmds.createNode("transform", name="t")
        cmds.createNode("transform", name="src")
        cmds.aliasAttr("move", "t.translate")
        Plug("t.move") << Node("src").worldMatrix[0]
        self.assertEqual(cmds.listConnections("t.translate", s=True, p=True), ["decomposeMatrix1.outputTranslate"])


class TestOldNames(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.t = Node(cmds.createNode("transform", name="t"))

    def test_an_old_read_raises_naming_the_new_spelling(self):
        tx = self.t.tx
        for old, new in _RENAMED.items():
            if old == new:
                continue
            with self.subTest(old=old):
                with self.assertRaisesRegex(AttributeError, re.escape(f"; plug.{old} is now ")):
                    getattr(tx, old)
        with self.assertRaisesRegex(AttributeError, re.escape("str(plug) or plug.get_full_name()")):
            tx.full_name

    def test_an_old_setter_raises_naming_the_new_spelling(self):
        tx = self.t.tx
        for old, new in (("alias", "set_alias"), ("default_value", "set_default_value"),
                         ("is_locked", "lock"), ("is_keyable", "set_keyable"),
                         ("is_channel_box", "set_channel_box")):
            with self.subTest(old=old):
                with self.assertRaisesRegex(AttributeError, re.escape(f"plug.{new}(")):
                    setattr(tx, old, True)
        self.assertFalse(cmds.getAttr("t.tx", lock=True))
        self.assertIsNone(cmds.aliasAttr("t", query=True))
        self.assertEqual(set(vars(tx)) & set(_RENAMED), set())

    def test_a_bare_attribute_has_no_public_state(self):
        attr = self.t.find_attr("tx")
        for name, value in (("alias", "x"), ("is_locked", True), ("anything", 1)):
            with self.subTest(name=name):
                with self.assertRaisesRegex(AttributeError, "a bare Attribute has no '=' sugar"):
                    setattr(attr, name, value)
                self.assertNotIn(name, vars(attr))
        attr._private = 1  # its own state
        self.assertEqual(vars(attr)["_private"], 1)

    def test_hasattr_answers_on_a_bare_attribute(self):
        attr = self.t.find_attr("tx")
        self.assertFalse(hasattr(attr, "nope"))
        self.assertFalse(hasattr(attr, "node"))
        with self.assertRaisesRegex(AttributeError, r"t\.translateX is not a compound: it has no child 'nope'"):
            attr.nope
        with self.assertRaisesRegex(AttributeError, re.escape("'node'; plug.node is now plug.get_node()")):
            attr.node
        self.assertTrue(hasattr(self.t.find_attr("t"), "translateX"))

    def test_next_index_is_gone(self):
        cmds.addAttr("t", ln="w", at="double", multi=True)
        with self.assertRaisesRegex(AttributeError, re.escape("plug.get_next_available_index()")):
            self.t.w.next_index
        self.assertEqual(self.t.w.get_next_available_index(), 0)


class TestBarePredicates(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.t = Node(cmds.createNode("transform", name="t"))

    def test_a_bare_predicate_has_no_truth_value(self):
        for name in _PREDICATES + ("has_category",):
            with self.subTest(name=name):
                bare = getattr(self.t.tx, name)
                self.assertIsInstance(bare, _BarePredicate)
                with self.assertRaisesRegex(TypeError, re.escape(f"{name} is a method: call it ({name}())")):
                    bool(bare)
                with self.assertRaises(TypeError):
                    if bare:  # noqa: SIM108 - the bug the guard stops
                        pass

    def test_a_bare_predicate_is_still_the_method(self):
        bare = self.t.tx.is_locked
        self.assertFalse(bare())
        self.assertTrue(callable(bare))
        self.assertEqual(bare.__name__, "is_locked")
        self.assertIs(bare.__func__, Attribute.is_locked)
        self.assertIs(bare.__self__.__class__, Plug)
        self.assertEqual(bare.__doc__, Attribute.is_locked.__doc__)
        self.assertEqual(list(inspect.signature(bare).parameters), [])
        self.assertFalse(inspect.ismethod(bare))

    def test_with_brackets_it_is_the_method(self):
        self.assertIs(self.t.tx.is_locked(), False)
        self.t.tx.lock()
        self.assertIs(self.t.tx.is_locked(), True)
        # other methods keep the bound method
        self.assertTrue(inspect.ismethod(self.t.tx.get_node))

    def test_a_child_named_like_a_predicate_is_the_plug(self):
        cmds.addAttr("t", ln="pair", at="compound", nc=1)
        cmds.addAttr("t", ln="is_multi", at="double", p="pair")
        child = self.t.pair.is_multi
        self.assertNotIsInstance(child, _BarePredicate)
        self.assertEqual(str(child), "t.is_multi")
        self.assertFalse(self.t.pair.is_multi())

    def test_over_a_list(self):
        u = Node(cmds.createNode("transform", name="u"))
        u.tx.lock()
        lst = List([self.t.tx, u.tx])
        with self.assertRaisesRegex(TypeError, re.escape("[x.is_locked() for x in items]")):
            bool(lst.is_locked)
        with self.assertRaisesRegex(TypeError, "does not broadcast"):
            lst.is_locked()
        self.assertEqual([x.is_locked() for x in lst], [False, True])
        # a method that is no predicate keeps a List's truth value
        self.assertTrue(bool(lst.get_node))


class TestPlugClassEnforcement(MayaTestCase):
    """A plug class (rig's, a user's) declares no public member but methods: a
    property, a cached_property or another descriptor, lowercase data or a nested
    class is refused when the class is made; UPPER_CASE data passes."""

    def test_refused_members(self):
        import functools

        class _Descriptor:
            def __get__(self, inst, owner=None):
                return 1

        for label, body in (
            ("property", {"side": property(lambda self: 1)}),
            ("cached_property", {"side": functools.cached_property(lambda self: 1)}),
            ("descriptor", {"side": _Descriptor()}),
            ("data", {"side": None}),
            ("class", {"Inner": type("Inner", (), {})}),
        ):
            for base in (Plug, Attribute):
                with self.subTest(member=label, base=base.__name__):
                    with self.assertRaisesRegex(TypeError, r"a plug class has no public"):
                        type("_Bad", (base,), dict(body))

    def test_methods_and_declarations_pass(self):
        def foo(self):
            return 1

        cls = type("_Good", (Plug,), {"UNITS": "cm", "foo": foo, "_side": None,
                                      "make": classmethod(lambda c: c)})
        self.assertIsInstance(cls.__dict__["foo"], _base._ChildOrMethod)
        self.assertEqual(cls.UNITS, "cm")
        self.assertIs(cls.foo, foo)


_RIG_DIR = os.path.dirname(os.path.abspath(rig.__file__))

# the plug-only names of the properties before the ban: no rig code reads one;
# `list.py`'s ``_NamedPlug`` keeps a slot named ``plug`` (not a plug's member)
_GONE_PLUG_ONLY = {"full_name", "alias", "attribute_type", "data_type", "default_value", "enums",
                   "num_children", "num_elements", "next_index", "plug"}
_ALLOWED = {("_internal/list.py", "plug")}


def _rig_sources():
    for base, dirs, files in os.walk(_RIG_DIR):
        dirs[:] = sorted(d for d in dirs if d not in ("_tests", "__pycache__"))
        for name in sorted(f for f in files if f.endswith(".py")):
            path = os.path.join(base, name)
            rel  = os.path.relpath(path, _RIG_DIR).replace(os.sep, "/")
            with open(path, encoding="utf-8") as fh:
                yield rel, ast.parse(fh.read())


def _callees(tree):
    return {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}


class TestRigSourceLint(MayaTestCase):
    """rig's own code (outside ``_tests``), read with ``ast``: what the ban needs
    of it, at no run-time cost."""

    def test_no_bare_predicate(self):
        # ``if plug.is_locked:`` would be the bound method (truthy): always called
        found = []
        for rel, tree in _rig_sources():
            callees = _callees(tree)
            for n in ast.walk(tree):
                if isinstance(n, ast.Attribute) and n.attr in _PREDICATES and id(n) not in callees:
                    found.append(f"{rel}:{n.lineno} .{n.attr}")
        self.assertEqual(found, [])

    def test_no_old_plug_name(self):
        found = []
        for rel, tree in _rig_sources():
            for n in ast.walk(tree):
                if isinstance(n, ast.Attribute) and n.attr in _GONE_PLUG_ONLY and (rel, n.attr) not in _ALLOWED:
                    found.append(f"{rel}:{n.lineno} .{n.attr}")
        self.assertEqual(found, [])

    def test_every_class_route_resolves(self):
        # ``type(x).get_node(x)``: a name no class has would raise AttributeError,
        # which a handler around it could swallow
        import sys

        import cgmath.geometry  # noqa: F401 - rig routes through its data classes

        known = set(dir(object)) | set(dir(type)) | set(dir(str)) | set(dir(list))
        for module in list(sys.modules.values()):
            if getattr(module, "__name__", "").startswith(("rig", "cgmath")):
                for value in list(vars(module).values()):
                    if isinstance(value, type):
                        known |= set(dir(value))
        found = []
        for rel, tree in _rig_sources():
            for n in ast.walk(tree):
                if (
                    isinstance(n, ast.Attribute)
                    and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Name)
                    and n.value.func.id == "type"
                    and n.attr not in known
                ):
                    found.append(f"{rel}:{n.lineno} type(...).{n.attr}")
        self.assertEqual(found, [])

"""Enforcement of the one rule on node and plug classes (user decisions 2026-10-16).

On a node or a plug, a bare name is the Maya attr (a plug's child) if one exists,
otherwise the method; with brackets it is the method. So a class declares, under a
public name, methods only.

* `TestClassBody`: a node / plug class body refuses a property, a cached_property,
  another descriptor, a nested class and lowercase data (the error names the
  ``_`` spelling); methods, classmethods, staticmethods and UPPER_CASE
  declarations pass.
* `TestClassData`: an UPPER_CASE declaration (``NATIVE_NODE_TYPE``,
  ``CUSTOM_NODE_TYPE``, ``FN_SET``, ``POINT_COMP_TYPE``, ``DEFAULT``, a user's
  ``SIZE``) is read on the class, never on an instance (``node.DEFAULT`` is the
  node's Maya attr ``DEFAULT``); a user ``CUSTOM_NODE_TYPE`` class still registers.
* `TestLateClassAssignment`: ``Cls.x = value`` after the class is made follows the
  class-body rule (``NodeMeta.__setattr__``, the plug classes' metaclass); a mock
  is kept, and ``mock.patch.object`` restores what it patched.
* `TestNoShadowingLeft`: ``spec._warn_shadowed``, ``_SHADOWED`` and
  ``is_plug_method`` are gone, and nothing hides a new attribute.
* `TestUnderscoreRuleOnNodes` / `...OnLists` / `...OnPlugs`: the '_' rule B. A
  ``_`` name is Python state only if it is rig's own state or its class body
  declares it; otherwise it is a Maya attr (a plug's child), else it raises.
* `TestScenePredicates`: the predicate guard on the scene handles
  (``faces.is_all``, ``cset.is_valid``, ``rig.container.is_active``).
* `TestPredicateLint`: rig's own code reads no bare ``.is_x`` / ``.has_x`` but
  the value objects' data.
"""

import functools
import inspect
import logging
from unittest import mock

from maya import cmds

import rig
from rig import Bool, Node
from rig._internal.plug import Plug
from rig.nodetypes import _base
from rig.nodetypes import DisplayLayer, Mesh, NurbsCurve, ShadingEngine
from rig.nodetypes._base import Attribute
from rig.nodetypes.dg_node import DGNode
from rig.nodetypes.transform import Transform
from rig._tests._base import MayaTestCase

# read by name, so that on a tree without the step each test fails on its own
# (the negative control) instead of the module failing to import
_ABSENT    = type("_Absent", (), {})
_ClassData = getattr(_base, "_ClassData", _ABSENT)

# rig's UPPER_CASE declarations: class -> name -> value
_DECLARED = (
    (DGNode, "NATIVE_NODE_TYPE", "entity"),
    (DGNode, "CUSTOM_NODE_TYPE", None),
    (Transform, "NATIVE_NODE_TYPE", "transform"),
    (Mesh, "POINT_COMP_TYPE", "vtx"),
    (NurbsCurve, "POINT_COMP_TYPE", "cv"),
    (DisplayLayer, "DEFAULT", "defaultLayer"),
    (ShadingEngine, "DEFAULT", "initialShadingGroup"),
)


class _Descriptor:
    def __get__(self, inst, owner=None):
        return 1


def _keep_registry(case):
    """Restore the node class registry when `case` ends: a class registered here
    (a CUSTOM_NODE_TYPE) never reaches another module's walk over it."""
    registered = dict(_base._NODE_CLASS_DICT)

    def restore():
        _base._NODE_CLASS_DICT.clear()
        _base._NODE_CLASS_DICT.update(registered)
        _base._CLASS_BY_TYPE.clear()
        _base._CASTABLE_TYPES.clear()

    case.addCleanup(restore)


def _members():
    """(label, member) for every public member a class body refuses."""
    return (
        ("property", property(lambda self: 1)),
        ("cached_property", functools.cached_property(lambda self: 1)),
        ("descriptor", _Descriptor()),
        ("class", type("Inner", (), {})),
        ("data", None),
        ("data", 1.5),
        ("data", "left"),
    )


class TestClassBody(MayaTestCase):
    def test_a_node_class_body_refuses_every_public_member_but_methods(self):
        for label, member in _members():
            with self.subTest(member=label, value=member):
                with self.assertRaisesRegex(
                    TypeError, rf"^_Bad\.side: a node class has no public {label} "
                ):
                    type(Transform)("_Bad", (Transform,), {"side": member})

    def test_the_data_refusal_names_the_private_spelling(self):
        with self.assertRaises(TypeError) as caught:
            class _Ctl(Transform):  # noqa: F841
                side = None
        text = str(caught.exception)
        self.assertIn("keep Python state under a '_' name (_side = None)", text)
        self.assertIn("SIDE = ..., read as type(node).SIDE", text)

    def test_a_plug_class_body_refuses_them_too(self):
        for base in (Attribute, Plug):
            for label, member in _members():
                with self.subTest(base=base.__name__, member=label, value=member):
                    with self.assertRaisesRegex(
                        TypeError, rf"^_Bad\.side: a plug class has no public {label} "
                    ):
                        type(base)("_Bad", (base,), {"side": member})

    def test_methods_private_members_and_class_data_pass(self):
        class _Fine(Transform):
            _side = None
            SIZE  = 2.0

            def get_size(self):
                return type(self).SIZE

            @classmethod
            def make(cls):
                return cls

            @staticmethod
            def helper():
                return 3

            @property
            def _private(self):
                return 4

        self.assertIsInstance(vars(_Fine)["get_size"], _base._PlugMethod)
        self.assertIsInstance(vars(_Fine)["make"], _base._PlugClassMethod)
        self.assertIsInstance(vars(_Fine)["helper"], _base._PlugStaticMethod)
        self.assertIsInstance(vars(_Fine)["_private"], property)
        self.assertIsNone(vars(_Fine)["_side"])
        self.assertIsInstance(vars(_Fine)["SIZE"], _ClassData)
        self.assertEqual(_Fine.SIZE, 2.0)

        class _FinePlug(Plug):
            _state = None
            UNITS  = "cm"

            def get_units(self):
                return type(self).UNITS

        self.assertIsInstance(vars(_FinePlug)["get_units"], _base._ChildOrMethod)
        self.assertIsInstance(vars(_FinePlug)["UNITS"], _ClassData)
        self.assertEqual(_FinePlug.UNITS, "cm")

    def test_a_class_body_copied_from_a_class(self):
        """guard: a class made from another's dict (its plug methods and class data
        already made) is made again, and registers its node type."""
        _keep_registry(self)

        class _EnforceCopied(Transform):
            CUSTOM_NODE_TYPE = "rigEnforceCopied"

            def get_size(self):
                return 1

        again = type(_EnforceCopied)(
            "_EnforceAgain", (Transform,), dict(vars(_EnforceCopied))
        )
        self.assertEqual(again.CUSTOM_NODE_TYPE, "rigEnforceCopied")
        self.assertIs(_base._NODE_CLASS_DICT["rigEnforceCopied"], again)
        self.assertIs(vars(again)["get_size"], vars(_EnforceCopied)["get_size"])


class TestClassData(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_the_class_reads_its_declarations(self):
        for cls, name, value in _DECLARED:
            with self.subTest(cls=cls.__name__, name=name):
                self.assertEqual(getattr(cls, name), value)
                self.assertIsInstance(_base._class_attr(cls, name), _ClassData)
        from maya.api import OpenMaya

        self.assertIs(Mesh.FN_SET, OpenMaya.MFnMesh)
        self.assertIs(Transform.FN_SET, OpenMaya.MFnTransform)

    def test_an_instance_never_does(self):
        t     = Transform.create(name="t")
        layer = DisplayLayer.create(name="L")
        cube  = Node(cmds.polyCube(name="cube", ch=False)[0]).get_shape()
        for node, name in (
            (t, "NATIVE_NODE_TYPE"), (t, "CUSTOM_NODE_TYPE"), (t, "FN_SET"),
            (cube, "POINT_COMP_TYPE"), (layer, "DEFAULT"),
        ):
            with self.subTest(node=str(node), name=name):
                with self.assertRaisesRegex(
                    AttributeError,
                    rf"Attribute not found: .*; {name} is class data: "
                    rf"{type(node).__name__}\.{name}",
                ):
                    getattr(node, name)
                self.assertFalse(hasattr(node, name))
                self.assertNotIn(name, dir(node))
        self.assertFalse(hasattr(t.tx, "NATIVE_NODE_TYPE"))

    def test_a_maya_attr_of_that_name_is_the_plug(self):
        layer = DisplayLayer.create(name="L")
        cmds.addAttr("L", longName="DEFAULT", attributeType="double")
        self.assertIsInstance(layer.DEFAULT, Plug)
        layer.DEFAULT = 2.5
        self.assertEqual(cmds.getAttr("L.DEFAULT"), 2.5)
        self.assertEqual(DisplayLayer.DEFAULT, "defaultLayer")
        self.assertNotIn("DEFAULT", vars(layer))

    def test_an_assignment_without_the_attr_raises(self):
        layer = DisplayLayer.create(name="L")
        with self.assertRaisesRegex(
            AttributeError, r"DEFAULT is class data: DisplayLayer\.DEFAULT"
        ):
            layer.DEFAULT = "other"
        self.assertNotIn("DEFAULT", vars(layer))
        self.assertEqual(DisplayLayer.DEFAULT, "defaultLayer")

    def test_a_user_custom_node_type_class(self):
        _keep_registry(self)

        class _EnforceCtl(Transform):
            CUSTOM_NODE_TYPE = "rigEnforceCtl"
            SIDES            = ("L", "R")

            def get_sides(self):
                return type(self).SIDES

        ctl = _EnforceCtl.create(name="ctl")
        self.assertIs(type(Node("ctl")), _EnforceCtl)
        self.assertEqual(ctl.get_node_type(), "rigEnforceCtl")
        self.assertEqual(ctl.get_sides(), ("L", "R"))
        self.assertIs(_base._NODE_CLASS_DICT["rigEnforceCtl"], _EnforceCtl)
        self.assertEqual(_EnforceCtl.CUSTOM_NODE_TYPE, "rigEnforceCtl")
        self.assertEqual(_EnforceCtl.NATIVE_NODE_TYPE, "transform")  # inherited
        with self.assertRaises(AttributeError):
            ctl.SIDES

    def test_getmembers_lists_no_class_data(self):
        t = Transform.create(name="t")
        names = [name for name, _ in inspect.getmembers(t)]
        for name in ("NATIVE_NODE_TYPE", "CUSTOM_NODE_TYPE", "FN_SET"):
            self.assertNotIn(name, names)
        self.assertIn("get_name", names)


class TestLateClassAssignment(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()

        class _Late(Transform):
            pass

        class _LatePlug(Plug):
            pass

        self.cls      = _Late
        self.plug_cls = _LatePlug

    def test_a_function_is_made_a_plug_method(self):
        def helper(self):
            return "helper"

        self.cls.helper = helper
        self.assertIsInstance(vars(self.cls)["helper"], _base._PlugMethod)
        node = self.cls._wrap(Transform.create(name="t"))
        self.assertEqual(node.helper(), "helper")
        cmds.addAttr("t", longName="helper", attributeType="double")
        self.assertIsInstance(node.helper, Plug)  # the rule: bare, the Maya attr
        self.assertEqual(node.helper(), "helper")
        self.cls.make = classmethod(lambda cls: cls)
        self.assertIsInstance(vars(self.cls)["make"], _base._PlugClassMethod)
        self.cls.util = staticmethod(lambda: 3)
        self.assertIsInstance(vars(self.cls)["util"], _base._PlugStaticMethod)

        self.plug_cls.helper = helper
        self.assertIsInstance(vars(self.plug_cls)["helper"], _base._ChildOrMethod)

    def test_upper_case_data_is_class_data(self):
        self.cls.SIZE = 3
        self.assertIsInstance(vars(self.cls)["SIZE"], _ClassData)
        self.assertEqual(self.cls.SIZE, 3)
        self.plug_cls.UNITS = "cm"
        self.assertIsInstance(vars(self.plug_cls)["UNITS"], _ClassData)

    def test_any_other_public_member_is_refused(self):
        for cls, kind in ((self.cls, "node"), (self.plug_cls, "plug")):
            for label, member in _members():
                with self.subTest(cls=kind, member=label, value=member):
                    with self.assertRaisesRegex(
                        TypeError, rf"\.side: a {kind} class has no public {label} "
                    ):
                        cls.side = member
                    self.assertNotIn("side", vars(cls))

    def test_a_private_name_is_stored(self):
        """guard: a '_' name of a class is the class's own business."""
        self.cls._count = 1
        self.cls._prop  = property(lambda self: 1)
        self.plug_cls._count = 1
        self.assertEqual(vars(self.cls)["_count"], 1)
        self.assertIsInstance(vars(self.cls)["_prop"], property)
        self.assertEqual(vars(self.plug_cls)["_count"], 1)

    def test_mock_patch_object_of_a_method_and_of_class_data(self):
        """guard: mock.patch.object patches and restores through the rule."""
        t = Transform.create(name="t")
        before = vars(DGNode)["get_name"]
        with mock.patch.object(DGNode, "get_name", return_value="patched") as patched:
            self.assertIs(vars(DGNode)["get_name"], patched)  # a mock is kept as is
            self.assertEqual(type(t).get_name(t), "patched")
        self.assertIs(vars(DGNode)["get_name"], before)
        self.assertEqual(t.get_name(), "t")
        before = vars(Mesh)["FN_SET"]
        with mock.patch.object(Mesh, "FN_SET", "patched"):
            self.assertEqual(Mesh.FN_SET, "patched")
            self.assertEqual(Transform.FN_SET.__name__, "MFnTransform")
        self.assertIs(vars(Mesh)["FN_SET"], before)
        self.assertEqual(Mesh.FN_SET.__name__, "MFnMesh")
        before = vars(Plug)["center"]
        with mock.patch.object(Plug, "center", mock.Mock()):
            pass
        self.assertIs(vars(Plug)["center"], before)
        # an inherited member: patched on the subclass, then deleted again
        with mock.patch.object(Transform, "get_name", return_value="x"):
            self.assertEqual(type(t).get_name(t), "x")
        self.assertNotIn("get_name", vars(Transform))
        self.assertEqual(t.get_name(), "t")


class TestNoShadowingLeft(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_the_shadow_helpers_are_gone(self):
        from rig.spec import _base as spec_base

        self.assertFalse(hasattr(spec_base, "_warn_shadowed"))
        self.assertFalse(hasattr(spec_base, "_SHADOWED"))
        self.assertFalse(hasattr(_base, "is_plug_method"))
        self.assertFalse(hasattr(rig.nodetypes, "is_plug_method"))

    def test_no_member_hides_a_new_attribute(self):
        t      = Transform.create(name="t")
        logger = logging.getLogger("rig.spec._base")
        for name in ("NATIVE_NODE_TYPE", "FN_SET", "side", "get_name"):
            with self.subTest(name=name):
                with self.assertRaises(AssertionError):
                    with self.assertLogs(logger, level=logging.WARNING):
                        t << Bool(name)
                self.assertIsInstance(getattr(t, name), Plug)



class TestUnderscoreRuleOnNodes(MayaTestCase):
    """The '_' rule B (user decision 2026-10-16): on a node a '_' name is Python
    state only if it is rig's node state or its class body (MRO) declares it;
    otherwise it is the node's Maya attr, else it raises."""

    TEST_START_NEW_SCENE = True

    def test_an_undeclared_name_raises(self):
        cube = cmds.polyCube(name="pCube1", ch=False)[0]
        t    = Transform.create(name="t")
        for node, name in ((t, "__tga__"), (t, "_tga"), (Node(cube), "_x")):
            with self.subTest(node=str(node), name=name):
                with self.assertRaisesRegex(
                    AttributeError,
                    rf"^Transform\.{name}: a '_' name on a node is its class's Python "
                    rf"state or the node's Maya attr, .*declare it in the class body "
                    rf"\({name} = None\)",
                ):
                    setattr(node, name, 5)
                self.assertNotIn(name, vars(node))
                self.assertFalse(hasattr(node, name))

    def test_a_maya_attr_of_that_name_is_set(self):
        """guard: the Maya attr of a '_' name is the assignment's target."""
        t = Transform.create(name="t")
        cmds.addAttr("t", longName="__tga__", attributeType="double")
        cmds.addAttr("t", longName="_tga", attributeType="double")
        t.__tga__ = 5
        t._tga = 6
        self.assertEqual(cmds.getAttr("t.__tga__"), 5)
        self.assertEqual(cmds.getAttr("t._tga"), 6)
        self.assertNotIn("_tga", vars(t))

    def test_a_declared_name_is_python_state(self):
        """guard: a '_' name the class body (MRO) declares is Python state."""
        class _Base(Transform):
            _side = None

            @property
            def _label(self):
                return vars(self).get("_label_value")

            @_label.setter
            def _label(self, value):
                vars(self)["_label_value"] = value

        class _Ctl(_Base):
            pass

        ctl = _Ctl._wrap(Transform.create(name="ctl"))
        cmds.addAttr("ctl", longName="_side", dataType="string")
        ctl._side = "L"  # declared: Python state, even with a Maya attr of the name
        self.assertEqual(vars(ctl)["_side"], "L")
        self.assertEqual(cmds.getAttr("ctl._side"), None)
        ctl._label = "x"  # a private property's setter runs
        self.assertEqual(ctl._label, "x")

    def test_rigs_own_state_and_a_name_already_stored(self):
        """guard: rig's node state and a name in vars(node) are stored."""
        t = Transform.create(name="t")
        handle = vars(t)["_objhandle"]
        t._objhandle = handle
        self.assertIs(vars(t)["_objhandle"], handle)
        vars(t)["_mine"] = 1  # the way to keep one on an instance (tests)
        t._mine = 2
        self.assertEqual(vars(t)["_mine"], 2)

    def test_a_deleted_a_freed_and_a_half_built_node(self):
        cmds.undoInfo(state=True, infinity=True)
        t = Transform.create(name="t")
        cmds.delete("t")
        with self.assertRaisesRegex(RuntimeError, "t already deleted!"):
            t._x = 1
        cmds.undo()
        f = Transform.create(name="f")
        cmds.file(new=True, force=True)
        with self.assertRaisesRegex(RuntimeError, "already deleted"):
            f._x = 1
        half = object.__new__(Transform)
        with self.assertRaises(AttributeError):
            half._x = 1
        self.assertNotIn("_x", vars(half))

    def test_tests_patch_the_class(self):
        t = Transform.create(name="t")
        with mock.patch.object(Transform, "_hook", "patched", create=True):
            self.assertEqual(t._hook, "patched")
        self.assertFalse(hasattr(t, "_hook"))
        # a name the class declares can be patched on one instance
        with mock.patch.object(t, "_attr_data_type_fallback", lambda attr: None):
            self.assertIn("_attr_data_type_fallback", vars(t))
        self.assertNotIn("_attr_data_type_fallback", vars(t))
        with self.assertRaises(AttributeError):
            with mock.patch.object(t, "_hook", "x", create=True):
                pass
        self.assertNotIn("_hook", vars(t))


class TestUnderscoreRuleOnLists(MayaTestCase):
    """A List mirrors the node rule (user decision 2026-10-16)."""

    TEST_START_NEW_SCENE = True

    def test_an_undeclared_name_raises(self):
        a = Transform.create(name="a")
        b = Transform.create(name="b")
        for items in ([a, b], [a.tx, b.tx], [1.0, 2.0], []):
            nodes = rig.List(items)
            with self.subTest(items=[str(x) for x in items]):
                with self.assertRaisesRegex(AttributeError, r"no element has '_x'"):
                    nodes._x = 1
                self.assertNotIn("_x", vars(nodes))

    def test_every_elements_maya_attr_is_set(self):
        """guard: the broadcast of a '_' name every element has."""
        a = Transform.create(name="a")
        b = Transform.create(name="b")
        for name in ("a", "b"):
            cmds.addAttr(name, longName="_tag", attributeType="double")
        rig.List([a, b])._tag = 3
        self.assertEqual([cmds.getAttr("a._tag"), cmds.getAttr("b._tag")], [3, 3])

    def test_its_own_and_its_declared_state(self):
        """guard: the List's own state and a subclass's declared state."""
        class _Sub(rig.List):
            _extra = None

        a = Transform.create(name="a")
        nodes = _Sub([a])
        nodes._extra = 1
        nodes._parent_multi = None
        self.assertEqual(vars(nodes)["_extra"], 1)
        self.assertIn("_parent_multi", vars(nodes))


class TestUnderscoreRuleOnPlugs(MayaTestCase):
    """Plugs follow the same rule (lead extension of the user decision): a '_'
    name is plug state if rig's or declared, else a child / Maya attr of that
    name, else it raises."""

    TEST_START_NEW_SCENE = True

    def test_a_child_of_that_name_is_set(self):
        t = Transform.create(name="t")
        cmds.addAttr("t", longName="pair", attributeType="compound", numberOfChildren=2)
        cmds.addAttr("t", longName="__child__", attributeType="double", parent="pair")
        cmds.addAttr("t", longName="other", attributeType="double", parent="pair")
        pair = t.pair
        pair.__child__ = 5
        self.assertEqual(cmds.getAttr("t.__child__"), 5)
        self.assertNotIn("__child__", vars(pair))

    def test_a_maya_attr_of_the_node_is_set(self):
        t = Transform.create(name="t")
        cmds.addAttr("t", longName="_tga", attributeType="double")
        plug = t.tx
        plug._tga = 2  # read as plug._tga: the node's attr
        self.assertEqual(cmds.getAttr("t._tga"), 2)
        self.assertNotIn("_tga", vars(plug))

    def test_an_undeclared_name_raises(self):
        t = Transform.create(name="t")
        for plug in (t.tx, t.translate, t.find_attr("tx")):
            with self.subTest(plug=type(plug).__name__, name=str(plug)):
                with self.assertRaisesRegex(
                    AttributeError, r"\.__child__: a '_' name on a plug is its class's "
                    r"Python state or a child / Maya attr of that name"
                ):
                    plug.__child__ = 5
                self.assertNotIn("__child__", vars(plug))

    def test_rigs_and_the_declared_plug_state(self):
        """guard: rig's plug state and a subclass's declared state are stored."""
        class _StatedPlug(Plug):
            _extra = None

        t     = Transform.create(name="t")
        plug  = t.tx
        owner = vars(plug)["_node"]
        plug._node = owner  # rig's plug state
        self.assertIs(vars(plug)["_node"], owner)
        stated = _StatedPlug(plug.get_mplug())
        stated._extra = 3
        self.assertEqual(vars(stated)["_extra"], 3)
        attr = t.find_attr("tx")
        attr._fn_set = None
        self.assertIsNone(vars(attr)["_fn_set"])

    def test_a_component_plug_declares_its_state(self):
        surface = Node(cmds.sphere(name="s", ch=False)[0]).get_shape()
        handle  = surface.cv
        for name in ("_comp_node", "_comp_alias", "_comp_ndims", "_comp_coords"):
            with self.subTest(name=name):
                self.assertIn(name, vars(type(handle)))
                self.assertIn(name, vars(handle))
        self.assertEqual(str(surface.cv[1, 2]), "sShape.cv[1][2]")

    def test_a_deleted_nodes_plug(self):
        cmds.undoInfo(state=True, infinity=True)
        t    = Transform.create(name="t")
        plug = t.tx
        cmds.delete("t")
        with self.assertRaisesRegex(RuntimeError, "already deleted!"):
            plug.__child__ = 5
        cmds.undo()


class TestScenePredicates(MayaTestCase):
    """The predicate guard (user decision 2026-10-16) on the scene handles: a
    bare ``is_x`` has no truth value, a call runs the method."""

    TEST_START_NEW_SCENE = True

    def _handles(self):
        from rig._internal.container import container
        from rig.nodetypes.mesh import ColorSet

        cube = Node(cmds.polyCube(name="cube", ch=False)[0]).get_shape()
        cset = cube.add_color_set("tint", ColorSet.Representation.RGBA)
        return ((cube.f, "is_all", True), (cube.f[:2], "is_all", False),
                (cset, "is_valid", True), (container, "is_active", False))

    def test_a_bare_predicate_has_no_truth_value(self):
        for handle, name, _ in self._handles():
            with self.subTest(handle=type(handle).__name__, name=name):
                bare = getattr(handle, name)
                self.assertIs(type(bare), _base._BarePredicate)
                with self.assertRaisesRegex(
                    TypeError, rf"^{name} is a method: call it \({name}\(\)\)$"
                ):
                    bool(bare)

    def test_a_call_runs_the_method(self):
        for handle, name, expected in self._handles():
            with self.subTest(handle=type(handle).__name__, name=name):
                self.assertIs(getattr(handle, name)(), expected)
        from rig._internal.container import container

        with container("box"):
            self.assertIs(container.is_active(), True)
        self.assertIs(container.is_active(), False)

    def test_class_access_is_the_function(self):
        from rig._internal.container import _ContainerStack
        from rig._internal.members import Components
        from rig.nodetypes.mesh import ColorSet

        for cls, name in ((Components, "is_all"), (ColorSet, "is_valid"),
                          (_ContainerStack, "is_active")):
            with self.subTest(cls=cls.__name__):
                self.assertTrue(inspect.isfunction(getattr(cls, name)))

    def test_nodes_and_plugs_too(self):
        """guard: the node and plug sides (the property ban steps)."""
        t = Transform.create(name="t")
        for owner, name in ((t, "is_locked"), (t.tx, "is_locked"), (t.tx, "has_category")):
            with self.subTest(owner=str(owner), name=name):
                self.assertIs(type(getattr(owner, name)), _base._BarePredicate)
        self.assertIs(t.tx.is_locked(), False)


# value objects keep their data as attributes (user decision 2026-10-16): the
# bare ``is_x`` reads of rig's own code that are no predicate method. Keyed by
# (file, receiver, name)
_VALUE_DATA = {
    # membership's selection groups (members._Selection / _Group), plain values
    ("membership.py", "selection", "is_all"),
    ("_internal/materials.py", "selection", "is_all"),   # re-pinned (user decision 2026-10-22, the shade module retired)
    ("_internal/members.py", "self", "is_all"),
    # an operand's facts (shorthand._Facts), plain values
    ("_internal/plug.py", "dst_facts", "is_compound"),
    ("_internal/plug.py", "src_facts", "is_compound"),
    ("_internal/shorthand.py", "src_facts", "is_matrix"),
    ("_internal/shorthand.py", "src_facts", "is_quaternion"),
    ("_internal/shorthand.py", "src_facts", "is_vector"),
    ("_internal/shorthand.py", "dst_facts", "is_matrix"),
    ("_internal/shorthand.py", "dst_facts", "is_quaternion"),
    ("_internal/shorthand.py", "dst_facts", "is_vector"),
    ("_internal/shorthand.py", "self", "is_compound"),
    # (re-pinned, fix round 2026-10-16: the cast's identity check of a class's
    # is_type reads its class-route mirror, cls_obj._raw_is_type, no bare
    # predicate, so its allowance is gone)
}


class TestPredicateLint(MayaTestCase):
    """rig's own code (outside ``_tests``), read with ``ast``: no bare ``.is_x`` /
    ``.has_x`` read but the value objects' data (user decision 2026-10-16: the
    lint goes with the guard)."""

    def test_no_bare_predicate_in_rig_source(self):
        import ast
        import os

        root  = os.path.dirname(rig.__file__)
        found = []
        seen  = set()
        for base, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in ("_tests", "__pycache__"))
            for name in sorted(f for f in files if f.endswith(".py")):
                path = os.path.join(base, name)
                rel  = os.path.relpath(path, root).replace(os.sep, "/")
                with open(path, encoding="utf-8") as fh:
                    tree = ast.parse(fh.read())
                callees = {id(c.func) for c in ast.walk(tree) if isinstance(c, ast.Call)}
                for n in ast.walk(tree):
                    if (
                        isinstance(n, ast.Attribute)
                        and isinstance(n.ctx, ast.Load)
                        and n.attr.startswith(("is_", "has_"))
                        and id(n) not in callees
                    ):
                        key = (rel, ast.unparse(n.value), n.attr)
                        seen.add(key)
                        if key not in _VALUE_DATA:
                            found.append(f"{rel}:{n.lineno} {key[1]}.{n.attr}")
        self.assertEqual(found, [])
        self.assertEqual(_VALUE_DATA - seen, set())  # no stale allowance

    def test_the_lint_finds_a_bare_read(self):
        """guard: the scan above flags a bare read and passes a call."""
        import ast

        tree    = ast.parse("if node.is_valid:\n    pass\nnode.is_valid()\nx = faces.is_all\n")
        callees = {id(c.func) for c in ast.walk(tree) if isinstance(c, ast.Call)}
        bare    = [
            n.attr for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and n.attr.startswith(("is_", "has_"))
            and id(n) not in callees
        ]
        self.assertEqual(sorted(bare), ["is_all", "is_valid"])

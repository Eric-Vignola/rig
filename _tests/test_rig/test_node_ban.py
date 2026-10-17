"""The property ban, node side (user decision 2026-10-16).

Every public member of a node class is a method, under the rule: a bare name is
the node's Maya attribute if one exists, otherwise the method; with brackets it
is the method. The 30 node properties became methods (``node.get_name()`` /
``str(node)``, ``node.get_uuid()``, ``node.get_fn_set()``, ``node.is_locked()``
...); the namespace setter is ``node.set_namespace(ns)``; ``file_path`` and
``file_path_with_copy_number`` are ``get_file_path(copy_number=False)``.

* `TestRenamedReads`: each method answers what its property did.
* `TestNoPublicProperty`: no public property is left on rig's node classes, and
  a node class body declares none (refused when the class is made).
* `TestTheClashesTheBanFixes`: names a property hid are Maya attrs now
  (``aiAOV.name``, a ``namespace`` attr and its ``=``, user attrs named
  ``uuid`` / ``engine`` / ``node_type``).
* `TestTheDunders`: ``str`` / ``repr`` / ``hash`` / ``==`` / ``os.fspath`` read
  the node's private naming, never a public method.
* `TestIntrospection`: ``inspect.getmembers`` on an unassigned Lambert and on a
  deleted node.
* `TestOldNames`: an old spelling raises naming the new one, on the miss path.
* `TestBarePredicates`: a node predicate named without brackets has no truth
  value (user decision: the guard), a classmethod's too, and over a List.
* `TestSetNamespace`: one undo step; a locked node is refused before anything
  is made.
* `TestRigSourceLint`: rig's own code reads no node-only old name and names no
  node predicate bare.
"""

import ast
import functools
import inspect
import os
import tempfile

from maya import cmds
from maya.api import OpenMaya

import rig
from rig import List, Node
from rig._internal.plug import Plug
from rig.nodetypes import _base
from rig.nodetypes.dg_node import DGNode
from rig.nodetypes.transform import Transform
from rig._tests._base import MayaTestCase

# read by name, so that on a tree without the ban each test fails on its own
# (the negative control) instead of the module failing to import
_ABSENT        = type("_Absent", (), {})
_BarePredicate = getattr(_base, "_BarePredicate", _ABSENT)

# the node properties before the ban, by class, and their methods
_RENAMED = {
    "name": "get_name", "long_name": "get_long_name", "short_name": "get_short_name",
    "clean_name": "get_clean_name", "uuid": "get_uuid", "namespace": "get_namespace",
    "node_type": "get_node_type", "is_locked": "is_locked", "is_valid": "is_valid",
    "mobject": "get_mobject", "fn_set": "get_fn_set", "mdagpath": "get_mdagpath",
    "is_shape": "is_shape", "local_shape_attr": "get_local_shape_attr",
    "world_shape_attr": "get_world_shape_attr", "injection_node": "get_injection_node",
    "component_tags": "get_component_tags", "num_vertices": "get_num_vertices",
    "num_polygons": "get_num_polygons", "num_weight_points": "get_num_weight_points",
    "num_cvs": "get_num_cvs", "uv_sets": "get_uv_sets", "current_uv_set": "get_current_uv_set",
    "paintable_maps": "get_paintable_maps", "engine": "get_engine", "is_default": "is_default",
    "num_targets": "get_num_targets", "num_keys": "get_num_keys", "file_path": "get_file_path",
    "file_path_with_copy_number": "get_file_path",
}
_PREDICATES = ("is_locked", "is_valid", "is_shape", "is_default")
_RIG_DIR    = os.path.dirname(rig.__file__)


def _node_classes():
    """Every node class rig defines (its modules' NodeMeta classes)."""
    import sys

    seen = []
    for name, module in list(sys.modules.items()):
        if not (name == "rig" or name.startswith("rig.")) or name.startswith("rig._tests"):
            continue
        for value in list(vars(module).values()):
            if (
                isinstance(value, type)
                and issubclass(value, _base.Node)
                and value.__module__.startswith("rig.")
                and value not in seen
            ):
                seen.append(value)
    return seen


def _cube(name="cube"):
    return Node(cmds.polyCube(name=name, ch=False)[0]).get_shape()


_AOV_CODE = """
import json, os, sys
import maya.standalone
maya.standalone.initialize(name="python")
try:
    from maya import cmds
    try:
        cmds.loadPlugin("mtoa", quiet=True)
    except RuntimeError:
        print("RESULT " + json.dumps("SKIP"))
    else:
        import rig
        from rig import Node
        aov = Node(cmds.createNode("aiAOV", name="aov"))
        plug = aov.name
        aov.name = "diffuse"
        print("RESULT " + json.dumps({
            "file": rig.__file__, "plug": type(plug).__name__, "str": str(plug),
            "set": cmds.getAttr("aov.name"), "node": str(aov), "get_name": aov.get_name(),
        }))
finally:
    sys.stdout.flush()
    try:
        maya.standalone.uninitialize()  # (os._exit alone crashes mtoa's shutdown)
    except BaseException:
        pass
    sys.stdout.flush()
    os._exit(0)
"""


def _in_own_mayapy(code):
    """What `code` prints after ``RESULT `` (json), run in a new mayapy on this
    rig (its file checked; "file" is dropped)."""
    import json
    import shutil
    import subprocess
    import sys

    env = dict(os.environ)
    # this rig's root only (the test process's whole path breaks mtoa's Qt import)
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(rig.__file__)))
    env["MAYA_DISABLE_CER"] = "1"
    cwd = tempfile.mkdtemp(prefix="rig_node_ban_")
    try:
        proc = subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env, timeout=600,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    finally:
        shutil.rmtree(cwd, ignore_errors=True)
    output = proc.stdout.decode("utf-8", "replace")
    lines  = [line for line in output.splitlines() if line.startswith("RESULT ")]
    assert len(lines) == 1, f"exit {proc.returncode}: " + output[-2000:]
    result = json.loads(lines[0][len("RESULT "):])
    if isinstance(result, dict) and "file" in result:
        here = os.path.normcase(os.path.dirname(rig.__file__))
        assert os.path.normcase(os.path.dirname(result.pop("file"))) == here, output[-2000:]
    return result


class TestRenamedReads(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_the_names(self):
        cmds.createNode("transform", name="g1")
        cmds.createNode("transform", name="g2")
        cmds.createNode("transform", name="dup", parent="g1")
        cmds.createNode("transform", name="dup", parent="g2")
        dup = Node("|g1|dup")
        self.assertEqual(dup.get_name(), "g1|dup")
        self.assertEqual(str(dup), "g1|dup")
        self.assertEqual(dup.get_long_name(), "|g1|dup")
        self.assertEqual(dup.get_short_name(), "dup")
        cmds.namespace(add="ns")
        ns = Node(cmds.createNode("multiplyDivide", name="ns:md"))
        self.assertEqual(ns.get_name(), "ns:md")
        self.assertEqual(ns.get_long_name(), "ns:md")  # a DG node's long name is its name
        self.assertEqual(ns.get_clean_name(), "md")
        self.assertEqual(ns.get_namespace(), "ns")

    def test_the_node_queries(self):
        t     = Transform.create(name="t")
        shape = _cube()
        self.assertEqual(t.get_uuid(), cmds.ls("t", uuid=True)[0])
        self.assertEqual(t.get_node_type(), "transform")
        self.assertIsInstance(t.get_mobject(), OpenMaya.MObject)
        self.assertIsInstance(t.get_fn_set(), OpenMaya.MFnTransform)
        self.assertEqual(t.get_mdagpath().fullPathName(), "|t")
        self.assertIs(t.is_valid(), True)
        self.assertIs(t.is_locked(), False)
        t.lock()
        self.assertIs(t.is_locked(), True)
        t.unlock()
        self.assertIs(t.is_shape(), False)
        self.assertIs(shape.is_shape(), True)
        self.assertIsInstance(shape.get_fn_set(), OpenMaya.MFnMesh)

    def test_the_geometry_members(self):
        mesh = _cube()
        self.assertEqual(mesh.get_num_vertices(), 8)
        self.assertEqual(mesh.get_num_polygons(), 6)
        self.assertEqual(mesh.get_num_weight_points(), 8)
        self.assertEqual(list(mesh.get_uv_sets()), ["map1"])
        self.assertEqual(mesh.get_current_uv_set(), "map1")
        self.assertEqual(mesh.get_paintable_maps(), [])
        self.assertEqual(str(mesh.get_local_shape_attr()), "cubeShape.outMesh")
        self.assertEqual(str(mesh.get_world_shape_attr()), "cubeShape.worldMesh")  # its instance's element
        self.assertEqual(str(mesh.get_injection_node()), "cubeShape")
        self.assertEqual(mesh.get_component_tags(), ["back", "bottom", "front", "left", "right", "top"])
        curve = Node(cmds.curve(p=[(0, 0, 0), (1, 0, 0), (2, 1, 0), (3, 0, 0)], d=3)).get_shape()
        self.assertEqual(curve.get_num_cvs(), 4)
        self.assertEqual(curve.get_num_weight_points(), 4)
        surface = Node(cmds.nurbsPlane(ch=False, u=1, v=1, d=1)[0]).get_shape()
        self.assertEqual(surface.get_num_cvs(), 4)
        self.assertEqual(surface.get_num_weight_points(), 4)

    def test_material_layer_deformer_and_curve(self):
        from rig.nodetypes import AnimCurve, DisplayLayer, Lambert

        cube = Node(cmds.polyCube(name="box", ch=False)[0])
        red  = Lambert.create(name="red")
        cube << red
        self.assertEqual(str(red.get_engine()), "redSG")
        self.assertIs(DisplayLayer("defaultLayer").is_default(), True)
        self.assertIs(DisplayLayer.create(name="L").is_default(), False)
        target = cmds.polyCube(name="smile", ch=False)[0]
        morph  = Node(cmds.blendShape(target, "box", name="morph")[0])
        self.assertEqual(morph.get_num_targets(), 1)
        cmds.setKeyframe("box.tx", time=1, value=0)
        cmds.setKeyframe("box.tx", time=5, value=2)
        curve = Node(cmds.listConnections("box.tx", source=True)[0])
        self.assertIsInstance(curve, AnimCurve)
        self.assertEqual(curve.get_num_keys(), 2)

    def test_a_reference_file_path(self):
        from rig.nodetypes import Reference

        folder = tempfile.mkdtemp(prefix="rig_node_ban_")
        path   = os.path.join(folder, "part.ma").replace("\\", "/")
        cmds.createNode("transform", name="part")
        cmds.file(rename=path)
        cmds.file(save=True, type="mayaAscii", force=True)
        cmds.file(new=True, force=True)
        cmds.file(path, reference=True, namespace="a")
        cmds.file(path, reference=True, namespace="b")
        refs = sorted(Reference.find_by_path(path), key=str)
        self.assertEqual([r.get_file_path() for r in refs], [path, path])
        self.assertEqual(
            sorted(r.get_file_path(copy_number=True) for r in refs), [path, path + "{1}"]
        )
        self.assertEqual(sorted(r.get_namespace() for r in refs), ["a", "b"])
        with self.assertRaises(NotImplementedError):
            refs[0].set_namespace("c")


class TestNoPublicProperty(MayaTestCase):
    def test_no_public_property_on_the_node_classes(self):
        found = []
        for cls in _node_classes():
            for name, value in vars(cls).items():
                if name[:1] != "_" and isinstance(value, (property, functools.cached_property)):
                    found.append(f"{cls.__name__}.{name}")
        self.assertEqual(found, [])

    def test_the_old_names_are_gone_and_the_new_ones_are_plug_methods(self):
        from rig.nodetypes import AnimCurve, BlendShape, DisplayLayer, Material, Mesh, NurbsCurve, Reference

        for cls in (DGNode, Transform, Mesh, NurbsCurve, Material, DisplayLayer, BlendShape, AnimCurve, Reference):
            for old, new in _RENAMED.items():
                if old in _PREDICATES or old == new:
                    continue
                with self.subTest(cls=cls.__name__, old=old):
                    self.assertIs(_base._class_attr(cls, old), _base._MISSING)
        for cls, names in (
            (DGNode, ("get_name", "get_long_name", "get_uuid", "set_namespace", "is_valid", "is_locked")),
            (Mesh, ("get_fn_set", "get_num_vertices", "get_uv_sets", "is_shape")),
            (Reference, ("get_file_path",)),
        ):
            for name in names:
                with self.subTest(cls=cls.__name__, new=name):
                    self.assertTrue(_base.is_plug_method(cls, name))

    def test_a_node_class_body_declares_no_property(self):
        class _Descriptor:
            def __get__(self, inst, owner=None):
                return 1

        for label, member in (
            ("property", property(lambda self: 1)),
            ("cached_property", functools.cached_property(lambda self: 1)),
            ("descriptor", _Descriptor()),
        ):
            with self.subTest(member=label):
                with self.assertRaisesRegex(TypeError, r"^_Bad\.size: a node class has no public"):
                    type(Transform)("_Bad", (Transform,), {"size": member})

    def test_methods_data_and_private_members_pass(self):
        """guard: the refusal leaves methods, class data and private members."""
        class _Fine(Transform):
            side = None  # class data: its own rule (not this step)
            SIZE = 2.0

            def get_size(self):
                return self.SIZE

            @classmethod
            def make(cls):
                return cls

            @staticmethod
            def helper():
                return 3

            @property
            def _private(self):
                return 4

        self.assertTrue(_base.is_plug_method(_Fine, "get_size"))
        self.assertTrue(_base.is_plug_method(_Fine, "make"))
        self.assertTrue(_base.is_plug_method(_Fine, "helper"))
        self.assertIsInstance(vars(_Fine)["_private"], property)


class TestTheClashesTheBanFixes(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_an_ai_aov_name(self):
        """In a mayapy of its own: mtoa loaded here would stay loaded for the rest
        of the suite (test_shade pins aiStandardSurface as unregistered). Skipped
        without mtoa."""
        result = _in_own_mayapy(_AOV_CODE)
        if result == "SKIP":
            self.skipTest("mtoa is not available")
        self.assertEqual(
            result,
            {"plug": "Plug", "str": "aov.name", "set": "diffuse", "node": "aov", "get_name": "aov"},
        )

    def test_a_namespace_attr_and_its_assignment(self):
        t = Transform.create(name="t")
        cmds.addAttr("t", ln="namespace", dt="string")
        t.namespace = "hero"  # the attr's '=' sugar, not a move of the node
        self.assertEqual(cmds.getAttr("t.namespace"), "hero")
        self.assertEqual(str(t), "t")
        self.assertEqual(t.get_namespace(), "")
        self.assertEqual(str(t.namespace), "t.namespace")

    def test_user_attrs_named_like_old_members(self):
        t = Transform.create(name="t")
        for name in ("uuid", "engine", "node_type", "long_name", "fn_set"):
            with self.subTest(name=name):
                cmds.addAttr("t", ln=name, at="double")
                plug = getattr(t, name)
                self.assertIsInstance(plug, Plug)
                self.assertEqual(str(plug), f"t.{name}")
                setattr(t, name, 2.0)
                self.assertEqual(cmds.getAttr(f"t.{name}"), 2.0)
        self.assertEqual(t.get_uuid(), cmds.ls("t", uuid=True)[0])
        self.assertEqual(t.get_node_type(), "transform")
        self.assertEqual(t.get_long_name(), "|t")


class TestTheDunders(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_they_read_the_private_naming(self):
        t = Transform.create(name="t")
        md = Node(cmds.createNode("multiplyDivide", name="md"))

        def refused(node):
            raise AssertionError("a dunder read a public method")

        with mock_patch(DGNode, "get_name", refused), mock_patch(DGNode, "get_long_name", refused):
            for node, name in ((t, "t"), (md, "md")):
                with self.subTest(node=name):
                    self.assertEqual(str(node), name)
                    self.assertEqual(repr(node), f'{type(node).__name__}("{name}")')
                    self.assertEqual(os.fspath(node), name)
                    self.assertEqual(node, Node(name))
                    self.assertEqual(hash(node), hash(Node(name)))
                    self.assertTrue(node > "a")
                    self.assertEqual(str(node.tx if node is t else node.input1X).split(".")[0], name)

    def test_a_maya_attr_named_name_does_not_reach_them(self):
        """guard: with the property gone, a Maya attr 'name' still names nothing."""
        t = Transform.create(name="t")
        cmds.addAttr("t", ln="name", dt="string")
        cmds.addAttr("t", ln="long_name", dt="string")
        self.assertEqual(str(t), "t")
        self.assertEqual(repr(t), 'Transform("t")')
        self.assertEqual(str(t.tx), "t.translateX")
        self.assertEqual(hash(t), hash("|t"))


def mock_patch(cls, name, value):
    from unittest import mock

    return mock.patch.object(cls, name, value)


class TestIntrospection(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_getmembers_of_an_unassigned_lambert(self):
        # its engine property raised ValueError (it feeds no engine)
        red     = Node(cmds.createNode("lambert", name="red"))
        members = dict(inspect.getmembers(red))
        self.assertTrue(inspect.ismethod(members["get_engine"]))
        with self.assertRaisesRegex(ValueError, "feeds no shading engine"):
            red.get_engine()

    def test_getmembers_of_a_deleted_node(self):
        # its name / uuid / fn_set properties raised "already deleted!"
        cmds.undoInfo(state=True, infinity=True)
        t = Transform.create(name="t")
        cmds.delete("t")
        members = dict(inspect.getmembers(t))
        self.assertTrue(inspect.ismethod(members["get_uuid"]))
        with self.assertRaisesRegex(RuntimeError, "already deleted!"):
            t.get_uuid()


class TestOldNames(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_an_old_read_raises_naming_the_new_spelling(self):
        t = Transform.create(name="t")
        for old, new in (
            ("name", "node.get_name() or str(node)"),
            ("uuid", "node.get_uuid()"),
            ("namespace", "node.get_namespace() / node.set_namespace(ns)"),
            ("file_path_with_copy_number", "node.get_file_path(copy_number=True)"),
        ):
            with self.subTest(old=old):
                with self.assertRaises(AttributeError) as caught:
                    getattr(t, old)
                self.assertEqual(
                    str(caught.exception), f"Attribute not found: t.{old}; node.{old} is now {new}"
                )
        self.assertFalse(hasattr(t, "uuid"))

    def test_an_old_setter_raises_naming_the_new_spelling(self):
        t = Transform.create(name="t")
        for old, new in (
            ("name", "node.name = x is now node.rename(x)"),
            ("namespace", "node.namespace = ns is now node.set_namespace(ns)"),
            ("is_locked", "node.is_locked = state is now node.lock() / node.unlock()"),
        ):
            with self.subTest(old=old):
                with self.assertRaises(AttributeError) as caught:
                    setattr(t, old, "x")
                self.assertTrue(str(caught.exception).endswith(new), str(caught.exception))
        self.assertEqual(str(t), "t")
        self.assertIs(t.is_locked(), False)


class TestBarePredicates(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_a_bare_predicate_has_no_truth_value(self):
        from rig.nodetypes import DisplayLayer

        t = Transform.create(name="t")
        for node, name in ((t, "is_locked"), (t, "is_valid"), (t, "is_shape"),
                           (DisplayLayer("defaultLayer"), "is_default"), (t, "is_type")):
            with self.subTest(name=name):
                bare = getattr(node, name)
                self.assertIs(type(bare), _BarePredicate)
                with self.assertRaisesRegex(TypeError, rf"^{name} is a method: call it \({name}\(\)\)$"):
                    bool(bare)

    def test_a_bare_predicate_is_still_the_method(self):
        t    = Transform.create(name="t")
        bare = t.is_valid
        self.assertIs(bare(), True)
        self.assertEqual(bare.__name__, "is_valid")
        self.assertIs(bare.__self__, t)
        self.assertIs(bare.__func__, DGNode.is_valid)
        self.assertEqual(bare.__doc__, DGNode.is_valid.__doc__)
        self.assertEqual(list(inspect.signature(bare).parameters), [])
        self.assertIs(t.is_type("t"), True)  # a classmethod's call

    def test_an_attr_named_like_a_predicate_is_the_plug(self):
        t = Transform.create(name="t")
        cmds.addAttr("t", ln="is_valid", at="bool")
        self.assertIsInstance(t.is_valid, Plug)
        self.assertIs(t.is_valid(), True)

    def test_over_a_list(self):
        nodes = List([Transform.create(name="a"), Transform.create(name="b")])
        with self.assertRaisesRegex(TypeError, r"call it on each one"):
            bool(nodes.is_valid)
        self.assertEqual([x.is_valid() for x in nodes], [True, True])


class TestSetNamespace(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_one_undo_step(self):
        cmds.undoInfo(state=True, infinity=True)
        t = Transform.create(name="t")
        cmds.flushUndo()
        t.set_namespace("ns")
        self.assertEqual(str(t), "ns:t")
        self.assertTrue(cmds.namespace(exists="ns"))
        cmds.undo()
        self.assertEqual(str(t), "t")
        self.assertFalse(cmds.namespace(exists="ns"))
        cmds.redo()
        self.assertEqual(str(t), "ns:t")

    def test_a_locked_node_is_refused_before_anything_is_made(self):
        t = Transform.create(name="t")
        t.lock()
        with self.assertRaisesRegex(RuntimeError, "it is locked"):
            t.set_namespace("ns")
        self.assertFalse(cmds.namespace(exists="ns"))
        self.assertEqual(str(t), "t")
        t.unlock()


_NODE_ONLY = {
    "long_name", "short_name", "clean_name", "mdagpath", "fn_set", "local_shape_attr",
    "world_shape_attr", "injection_node", "component_tags", "num_vertices", "num_polygons",
    "num_weight_points", "num_cvs", "uv_sets", "current_uv_set", "paintable_maps", "num_targets",
    "file_path", "file_path_with_copy_number",
}
# ColorSet (a scene object outside nodes: its own step) keeps its is_valid property
_ALLOWED_BARE = {("nodetypes/mesh.py", "is_valid")}


def _rig_sources():
    for base, dirs, files in os.walk(_RIG_DIR):
        dirs[:] = sorted(d for d in dirs if d not in ("_tests", "__pycache__"))
        for name in sorted(f for f in files if f.endswith(".py")):
            path = os.path.join(base, name)
            rel  = os.path.relpath(path, _RIG_DIR).replace(os.sep, "/")
            with open(path, encoding="utf-8") as fh:
                yield rel, ast.parse(fh.read())


class TestRigSourceLint(MayaTestCase):
    """rig's own code (outside ``_tests``), read with ``ast``."""

    def test_no_old_node_name(self):
        found = []
        for rel, tree in _rig_sources():
            for n in ast.walk(tree):
                if isinstance(n, ast.Attribute) and n.attr in _NODE_ONLY:
                    found.append(f"{rel}:{n.lineno} .{n.attr}")
        self.assertEqual(found, [])

    def test_no_bare_node_predicate(self):
        found = []
        for rel, tree in _rig_sources():
            callees = {id(c.func) for c in ast.walk(tree) if isinstance(c, ast.Call)}
            for n in ast.walk(tree):
                if (
                    isinstance(n, ast.Attribute)
                    and n.attr in ("is_valid", "is_shape", "is_default")
                    and id(n) not in callees
                    and (rel, n.attr) not in _ALLOWED_BARE
                ):
                    found.append(f"{rel}:{n.lineno} .{n.attr}")
        self.assertEqual(found, [])

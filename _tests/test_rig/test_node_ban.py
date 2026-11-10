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
* `TestOldNames`: an old spelling raises naming the new one, on the miss path: a
  property before the ban, or a method the names review retired (user decision
  2026-10-30), which also leaves its class.
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
import re
import tempfile

from maya import cmds
from maya.api import OpenMaya

import rig
from rig import List, Node
from rig._internal.plug import Plug
from rig.nodetypes import _base, AnimCurveTL, dg_node, DisplayLayer
from rig.nodetypes.dg_node import _OLD_NODE_NAMES, DGNode
from rig.nodetypes.transform import Transform
from rig._tests._base import MayaTestCase, own_interpreter

# read by name, so that on a tree without the ban each test fails on its own
# (the negative control) instead of the module failing to import
_ABSENT        = type("_Absent", (), {})
_BarePredicate = getattr(_base, "_BarePredicate", _ABSENT)

# the node properties before the ban, by class, and their methods
_RENAMED = {
    "name": "get_name", "long_name": "get_long_name", "short_name": "get_short_name",
    "clean_name": "get_short_name", "uuid": "get_uuid", "namespace": "get_namespace",
    "node_type": "get_node_type", "is_locked": "is_locked", "is_valid": "is_valid",
    "mobject": "get_mobject", "fn_set": "get_fn_set", "mdagpath": "get_mdagpath",
    "is_shape": "is_shape", "local_shape_attr": "get_output_attr",
    "world_shape_attr": "get_output_attr", "injection_node": "get_injection_node",
    "component_tags": "get_tag_names", "num_vertices": "get_vertex_count",
    "num_polygons": "get_face_count", "num_weight_points": "get_weight_point_count",
    "num_cvs": "get_cv_count", "uv_sets": "get_uv_sets", "current_uv_set": "get_current_uv_set",
    "paintable_maps": "get_map_names", "engine": "get_engine", "is_default": "is_default",
    "num_targets": "get_target_count", "num_keys": "get_key_count", "file_path": "get_file_path",
    "file_path_with_copy_number": "get_file_path",
}
_PREDICATES = ("is_locked", "is_valid", "is_shape", "is_default")
_RIG_DIR    = os.path.dirname(rig.__file__)

# the node methods the names review retired (user decision 2026-10-30): the classes
# that had one (keys of `_SAMPLES`, which builds a node of each) and the new spelling
# (None: a stub that went, with no new spelling)
_OLD_METHODS = {
    "ensure_valid":   ("DGNode", "is_valid"),
    "get_clean_name": ("DGNode", "get_short_name"),
    "for_node":       ("DisplayLayer", "of"),
    "get_num_keys":   ("AnimCurve", "get_key_count"),
    # a geometry's component tags
    "add_component_tag":             ("Mesh NurbsCurve NurbsSurface", "add_tag"),
    "remove_component_tag":          ("Mesh NurbsCurve NurbsSurface", "delete_tag"),
    "rename_component_tag":          ("Mesh NurbsCurve NurbsSurface", "rename_tag"),
    "set_component_tag_contents":    ("Mesh NurbsCurve NurbsSurface", "set_tag_contents"),
    "select_component_tag_contents": ("Mesh NurbsCurve NurbsSurface", "select_tag_contents"),
    "get_component_tags":            ("Mesh NurbsCurve NurbsSurface", "get_tag_names"),
    "has_component_tag":             ("Mesh NurbsCurve NurbsSurface", "has_tag"),
    "get_component_tag_index":       ("Mesh NurbsCurve NurbsSurface", "get_tag_index"),
    "get_component_tag_name":        ("Mesh NurbsCurve NurbsSurface", "get_tag_name"),
    "get_component_tag_history":     ("Mesh NurbsCurve NurbsSurface", "get_tag_history"),
    "get_component_tag_category":    ("Mesh NurbsCurve NurbsSurface", "get_tag_category"),
    "set_component_tag_category":    ("Mesh NurbsCurve NurbsSurface", "set_tag_category"),
    "get_component_tag_contents":    ("Mesh NurbsCurve NurbsSurface", "get_tag_contents"),
    "get_component_tag_indices":     ("Mesh NurbsCurve NurbsSurface", "get_tag_component_indices"),
    "is_component_tag_empty":        ("Mesh NurbsCurve NurbsSurface", "is_tag_empty"),
    "get_component_tag_data":        ("Mesh NurbsCurve NurbsSurface", "get_tag_data"),
    "serialize_component_tags":      ("Mesh NurbsCurve NurbsSurface", "serialize_tags"),
    "transfer_component_tags":       ("Mesh", "transfer_tags"),
    # the component tag read that had no verb
    "component_tag_expression_subset_state": ("Mesh NurbsCurve NurbsSurface", "get_tag_subset_state"),
    # a shape's output and its normals
    "get_local_shape_attr":        ("Mesh NurbsCurve NurbsSurface", "get_output_attr"),
    "get_world_shape_attr":        ("Mesh NurbsCurve NurbsSurface", "get_output_attr"),
    "get_per_face_vertex_normals": ("Mesh", "get_face_vertex_normals"),
    # the counts
    "get_num_vertices":      ("Mesh", "get_vertex_count"),
    "get_num_polygons":      ("Mesh", "get_face_count"),
    "get_num_weight_points": ("Mesh NurbsCurve NurbsSurface", "get_weight_point_count"),
    "get_num_cvs":           ("NurbsCurve NurbsSurface", "get_cv_count"),
    # the maps
    "get_paintable_maps":  ("Mesh", "get_map_names"),
    "paint_map":           ("Mesh", "enter_paint_tool"),
    "sanitize_map_values": ("Mesh", "set_map_values"),
    "merge_maps":          ("Mesh", None),
    "merge_maps_list":     ("Mesh", None),
    # a follicle's
    "set_uv_values": ("Follicle", "set_uv"),
    # a joint's
    "convert_orients_to_rotation": ("Joint", "orients_to_rotations"),
    "convert_rotation_to_orients": ("Joint", "rotations_to_orients"),
    "hierarchy_to_rotations":      ("Joint", "orients_to_rotations"),
    "hierarchy_to_orients":        ("Joint", "rotations_to_orients"),
    "orient_joint":                ("Joint", "orient"),
    "orient_chain":                ("Joint", "orient"),
    # a blendShape's
    "target_data_attrs": ("BlendShape", "get_target_data_attrs"),
    "get_num_targets":   ("BlendShape", "get_target_count"),
    "set_targets_data":  ("BlendShape", "set_target_data"),
    # a skinCluster's
    "copy_skincluster":          ("SkinCluster", "transfer_from_meshes"),
    "connect_bind_pre_matrices": ("SkinCluster", "connect_bind_pre_matrix"),
    # a transform's, which a joint inherits
    "set_xfrom_attrs_locked": ("Transform Joint", "set_xform_attrs_locked"),
}
_SAMPLES = {
    "DGNode":       lambda: Node(cmds.createNode("multiplyDivide", name="md")),
    "DisplayLayer": lambda: DisplayLayer.define("layer"),
    "AnimCurve":    lambda: AnimCurveTL.create([0, 1], name="curve"),
    "Mesh":         lambda: _cube(),
    "NurbsCurve":   lambda: Node(cmds.curve(p=[(0, 0, 0), (1, 0, 0), (2, 1, 0), (3, 0, 0)], d=3)).get_shape(),
    "NurbsSurface": lambda: Node(cmds.nurbsPlane(ch=False)[0]).get_shape(),
    "Follicle":     lambda: Node(cmds.createNode("follicle")),
    "Transform":    lambda: Node(cmds.createNode("transform", name="xform")),
    "Joint":        lambda: Node(cmds.createNode("joint", name="jnt")),
    "BlendShape":   lambda: _blendshape(),
    "SkinCluster":  lambda: _skincluster(),
}
# a handle's method the names review retired: its table is the handle's, not the nodes'
_OLD_NAMES = {"as_current_color_set": "as_current"}
# the hints that need their keyword to reach the same member: `_OLD_METHODS` reads a
# hint up to its bracket
_OLD_KEYWORDS = {
    "ensure_valid":         "is_valid(failfast=True)",
    "clean_name":           "get_short_name(namespace=False)",
    "get_clean_name":       "get_short_name(namespace=False)",
    "world_shape_attr":     "get_output_attr(world_space=True)",
    "get_world_shape_attr": "get_output_attr(world_space=True)",
}


def _is_plug_method(cls, name):
    """True if ``name`` on node class or plug class `cls` is a plug method. Re-pinned
    (user decision 2026-10-16, class data: ``rig.nodetypes._base.is_plug_method`` was
    deleted with ``spec._warn_shadowed``, its one user): read off the class dicts."""
    return isinstance(
        _base._class_attr(cls, name), _base._PLUG_DESCRIPTORS + (_base._ChildOrMethod,)
    )


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


def _blendshape():
    target = cmds.polyCube(name="target", ch=False)[0]
    return Node(cmds.blendShape(target, cmds.polyCube(name="base", ch=False)[0], name="morph")[0])


def _skincluster():
    joint = cmds.createNode("joint", name="bone")
    return Node(cmds.skinCluster(joint, cmds.polyCube(name="skinned", ch=False)[0], name="skin")[0])


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
    """What `code` prints after ``RESULT `` (json), which ran in a new mayapy on this rig (its
    file checked; "file" dropped) until 2026-11-10: the suite runs in this session only now, so
    the test is skipped (`own_interpreter`); the cases stay as they ran."""
    own_interpreter("a case in a fresh mayapy (mtoa loaded here would stay loaded)")


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
        self.assertEqual(ns.get_short_name(namespace=False), "md")
        self.assertEqual(ns.get_namespace(), "ns")

    def test_the_short_name_and_its_namespace(self):
        cmds.namespace(add="ns")
        cmds.namespace(add="deep", parent="ns")
        cmds.createNode("transform", name="ns:g1")
        cmds.createNode("transform", name="ns:g2")
        cmds.createNode("transform", name="ns:deep:box", parent="ns:g1")
        cmds.createNode("transform", name="ns:deep:box", parent="ns:g2")
        box = Node("|ns:g1|ns:deep:box")
        self.assertEqual(box.get_name(), "ns:g1|ns:deep:box")
        self.assertEqual(box.get_short_name(), "ns:deep:box")
        self.assertEqual(box.get_short_name(namespace=True), "ns:deep:box")
        self.assertEqual(box.get_short_name(namespace=False), "box")
        self.assertEqual(Transform.get_short_name(box, namespace=False), "box")
        self.assertEqual(Node(cmds.createNode("transform", name="plain")).get_short_name(namespace=False), "plain")
        # the module's function, on a name or a node
        self.assertEqual(dg_node.get_short_name("|grp|ns:cube"), "ns:cube")
        self.assertEqual(dg_node.get_short_name("|grp|ns:cube", namespace=False), "cube")
        self.assertEqual(dg_node.get_short_name(box, namespace=False), "box")
        # the keyword is keyword-only, both ways
        with self.assertRaises(TypeError):
            box.get_short_name(False)
        with self.assertRaises(TypeError):
            dg_node.get_short_name(box, False)

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
        self.assertEqual(mesh.get_vertex_count(), 8)
        self.assertEqual(mesh.get_face_count(), 6)
        self.assertEqual(mesh.get_weight_point_count(), 8)
        self.assertEqual(list(mesh.get_uv_sets()), ["map1"])
        self.assertEqual(mesh.get_current_uv_set(), "map1")
        self.assertEqual(mesh.get_map_names(), [])
        self.assertEqual(str(mesh.get_output_attr()), "cubeShape.outMesh")
        self.assertEqual(str(mesh.get_output_attr(world_space=True)), "cubeShape.worldMesh")  # its instance's element
        self.assertEqual(str(mesh.get_injection_node()), "cubeShape")
        self.assertEqual(mesh.get_tag_names(), ["back", "bottom", "front", "left", "right", "top"])
        curve = Node(cmds.curve(p=[(0, 0, 0), (1, 0, 0), (2, 1, 0), (3, 0, 0)], d=3)).get_shape()
        self.assertEqual(curve.get_cv_count(), 4)
        self.assertEqual(curve.get_weight_point_count(), 4)
        surface = Node(cmds.nurbsPlane(ch=False, u=1, v=1, d=1)[0]).get_shape()
        self.assertEqual(surface.get_cv_count(), 4)
        self.assertEqual(surface.get_weight_point_count(), 4)

    def test_a_shape_output_and_its_space(self):
        for shape, local, world in (
            (_cube(), "outMesh", "worldMesh"),
            (_SAMPLES["NurbsCurve"](), "local", "worldSpace"),
            (_SAMPLES["NurbsSurface"](), "local", "worldSpace"),
        ):
            with self.subTest(shape=type(shape).__name__):
                self.assertEqual(shape.get_output_attr().get_name(), local)
                self.assertEqual(shape.get_output_attr(world_space=False).get_name(), local)
                self.assertEqual(shape.get_output_attr(world_space=True).get_name(), world)
                self.assertEqual(shape.get_output_attr(world_space=True).get_logical_index(), 0)
                # each is read once, and held
                self.assertIs(shape.get_output_attr(), shape.get_output_attr())
                self.assertIs(shape.get_output_attr(world_space=True), shape.get_output_attr(world_space=True))
                # the keyword is keyword-only
                with self.assertRaises(TypeError):
                    shape.get_output_attr(True)

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
        self.assertEqual(morph.get_target_count(), 1)
        cmds.setKeyframe("box.tx", time=1, value=0)
        cmds.setKeyframe("box.tx", time=5, value=2)
        curve = Node(cmds.listConnections("box.tx", source=True)[0])
        self.assertIsInstance(curve, AnimCurve)
        self.assertEqual(curve.get_key_count(), 2)

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
            (Mesh, ("get_fn_set", "get_vertex_count", "get_uv_sets", "is_shape")),
            (Reference, ("get_file_path",)),
        ):
            for name in names:
                with self.subTest(cls=cls.__name__, new=name):
                    self.assertTrue(_is_plug_method(cls, name))

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
        """guard: the refusal leaves methods, class data and private members.
        Re-pinned (user decision 2026-10-16, class data): lowercase public data is
        refused now (`TestClassData`), so the class keeps Python state under
        ``_side``; ``SIZE`` is class data, read through the class."""
        class _Fine(Transform):
            _side = None
            SIZE = 2.0

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

        self.assertTrue(_is_plug_method(_Fine, "get_size"))
        self.assertTrue(_is_plug_method(_Fine, "make"))
        self.assertTrue(_is_plug_method(_Fine, "helper"))
        self.assertIsInstance(vars(_Fine)["_private"], property)


class TestTheClashesTheBanFixes(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_an_ai_aov_name(self):
        """In a mayapy of its own: mtoa loaded here would stay loaded for the rest
        of the suite (test_shade pins aiStandardSurface as unregistered). Skipped
        without mtoa, and since 2026-11-10 skipped as a whole (`_in_own_mayapy`)."""
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

    def test_an_old_method_raises_naming_a_new_spelling_that_exists(self):
        nodes = {owner: build() for owner, build in _SAMPLES.items()}
        for old, (owners, new) in _OLD_METHODS.items():
            for owner in owners.split():
                node = nodes[owner]
                with self.subTest(old=old, owner=owner):
                    self.assertIs(_base._class_attr(type(node), old), _base._MISSING)
                    with self.assertRaises(AttributeError) as caught:
                        getattr(node, old)()
                    self.assertIn(f"; node.{old} is now ", str(caught.exception))
                    if new is not None:
                        self.assertTrue(_is_plug_method(type(node), new), new)
                        self.assertIn(f"{new}(", str(caught.exception).split(" is now ")[1])

    def test_a_hint_keeps_its_keyword(self):
        mesh = _cube()
        for old, new in _OLD_KEYWORDS.items():
            with self.subTest(old=old):
                with self.assertRaises(AttributeError) as caught:
                    getattr(mesh, old)
                self.assertEqual(str(caught.exception).split("; ", 1)[1], f"node.{old} is now node.{new}")

    def test_a_colour_set_names_the_method_it_retired(self):
        from rig.nodetypes.mesh import ColorSet

        cset = _cube().add_color_set("tint", ColorSet.Representation.RGBA)
        for old, new in _OLD_NAMES.items():
            with self.subTest(old=old):
                self.assertFalse(hasattr(ColorSet, old))
                self.assertTrue(inspect.isfunction(getattr(ColorSet, new)), new)
                with self.assertRaises(AttributeError) as caught:
                    getattr(cset, old)()
                self.assertIn(f"; cset.{old} is now cset.{new}()", str(caught.exception))

    def test_every_hint_names_a_method_that_exists(self):
        classes = _node_classes()
        for old, hint in _OLD_NODE_NAMES.items():
            for name in set(re.findall(r"(\w+)\(", hint)) - {"str"}:
                with self.subTest(old=old, name=name):
                    self.assertTrue(any(_is_plug_method(cls, name) for cls in classes), hint)
        for old, new in _RENAMED.items():
            with self.subTest(old=old, new=new):
                self.assertTrue(any(_is_plug_method(cls, new) for cls in classes), new)

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
        # re-pinned (user decision 2026-10-30): `is_valid(*, failfast=False)`
        self.assertEqual(list(inspect.signature(bare).parameters), ["failfast"])
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
# re-pinned (user decision 2026-10-16, the predicate guard): ColorSet's is_valid
# is a guarded method too, so nothing is allowed bare
_ALLOWED_BARE = set()


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

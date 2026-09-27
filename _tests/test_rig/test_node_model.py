"""Tests for the node model: one node class hierarchy, the owner rule, typed
nodes speaking the DSL.

Each class names the round-4a step it belongs to:

* M1: the typed internals (``rig/nodetypes``) read Maya attributes through
  ``find_attr`` (an AST lint and a runtime tripwire), so they keep getting
  Attributes once typed nodes speak the DSL (``node.<attr>`` gives a Plug);
  the package sites outside ``rig/nodetypes`` that read a typed node's attr
  by name are pinned where they are fine with a Plug.
* M2: typed constructors (``__dict__`` writes in the usual key order, copy only
  within the class, ``Mesh(transform_node)``) and the ``_`` probe guard on
  typed nodes, also on a node a new scene, a file open or a reference unload
  freed (K S3 ``ff99c29``); the API 1.0 handle helpers and the complete list
  of the package's ``_objhandle1`` / ``_fn_set1`` readers (for round 5, NW6).
* M3: typed nodes speak the DSL while the ``Node`` wrapper is still there (K
  S4a ``f7511a2`` and the merge-only fixes of ``82547f5``, on round 3's owner
  rule): ``PyNode(x).attr`` is a Plug owned by the typed node, the ``=`` sugar
  (variant K), ``<<`` / ``>>``, the component fallbacks, the Plug ``_`` rule,
  ``find_attr(Plug)``; and the edge cases of the round-4a checklist (deleted,
  freed, instanced, namespaced nodes, components, dynamic attrs, identity).
* M4: the class swap (K S4b ``ca9e345`` / ``4d3a02e`` and the merge-only parts
  of ``82547f5`` / ``9244e90``): ``Node`` is the root class and the DSL factory
  (``Node(x) is x``, typed repr, ``isinstance(PyNode(x), Node)``), the wrapper is
  gone, ``Container`` is a ``DGNode`` subclass (symmetric equality, owner,
  lookup order, the publish guard), ``Node.wrap`` on the metaclass; round 3's
  one-key plug hash is kept.
"""

import ast
import copy
import os
import re
import shutil
import sys
import tempfile
from unittest import mock

import numpy as np
from maya import cmds
from maya.api import OpenMaya
from rig import container, Node, Plug
from rig.nodetypes import DGNode, PyNode, Transform
from rig._internal.math_nodes import _decompose_matrix
from rig._internal.members import Components
from rig._tests._base import MayaTestCase


def _mobject(name):
    sel = OpenMaya.MSelectionList()
    sel.add(name)
    return sel.getDependNode(0)


def _all_subclasses(cls):
    found = {cls}
    for sub in cls.__subclasses__():
        found |= _all_subclasses(sub)
    return found


def _maya_attr_loads_via_self():
    """Every ``self.<name>`` load in a method of a ``rig/nodetypes`` node class
    whose name is no Python member of the class (MRO names, and names some
    ``self.<name> = ...`` in the class hierarchy stores): a read that falls
    through to ``DGNode.__getattr__`` and would get a Maya attribute."""
    import rig.nodetypes as nodetypes

    root  = os.path.dirname(nodetypes.__file__)
    files = [os.path.join(root, f) for f in sorted(os.listdir(root)) if f.endswith(".py")]
    plugins = os.path.join(root, "plugins")
    if os.path.isdir(plugins):
        files += [
            os.path.join(plugins, f) for f in sorted(os.listdir(plugins)) if f.endswith(".py")
        ]
    classes = {cls.__name__: cls for cls in _all_subclasses(DGNode)}
    trees   = {path: ast.parse(open(path, encoding="utf-8").read()) for path in files}

    stored = {}
    for tree in trees.values():
        for cnode in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            names = stored.setdefault(cnode.name, set())
            for sub in ast.walk(cnode):
                if (
                    isinstance(sub, ast.Attribute)
                    and isinstance(sub.ctx, ast.Store)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id in ("self", "inst")
                ):
                    names.add(sub.attr)
                    if sub.attr.startswith("__") and not sub.attr.endswith("__"):
                        names.add(f"_{cnode.name}{sub.attr}")

    def python_names(cls):
        names = set(dir(cls))
        for base in cls.__mro__:
            names |= stored.get(base.__name__, set())
        # Python state written through ``__dict__`` (never a Maya attr name):
        # the API handles and caches, plus round 3's node serial and the
        # taken DAG path of a stale instance path.
        return names | {
            "_mobject", "_fn_set", "_fn_set1", "_objhandle1", "_attr_dict", "_mdagpath",
            "_node_serial", "_taken_mdagpath",
        }

    loads = []
    for path, tree in trees.items():
        for cnode in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            cls = classes.get(cnode.name)
            if cls is None:
                continue
            names = python_names(cls)
            for fn in (n for n in cnode.body if isinstance(n, ast.FunctionDef)):
                if any(
                    isinstance(d, ast.Name) and d.id in ("classmethod", "staticmethod")
                    for d in fn.decorator_list
                ):
                    continue
                for sub in ast.walk(fn):
                    if not (
                        isinstance(sub, ast.Attribute)
                        and isinstance(sub.value, ast.Name)
                        and sub.value.id == "self"
                        and isinstance(sub.ctx, ast.Load)
                    ):
                        continue
                    name = sub.attr
                    if name.startswith("__") and not name.endswith("__"):
                        name = f"_{cnode.name}{name}"
                    if name not in names:
                        loads.append(
                            f"{os.path.basename(path)}:{sub.lineno} "
                            f"{cnode.name}.{fn.name} -> self.{sub.attr}"
                        )
    return loads


def _nodetypes_lookups(func):
    """The non-``_`` names ``DGNode.__getattr__`` resolves for callers in
    ``rig/nodetypes`` while ``func`` runs, as ``(file, line, name)``."""
    import rig.nodetypes as nodetypes

    root     = os.path.normcase(os.path.dirname(nodetypes.__file__))
    seen     = []
    original = DGNode.__getattr__

    def recording(self, name):
        if name[:1] != "_":
            frame  = sys._getframe(1)
            caller = os.path.normcase(frame.f_code.co_filename)
            if caller.startswith(root):
                seen.append((os.path.basename(caller), frame.f_lineno, name))
        return original(self, name)

    with mock.patch.object(DGNode, "__getattr__", recording):
        func()
    return seen


class TestNodetypesReadsThroughFindAttr(MayaTestCase):
    """M1: the typed internals read Maya attrs through ``find_attr`` (Attributes),
    never through ``node.<attr>`` (which gives a DSL Plug on a DSL node)."""

    TEST_START_NEW_SCENE = True

    def test_self_loads_are_python_members(self):
        self.assertEqual(_maya_attr_loads_via_self(), [])

    def test_runtime_tripwire(self):
        from rig.nodetypes import BlendShape, Choice, Follicle, SkinCluster

        def transforms():
            joint = PyNode(cmds.createNode("joint", name="jnt"))
            loc   = PyNode(cmds.spaceLocator(name="loc")[0])
            for node in (joint, loc):
                node.serialize()
                node.get_matrix(world_space=True)
                node.get_matrix(world_space=False)
            cube = PyNode(cmds.polyCube(name="cube", ch=False)[0])
            cube.duplicate_geometry()
            shape = cube.get_children(type="mesh")[0]
            shape.serialize(world_space=False)

        def blendshapes():
            base   = PyNode(cmds.polyCube(name="base", ch=False)[0])
            target = PyNode(cmds.polyCube(name="target", ch=False)[0])
            cmds.xform(f"{target}.vtx[3]", ws=True, t=(10, 2, 3))
            bls = PyNode.create("blendShape", target, base)
            self.assertIsInstance(bls, BlendShape)
            index = bls.add_empty_target("extra")
            bls.set_target_name(index, "renamed")
            bls.get_target_weight_attr(0)
            bls.get_target_data(0)
            bls.serialize()
            self.assertEqual(bls.num_targets, 2)
            self.assertEqual(list(bls.get_target_indices()), [0, 1])

        def follicles():
            plane = cmds.polyPlane(name="plane", ch=False)[0]
            ref   = cmds.spaceLocator(name="ref")[0]
            cmds.xform(ref, t=(0.1, 0, 0.2), ws=True)
            follicle = Follicle.create_on_mesh(plane, ref, name="fol")
            follicle.set_uv_values([0.25, 0.75])

        def skinclusters():
            j1   = PyNode.create("joint", name="j1")
            j2   = PyNode.create("joint", name="j2", parent=j1)
            cube = PyNode(cmds.polyCube(name="skinned", ch=False)[0])
            skin = PyNode.create("skinCluster", cube, (j1, j2))
            self.assertIsInstance(skin, SkinCluster)
            cmds.select(clear=True)
            PyNode.create("joint", name="drv_j1")
            PyNode.create("joint", name="drv_j2")
            skin.connect_bind_pre_matrices(lambda name: "drv_" + name)

        def choices():
            cmds.createNode("transform", name="src")
            pick = PyNode(cmds.createNode("choice", name="pick"))
            self.assertIsInstance(pick, Choice)
            cmds.connectAttr("src.translate", "pick.input[0]")
            self.assertEqual(pick.find_attr("output").data_type, "double3")
            # a message input: getAttr answers "Tdata", so Choice's fallback
            # hook reads the selector to resolve the output
            cmds.connectAttr("src.message", "pick.input[1]")
            cmds.setAttr("pick.selector", 1)
            self.assertEqual(pick.find_attr("output").data_type, "message")

        def component_tags():
            ball = PyNode(cmds.polySphere(name="ball", ch=False)[0])
            mesh = ball.get_children(type="mesh")[0]
            mesh.add_component_tag("t1")
            mesh.set_component_tag_contents("t1", [2, 3], category="v")
            self.assertEqual(mesh.component_tags, ["t1"])
            self.assertTrue(mesh.has_component_tag(0))
            self.assertEqual(mesh.get_component_tag_index("t1"), 0)
            self.assertEqual(mesh.get_component_tag_name(0), "t1")
            mesh.rename_component_tag("t1", "t2")
            self.assertEqual(mesh.component_tags, ["t2"])
            mesh.remove_component_tag("t2")
            self.assertEqual(mesh.component_tags, [])

        for label, func in (
            ("transforms", transforms),
            ("blendshapes", blendshapes),
            ("follicles", follicles),
            ("skinclusters", skinclusters),
            ("choices", choices),
            ("component_tags", component_tags),
        ):
            with self.subTest(case=label):
                cmds.file(new=True, force=True)
                self.assertEqual(_nodetypes_lookups(func), [])


class TestPackageTypedDottedSites(MayaTestCase):
    """M1: the package code outside ``rig/nodetypes`` that reads a Maya attr of a
    typed node by name (the one-off probe of round 4a, M1). The one site is
    ``shorthand._matrix_to_point``: ``_safe_attr(transform_node, "ro")`` on the
    typed parent ``get_parent()`` returns, handed to ``_decompose_matrix`` as
    ``rotate_order=``. It uses the attr as an operand only, so a Plug (typed
    dotted access from M3 on) builds the same network as today's Attribute."""

    TEST_START_NEW_SCENE = True

    def _curve_and_locator(self):
        cmds.curve(d=1, p=[(0, 0, 0), (1, 0, 0), (2, 0, 0)], name="crv")
        cmds.rename(cmds.listRelatives("crv", shapes=True)[0], "crvShape")
        cmds.setAttr("crv.rotateOrder", 3)
        cmds.spaceLocator(name="loc")

    def test_matrix_to_point_connects_the_parent_rotate_order(self):
        self._curve_and_locator()
        Node("crvShape").cv[1] << Node("loc").worldMatrix[0]
        decomposes = cmds.ls(type="decomposeMatrix")
        self.assertEqual(len(decomposes), 1)
        decompose = decomposes[0]
        self.assertEqual(
            cmds.listConnections(f"{decompose}.inputRotateOrder", s=True, d=False, plugs=True),
            ["crv.rotateOrder"],
        )
        self.assertEqual(
            cmds.listConnections("crvShape.controlPoints[1]", s=True, d=False, plugs=True),
            [f"{decompose}.outputTranslate"],
        )
        # the same matrix into another point of the curve: one decompose
        Node("crvShape").cv[2] << Node("loc").worldMatrix[0]
        self.assertEqual(cmds.ls(type="decomposeMatrix"), [decompose])
        self.assertEqual(
            cmds.listConnections("crvShape.controlPoints[2]", s=True, d=False, plugs=True),
            [f"{decompose}.outputTranslate"],
        )

    def test_rotate_order_attribute_and_plug_share_one_decompose(self):
        # what `_matrix_to_point` hands `_decompose_matrix` today (the typed
        # Attribute) and from M3 on (a Plug of the same plug) is one memo entry
        self._curve_and_locator()
        wm      = Node("loc").worldMatrix[0]
        typed   = PyNode("crv").find_attr("ro")
        plug    = Node("crv").ro
        self.assertEqual(type(typed).__name__, "Attribute")
        self.assertEqual(type(plug).__name__, "Plug")
        first   = _decompose_matrix(wm, rotate_order=typed)
        second  = _decompose_matrix(wm, rotate_order=plug)
        self.assertIs(second, first)
        self.assertEqual(str(first), "decomposeMatrix1.outputTranslate")
        self.assertEqual(cmds.ls(type="decomposeMatrix"), ["decomposeMatrix1"])


_DG_KEYS  = ["_mobject", "_fn_set", "_fn_set1", "_objhandle1", "_attr_dict"]
_DAG_KEYS = ["_mdagpath", "_mobject", "_fn_set", "_fn_set1", "_objhandle1", "_attr_dict"]
_GEO_KEYS = ["_Geometry__local_shape_attr", "_Geometry__world_shape_attr"]


def _base_module():
    from rig.nodetypes import _base

    _base._canonical_kind(DGNode)  # binds the constructor parts _copy_wrapper reads
    return _base


class TestTypedLayerPrep(MayaTestCase):
    """M2 (K S3 ``ff99c29``): typed constructors write ``__dict__`` in their usual
    key order, copy only a node of their own class, and ``_`` probes on typed
    nodes are cheap. The key orders are the ones the constructors had at M1."""

    TEST_START_NEW_SCENE = True

    def tearDown(self):
        PyNode._CLASS_BY_TYPE.clear()
        PyNode._CASTABLE_TYPES.clear()
        super().tearDown()

    def test_copy_shares_the_internals(self):
        cmds.createNode("transform", name="a")
        dg  = PyNode("a")
        dup = copy.copy(dg)
        self.assertIs(type(dup), type(dg))
        self.assertIsNot(dup, dg)
        for key in _DAG_KEYS:
            self.assertIs(vars(dup)[key], vars(dg)[key])
        self.assertEqual(dup.name, "a")

    def test_dunder_probe_on_a_freed_node(self):
        cmds.createNode("transform", name="a")
        dg = PyNode("a")
        self.assertFalse(hasattr(dg, "__array__"))
        cmds.file(new=True, force=True)
        # a freed node's API 2.0 objects must not be used: only the handle is read
        probe = mock.Mock()
        vars(dg)["_fn_set"] = probe
        self.assertFalse(hasattr(dg, "__array__"))
        self.assertFalse(hasattr(dg, "_x"))
        self.assertEqual(probe.mock_calls, [])

    def test_private_probe_on_a_deleted_node(self):
        cmds.undoInfo(state=True, infinity=True)
        cmds.createNode("multiplyDivide", name="md")
        dg = PyNode("md")
        cmds.delete("md")
        self.assertFalse(hasattr(dg, "_x"))
        self.assertFalse(hasattr(dg, "__deepcopy__"))
        cmds.undo()
        # a Maya attr of a "_" name on the live node still resolves
        cmds.addAttr("md", longName="__parked__", attributeType="double")
        self.assertEqual(str(dg.__parked__), "md.__parked__")

    def test_mesh_of_a_transform_node_is_its_shape(self):
        from rig.nodetypes import Mesh

        cube  = cmds.polyCube(name="cube", ch=False)[0]
        shape = cmds.listRelatives(cube, shapes=True)[0]
        mesh  = Mesh(PyNode(cube))
        self.assertIs(type(mesh), Mesh)
        self.assertEqual(mesh.name, shape)
        self.assertTrue(mesh.mobject == _mobject(shape))
        # a node of the class itself (or a subclass) still shares its internals
        again = Mesh(mesh)
        self.assertIs(vars(again)["_fn_set"], vars(mesh)["_fn_set"])
        source = PyNode(cube)
        self.assertIs(vars(Transform(source))["_fn_set"], vars(source)["_fn_set"])

    def test_constructor_key_order(self):
        from rig.nodetypes import DAGNode, Mesh

        md    = cmds.createNode("multiplyDivide", name="md")
        xform = cmds.createNode("transform", name="xf")
        cube  = cmds.polyCube(name="cube", ch=False)[0]
        shape = cmds.listRelatives(cube, shapes=True)[0]
        PyNode(_mobject(md))
        PyNode(_mobject(xform))
        PyNode(_mobject(shape))
        from_path = OpenMaya.MSelectionList()
        from_path.add(xform)
        cases = {
            "dg": (DGNode(md), _DG_KEYS),
            "dg_copy": (DGNode(DGNode(md)), _DG_KEYS),
            "dg_mobject": (DGNode(_mobject(md)), _DG_KEYS),
            "dag": (DAGNode(xform), _DAG_KEYS),
            "dag_path": (DAGNode(from_path.getDagPath(0)), _DAG_KEYS),
            "dag_mobject": (DAGNode(_mobject(xform)), [_DAG_KEYS[1], _DAG_KEYS[0]] + _DAG_KEYS[2:]),
            "dag_copy": (
                DAGNode(DAGNode(xform)),
                [_DAG_KEYS[1], _DAG_KEYS[0]] + _DAG_KEYS[2:],
            ),
            "geometry": (Mesh(shape), _DAG_KEYS + _GEO_KEYS),
            "checked_dg": (PyNode(_mobject(md)), _DG_KEYS),
            "checked_dag": (PyNode(_mobject(xform)), _DAG_KEYS),
            "copy_dg": (_base_module()._copy_wrapper(PyNode(_mobject(md))), _DG_KEYS),
            "copy_dag": (_base_module()._copy_wrapper(PyNode(_mobject(xform))), _DAG_KEYS),
            "copy_geometry": (
                _base_module()._copy_wrapper(PyNode(_mobject(shape))),
                _DAG_KEYS + _GEO_KEYS,
            ),
        }
        for label, (node, keys) in cases.items():
            with self.subTest(case=label):
                self.assertEqual(list(vars(node)), keys)


# `_` names a typed node is probed with: a missing name, two dunders Python and
# numpy probe (`copy.deepcopy`, `numpy.asarray`) and a Maya attr of a `_` name
_PROBES = ("_x", "__array__", "__deepcopy__", "__parked__")


class TestPrivateProbeOnAFreedNode(MayaTestCase):
    """M2, E2: the ``_`` probe guard of ``DGNode.__getattr__`` on typed nodes a new
    scene, a file open (bringing nodes of the same names) or a reference unload
    freed. Only the API 1.0 handle is read: ``hasattr`` answers False, a Maya
    attr of a ``_`` name no longer resolves, ``copy.copy`` works and shares the
    handle, and the freed node's API 2.0 fn set is never called."""

    TEST_START_NEW_SCENE = True

    def _build(self):
        md = cmds.createNode("multiplyDivide", name="md")
        cmds.addAttr(md, longName="__parked__", attributeType="double")
        cmds.createNode("transform", name="xf")
        cmds.polyCube(name="cube", ch=False)
        return ["md", "xf", "cubeShape"]

    def _hold(self, names):
        held = [PyNode(name) for name in names]
        self.assertEqual([type(n).__name__ for n in held], ["DGNode", "Transform", "Mesh"])
        self.assertTrue(hasattr(held[0], "__parked__"))
        self.assertEqual(str(held[0].__parked__), f"{names[0]}.__parked__")
        for node in held:
            self.assertFalse(hasattr(node, "_x"))
            self.assertFalse(hasattr(node, "__array__"))
        return held

    def _assert_freed(self, held):
        for node in held:
            with self.subTest(node=type(node).__name__):
                probe = mock.Mock()
                vars(node)["_fn_set"] = probe
                for name in _PROBES:
                    self.assertFalse(hasattr(node, name), name)
                    with self.assertRaisesRegex(AttributeError, f"^{name}$"):
                        getattr(node, name)
                dup = copy.copy(node)
                self.assertIs(type(dup), type(node))
                self.assertIs(vars(dup)["_objhandle1"], vars(node)["_objhandle1"])
                self.assertFalse(dup.is_valid)
                with self.assertRaisesRegex(RuntimeError, "already deleted!$"):
                    node.ensure_valid()
                self.assertEqual(probe.mock_calls, [])

    def test_new_scene(self):
        held = self._hold(self._build())
        cmds.file(new=True, force=True)
        for _ in range(50):
            cmds.createNode("multiplyDivide")  # reuse the freed memory
        self._assert_freed(held)

    def test_file_open_reusing_the_names(self):
        folder = tempfile.mkdtemp(prefix="rig_m2_probe_open_")
        path   = os.path.join(folder, "probe_open.ma").replace("\\", "/")
        try:
            names = self._build()
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
            held = self._hold(names)
            cmds.file(path, open=True, force=True)
            self.assertTrue(all(cmds.objExists(name) for name in names))
            self._assert_freed(held)
            # the node of the same name the file brought is its own
            self.assertEqual(str(PyNode("md").__parked__), "md.__parked__")
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)

    def test_reference_unload(self):
        folder = tempfile.mkdtemp(prefix="rig_m2_probe_ref_")
        path   = os.path.join(folder, "probe_ref.ma").replace("\\", "/")
        try:
            self._build()
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
            cmds.file(new=True, force=True)
            cmds.file(path, reference=True, namespace="ref")
            held = self._hold(["ref:md", "ref:xf", "ref:cubeShape"])
            cmds.file(unloadReference=cmds.referenceQuery(path, referenceNode=True))
            self._assert_freed(held)
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)


# The API 1.0 node handle and fn set, which round 5 (NW6) moves off API 1.0
_API1_NAMES = frozenset({"_objhandle1", "_fn_set1"})
_NW6_MARKER = "NW6: API 1.0 handle"

# Every (module, function) of the package that reads or stores `_objhandle1` /
# `_fn_set1`. The cold readers ask `_handle_valid` / `_handle_alive`; each site
# below reads the objects inline, on a hot path or because it keeps them, and
# marks the line "NW6: API 1.0 handle". A new reader is added here (and marked).
_API1_SITES = frozenset(
    {
        # the two helpers
        ("rig.nodetypes._base", "_handle_valid"),
        ("rig.nodetypes._base", "_handle_alive"),
        # the constructors, which store them (a copy constructor reads them)
        ("rig.nodetypes.dg_node", "DGNode.__init__"),
        ("rig.nodetypes.dg_node", "DGNode._cache_api1_objects"),
        ("rig.nodetypes.dag_node", "DAGNode.__init__"),
        # the names of the node state the `=` sugar stores (M3)
        ("rig.nodetypes.dg_node", "<module>"),
        # hot: `DGNode.ensure_valid` (and its name of a deleted node), the owner
        # checks of a plug, the plug hash and node serial, the fn set of a
        # plug's node, the memo identity of a DG node, the cache hit of a
        # typed node's attribute lookup (the wrapper's until M3)
        ("rig.nodetypes.dg_node", "DGNode.ensure_valid"),
        ("rig.nodetypes._base", "_ensure_owner_alive"),
        ("rig.nodetypes._base", "_node_serial"),
        ("rig.nodetypes._base", "_plug_hash"),
        ("rig.nodetypes._base", "_plug_node_fn_set"),
        ("rig._internal.plug", "_owner_alive"),
        ("rig._internal.memoize", "_named_dg_identity"),
        ("rig.nodetypes.dg_node", "DGNode.__getattr__"),
        # they keep the handle or the fn set: the attribute handles of
        # `find_attr` / `find_alias`, a plug of a held node, a memo attr check
        ("rig.nodetypes.dg_node", "DGNode.find_attr"),
        ("rig.nodetypes.dg_node", "DGNode.find_alias"),
        ("rig._internal.plug", "_named_plug"),
        ("rig._internal.memoize", "_attr_check"),
        # the dead canonical-wrapper helpers (round 4a M8 deletes them)
        ("rig.nodetypes._base", "_wrapper_is_canonical"),
        ("rig.nodetypes._base", "_copy_wrapper"),
    }
)


def _api1_accesses():
    """Every access to `_objhandle1` / `_fn_set1` in the rig package outside
    ``_tests``: an attribute (``node._objhandle1``) or a str constant (a
    subscript, ``d.get(...)``, ``in``), as ``(module, function, path, line,
    marked)``; ``marked`` is True if the line, or the line above it, has the
    "NW6: API 1.0 handle" marker."""
    import rig

    root   = os.path.dirname(rig.__file__)
    parent = os.path.dirname(root)
    found  = []
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in ("_tests", "__pycache__"))
        for name in sorted(f for f in files if f.endswith(".py")):
            path   = os.path.join(base, name)
            module = os.path.relpath(path, parent)[:-3].replace(os.sep, ".")
            if module.endswith(".__init__"):
                module = module[: -len(".__init__")]
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
            lines = source.splitlines()

            def visit(node, scope):
                for child in ast.iter_child_nodes(node):
                    if isinstance(
                        child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    ):
                        visit(child, scope + [child.name])
                        continue
                    if (isinstance(child, ast.Attribute) and child.attr in _API1_NAMES) or (
                        isinstance(child, ast.Constant) and child.value in _API1_NAMES
                    ):
                        near   = lines[child.lineno - 1] + lines[child.lineno - 2]
                        marked = _NW6_MARKER in near
                        found.append(
                            (module, ".".join(scope) or "<module>", path, child.lineno, marked)
                        )
                    visit(child, scope)

            visit(ast.parse(source), [])
    return found


class TestApi1HandleReaders(MayaTestCase):
    """M2: the API 1.0 handle helpers (`_handle_valid` / `_handle_alive`) and the
    complete list of the package's `_objhandle1` / `_fn_set1` readers, so that
    round 5 (NW6, the handles off API 1.0) has one list of sites to edit."""

    TEST_START_NEW_SCENE = True

    def test_readers_are_known(self):
        sites = {(module, func) for module, func, _, _, _ in _api1_accesses()}
        self.assertEqual(sites, _API1_SITES)

    def test_inline_readers_are_marked(self):
        helpers  = {("rig.nodetypes._base", "_handle_valid"), ("rig.nodetypes._base", "_handle_alive")}
        unmarked = [
            f"{os.path.basename(path)}:{line} {func}"
            for module, func, path, line, marked in _api1_accesses()
            if not marked and (module, func) not in helpers
        ]
        self.assertEqual(unmarked, [])

    def test_the_helpers(self):
        from rig.nodetypes._base import _handle_alive, _handle_valid

        cmds.undoInfo(state=True, infinity=True)
        live    = PyNode(cmds.createNode("transform", name="live"))
        deleted = PyNode(cmds.createNode("multiplyDivide", name="deleted"))
        freed   = PyNode(cmds.createNode("transform", name="freed"))
        half    = object.__new__(Transform)
        cmds.delete("deleted")
        cases = {
            "live": (live, True, True),
            "deleted": (deleted, False, True),
            "half_built": (half, False, False),
        }
        for label, (node, valid, alive) in cases.items():
            with self.subTest(case=label):
                self.assertIs(_handle_valid(vars(node)), valid)
                self.assertIs(_handle_alive(vars(node)), alive)
                self.assertIs(node.is_valid, valid)
        self.assertIs(_handle_valid({}), False)
        self.assertIs(_handle_alive({}), False)
        cmds.undo()
        self.assertTrue(deleted.is_valid)
        cmds.file(new=True, force=True)
        # a freed node: only its API 1.0 handle is read
        probe = mock.Mock()
        vars(freed)["_fn_set"] = probe
        self.assertIs(_handle_valid(vars(freed)), False)
        self.assertIs(_handle_alive(vars(freed)), False)
        self.assertIs(freed.is_valid, False)
        self.assertEqual(probe.mock_calls, [])


def _scene_nodes():
    """The names of every node in the scene (to count what an expression built)."""
    return set(cmds.ls())


class TestTypedNodesSpeakTheDsl(MayaTestCase):
    """M3 (K S4a ``f7511a2``): typed nodes return Plugs owned by the typed node,
    keep the ``=`` sugar (variant K), inject with ``<<`` / ``>>`` and resolve
    the component fallbacks; the Plug ``_`` rule."""

    TEST_START_NEW_SCENE = True

    def test_attribute_access_gives_plugs(self):
        cmds.createNode("transform", name="a")
        dg = PyNode("a")
        for plug in (dg.tx, dg.t, dg.t[0], dg.t.ty, dg.worldMatrix[0]):
            with self.subTest(plug=str(plug)):
                self.assertIsInstance(plug, Plug)
                self.assertIs(plug.node, dg)
                self.assertTrue(plug.node.mobject == _mobject("a"))
        # the typed API still reads Attributes
        self.assertNotIsInstance(dg.find_attr("tx"), Plug)
        dg.tx << 3.0
        self.assertEqual(cmds.getAttr("a.tx"), 3.0)

    def test_typed_plugs_have_plug_semantics(self):
        # C7: what a typed node's attr does now that it is a Plug
        cmds.createNode("transform", name="a")
        cmds.createNode("transform", name="b")
        a, b = PyNode("a"), PyNode("b")
        with self.assertRaisesRegex(TypeError, r"^'>>' does not connect plugs: write b\.translateY"):
            a.tx >> b.ty
        self.assertIsNone(cmds.listConnections("b.ty", source=True))
        self.assertEqual(str(a.tx >> "txCopy"), "a.txCopy")
        for label, build in (
            ("+", lambda: a.tx + b.tx),
            ("%", lambda: a.tx % b.tx),
            ("//", lambda: a.tx // b.tx),
            ("==", lambda: a.tx == b.tx),
            ("sorted", lambda: sorted([b.tx, a.tx])),
        ):
            with self.subTest(op=label):
                before = _scene_nodes()
                result = build()
                self.assertTrue(_scene_nodes() - before, label)
                if label != "sorted":
                    self.assertIsInstance(result, Plug)
        # get() is numpy-shaped; the typed API reads the getAttr shape
        np.testing.assert_array_equal(a.t.get(), [0.0, 0.0, 0.0])
        self.assertEqual(a.find_attr("t").get(), [(0.0, 0.0, 0.0)])
        # `plug << typed attr` connects
        b.tz << a.find_attr("tz")
        self.assertEqual(cmds.listConnections("b.tz", source=True, plugs=True), ["a.translateZ"])

    def test_setattr_sugar_variant_k(self):
        cmds.undoInfo(state=True, infinity=True)
        cmds.createNode("transform", name="a")
        cmds.addAttr("a", longName="__parked__", attributeType="double")
        dg     = PyNode("a")
        before = list(vars(dg))

        dg.tx = 5
        self.assertEqual(cmds.getAttr("a.tx"), 5.0)
        dg.__parked__ = 4.0
        self.assertEqual(cmds.getAttr("a.__parked__"), 4.0)
        self.assertEqual(list(vars(dg)), before)

        dg.namespace = "ns"
        self.assertEqual(dg.name, "ns:a")
        with self.assertRaises(AttributeError) as ctx:
            dg.name = "x"
        self.assertIn("setter", str(ctx.exception))
        with self.assertRaises(AttributeError) as ctx:
            dg.rename = 1
        self.assertIn("find_attr('rename')", str(ctx.exception))
        curve = PyNode(
            cmds.listRelatives(cmds.curve(point=[(0, 0, 0), (1, 0, 0)], degree=1), shapes=True)[0]
        )
        with self.assertRaises(AttributeError) as ctx:
            curve.create = 1
        self.assertIn("find_attr('create')", str(ctx.exception))
        self.assertIs(type(curve.find_attr("create")), _base_module().Attribute)
        with self.assertRaises(AttributeError) as ctx:
            dg.tranlsateX = 1
        self.assertIn("Attribute not found", str(ctx.exception))
        with self.assertRaises(AttributeError):
            dg.user_tag = 1
        self.assertNotIn("user_tag", vars(dg))
        dg._cache = 1
        self.assertEqual(vars(dg)["_cache"], 1)

        with mock.patch.object(dg, "_attr_data_type_fallback", create=True) as hook:
            self.assertIs(dg._attr_data_type_fallback, hook)
        self.assertNotIn("_attr_data_type_fallback", vars(dg))

        cmds.delete("ns:a")
        dg._x = 1
        self.assertEqual(vars(dg)["_x"], 1)
        with self.assertRaises(RuntimeError) as ctx:
            dg.tx = 1
        self.assertTrue(str(ctx.exception).endswith("a already deleted!"))

    def test_setattr_through_node(self):
        # renamed at round 4a M4 (was test_setattr_through_the_wrapper, an M3
        # id): the Node factory returns the typed node, so this is the typed
        # node's sugar reached through Node(...), and a Container's (a DGNode
        # subclass since M4) published names
        cmds.createNode("transform", name="a")
        node = Node("a")
        self.assertIs(type(node), Transform)
        node.tx = 2
        self.assertEqual(cmds.getAttr("a.tx"), 2.0)
        node.namespace = "ns"
        self.assertEqual(str(node), "ns:a")
        with self.assertRaises(AttributeError):
            node.rename = 1
        with container("box") as box:
            inner = Node.create("transform", name="inner")
            container.publish_input(inner.tx, "blend")
        self.assertIsInstance(box, DGNode)
        box.blend = 1.5
        self.assertEqual(cmds.getAttr("inner.tx"), 1.5)

    def test_rshift_none_and_lshift_spec(self):
        from rig.spec import Float

        cmds.createNode("transform", name="a")
        dg = PyNode("a")
        self.assertIs(dg >> None, dg)
        plug = dg << Float("knob", dv=0.5)
        self.assertEqual(str(plug), "a.knob")
        self.assertEqual(cmds.getAttr("a.knob"), 0.5)
        out = dg >> Float("readout")
        self.assertEqual(str(out), "a.readout")
        self.assertFalse(cmds.attributeQuery("readout", node="a", writable=True))
        self.assertEqual(os.fspath(dg), "a")
        # the error texts of the wrapper's operators
        with self.assertRaisesRegex(TypeError, r"^Cannot inject int into a bare Node; "):
            dg << 5
        with self.assertRaisesRegex(TypeError, r"^'>>' on a Node supports `>> None`"):
            dg >> 5

    def test_matrix_sources_drive_a_typed_transform(self):
        cmds.createNode("transform", name="a")
        cmds.createNode("transform", name="b")
        b = PyNode("b")
        self.assertIs(b << PyNode("a").matrix, b)
        for channel in ("t", "r", "s"):
            with self.subTest(channel=channel):
                sources = cmds.listConnections(f"b.{channel}", source=True, destination=False)
                self.assertEqual([cmds.nodeType(s) for s in sources], ["decomposeMatrix"])
        cmds.createNode("transform", name="c")
        c        = PyNode("c")
        m        = np.eye(4)
        m[3, :3] = [7.0, 8.0, 9.0]
        self.assertIs(c << m, c)
        self.assertEqual(cmds.getAttr("c.t")[0], (7.0, 8.0, 9.0))

    def test_component_fallbacks_on_typed_nodes(self):
        cube  = cmds.polyCube(name="cube", ch=False)[0]
        shape = cmds.listRelatives(cube, shapes=True)[0]
        surf  = cmds.sphere(name="ball", constructionHistory=False)[0]
        self.assertEqual(str(PyNode(cube).vtx[3]), f"{shape}.controlPoints[3]")
        faces = PyNode(cube).f
        self.assertIsInstance(faces, Components)
        self.assertTrue(faces.shape.mobject == _mobject(shape))
        self.assertIsInstance(PyNode(shape).e, Components)
        from rig._internal.plug import ComponentPlug

        cv = PyNode(surf).cv
        self.assertIsInstance(cv, ComponentPlug)
        self.assertEqual(str(cv[1, 2]), "ballShape.cv[1][2]")
        with self.assertRaises(AttributeError):
            PyNode(shape).cv
        # a curve shape's `f` is its Maya attr `form`, not a component
        crv = cmds.listRelatives(cmds.curve(point=[(0, 0, 0), (1, 0, 0)], degree=1), shapes=True)[0]
        self.assertEqual(str(PyNode(crv).f), f"{crv}.form")

    def test_plug_private_names(self):
        cmds.createNode("transform", name="a")
        plug = PyNode("a").tx
        with self.assertRaises(AttributeError):
            plug._attr_dict
        with self.assertRaises(AttributeError):
            plug._objhandle1
        with mock.patch.object(cmds, "container", wraps=cmds.container) as probe:
            self.assertFalse(hasattr(plug, "__array_interface__"))
            self.assertFalse(hasattr(Node("a").t, "_ipython_canary_method_should_not_exist_"))
            self.assertFalse(hasattr(Plug("a.tx"), "__array_struct__"))
        self.assertEqual(probe.call_count, 0)
        # a Maya attr of a "_" name is still a sibling, and a compound child too
        cmds.addAttr("a", longName="__parked__", attributeType="double")
        self.assertEqual(str(plug.__parked__), "a.__parked__")
        cmds.addAttr("a", longName="_pair", attributeType="double2")
        cmds.addAttr("a", longName="_first", attributeType="double", parent="_pair")
        cmds.addAttr("a", longName="_second", attributeType="double", parent="_pair")
        self.assertEqual(str(PyNode("a")._pair._second), "a._second")
        # a half-built plug has none
        self.assertFalse(hasattr(Plug.__new__(Plug, "a.tx"), "_x"))

    def test_plug_private_names_on_a_freed_node(self):
        cmds.createNode("transform", name="a")
        plug = PyNode("a").tx
        cmds.file(new=True, force=True)
        with mock.patch.object(cmds, "container", wraps=cmds.container) as probe:
            for name in ("__array__", "_x", "__parked__"):
                self.assertFalse(hasattr(plug, name), name)
        self.assertEqual(probe.call_count, 0)

    def test_find_attr_of_a_plug_is_the_attribute(self):
        attribute_cls = _base_module().Attribute
        cmds.createNode("transform", name="a")
        cmds.createNode("transform", name="b")
        dg = PyNode("a")
        for plug in (dg.tx, Node("a").tx, Plug("a.tx"), dg.t[0]):
            with self.subTest(plug=type(plug.node).__name__):
                attr = dg.find_attr(plug)
                self.assertIs(type(attr), attribute_cls)
                self.assertIs(attr.node, plug.node)
                self.assertEqual(str(attr), "a.translateX")
                self.assertEqual(attr.get(), 0.0)
        # another node's plug is refused, quietly too
        with self.assertRaisesRegex(AttributeError, "doesn't belong to a"):
            dg.find_attr(PyNode("b").tx)
        self.assertIsNone(dg.find_attr(PyNode("b").tx, quiet=True))
        # a typed Attribute is handed back as is
        typed = dg.find_attr("ty")
        self.assertIs(dg.find_attr(typed), typed)

    def test_find_attr_of_a_freed_plug_raises(self):
        cmds.createNode("transform", name="a")
        plug = PyNode("a").tx
        cmds.file(new=True, force=True)
        cmds.createNode("transform", name="a")
        with self.assertRaisesRegex(RuntimeError, "already deleted!$"):
            PyNode("a").find_attr(plug)

    def test_str_operand_and_same_plug_compare_build_nothing(self):
        # X1 and the one operand check reach typed plugs too
        cmds.createNode("transform", name="cube")
        before = _scene_nodes()
        with self.assertRaises(TypeError):
            PyNode("cube").tx == "cube.ty"
        self.assertEqual(_scene_nodes(), before)
        self.assertIs(PyNode("cube").tx == PyNode("cube").tx, True)
        self.assertIs(PyNode("cube").tx != Node("cube").tx, False)
        self.assertEqual({PyNode("cube").tx: 1}[PyNode("cube").tx], 1)
        self.assertEqual({Node("cube").tx: 2}[PyNode("cube").tx], 2)
        self.assertIn(PyNode("cube").tx, {PyNode("cube").tx, PyNode("cube").ty})
        self.assertEqual(_scene_nodes(), before)


def _source_index(dst):
    """The logical index of the element connected into the plug named `dst`."""
    sel = OpenMaya.MSelectionList()
    sel.add(dst)
    return sel.getPlug(0).source().logicalIndex()


def _public_self_stores():
    """Every ``self.<name> = ...`` store of a public name in a method of a
    ``rig/nodetypes`` node class, unless the class attribute of that name is a
    property with a setter or another data descriptor: such a store goes through
    ``DGNode.__setattr__``, which writes a Maya attr of that name (or raises)."""
    import rig.nodetypes as nodetypes
    from rig.nodetypes._base import _MISSING, _class_attr

    root    = os.path.dirname(nodetypes.__file__)
    files   = [os.path.join(root, f) for f in sorted(os.listdir(root)) if f.endswith(".py")]
    classes = {cls.__name__: cls for cls in _all_subclasses(DGNode)}
    stores  = []
    for path in files:
        tree = ast.parse(open(path, encoding="utf-8").read())
        for cnode in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            cls = classes.get(cnode.name)
            if cls is None:
                continue
            for sub in ast.walk(cnode):
                if not (
                    isinstance(sub, ast.Attribute)
                    and isinstance(sub.ctx, ast.Store)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "self"
                    and not sub.attr.startswith("_")
                ):
                    continue
                found = _class_attr(cls, sub.attr)
                if isinstance(found, property):
                    settable = found.fset is not None
                else:
                    settable = found is not _MISSING and hasattr(type(found), "__set__")
                if not settable:
                    stores.append(f"{os.path.basename(path)}:{sub.lineno} {cnode.name} self.{sub.attr}")
    return stores


class TestMergeReviewFixes(MayaTestCase):
    """M3 / M4: K's merge-only review fixes (``82547f5``): Plugs handed to the
    typed API, instance monkeypatching, public stores in the node classes, owner
    propagation of the typed accessors (M3); ``Container.create`` is the
    container-aware factory and a node's Maya attr ``wrap`` is reachable (M4)."""

    TEST_START_NEW_SCENE = True

    def test_typed_map_api_takes_the_plug_of_a_map(self):
        attribute_cls = _base_module().Attribute
        cube = PyNode(cmds.polyCube(name="cube", ch=False)[0])
        mesh = cube.get_children(type="mesh")[0]
        mesh.add_map("skinMask", values=[0, 0.5, 1, 0, 0, 0.25, 0, 0])
        plug = mesh.skinMask
        self.assertIsInstance(plug, Plug)
        attr = mesh.find_attr(plug)
        self.assertIs(type(attr), attribute_cls)
        self.assertIs(attr.node, mesh)
        self.assertEqual(mesh.get_map_values(plug)[:3], [0.0, 0.5, 1.0])
        self.assertEqual(list(mesh.get_map_data(plug).indices), [1, 2, 5])
        self.assertEqual(plug.filter_array_values(0.0), ([1, 2, 5], [0.5, 1.0, 0.25]))
        copy_attr = mesh.duplicate_map(plug, "w2")
        self.assertIs(type(copy_attr), attribute_cls)
        self.assertEqual(mesh.get_map_values("w2"), mesh.get_map_values("skinMask"))
        mesh.mirror_map(plug)
        mesh.set_map_values(plug, 0.5)
        self.assertEqual(mesh.get_map_values("skinMask"), [0.5] * 8)

    def test_instance_monkeypatching_of_node_methods(self):
        node = PyNode(cmds.createNode("transform", name="a"))
        with mock.patch.object(node, "get_parent", return_value="P"):
            self.assertEqual(node.get_parent(), "P")
        self.assertIsNone(node.get_parent())
        self.assertNotIn("get_parent", vars(node))
        node.get_parent = lambda: "Q"
        self.assertEqual(node.get_parent(), "Q")
        del node.get_parent
        self.assertIsNone(node.get_parent())
        # a Python attribute stored on the instance is set there again
        vars(node)["user_tag"] = 1
        node.user_tag = 2
        self.assertEqual(vars(node)["user_tag"], 2)
        # still refused: a non-callable over a method, a name that is no Maya attr
        with self.assertRaisesRegex(AttributeError, r"find_attr\('rename'\)"):
            node.rename = 1
        with self.assertRaisesRegex(AttributeError, "Attribute not found"):
            node.side = "L"

    def test_public_self_stores_are_settable_members(self):
        # a public `self.<name> = ...` in a node class goes through
        # DGNode.__setattr__ (a Maya write, or "Attribute not found"), so it must
        # name a property with a setter or another data descriptor
        self.assertEqual(_public_self_stores(), [])

    def test_owner_propagation_of_typed_accessors(self):
        # the accessors of a node's plugs and attrs keep the node as their owner,
        # through PyNode and (M4) through the Node factory, which is the same
        # typed node; the class of a Plug's element_by_* result is M4B's (D29)
        attribute_cls = _base_module().Attribute
        for factory in (PyNode, Node):
            with self.subTest(factory=factory.__name__):
                cmds.file(new=True, force=True)
                node   = factory(cmds.createNode("transform", name="n"))
                parent = node.tx.get_parent()
                self.assertIs(type(parent), attribute_cls)
                self.assertIs(parent.node, node)
                child = node.find_attr("t").child(0)
                self.assertIs(type(child), attribute_cls)
                self.assertIs(child.node, node)
                pma = factory(cmds.createNode("plusMinusAverage", name="pma"))
                pma.input1D[0] << 1
                element = pma.input1D[0]
                self.assertIs(type(element), Plug)
                self.assertIs(element.node, pma)
                element = pma.input1D.element_by_physical_index(0)
                self.assertIsInstance(element, attribute_cls)
                self.assertIs(element.node, pma)
                element = pma.find_attr("input1D").element_by_physical_index(0)
                self.assertIs(type(element), attribute_cls)
                self.assertIs(element.node, pma)
                # plugs are not cached (D30)
                self.assertIsNot(node.tx, node.tx)

    def test_container_create_is_the_container_aware_factory(self):
        # M4 (82547f5): DGNode's typed create would build an "entity" node of
        # the unregistered Container class; Container.create is Node.create
        from rig._internal.container import Container

        self.assertIs(Container.__dict__["create"].__func__, Node.create.__func__)
        with container("box") as box:
            made = type(box).create("transform", name="viaCtn")
        self.assertIs(type(made), Transform)
        self.assertIn(made.name, cmds.container(str(box), query=True, nodeList=True))

    def test_maya_attr_wrap_is_not_shadowed(self):
        # M4 (82547f5): Node.wrap lives on the metaclass, so a node's Maya attr
        # `wrap` (3D textures) is reachable, and the helper is on every class
        brownian = Node(cmds.shadingNode("brownian", asTexture=True, name="brown"))
        self.assertIs(type(brownian.wrap), Plug)
        self.assertEqual(str(brownian.wrap), "brown.wrap")
        self.assertIs(brownian.wrap.node, brownian)
        brownian.wrap << 0
        self.assertEqual(cmds.getAttr("brown.wrap"), 0)
        brownian.wrap = 1
        self.assertEqual(cmds.getAttr("brown.wrap"), 1)
        self.assertEqual(Node.wrap("brown").name, "brown")
        self.assertEqual(Transform.wrap(["persp"])[0].name, "persp")
        self.assertEqual(type(brownian).wrap("brown").name, "brown")
        self.assertIsNone(Node.wrap(None))


class TestTypedDslEdges(MayaTestCase):
    """M3: the round-4a edge checklist on typed nodes' plugs: E1 deleted / undone
    / renamed / same-name reuse, E2 new scene / file open / reference unload,
    E3 instancing, E4 namespaces, E6 components, E7 dynamic attrs, E8 identity."""

    TEST_START_NEW_SCENE = True

    _FREED = (
        r"^Transform node \(freed by a new scene, a file open or a reference unload\) "
        r"already deleted!$"
    )

    def test_deleted_undone_renamed_and_reused(self):
        # E1
        cmds.undoInfo(state=True, infinity=True)
        dg   = PyNode(cmds.createNode("transform", name="gone"))
        plug = dg.tx
        cmds.delete("gone")
        for label, op in (
            ("str(plug)", lambda: str(plug)),
            ("str(dg.tx)", lambda: str(dg.tx)),
            ("dg.ty", lambda: dg.ty),
            ("dg.tx << 1", lambda: dg.tx << 1),
            ("dg.tx = 1", lambda: setattr(dg, "tx", 1)),
            ("plug.get()", lambda: plug.get()),
        ):
            with self.subTest(op=label):
                with self.assertRaisesRegex(RuntimeError, "^gone already deleted!$"):
                    op()
        cmds.undo()
        self.assertEqual(str(plug), "gone.translateX")
        dg.tx = 2
        self.assertEqual(cmds.getAttr("gone.tx"), 2.0)
        # a rename: held and new plugs name the new name
        cmds.rename("gone", "back")
        self.assertEqual(str(plug), "back.translateX")
        self.assertEqual(str(dg.ty), "back.translateY")
        # the name taken by a new node: the held node and its plug raise, the
        # new node is another one
        cmds.delete("back")
        cmds.createNode("multiplyDivide", name="back")
        for op in (lambda: str(plug), lambda: str(dg.tx), lambda: dg.tz):
            with self.assertRaisesRegex(RuntimeError, "^back already deleted!$"):
                op()
        self.assertEqual(str(PyNode("back").input1X), "back.input1X")
        self.assertIsNot(PyNode("back").input1X.node, dg)

    def _held(self, names):
        held = []
        for name in names:
            node     = PyNode(name)
            compound = node.t
            compound.tx  # a cached child
            held.append((node, node.tx, compound, node.worldMatrix[0], node.find_attr("ty")))
        return held

    def _assert_freed(self, held):
        from rig.spec import Float

        persp = Node("persp")
        ops = {
            "node.tx (cached)":  lambda n, p, t, w, a: n.tx,
            "node.ty":           lambda n, p, t, w, a: n.ty,
            "node.tx = 1":       lambda n, p, t, w, a: setattr(n, "tx", 1),
            "node << spec":      lambda n, p, t, w, a: n << Float("k"),
            "str(node)":         lambda n, p, t, w, a: str(n),
            "str(plug)":         lambda n, p, t, w, a: str(p),
            "plug.get()":        lambda n, p, t, w, a: p.get(),
            "plug << 1":         lambda n, p, t, w, a: p << 1,
            "plug + 1":          lambda n, p, t, w, a: p + 1,
            "plug == plug":      lambda n, p, t, w, a: p == p,
            "compound.tx":       lambda n, p, t, w, a: t.tx,
            "compound[0]":       lambda n, p, t, w, a: t[0],
            "element.get()":     lambda n, p, t, w, a: w.get(),
            "attribute.get()":   lambda n, p, t, w, a: a.get(),
            "find_attr(plug)":   lambda n, p, t, w, a: persp.find_attr(p),
            "persp.tx << plug":  lambda n, p, t, w, a: persp.tx << p,
        }
        for label, op in ops.items():
            with self.subTest(op=label):
                for entry in held:
                    with self.assertRaisesRegex(RuntimeError, self._FREED):
                        op(*entry)
        for node, plug, *_ in held:
            self.assertFalse(node.is_valid)
            self.assertIsInstance(hash(plug), int)
            self.assertFalse(hasattr(plug, "__array__"))
            self.assertFalse(hasattr(node, "__array__"))
            self.assertFalse(hasattr(node, "__deepcopy__"))
            self.assertIs(plug.node, node)
            self.assertIs(node >> None, node)

    def test_held_across_a_new_scene_raise(self):
        # E2: a freed node's fn sets and MPlugs point at freed memory
        held = self._held([cmds.createNode("transform", name=f"held{i}") for i in range(3)])
        cmds.file(new=True, force=True)
        for _ in range(50):
            cmds.createNode("multiplyDivide")  # reuse the freed memory
        self._assert_freed(held)

    def test_held_across_a_file_open_reusing_the_names_raise(self):
        folder = tempfile.mkdtemp(prefix="rig_m3_open_")
        path   = os.path.join(folder, "m3_open.ma").replace("\\", "/")
        try:
            names = [cmds.createNode("transform", name=f"held{i}") for i in range(2)]
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
            held = self._held(names)
            cmds.file(path, open=True, force=True)
            self.assertTrue(all(cmds.objExists(name) for name in names))
            self._assert_freed(held)
            # the node of the same name the file brought is its own
            self.assertEqual(str(PyNode("held0").tx), "held0.translateX")
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)

    def test_held_across_a_reference_unload_raise(self):
        folder = tempfile.mkdtemp(prefix="rig_m3_ref_")
        path   = os.path.join(folder, "m3_ref.ma").replace("\\", "/")
        try:
            for i in range(2):
                cmds.createNode("transform", name=f"held{i}")
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
            cmds.file(new=True, force=True)
            cmds.file(path, reference=True, namespace="ref")
            held = self._held([f"ref:held{i}" for i in range(2)])
            cmds.file(unloadReference=cmds.referenceQuery(path, referenceNode=True))
            self._assert_freed(held)
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)

    def test_instanced_plugs_are_named_through_the_path_taken(self):
        # E3
        cmds.loadPlugin("matrixNodes", quiet=True)
        cmds.createNode("transform", name="T1")
        cmds.createNode("locator", name="S", parent="T1")
        cmds.createNode("transform", name="T2")
        cmds.parent("|T1|S", "T2", add=True, shape=True)
        cmds.setAttr("T2.tx", 7)
        second = PyNode("|T2|S")
        self.assertEqual(str(second.v), "T2|S.visibility")
        self.assertEqual(str(PyNode("|T1|S").v), "T1|S.visibility")
        self.assertIs(second.v.node, second)
        for path, index, name in (
            ("|T2|S", 0, "T2|S.worldMatrix[0]"),
            ("|T2|S", 1, "T2|S.worldMatrix"),
            ("|T1|S", 0, "T1|S.worldMatrix"),
            ("|T1|S", 1, "T1|S.worldMatrix[1]"),
        ):
            with self.subTest(path=path, index=index):
                plug = PyNode(path).worldMatrix[index]
                self.assertEqual(str(plug), name)
                self.assertEqual(plug.get()[3][0], 7.0 if index else 0.0)
                dst = cmds.createNode("multMatrix")
                PyNode(dst).matrixIn[0] << plug
                self.assertEqual(_source_index(dst + ".matrixIn[0]"), index)
        # one plug read through two paths is one key, and compares the same
        first = PyNode("|T1|S").v
        self.assertEqual({first: 1}.get(second.v), 1)
        self.assertIs(first == second.v, True)
        # a removed instance: the plug is named through the path left, and the
        # taken path again once the removal is undone
        cmds.undoInfo(state=True, infinity=True)
        cmds.parent("T2|S", removeObject=True, shape=True)
        self.assertEqual(str(second.v), "S.visibility")
        cmds.undo()
        self.assertEqual(str(second.v), "T2|S.visibility")

    def test_namespaces(self):
        # E4
        cmds.namespace(add="ns")
        cmds.createNode("transform", name="ns:n")
        dg = PyNode("ns:n")
        self.assertEqual(str(dg.tx), "ns:n.translateX")
        self.assertIs(dg.tx.node, dg)
        dg.tx = 3
        self.assertEqual(cmds.getAttr("ns:n.tx"), 3.0)
        try:
            cmds.namespace(set="ns")
            made = cmds.createNode("transform", name="m")
        finally:
            cmds.namespace(set=":")
        self.assertEqual(made, "ns:m")
        other = PyNode(made)
        other.namespace = "other"
        self.assertEqual(str(other.ty), "other:m.translateY")
        self.assertEqual(str(dg.ty), "ns:n.translateY")

    def test_components_of_typed_nodes(self):
        # E6
        from rig._internal.plug import ComponentPlug

        cube  = cmds.polyCube(name="cube", ch=False)[0]
        shape = cmds.listRelatives(cube, shapes=True)[0]
        mesh  = PyNode(shape)
        self.assertIs(mesh.vtx.node, mesh)
        self.assertEqual(str(mesh.vtx[2]), f"{shape}.controlPoints[2]")
        self.assertEqual(
            [str(p) for p in mesh.vtx[0:2]],
            [f"{shape}.controlPoints[0]", f"{shape}.controlPoints[1]"],
        )
        self.assertEqual((PyNode(cube).e >> None).tolist(), list(range(12)))
        self.assertEqual(PyNode(cube).f[1:3].count, 2)
        surf  = cmds.sphere(name="ball", ch=False)[0]
        shell = PyNode(cmds.listRelatives(surf, shapes=True)[0])
        cv    = shell.cv
        self.assertIsInstance(cv, ComponentPlug)
        self.assertIs(cv.node, shell)
        element = cv[1, 2]
        self.assertIs(element.node, shell)
        self.assertTrue(element.equals(Plug(f"{shell}.cv[1][2]")))
        lattice = cmds.lattice(cmds.polySphere(name="lsph", ch=False)[0], divisions=(2, 3, 2))[1]
        pt = PyNode(cmds.listRelatives(lattice, shapes=True)[0]).pt
        self.assertIsInstance(pt, ComponentPlug)
        self.assertEqual(str(pt[1, 2, 0]), f"{pt.node}.pt[1][2][0]")
        # a 1-D curve cv is a plain Plug
        crv = PyNode(
            cmds.listRelatives(
                cmds.curve(point=[(0, 0, 0), (1, 0, 0), (2, 0, 0)], degree=1), shapes=True
            )[0]
        )
        self.assertIs(type(crv.cv), Plug)
        crv.cv[1] << [0.0, 5.0, 0.0]
        self.assertEqual(cmds.getAttr(f"{crv}.controlPoints[1]")[0], (0.0, 5.0, 0.0))

    def test_dynamic_attr_freed_after_delete_and_flush(self):
        # E7
        cmds.undoInfo(state=True, infinity=True)
        cmds.createNode("transform", name="a")
        cmds.addAttr("a", longName="userDyn", attributeType="double")
        dg   = PyNode("a")
        plug = dg.userDyn
        self.assertIsNotNone(plug.__dict__["_attr1"])
        cmds.deleteAttr("a.userDyn")
        cmds.flushUndo()
        for label, op in (
            ("str", lambda: str(plug)),
            ("get", lambda: plug.get()),
            ("<<", lambda: plug << 1),
            ("+", lambda: plug + 1),
        ):
            with self.subTest(op=label):
                with self.assertRaisesRegex(RuntimeError, r"^a\.userDyn already deleted!$"):
                    op()
        self.assertFalse(hasattr(plug, "__array__"))
        # re-added with another type: a new plug of the new attribute
        cmds.addAttr("a", longName="userDyn", dataType="string")
        dg.userDyn = "text"
        self.assertEqual(cmds.getAttr("a.userDyn"), "text")
        self.assertEqual(str(dg.userDyn), "a.userDyn")

    def test_identity(self):
        # E8
        cmds.createNode("transform", name="a")
        dg   = PyNode("a")
        node = Node("a")
        self.assertIs(dg.tx.node, dg)
        self.assertIs(dg.t[0].node, dg)
        self.assertIs(dg.t.ty.node, dg)
        self.assertIs(node.tx.node, node)
        self.assertIs(node.t[0].node, node)
        self.assertIs(node.t.ty.node, node)
        # re-pinned (round 4a M4, C8): Node("a") is the typed node, no wrapper
        self.assertIs(type(node.tx.node), Transform)
        self.assertIs(type(dg.tx.node), Transform)
        self.assertIsInstance(node, Node)
        self.assertIs(Node(dg), dg)
        # the typed class is kept, and ">> None" is the node itself
        self.assertIs(node >> None, node)
        self.assertIs(dg._dg_node, dg)
        self.assertEqual(repr(node), 'Transform("a")')
        # one key for one Maya plug, whatever the object it was read from
        self.assertEqual(hash(dg.tx), hash(node.tx))
        self.assertEqual(hash(dg.tx), hash(Plug("a.tx")))
        self.assertTrue(dg.tx.equals(node.tx))
        # a Plug of a typed Attribute is owned by its typed node
        self.assertIs(Plug(dg.find_attr("tx")).node, dg)


class TestOneNodeHierarchy(MayaTestCase):
    """M4: ``Node`` is the root of the node classes and the DSL factory (K S4b
    ``ca9e345`` / ``4d3a02e``, the merge-only parts of ``82547f5`` / ``9244e90``,
    re-implemented on round 3); the wrapper class is gone (C8)."""

    TEST_START_NEW_SCENE = True

    def test_factory_input_table(self):
        # D10: a node is returned as is, an attr / plug gives its owner, a name,
        # dotted name, MPlug, MObject, MDagPath or uuid the typed node
        from rig.nodetypes import DAGNode, Mesh

        cube  = cmds.polyCube(name="cube", ch=False)[0]
        shape = cmds.listRelatives(cube, shapes=True)[0]
        node  = PyNode(cube)
        sel   = OpenMaya.MSelectionList()
        sel.add(cube)
        uuid  = cmds.ls(cube, uuid=True)[0]
        self.assertTrue(issubclass(DGNode, Node))
        self.assertIsInstance(node, Node)
        self.assertIs(Node(node), node)
        self.assertIs(Node(node.tx), node)
        self.assertIs(Node(node.t[1]), node)
        self.assertIs(Node(node.find_attr("tx")), node)
        for label, value, cls in (
            ("name", cube, Transform),
            ("long name", f"|{cube}", Transform),
            ("dotted", f"{cube}.tx", Transform),
            ("dotted compound", f"{cube}.translate", Transform),
            ("dotted component", f"{shape}.vtx[0]", Mesh),
            ("mplug", node.tx.plug, Transform),
            ("shape mplug", PyNode(shape).find_attr("outMesh").plug, Mesh),
            ("mobject", _mobject(cube), Transform),
            ("mdagpath", sel.getDagPath(0), Transform),
            ("uuid", uuid, Transform),
            ("plug of a name", Plug(f"{cube}.tx"), Transform),
        ):
            with self.subTest(case=label):
                result = Node(value)
                self.assertIs(type(result), cls)
                self.assertIsInstance(result, DAGNode)
                self.assertIsInstance(result, Node)
        for bad in (3, None, 1.5):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "is not a str, MObject, MDagPath"):
                    Node(bad)
        self.assertIs(type(Node(cube)), type(PyNode(cube)))
        # a node class constructs as usual; PyNode keeps a dotted name an Attribute
        self.assertIs(type(Transform(cube)), Transform)
        self.assertIs(type(PyNode(f"{cube}.tx")), _base_module().Attribute)

    def test_the_class_call_has_one_branch_point(self):
        # NodeMeta.__call__ runs for every node class call and dispatches the
        # root only; PyNode's cast constructs without it (round 4b builds on both)
        from rig.nodetypes._base import NodeMeta

        cmds.createNode("transform", name="a")
        calls    = []
        original = NodeMeta.__call__

        def counting(cls, *args, **kwargs):
            calls.append(cls.__name__)
            return original(cls, *args, **kwargs)

        with mock.patch.object(NodeMeta, "__call__", counting):
            PyNode("a")
            PyNode(_mobject("a"))
            self.assertEqual(calls, [])
            Node("a")
            self.assertEqual(calls, ["Node"])
            Transform("a")
            self.assertEqual(calls, ["Node", "Transform"])

    def test_plugs_of_typed_nodes_are_owned_by_them(self):
        cmds.createNode("transform", name="a")
        node = Node("a")
        for plug in (node.tx, node.t[0], node.t.ty, node.translate.child(2), node.wm[0]):
            with self.subTest(plug=str(plug)):
                self.assertIs(type(plug), Plug)
                self.assertIs(plug.node, node)
        self.assertIs(node.find_attr("tx").node, node)
        self.assertIs(node >> None, node)
        self.assertEqual(repr(node), 'Transform("a")')
        self.assertEqual(repr(Node("persp").tx), 'Plug("persp.translateX")')

    def test_plug_state_matches_a_plug_of_its_mplug(self):
        # (K's _bound parity, on round 3's constructors): a node's plug holds the
        # state keys a Plug built from its MPlug holds, in the same order
        cmds.createNode("transform", name="a")
        mplug = PyNode("a").find_attr("tx").plug
        self.assertEqual(list(vars(Node("a").tx)), list(vars(Plug(mplug))))
        self.assertEqual(str(Node("a").tx), str(Plug(mplug)))
        self.assertEqual(list(vars(Node("a").t[0])), list(vars(Plug(mplug))))
        self.assertEqual(str(Node("a").t[0]), str(Plug(mplug)))

    def test_plug_hash_is_rename_stable(self):
        # round 3's one-key hash (a node serial and the attr with its indices),
        # not the prototype's (handle hash code, alias)
        base = _base_module()
        cmds.createNode("transform", name="a")
        plug   = Node("a").tx
        before = hash(plug)
        self.assertEqual(before, hash((base._node_serial(plug.node), "translateX")))
        self.assertNotEqual(before, hash((plug.node._objhandle1.hashCode(), "translateX")))
        cmds.rename("a", "b")
        self.assertEqual(str(plug), "b.translateX")
        self.assertEqual(hash(plug), before)
        self.assertEqual(hash(Node("b").tx), before)
        self.assertEqual(hash(Plug("b.tx")), before)

    def test_create_joins_the_container(self):
        with container("box") as box:
            node = Node.create("multiplyDivide", name="md")
        self.assertIs(type(node), DGNode)
        self.assertIn("md", cmds.container(str(box), query=True, nodeList=True))

    def test_container_equality_owner_and_lookup(self):
        from rig._internal.container import Container

        with container("box") as box:
            inner = Node.create("transform", name="inner")
            container.publish_input(inner.tx, "slide")
        plain = PyNode(str(box))
        self.assertIsInstance(box, Container)
        self.assertIsInstance(box, DGNode)
        self.assertIsInstance(box, Node)
        self.assertNotIsInstance(plain, Container)
        self.assertIs(Node(box), box)
        # symmetric equality, equal hashes
        self.assertTrue(box == plain)
        self.assertTrue(plain == box)
        self.assertFalse(box != plain)
        self.assertEqual(hash(box), hash(plain))
        self.assertEqual(len({box, plain}), 1)
        self.assertFalse(box == PyNode("inner"))
        self.assertEqual(repr(box), 'Container("box")')
        # a genuine attr is owned by the container, a published one by its node
        self.assertIs(box.blackBox.node, box)
        self.assertEqual(str(box.slide), "inner.translateX")
        # class members win over published names
        self.assertEqual(box.name, "box")
        box.slide = 2.0
        self.assertEqual(cmds.getAttr("inner.tx"), 2.0)
        # the "_" rule: only a Maya attr of the live container
        self.assertFalse(hasattr(box, "_nope"))
        self.assertFalse(hasattr(box, "__array__"))
        cmds.addAttr(str(box), longName="__tag__", attributeType="double")
        self.assertEqual(str(box.__tag__), "box.__tag__")
        self.assertIs(box >> None, box)

    def test_publish_refuses_a_container_member_name(self):
        with container("box") as box:
            inner = Node.create("transform", name="inner")
            for name in ("name", "cleanup", "uuid", "rename"):
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, "is a Container attribute"):
                        container.publish_input(inner.tx, name)
                    with self.assertRaisesRegex(ValueError, "is a Container attribute"):
                        container.publish_output(inner.worldMatrix[0], name)
                    # the external-source form too
                    with self.assertRaisesRegex(ValueError, "is a Container attribute"):
                        container.publish_input(1.0, name)
            # a plain name still publishes
            container.publish_input(inner.ty, "lift")
        self.assertEqual(
            cmds.container(str(box), query=True, publishName=True) or [], ["lift"]
        )

    def test_node_level_class_attribute_refuses_sugar(self):
        # a Maya attr named like a node method (a curve shape's `create`) is
        # reached through find_attr (`wrap` is a Maya attr now: TestMergeReviewFixes)
        curve = Node(cmds.listRelatives(cmds.curve(p=[(0, 0, 0), (1, 0, 0)], d=1), shapes=True)[0])
        self.assertTrue(cmds.attributeQuery("create", node=str(curve), exists=True))
        with self.assertRaises(AttributeError) as ctx:
            curve.create = 1
        self.assertIn("find_attr('create')", str(ctx.exception))
        self.assertIs(type(curve.find_attr("create")), _base_module().Attribute)

    def test_lift(self):
        from rig import lift

        cmds.createNode("transform", name="a")
        node = Node("a")
        self.assertIs(lift(node), node)
        attr = node.find_attr("tx")
        plug = lift(attr)
        self.assertIs(type(plug), Plug)
        self.assertIs(plug.node, node)
        self.assertIs(lift(plug), plug)
        self.assertIs(type(lift("a")), Transform)
        self.assertIs(type(lift("a.tx")), Plug)
        with self.assertRaises(TypeError):
            lift(42)

    def test_membership_with_a_typed_node_on_the_left(self):
        # a node on the left of a member spec is a Node, whatever the object
        # (at M3 a PyNode on the left raised "... is not a Node")
        from rig import Layer, Tag

        cube = PyNode(cmds.polyCube(name="cube", ch=False)[0])
        self.assertIs(cube << Tag("t1"), cube)
        self.assertIs(cube << Layer("L"), cube)
        self.assertTrue(cube >> Layer("L"))
        faces = cube.f[0:2]
        self.assertIs(faces << Tag("t2"), faces)

    def test_a_typed_node_on_the_right_of_a_plug(self):
        # typed nodes are clone targets of plug >> node and bare nodes for <<
        from rig.spec import Float

        src = PyNode(cmds.createNode("transform", name="src"))
        dst = PyNode(cmds.createNode("transform", name="dst"))
        knob = src << Float("knob", dv=0.25)
        clone = knob >> dst
        self.assertIs(type(clone), Plug)
        self.assertEqual(str(clone), "dst.knob")
        # the clone is a plug of dst (owned by a cast of it; binding the spec's
        # result to the node object is M10's)
        self.assertEqual(clone.node, dst)
        with self.assertRaisesRegex(TypeError, "^Cannot inject bare Node 'src' into matrix Plug"):
            dst.offsetParentMatrix << src
        with self.assertRaisesRegex(TypeError, r"^Cannot inject Node into a bare Node"):
            dst << src

    def test_wrap_on_the_node_classes(self):
        cmds.createNode("transform", name="a")
        self.assertEqual(repr(Node.wrap("a")), 'Transform("a")')
        self.assertEqual(
            repr(Transform.wrap(["a", "persp"])),
            'PlugList([Transform("a"), Transform("persp")])',
        )
        self.assertEqual(Node.wrap("not_a_node_xyz"), "not_a_node_xyz")
        self.assertEqual(Node.wrap(5.0), 5.0)
        self.assertIsNone(Node.wrap(None))

    def test_same_plug_keys_after_the_swap(self):
        # P1 / P2 pins: one Maya plug is one key through two objects (0 nodes),
        # world space elements of two instances are two keys, a plug read
        # through the second instance path is named through it
        def dep():
            return set(cmds.ls(dep=True))

        t = Node(cmds.createNode("transform", name="t"))
        before = dep()
        self.assertEqual({t.tx: 1}[t.tx], 1)
        self.assertIn(t.tx, {t.tx, t.ty})
        self.assertEqual(dep() - before, set())
        cmds.file(new=True, force=True)
        top = cmds.createNode("transform", name="T1")
        cmds.createNode("locator", name="S", parent=top)
        cmds.createNode("transform", name="T2")
        cmds.parent("T1|S", "T2", addObject=True, shape=True)
        n1, n2 = Node("|T1|S"), Node("|T2|S")
        wm0, wm1 = n1.worldMatrix[0], n1.worldMatrix[1]
        self.assertNotEqual(hash(wm0), hash(wm1))
        before = dep()
        self.assertIsNone({wm0: "hit"}.get(wm1))
        v1, v2 = n1.v, n2.v
        self.assertEqual(str(v1), "T1|S.visibility")
        self.assertEqual(str(v2), "T2|S.visibility")
        self.assertEqual(hash(v1), hash(v2))
        self.assertEqual({v1: "hit"}.get(v2), "hit")
        self.assertEqual(dep() - before, set())

    def test_held_typed_node_across_a_new_scene(self):
        # E2: the factory and >> None hand a freed node back as is, and read
        # nothing from it; its plugs raise the freed error
        node = Node(cmds.createNode("transform", name="held"))
        plug = node.tx
        cmds.file(new=True, force=True)
        for _ in range(50):
            cmds.createNode("multiplyDivide")  # reuse the freed memory
        self.assertIs(Node(node), node)
        self.assertIs(Node(plug), node)
        self.assertIs(node >> None, node)
        self.assertFalse(node.is_valid)
        self.assertFalse(hasattr(node, "__array__"))
        self.assertIsInstance(hash(plug), int)
        for label, op in (
            ("node.tx", lambda: node.tx),
            ("str(node)", lambda: str(node)),
            ("repr(node)", lambda: repr(node)),
            ("plug.get()", lambda: plug.get()),
            ("lift(plug.node)", lambda: str(Node(plug.node).ty)),
        ):
            with self.subTest(op=label):
                with self.assertRaisesRegex(RuntimeError, "already deleted!$"):
                    op()

    def test_container_held_across_a_new_scene(self):
        # E5: a Container held across a new scene raises the freed error on a
        # published name, a genuine attr and its name, and hands itself back
        with container("box") as box:
            inner = Node.create("transform", name="inner")
            container.publish_input(inner.tx, "slide")
        cmds.file(new=True, force=True)
        for _ in range(50):
            cmds.createNode("multiplyDivide")  # reuse the freed memory
        self.assertIs(Node(box), box)
        self.assertIs(box >> None, box)
        self.assertFalse(box.is_valid)
        self.assertFalse(hasattr(box, "__array__"))
        self.assertFalse(hasattr(box, "_nope"))
        for label, op in (
            ("published", lambda: box.slide),
            ("genuine", lambda: box.blackBox),
            ("sugar", lambda: setattr(box, "slide", 1.0)),
            ("name", lambda: box.name),
            ("repr", lambda: repr(box)),
        ):
            with self.subTest(op=label):
                with self.assertRaisesRegex(
                    RuntimeError, r"^Container node \(freed by a new scene.*already deleted!$"
                ):
                    op()

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
  rule): a typed node's ``.attr`` is a Plug owned by it, the ``=`` sugar
  (variant K), ``<<`` / ``>>``, the component fallbacks, the Plug ``_`` rule,
  ``find_attr(Plug)``; and the edge cases of the round-4a checklist (deleted,
  freed, instanced, namespaced nodes, components, dynamic attrs, identity).
* M4: the class swap (K S4b ``ca9e345`` / ``4d3a02e`` and the merge-only parts
  of ``82547f5`` / ``9244e90``): ``Node`` is the root class and the DSL factory
  (``Node(x) is x``, typed repr, every typed node is a ``Node``), the wrapper is
  gone, ``Container`` is a ``DGNode`` subclass (symmetric equality, owner,
  lookup order, the publish guard), ``Node.wrap`` on the metaclass; round 3's
  one-key plug hash is kept.
* M4B: a Plug's elements are Plugs built once (D29, ``_CHILD_CLASS``) in the
  state round 3 gave them, and ``_cast`` is the one cast core (D12) the
  ``Node`` factory calls.
* M8: the dead canonical-wrapper helpers and the ``_dg_node`` shim are gone
  (``TestNoWrapperLeft``).
* R2: PyNode is gone; ``Node`` is the only node factory (its full input table,
  ``Node.create`` and ``Node.find_all``: ``test_r4a_fixes.TestNodeOnly``).

The round-3 classes of the former ``test_node_merge.py`` follow the round-4a
ones (moved in M8): ``TestPlugQuickWins`` (S0: Plug property setters,
``bool(plug)``, ``find_attr`` filters and caching, ``rename_attr`` with a
Plug), ``TestOwnerRule`` (S1: a plug's node is the node object it was read
from; foreign plugs, stale DAG paths, held plugs across delete / reuse /
undo), ``TestReviewFixes`` (the merge prototype's review fixes, with its test
ids) and ``TestInstancedPlugIdentity`` (decision D-B: one identity per Maya
plug, whatever the instance path).
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
from rig import Container, container, List, Node, NodeNotFoundError, Plug
from rig.nodetypes import DGNode, Transform, _base
from rig._internal.math_nodes import _decompose_matrix
from rig._internal.members import Components
from rig._internal.memoize import _attribute_key
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
            joint = Node(cmds.createNode("joint", name="jnt"))
            loc   = Node(cmds.spaceLocator(name="loc")[0])
            for node in (joint, loc):
                node.serialize()
                node.get_matrix(world_space=True)
                node.get_matrix(world_space=False)
            cube = Node(cmds.polyCube(name="cube", ch=False)[0])
            cube.duplicate_geometry()
            shape = cube.get_children(type="mesh")[0]
            shape.serialize(world_space=False)

        def blendshapes():
            base   = Node(cmds.polyCube(name="base", ch=False)[0])
            target = Node(cmds.polyCube(name="target", ch=False)[0])
            cmds.xform(f"{target}.vtx[3]", ws=True, t=(10, 2, 3))
            bls = Node.create("blendShape", target, base)
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
            j1   = Node.create("joint", name="j1")
            j2   = Node.create("joint", name="j2", parent=j1)
            cube = Node(cmds.polyCube(name="skinned", ch=False)[0])
            skin = Node.create("skinCluster", cube, (j1, j2))
            self.assertIsInstance(skin, SkinCluster)
            cmds.select(clear=True)
            Node.create("joint", name="drv_j1")
            Node.create("joint", name="drv_j2")
            skin.connect_bind_pre_matrices(lambda name: "drv_" + name)

        def choices():
            cmds.createNode("transform", name="src")
            pick = Node(cmds.createNode("choice", name="pick"))
            self.assertIsInstance(pick, Choice)
            cmds.connectAttr("src.translate", "pick.input[0]")
            self.assertEqual(pick.find_attr("output").data_type, "double3")
            # a message input: getAttr answers "Tdata", so Choice's fallback
            # hook reads the selector to resolve the output
            cmds.connectAttr("src.message", "pick.input[1]")
            cmds.setAttr("pick.selector", 1)
            self.assertEqual(pick.find_attr("output").data_type, "message")

        def component_tags():
            ball = Node(cmds.polySphere(name="ball", ch=False)[0])
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
        typed   = Node("crv").find_attr("ro")
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

    return _base


class TestTypedLayerPrep(MayaTestCase):
    """M2 (K S3 ``ff99c29``): typed constructors write ``__dict__`` in their usual
    key order, copy only a node of their own class, and ``_`` probes on typed
    nodes are cheap. The key orders are the ones the constructors had at M1."""

    TEST_START_NEW_SCENE = True

    def tearDown(self):
        _base._CLASS_BY_TYPE.clear()
        _base._CASTABLE_TYPES.clear()
        super().tearDown()

    def test_copy_shares_the_internals(self):
        cmds.createNode("transform", name="a")
        dg  = Node("a")
        dup = copy.copy(dg)
        self.assertIs(type(dup), type(dg))
        self.assertIsNot(dup, dg)
        for key in _DAG_KEYS:
            self.assertIs(vars(dup)[key], vars(dg)[key])
        self.assertEqual(dup.name, "a")

    def test_dunder_probe_on_a_freed_node(self):
        cmds.createNode("transform", name="a")
        dg = Node("a")
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
        dg = Node("md")
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
        mesh  = Mesh(Node(cube))
        self.assertIs(type(mesh), Mesh)
        self.assertEqual(mesh.name, shape)
        self.assertTrue(mesh.mobject == _mobject(shape))
        # a node of the class itself (or a subclass) still shares its internals
        again = Mesh(mesh)
        self.assertIs(vars(again)["_fn_set"], vars(mesh)["_fn_set"])
        source = Node(cube)
        self.assertIs(vars(Transform(source))["_fn_set"], vars(source)["_fn_set"])

    def test_constructor_key_order(self):
        from rig.nodetypes import DAGNode, Mesh

        md    = cmds.createNode("multiplyDivide", name="md")
        xform = cmds.createNode("transform", name="xf")
        cube  = cmds.polyCube(name="cube", ch=False)[0]
        shape = cmds.listRelatives(cube, shapes=True)[0]
        Node(_mobject(md))
        Node(_mobject(xform))
        Node(_mobject(shape))
        from_path = OpenMaya.MSelectionList()
        from_path.add(xform)
        # re-pinned (round 4b NC2): the constructors are reached by
        # Cls._wrap(x); Cls(x) is the reference, which returns the node of
        # the most derived class (built by the cast)
        cases = {
            "dg": (DGNode._wrap(md), _DG_KEYS),
            "dg_copy": (DGNode._wrap(DGNode._wrap(md)), _DG_KEYS),
            "dg_mobject": (DGNode._wrap(_mobject(md)), _DG_KEYS),
            "dag": (DAGNode._wrap(xform), _DAG_KEYS),
            "dag_path": (DAGNode._wrap(from_path.getDagPath(0)), _DAG_KEYS),
            "dag_mobject": (
                DAGNode._wrap(_mobject(xform)),
                [_DAG_KEYS[1], _DAG_KEYS[0]] + _DAG_KEYS[2:],
            ),
            "dag_copy": (
                DAGNode._wrap(DAGNode._wrap(xform)),
                [_DAG_KEYS[1], _DAG_KEYS[0]] + _DAG_KEYS[2:],
            ),
            "geometry": (Mesh._wrap(shape), _DAG_KEYS + _GEO_KEYS),
            "checked_dg": (Node(_mobject(md)), _DG_KEYS),
            "checked_dag": (Node(_mobject(xform)), _DAG_KEYS),
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
        held = [Node(name) for name in names]
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
            self.assertEqual(str(Node("md").__parked__), "md.__parked__")
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
# `_fn_set1`. The cold readers ask `_handle_valid`; each site
# below reads the objects inline, on a hot path or because it keeps them, and
# marks the line "NW6: API 1.0 handle". A new reader is added here (and marked).
_API1_SITES = frozenset(
    {
        # the helper
        ("rig.nodetypes._base", "_handle_valid"),
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
        # the attribute handle of an owned spec plug (round 4a M10)
        ("rig.spec._base", "_plug_of"),
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
    """M2: the API 1.0 handle helper (`_handle_valid`) and the
    complete list of the package's `_objhandle1` / `_fn_set1` readers, so that
    round 5 (NW6, the handles off API 1.0) has one list of sites to edit."""

    TEST_START_NEW_SCENE = True

    def test_readers_are_known(self):
        sites = {(module, func) for module, func, _, _, _ in _api1_accesses()}
        self.assertEqual(sites, _API1_SITES)

    def test_inline_readers_are_marked(self):
        helpers  = {("rig.nodetypes._base", "_handle_valid")}
        unmarked = [
            f"{os.path.basename(path)}:{line} {func}"
            for module, func, path, line, marked in _api1_accesses()
            if not marked and (module, func) not in helpers
        ]
        self.assertEqual(unmarked, [])

    def test_the_helpers(self):
        from rig.nodetypes import _base
        from rig.nodetypes._base import _handle_valid

        # the other helper had no package caller once the wrapper was gone
        self.assertFalse(hasattr(_base, "_handle_alive"))

        cmds.undoInfo(state=True, infinity=True)
        live    = Node(cmds.createNode("transform", name="live"))
        deleted = Node(cmds.createNode("multiplyDivide", name="deleted"))
        freed   = Node(cmds.createNode("transform", name="freed"))
        half    = object.__new__(Transform)
        cmds.delete("deleted")
        cases = {
            "live": (live, True),
            "deleted": (deleted, False),
            "half_built": (half, False),
        }
        for label, (node, valid) in cases.items():
            with self.subTest(case=label):
                self.assertIs(_handle_valid(vars(node)), valid)
                self.assertIs(node.is_valid, valid)
        self.assertIs(_handle_valid({}), False)
        cmds.undo()
        self.assertTrue(deleted.is_valid)
        cmds.file(new=True, force=True)
        # a freed node: only its API 1.0 handle is read
        probe = mock.Mock()
        vars(freed)["_fn_set"] = probe
        self.assertIs(_handle_valid(vars(freed)), False)
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
        dg = Node("a")
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
        a, b = Node("a"), Node("b")
        with self.assertRaisesRegex(TypeError, r"^'>>' does not connect plugs: write b\.translateY"):
            a.tx >> b.ty
        self.assertIsNone(cmds.listConnections("b.ty", source=True))
        self.assertEqual(str(a.tx >> "txCopy"), "a.txCopy")
        for label, build in (
            ("+", lambda: a.tx + b.tx),
            ("%", lambda: a.tx % b.tx),
            ("//", lambda: a.tx // b.tx),
            ("==", lambda: a.tx == b.tx),
        ):
            with self.subTest(op=label):
                before = _scene_nodes()
                result = build()
                self.assertTrue(_scene_nodes() - before, label)
                self.assertIsInstance(result, Plug)
        # an ordering comparison builds its node and has no truth value, so
        # sorted() raises at its first comparison instead of ordering by a
        # condition plug that is always true
        before = _scene_nodes()
        with self.assertRaisesRegex(TypeError, r"ordering comparison"):
            sorted([b.tx, a.tx])
        self.assertEqual(len(_scene_nodes() - before), 1)
        self.assertEqual(
            [str(p) for p in sorted([b.tx, a.tx], key=str)], ["a.translateX", "b.translateX"]
        )
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
        dg     = Node("a")
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
        curve = Node(
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
        # the Node factory returns the typed node, so this is the typed node's
        # sugar reached through Node(...), and a Container's (a DGNode subclass)
        # published names
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
        dg = Node("a")
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
        b = Node("b")
        self.assertIs(b << Node("a").matrix, b)
        for channel in ("t", "r", "s"):
            with self.subTest(channel=channel):
                sources = cmds.listConnections(f"b.{channel}", source=True, destination=False)
                self.assertEqual([cmds.nodeType(s) for s in sources], ["decomposeMatrix"])
        cmds.createNode("transform", name="c")
        c        = Node("c")
        m        = np.eye(4)
        m[3, :3] = [7.0, 8.0, 9.0]
        self.assertIs(c << m, c)
        self.assertEqual(cmds.getAttr("c.t")[0], (7.0, 8.0, 9.0))

    def test_component_fallbacks_on_typed_nodes(self):
        cube  = cmds.polyCube(name="cube", ch=False)[0]
        shape = cmds.listRelatives(cube, shapes=True)[0]
        surf  = cmds.sphere(name="ball", constructionHistory=False)[0]
        self.assertEqual(str(Node(cube).vtx[3]), f"{shape}.controlPoints[3]")
        faces = Node(cube).f
        self.assertIsInstance(faces, Components)
        self.assertTrue(faces.shape.mobject == _mobject(shape))
        self.assertIsInstance(Node(shape).e, Components)
        from rig._internal.plug import ComponentPlug

        cv = Node(surf).cv
        self.assertIsInstance(cv, ComponentPlug)
        self.assertEqual(str(cv[1, 2]), "ballShape.cv[1][2]")
        with self.assertRaises(AttributeError):
            Node(shape).cv
        # a curve shape's `f` is its Maya attr `form`, not a component
        crv = cmds.listRelatives(cmds.curve(point=[(0, 0, 0), (1, 0, 0)], degree=1), shapes=True)[0]
        self.assertEqual(str(Node(crv).f), f"{crv}.form")

    def test_plug_private_names(self):
        cmds.createNode("transform", name="a")
        plug = Node("a").tx
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
        self.assertEqual(str(Node("a")._pair._second), "a._second")
        # a half-built plug has none
        self.assertFalse(hasattr(Plug.__new__(Plug, "a.tx"), "_x"))

    def test_plug_private_names_on_a_freed_node(self):
        cmds.createNode("transform", name="a")
        plug = Node("a").tx
        cmds.file(new=True, force=True)
        with mock.patch.object(cmds, "container", wraps=cmds.container) as probe:
            for name in ("__array__", "_x", "__parked__"):
                self.assertFalse(hasattr(plug, name), name)
        self.assertEqual(probe.call_count, 0)

    def test_find_attr_of_a_plug_is_the_attribute(self):
        attribute_cls = _base_module().Attribute
        cmds.createNode("transform", name="a")
        cmds.createNode("transform", name="b")
        dg = Node("a")
        for plug in (dg.tx, Node("a").tx, Plug("a.tx"), dg.t[0]):
            with self.subTest(plug=type(plug.node).__name__):
                attr = dg.find_attr(plug)
                self.assertIs(type(attr), attribute_cls)
                self.assertIs(attr.node, plug.node)
                self.assertEqual(str(attr), "a.translateX")
                self.assertEqual(attr.get(), 0.0)
        # another node's plug is refused, quietly too
        with self.assertRaisesRegex(AttributeError, "doesn't belong to a"):
            dg.find_attr(Node("b").tx)
        self.assertIsNone(dg.find_attr(Node("b").tx, quiet=True))
        # a typed Attribute is handed back as is
        typed = dg.find_attr("ty")
        self.assertIs(dg.find_attr(typed), typed)

    def test_find_attr_of_a_freed_plug_raises(self):
        cmds.createNode("transform", name="a")
        plug = Node("a").tx
        cmds.file(new=True, force=True)
        cmds.createNode("transform", name="a")
        with self.assertRaisesRegex(RuntimeError, "already deleted!$"):
            Node("a").find_attr(plug)

    def test_str_operand_and_same_plug_compare_build_nothing(self):
        # X1 and the one operand check reach typed plugs too
        cmds.createNode("transform", name="cube")
        before = _scene_nodes()
        with self.assertRaises(TypeError):
            Node("cube").tx == "cube.ty"
        self.assertEqual(_scene_nodes(), before)
        self.assertIs(Node("cube").tx == Node("cube").tx, True)
        self.assertIs(Node("cube").tx != Node("cube").tx, False)
        self.assertEqual({Node("cube").tx: 1}[Node("cube").tx], 1)
        self.assertEqual({Node("cube").tx: 2}[Node("cube").tx], 2)
        self.assertIn(Node("cube").tx, {Node("cube").tx, Node("cube").ty})
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
        cube = Node(cmds.polyCube(name="cube", ch=False)[0])
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
        node = Node(cmds.createNode("transform", name="a"))
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
        # through the cast core and (M4) through the Node factory, which give the
        # same typed node; a Plug's element_by_* result is a Plug (M4B, D29)
        attribute_cls = _base_module().Attribute
        for factory in (_base._cast, Node):
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
                # M4B (D29): a Plug's element is a Plug
                self.assertIs(type(element), Plug)
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
        dg   = Node(cmds.createNode("transform", name="gone"))
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
        self.assertEqual(str(Node("back").input1X), "back.input1X")
        self.assertIsNot(Node("back").input1X.node, dg)

    def _held(self, names):
        held = []
        for name in names:
            node     = Node(name)
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
            self.assertEqual(str(Node("held0").tx), "held0.translateX")
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
        second = Node("|T2|S")
        self.assertEqual(str(second.v), "T2|S.visibility")
        self.assertEqual(str(Node("|T1|S").v), "T1|S.visibility")
        self.assertIs(second.v.node, second)
        for path, index, name in (
            ("|T2|S", 0, "T2|S.worldMatrix[0]"),
            ("|T2|S", 1, "T2|S.worldMatrix"),
            ("|T1|S", 0, "T1|S.worldMatrix"),
            ("|T1|S", 1, "T1|S.worldMatrix[1]"),
        ):
            with self.subTest(path=path, index=index):
                plug = Node(path).worldMatrix[index]
                self.assertEqual(str(plug), name)
                self.assertEqual(plug.get()[3][0], 7.0 if index else 0.0)
                dst = cmds.createNode("multMatrix")
                Node(dst).matrixIn[0] << plug
                self.assertEqual(_source_index(dst + ".matrixIn[0]"), index)
        # one plug read through two paths is one key, and compares the same
        first = Node("|T1|S").v
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
        dg = Node("ns:n")
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
        other = Node(made)
        other.namespace = "other"
        self.assertEqual(str(other.ty), "other:m.translateY")
        self.assertEqual(str(dg.ty), "ns:n.translateY")

    def test_components_of_typed_nodes(self):
        # E6
        from rig._internal.plug import ComponentPlug

        cube  = cmds.polyCube(name="cube", ch=False)[0]
        shape = cmds.listRelatives(cube, shapes=True)[0]
        mesh  = Node(shape)
        self.assertIs(mesh.vtx.node, mesh)
        self.assertEqual(str(mesh.vtx[2]), f"{shape}.controlPoints[2]")
        self.assertEqual(
            [str(p) for p in mesh.vtx[0:2]],
            [f"{shape}.controlPoints[0]", f"{shape}.controlPoints[1]"],
        )
        self.assertEqual((Node(cube).e >> None).tolist(), list(range(12)))
        self.assertEqual(Node(cube).f[1:3].count, 2)
        surf  = cmds.sphere(name="ball", ch=False)[0]
        shell = Node(cmds.listRelatives(surf, shapes=True)[0])
        cv    = shell.cv
        self.assertIsInstance(cv, ComponentPlug)
        self.assertIs(cv.node, shell)
        element = cv[1, 2]
        self.assertIs(element.node, shell)
        self.assertTrue(element.equals(Plug(f"{shell}.cv[1][2]")))
        lattice = cmds.lattice(cmds.polySphere(name="lsph", ch=False)[0], divisions=(2, 3, 2))[1]
        pt = Node(cmds.listRelatives(lattice, shapes=True)[0]).pt
        self.assertIsInstance(pt, ComponentPlug)
        self.assertEqual(str(pt[1, 2, 0]), f"{pt.node}.pt[1][2][0]")
        # a 1-D curve cv is a plain Plug
        crv = Node(
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
        dg   = Node("a")
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
        dg   = Node("a")
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
        # re-pinned (round 4a M8): the _dg_node shim that returned the node is gone
        self.assertFalse(hasattr(dg, "_dg_node"))
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
        node  = Node(cube)
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
            ("shape mplug", Node(shape).find_attr("outMesh").plug, Mesh),
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
        self.assertIs(type(Node(cube)), type(_base._cast(cube)))
        # a node class constructs as usual; the attribute of a dotted name is
        # Attribute(...) (R2), and the private cast core keeps its attribute branch
        self.assertIs(type(Transform(cube)), Transform)
        self.assertIs(type(_base.Attribute(f"{cube}.tx")), _base.Attribute)
        self.assertIs(type(_base._cast(f"{cube}.tx")), _base.Attribute)

    def test_the_class_call_has_one_branch_point(self):
        # NodeMeta.__call__ runs for every node class call and dispatches the
        # root only; the cast core constructs without it (round 4b builds on both)
        from rig.nodetypes._base import NodeMeta

        cmds.createNode("transform", name="a")
        calls    = []
        original = NodeMeta.__call__

        def counting(cls, *args, **kwargs):
            calls.append(cls.__name__)
            return original(cls, *args, **kwargs)

        with mock.patch.object(NodeMeta, "__call__", counting):
            _base._cast("a")
            _base._cast(_mobject("a"))
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
        mplug = Node("a").find_attr("tx").plug
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
        plain = Node(str(box))
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
        self.assertFalse(box == Node("inner"))
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
        # (at M3 a typed node on the left raised "... is not a Node")
        from rig import Layer, Tag

        cube = Node(cmds.polyCube(name="cube", ch=False)[0])
        self.assertIs(cube << Tag("t1"), cube)
        self.assertIs(cube << Layer("L"), cube)
        self.assertTrue(cube >> Layer("L"))
        faces = cube.f[0:2]
        self.assertIs(faces << Tag("t2"), faces)

    def test_a_typed_node_on_the_right_of_a_plug(self):
        # typed nodes are clone targets of plug >> node and bare nodes for <<
        from rig.spec import Float

        src = Node(cmds.createNode("transform", name="src"))
        dst = Node(cmds.createNode("transform", name="dst"))
        knob = src << Float("knob", dv=0.25)
        clone = knob >> dst
        self.assertIs(type(clone), Plug)
        self.assertEqual(str(clone), "dst.knob")
        # tightened (round 4a M10, spec S5): the clone is owned by dst itself,
        # and the spec's plug by src
        self.assertIs(clone.node, dst)
        self.assertIs(knob.node, src)
        with self.assertRaisesRegex(TypeError, "^Cannot inject bare Node 'src' into matrix Plug"):
            dst.offsetParentMatrix << src
        with self.assertRaisesRegex(TypeError, r"^Cannot inject Node into a bare Node"):
            dst << src

    def test_wrap_on_the_node_classes(self):
        cmds.createNode("transform", name="a")
        self.assertEqual(repr(Node.wrap("a")), 'Transform("a")')
        self.assertEqual(
            repr(Transform.wrap(["a", "persp"])),
            'List([Transform("a"), Transform("persp")])',
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


def _oracle(parent, mplug, cls):
    """What round 3 built for a child / element / parent plug `mplug` of `parent`:
    a fresh `cls` of it handed `parent`'s owner, handles and path naming."""
    base = _base_module()
    return base._inherit_owner(parent, base._new_attr(cls, mplug))


def _name_or_error(attr):
    try:
        return str(attr)
    except Exception as exc:
        return (type(exc).__name__, str(exc))


class TestOwnerPropagatingConstruction(MayaTestCase):
    """M4B (D29): a Plug's elements are Plugs built once (``_CHILD_CLASS``), in the
    state round 3 gave them (``_new_attr`` + ``_inherit_owner``: owner, handles,
    naming through the owner's path), for Plug and typed Attribute parents, owned
    or not, on DG, DAG, instanced (E3), dynamic (E7) and component (E6) attrs,
    deleted to the undo queue and undone (E1), freed (E2); and D12, the one cast
    core (``_cast``) the ``Node`` factory calls."""

    TEST_START_NEW_SCENE = True

    def _scene(self):
        cmds.createNode("transform", name="t")
        pma = cmds.createNode("plusMinusAverage", name="pma")
        for i in range(3):
            cmds.setAttr(f"{pma}.input1D[{i}]", i)
        top = cmds.createNode("transform", name="T1")
        cmds.createNode("transform", name="S", parent=top)
        cmds.createNode("transform", name="T2")
        cmds.parent("T1|S", "T2", add=True, relative=True)
        cmds.createNode("transform", name="dyn")
        cmds.addAttr("dyn", longName="dc", attributeType="compound", numberOfChildren=2, multi=True)
        cmds.addAttr("dyn", longName="dcx", attributeType="double", parent="dc")
        cmds.addAttr("dyn", longName="dcy", attributeType="double", parent="dc")
        cmds.addAttr("dyn", longName="dv", attributeType="double3")
        for axis in "xyz":
            cmds.addAttr("dyn", longName=f"dv{axis}", attributeType="double", parent="dv")
        cmds.addAttr("dyn", longName="arr", attributeType="double", multi=True)
        cmds.setAttr("dyn.arr[2]", 1)
        cmds.setAttr("dyn.dc[0].dcx", 1)
        cmds.polyCube(name="cube", ch=False)
        cmds.nurbsPlane(name="surf", ch=False, u=3, v=3)

    # parent label -> (parent factory, kind): a "multi" parent is read by index,
    # a "compound" parent by child index and name
    _PARENTS = {
        "owned multi":            (lambda: Node("pma").input1D, "multi"),
        "owned multi compound":   (lambda: Node("pma").input3D, "multi"),
        "owned compound":         (lambda: Node("t").t, "compound"),
        "owned element":          (lambda: Node("pma").input3D[1], "compound"),
        "owned world matrix":     (lambda: Node("t").worldMatrix, "multi"),
        "instanced T2 compound":  (lambda: Node("|T2|S").t, "compound"),
        "instanced T2 matrix":    (lambda: Node("|T2|S").worldMatrix, "multi"),
        "instanced T1 matrix":    (lambda: Node("|T1|S").worldMatrix, "multi"),
        "dynamic multi":          (lambda: Node("dyn").arr, "multi"),
        "dynamic compound multi": (lambda: Node("dyn").dc, "multi"),
        "dynamic element":        (lambda: Node("dyn").dc[0], "compound"),
        "dynamic compound":       (lambda: Node("dyn").dv, "compound"),
        "mesh vtx":               (lambda: Node("cube").vtx, "multi"),
        "unowned multi":          (lambda: Plug("pma.input1D"), "multi"),
        "unowned compound":       (lambda: Plug("t.t"), "compound"),
        "unowned instanced":      (lambda: Plug("|T2|S.t"), "compound"),
        "unowned dynamic":        (lambda: Plug("dyn.dv"), "compound"),
        "unowned dynamic multi":  (lambda: Plug("dyn.arr"), "multi"),
        "plug of a typed attr":   (lambda: Plug(Node("|T2|S").find_attr("t")), "compound"),
        "typed multi":            (lambda: Node("pma").find_attr("input1D"), "multi"),
        "typed compound":         (lambda: Node("t").find_attr("t"), "compound"),
        "typed instanced":        (lambda: Node("|T2|S").find_attr("t"), "compound"),
        "typed instanced matrix": (lambda: Node("|T2|S").find_attr("worldMatrix"), "multi"),
        "typed dynamic":          (lambda: Node("dyn").find_attr("dv"), "compound"),
        "typed dynamic multi":    (lambda: Node("dyn").find_attr("arr"), "multi"),
        "named typed multi":      (lambda: _base_module().Attribute("pma.input1D"), "multi"),
    }

    def _cases(self, parent, kind):
        """{label: (op, [(mplug, class)])}: every element / child / parent
        accessor of `parent`, with the MPlugs and classes round 3 built."""
        Attribute = _base_module().Attribute
        cls       = Plug if isinstance(parent, Plug) else Attribute
        mplug     = parent.plug
        if kind == "multi":
            cases = {
                "[1]": (lambda: parent[1], [(mplug.elementByLogicalIndex(1), cls)]),
                "[0:2]": (
                    lambda: parent[0:2],
                    [(mplug.elementByLogicalIndex(i), cls) for i in (0, 1)],
                ),
                "[[2, 0]]": (
                    lambda: parent[[2, 0]],
                    [(mplug.elementByLogicalIndex(i), cls) for i in (2, 0)],
                ),
                "element_by_logical_index(2)": (
                    lambda: parent.element_by_logical_index(2),
                    [(mplug.elementByLogicalIndex(2), cls)],
                ),
            }
            try:
                physical = mplug.elementByPhysicalIndex(0)
            except RuntimeError:
                physical = None  # no element yet, or component storage (controlPoints)
            if physical is not None:
                cases["element_by_physical_index(0)"] = (
                    lambda: parent.element_by_physical_index(0),
                    [(physical, cls)],
                )
            return cases
        count = mplug.numChildren()
        name  = OpenMaya.MFnAttribute(mplug.child(1).attribute()).name
        cases = {
            "child(1)": (lambda: parent.child(1), [(mplug.child(1), cls)]),
            f".{name}": (lambda: getattr(parent, name), [(mplug.child(1), cls)]),
            # the parent of a child is an Attribute, as round 3 built it
            "child(0).get_parent()": (lambda: parent.child(0).get_parent(), [(mplug, Attribute)]),
        }
        if cls is Plug:
            # only a Plug indexes a compound's children
            cases["[1]"]   = (lambda: parent[1], [(mplug.child(1), cls)])
            cases["[-1]"]  = (lambda: parent[-1], [(mplug.child(count - 1), cls)])
            cases["[0:2]"] = (lambda: parent[0:2], [(mplug.child(i), cls) for i in (0, 1)])
        return cases

    def _assert_built_as_round_3(self, parent, built, expected):
        from rig._internal.list import List

        items = built if isinstance(built, list) else [built]
        if isinstance(built, list):
            self.assertIs(type(built), List if isinstance(parent, Plug) else list)
        self.assertEqual(len(items), len(expected))
        for item, (mplug, cls) in zip(items, expected):
            oracle = _oracle(parent, mplug, cls)
            # the state first: naming an unowned plug casts its owner
            self.assertIs(type(item), type(oracle))
            self.assertEqual(list(vars(item)), list(vars(oracle)))
            self.assertEqual(str.__str__(item), str.__str__(oracle))
            for key, value in vars(oracle).items():
                if key in ("_node", "_handle1", "_attr1"):
                    self.assertIs(vars(item)[key], value, key)
                elif key == "_mplug":
                    self.assertEqual(vars(item)[key], mplug)
                else:
                    self.assertEqual(vars(item)[key], value, key)
            self.assertEqual(_name_or_error(item), _name_or_error(oracle))
            if vars(parent)["_node"] is not None:
                self.assertIs(item.node, parent.node)

    def _sweep(self, parents):
        for label, (make, kind) in parents.items():
            for op_label in self._cases(make(), kind):
                with self.subTest(parent=label, op=op_label):
                    parent = make()
                    op, expected = self._cases(parent, kind)[op_label]
                    self._assert_built_as_round_3(parent, op(), expected)

    def test_elements_children_and_parents_are_built_as_round_3(self):
        self._scene()
        self._sweep(self._PARENTS)

    def test_on_a_node_deleted_to_the_undo_queue_and_undone(self):
        # E1: the plugs of a deleted node still build (its MPlugs are valid in
        # the undo queue) as round 3 built them, and again after the undo
        self._scene()
        cmds.undoInfo(state=True, infinity=True)
        held = {
            label: (make(), kind)
            for label, (make, kind) in self._PARENTS.items()
            if not label.startswith(("instanced", "mesh", "typed instanced", "plug of"))
        }
        cmds.undoInfo(openChunk=True)
        cmds.delete("pma", "t", "dyn")
        cmds.undoInfo(closeChunk=True)
        parents = {label: (lambda p=p: p, kind) for label, (p, kind) in held.items()}
        self._sweep(parents)
        cmds.undo()
        self._sweep(parents)

    def test_held_across_a_new_scene_raise(self):
        # E2: every accessor of a parent a new scene freed raises the freed
        # error (its MPlug points at freed memory) before it builds anything
        self._scene()
        held = {label: make() for label, (make, _) in self._PARENTS.items()}
        cmds.file(new=True, force=True)
        for _ in range(50):
            cmds.createNode("multiplyDivide")  # reuse the freed memory
        ops = {
            "[1]":                          lambda p: p[1],
            "[0:2]":                        lambda p: p[0:2],
            "child(0)":                     lambda p: p.child(0),
            "element_by_logical_index(0)":  lambda p: p.element_by_logical_index(0),
            "element_by_physical_index(0)": lambda p: p.element_by_physical_index(0),
            "get_parent()":                 lambda p: p.get_parent(),
        }
        for label, parent in held.items():
            for op_label, op in ops.items():
                with self.subTest(parent=label, op=op_label):
                    with self.assertRaisesRegex(RuntimeError, "already deleted!$"):
                        op(parent)

    def test_the_child_class(self):
        from rig._internal.plug import ComponentPlug

        Attribute = _base_module().Attribute
        self.assertIs(Attribute._CHILD_CLASS, Attribute)
        self.assertIs(Plug._CHILD_CLASS, Plug)
        self.assertIs(ComponentPlug._CHILD_CLASS, Plug)
        self._scene()
        # a component handle's element by index is the flat storage, a Plug;
        # its element by coordinates stays the ComponentPlug element
        handle = Node("surf").cv
        flat   = handle.element_by_logical_index(3)
        self.assertIs(type(handle), ComponentPlug)
        self.assertIs(type(flat), Plug)
        self.assertIs(flat.node, handle.node)
        self.assertEqual(str(flat), "surfShape.controlPoints[3]")
        self.assertIs(type(handle[1, 2]), ComponentPlug)
        self.assertIs(handle[1, 2].node, handle.node)
        # a Plug's element by index is a Plug, a typed attr's an Attribute
        self.assertIs(type(Node("pma").input1D.element_by_physical_index(0)), Plug)
        self.assertIs(type(Node("pma").find_attr("input1D").element_by_physical_index(0)), Attribute)

    def test_plugs_are_never_cached(self):
        # D30: every lookup builds a new Plug; the child caches hold Attributes
        self._scene()
        Attribute = _base_module().Attribute
        multi = Node("pma").input1D
        comp  = Node("t").t
        self.assertIsNot(multi[1], multi[1])
        self.assertIsNot(comp[1], comp[1])
        self.assertIsNot(comp.child(1), comp.child(1))
        self.assertIsNot(comp.translateY, comp.translateY)
        for cache in ("_Attribute__child_id_dict", "_Attribute__child_name_dict"):
            with self.subTest(cache=cache):
                self.assertTrue(vars(comp)[cache])
                self.assertTrue(all(type(v) is Attribute for v in vars(comp)[cache].values()))
        self.assertEqual(vars(multi)["_Attribute__child_id_dict"], {})

    def test_a_slice_keeps_its_parent_multi(self):
        # the list results are Lists; a multi's slice is tagged with it
        self._scene()
        multi = Node("pma").input1D
        self.assertIs(multi[0:2]._parent_multi, multi)
        self.assertIsNone(multi[[0, 1]]._parent_multi)
        self.assertIsNone(Node("t").t[0:2]._parent_multi)
        self.assertEqual(list(multi[5:5]), [])
        self.assertIs(multi[5:5]._parent_multi, multi)

    def test_an_element_is_built_once(self):
        # D29: one construction per element (round 3 built an Attribute, then
        # a Plug of it)
        import rig._internal.plug as plug_module

        base = _base_module()
        self._scene()
        multi    = Node("pma").input1D
        unowned  = Plug("pma.input1D")
        typed    = Node("pma").find_attr("input1D")
        compound = Node("t").t
        compound.translateY
        compound.child(1)
        for label, op, count in (
            ("pma.input1D[3]", lambda: multi[3], 1),
            ("pma.input1D[0:3]", lambda: multi[0:3], 3),
            ("pma.input1D[[0, 2]]", lambda: multi[[0, 2]], 2),
            ("unowned[3]", lambda: unowned[3], 1),
            ("typed[3]", lambda: typed[3], 1),
            ("element_by_logical_index", lambda: multi.element_by_logical_index(3), 1),
            ("t[0]", lambda: compound[0], 1),
            ("t[:]", lambda: compound[:], 3),
            ("t.child(1) (cached child)", lambda: compound.child(1), 1),
            ("t.translateY (cached child)", lambda: compound.translateY, 1),
        ):
            with self.subTest(op=label):
                in_base = mock.Mock(wraps=base._new_attr)
                in_plug = mock.Mock(wraps=plug_module._new_attr)
                with mock.patch.object(base, "_new_attr", in_base), mock.patch.object(
                    plug_module, "_new_attr", in_plug
                ):
                    op()
                self.assertEqual(in_base.call_count + in_plug.call_count, count)

    def test_node_factory_makes_one_cast_frame(self):
        # D12: Node(x) runs the cast core once (R2: there is no other cast door)
        base = _base_module()
        cmds.createNode("transform", name="a")
        cast = base._cast.__code__

        def frames(func):
            seen = []

            def profile(frame, event, arg):
                if event == "call" and frame.f_code is cast:
                    seen.append("_cast")

            previous = sys.getprofile()
            sys.setprofile(profile)
            try:
                func()
            finally:
                sys.setprofile(previous)
            return seen

        for label, value in (("name", "a"), ("mobject", _mobject("a")), ("dotted", "a.tx")):
            with self.subTest(value=label):
                self.assertEqual(frames(lambda: Node(value)), ["_cast"])
        node = Node("a")
        self.assertEqual(frames(lambda: Node(node)), [])
        self.assertEqual(frames(lambda: Node(node.tx)), [])
        self.assertEqual(frames(lambda: base._cast(node)), ["_cast"])

    def test_one_cast_core_gives_one_result(self):
        # Node(x) and the private cast core are one function for a node: same
        # classes, same errors (R2: Node is the only public door); the core keeps
        # its attribute branch, Node gives the node of a dotted name
        base = _base_module()
        cube = cmds.polyCube(name="cube", ch=False)[0]
        for value in (cube, "cubeShape", _mobject(cube), cmds.ls(cube, uuid=True)[0]):
            with self.subTest(value=str(value)):
                self.assertIs(type(base._cast(value)), type(Node(value)))
        self.assertIs(type(base._cast("cube.tx")), base.Attribute)
        self.assertIs(type(Node("cube.tx")), Transform)
        for bad in (3, None):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "is not a str, MObject"):
                    Node(bad)
                with self.assertRaisesRegex(ValueError, "is not a str, MObject"):
                    base._cast(bad)
        gone    = cmds.createNode("transform", name="gone")
        missing = cmds.ls(gone, uuid=True)[0]
        cmds.delete(gone)
        cmds.flushUndo()
        errors = []
        for cast in (Node, base._cast):
            try:
                cast(missing)
            except Exception as exc:
                errors.append((type(exc), str(exc)))
        # re-pinned (round 4b NC1): Node() raises the lookup family's
        # NodeNotFoundError naming the uuid (a TypeError, as the core's is); the
        # core keeps its own error
        self.assertEqual(
            errors,
            [
                (NodeNotFoundError, f"no node has the uuid {missing!r}"),
                (TypeError, f"No object matches uuid: {missing}."),
            ],
        )
        self.assertTrue(issubclass(errors[0][0], TypeError))
        for args, kwargs in ((("x",), {}), ((), {"k": 1})):
            with self.subTest(extra=(args, kwargs)):
                with self.assertRaises(TypeError):
                    Node(cube, *args, **kwargs)
        self.assertIs(type(Node(obj=cube)), Transform)
        with self.assertRaises(TypeError):
            Node()

    def test_every_cast_goes_through_the_cast_core(self):
        # R2: the Node factory and Attribute.node's lazy cast both call the module
        # global _cast, so a test that counts casts patches that one function; a
        # node object or a plug held by its node casts nothing
        base = _base_module()
        cmds.createNode("transform", name="a")
        counting = mock.Mock(side_effect=base._cast)
        with mock.patch.object(base, "_cast", counting):
            node = Node("a")
            self.assertIs(type(node), Transform)
            self.assertEqual(counting.call_count, 1)
            self.assertIs(Node(node), node)
            self.assertIs(Node(node.tx), node)
            self.assertEqual(counting.call_count, 1)
            self.assertEqual(str(Plug("a.tx")), "a.translateX")
            self.assertEqual(counting.call_count, 2)


# ---------------------------------------------------------------------------
# Round 3, moved from test_node_merge.py in round 4a M8 (class and method names
# kept): S0 Plug quick wins, S1 the owner rule, the merge prototype's review
# fixes, D-B one identity per Maya plug
# ---------------------------------------------------------------------------


def _instanced_locator():
    """Locator shape ``S`` instanced under ``T1`` (instance 0, tx 0) and ``T2``
    (instance 1, tx 7)."""
    cmds.loadPlugin("matrixNodes", quiet=True)
    cmds.createNode("transform", name="T1")
    cmds.createNode("locator", name="S", parent="T1")
    cmds.createNode("transform", name="T2")
    cmds.parent("|T1|S", "T2", add=True, shape=True)
    cmds.setAttr("T2.tx", 7)


class TestPlugQuickWins(MayaTestCase):
    """S0: the Plug and ``find_attr`` fixes that do not depend on the merge."""

    TEST_START_NEW_SCENE = True

    def test_property_setters_work_on_a_plug(self):
        net = cmds.createNode("network", name="net")
        cmds.addAttr(net, ln="knob", at="double", keyable=True)
        plug = Node(net).knob
        plug.alias = "dial"
        self.assertEqual(Plug("net.knob").alias, "dial")
        plug.alias = None
        self.assertEqual(Plug("net.knob").alias, "knob")
        plug.is_keyable = False
        self.assertFalse(cmds.getAttr("net.knob", keyable=True))
        plug.is_channel_box = True
        self.assertTrue(cmds.getAttr("net.knob", channelBox=True))
        plug.default_value = 2.5
        self.assertEqual(cmds.addAttr("net.knob", query=True, defaultValue=True), 2.5)
        plug.is_locked = True
        self.assertTrue(cmds.getAttr("net.knob", lock=True))
        # none of them became a Python attribute of the plug
        for name in ("alias", "is_keyable", "is_channel_box", "default_value", "is_locked"):
            self.assertNotIn(name, vars(plug))

    def test_str_method_names_stay_sugar(self):
        net = cmds.createNode("network", name="net")
        cmds.addAttr(net, ln="center", at="double")
        cmds.addAttr(net, ln="other", at="double")
        plug = Plug("net.other")
        plug.center = 5
        self.assertEqual(cmds.getAttr("net.center"), 5.0)
        self.assertNotIn("center", vars(plug))

    def test_bool_makes_no_container_query(self):
        cmds.createNode("transform", name="a")
        plugs = (Node("a").tx, Plug("a.t"), Node("a").t[0])
        with mock.patch.object(cmds, "container", wraps=cmds.container) as probe:
            self.assertEqual([bool(plug) for plug in plugs], [True] * 3)
        self.assertEqual(probe.call_count, 0)
        # a comparison result still reports whether its operands match
        self.assertTrue(Node("a").tx == Plug("a.tx"))
        self.assertFalse(Node("a").tx == Node("a").ty)

    def test_find_attr_filters_a_cache_hit(self):
        cmds.createNode("transform", name="a")
        dg     = Node("a")
        cached = dg.find_attr("tx")
        self.assertIsNone(dg.find_attr("tx", data_type="string"))
        self.assertIsNone(dg.find_attr("translateX", category="noSuchCategory"))
        self.assertIs(dg.find_attr("tx", data_type="doubleLinear"), cached)
        self.assertIs(dg.find_attr("translateX"), cached)

    def test_readded_extension_attr_through_the_same_node(self):
        cmds.createNode("transform", name="a")
        dg = Node("a")
        try:
            cmds.addExtension(nodeType="transform", longName="mergeExt", at="double")
            self.assertEqual(dg.find_attr("mergeExt").data_type, "double")
            cmds.deleteExtension(
                nodeType="transform", attribute="mergeExt", forceDelete=True
            )
            cmds.addExtension(nodeType="transform", longName="mergeExt", dataType="string")
            attr = dg.find_attr("mergeExt")
            self.assertEqual(str(attr), "a.mergeExt")
            self.assertEqual(attr.data_type, "string")
            self.assertNotIn("mergeExt", dg._attr_dict)
            # normal attrs are still cached
            dg.find_attr("tx")
            self.assertIn("translateX", dg._attr_dict)
        finally:
            if cmds.attributeQuery("mergeExt", type="transform", exists=True):
                cmds.deleteExtension(
                    nodeType="transform", attribute="mergeExt", forceDelete=True
                )

    def test_rename_attr_with_a_plug_creates_no_node(self):
        net = cmds.createNode("network", name="net")
        cmds.addAttr(net, ln="foo", at="double")
        dg     = Node(net)
        before = sorted(cmds.ls())
        result = dg.rename_attr(Node(net).foo, "bar")
        self.assertEqual(sorted(cmds.ls()), before)
        self.assertTrue(cmds.attributeQuery("bar", node=net, exists=True))
        self.assertEqual(str(result), "net.bar")


class TestOwnerRule(MayaTestCase):
    """S1: a plug's node is the node object it was read from."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self._registered = dict(_base._NODE_CLASS_DICT)

    def tearDown(self):
        _base._NODE_CLASS_DICT.clear()
        _base._NODE_CLASS_DICT.update(self._registered)
        _base._CLASS_BY_TYPE.clear()
        _base._CASTABLE_TYPES.clear()
        super().tearDown()

    def test_plugs_children_and_elements_share_the_node(self):
        node = Node(cmds.createNode("transform", name="a"))
        pma  = Node(cmds.createNode("plusMinusAverage", name="pma"))
        for plug in (
            node.tx,
            node.t[0],
            node.t.tx,
            node.t[:][1],
            node.translate.child(2),
        ):
            with self.subTest(plug=str(plug)):
                self.assertIs(plug.node, node)
        for plug in (pma.input1D[3], pma.input3D[1], pma.input3D[1].input3Dx):
            with self.subTest(plug=str(plug)):
                self.assertIs(plug.node, pma)
        self.assertIs(node.find_attr("tx").node, node)
        self.assertIs(node.find_attr("tx"), node.find_attr("translateX"))

    def test_foreign_plug_gets_its_own_node(self):
        points = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
        curve  = cmds.curve(point=points, name="crv")
        shape  = cmds.listRelatives(curve, shapes=True)[0]
        plug   = Node(curve).controlPoints
        self.assertEqual(plug.node.name, shape)
        self.assertTrue(plug.node.mobject == _mobject(shape))
        self.assertEqual(str(plug), f"{shape}.controlPoints")
        # the transform's own attrs are still its own
        self.assertEqual(Node(curve).tx.node.name, curve)

    def test_stale_path_names_the_surviving_instance(self):
        cmds.createNode("transform", name="T1")
        cmds.createNode("transform", name="T2")
        cmds.createNode("locator", name="S", parent="T1")
        cmds.parent("T1|S", "T2", add=True, shape=True, relative=True)
        node = Node("|T2|S")
        held = node.visibility
        self.assertEqual(str(held), "T2|S.visibility")
        cmds.parent("T2|S", removeObject=True, shape=True)
        self.assertEqual(node.name, "S")
        self.assertEqual(node.long_name, "|T1|S")
        self.assertTrue(node.mdagpath.isValid())
        self.assertEqual(str(held), "S.visibility")
        self.assertEqual(str(node.localPositionX), "S.localPositionX")

    def test_held_plug_across_delete_reuse_and_undo(self):
        cmds.undoInfo(state=True, infinity=True)
        node = Node(cmds.createNode("transform", name="held"))
        plug = node.tx
        cmds.delete("held")
        for func in (str, lambda p: p.get()):
            with self.assertRaises(RuntimeError) as ctx:
                func(plug)
            self.assertEqual(str(ctx.exception), "held already deleted!")
        cmds.createNode("transform", name="held")
        with self.assertRaises(RuntimeError) as ctx:
            str(plug)
        self.assertEqual(str(ctx.exception), "held already deleted!")
        # a plug built from the name finds the new node
        self.assertEqual(str(Plug("held.tx")), "held.translateX")
        cmds.undo()
        cmds.undo()
        self.assertEqual(str(plug), "held.translateX")
        plug << 3.0
        self.assertEqual(cmds.getAttr("held.tx"), 3.0)

    def test_owner_soundness_sweep(self):
        md         = cmds.createNode("multiplyDivide", name="md")
        xform      = cmds.createNode("transform", name="xf")
        jnt        = cmds.createNode("joint", name="jnt")
        cube       = cmds.polyCube(name="cube")[0]
        mesh       = cmds.listRelatives(cube, shapes=True)[0]
        surf       = cmds.sphere(name="ball", constructionHistory=False)[0]
        surf_shape = cmds.listRelatives(surf, shapes=True)[0]
        lattice    = cmds.lattice(cube, name="lat")[1]
        lat_shape  = cmds.listRelatives(lattice, shapes=True)[0]
        target     = cmds.polyCube(name="target")[0]
        base       = cmds.polyCube(name="base")[0]
        bs         = cmds.blendShape(target, base, name="bs")[0]
        pma        = cmds.createNode("plusMinusAverage", name="pma")
        cmds.createNode("transform", name="T1")
        cmds.createNode("transform", name="T2")
        cmds.createNode("locator", name="S", parent="T1")
        cmds.parent("T1|S", "T2", add=True, shape=True, relative=True)
        cmds.namespace(add="ns")
        cmds.createNode("transform", name="ns:n")
        with container("box") as box:
            inner = Node.create("transform", name="inner")
            container.publish_input(inner.tx, "slide")
        self.assertIsInstance(box, Container)

        plugs = {
            "dg": lambda: Node(md).input1X,
            "dg_child": lambda: Node(md).input1[1],
            "transform": lambda: Node(xform).tx,
            "transform_child": lambda: Node(xform).t.ty,
            "world_matrix": lambda: Node(xform).worldMatrix[0],
            "joint": lambda: Node(jnt).jointOrientX,
            "mesh": lambda: Node(mesh).outMesh,
            "mesh_vtx": lambda: Node(mesh).vtx[3],
            "vtx_via_transform": lambda: Node(cube).vtx[3],
            "surface_cv": lambda: Node(surf_shape).cv[1, 2],
            "surface_cv_handle": lambda: Node(surf_shape).cv,
            "surface_cv_via_transform": lambda: Node(surf).cv[1, 2],
            "lattice_pt": lambda: Node(lat_shape).pt[0, 1, 0],
            "blendshape_alias": lambda: getattr(Node(bs), target),
            "blendshape_weight": lambda: Node(bs).weight[0],
            "container_genuine": lambda: box.blackBox,
            "container_published": lambda: box.slide,
            "instanced": lambda: Node("|T2|S").visibility,
            "namespaced": lambda: Node("ns:n").tx,
            "element": lambda: Node(pma).input3D[1],
            "element_child": lambda: Node(pma).input3D[1].input3Dx,
            "string": lambda: Plug("xf.tx"),
        }
        for label, factory in plugs.items():
            with self.subTest(pattern=label):
                plug = factory()
                self.assertTrue(plug.node.mobject == plug.plug.node(), str(plug))
                self.assertIsInstance(plug.node, Node)
        for kind in ("f", "e"):
            with self.subTest(pattern=kind):
                components = getattr(Node(cube), kind)
                self.assertIsInstance(components, Components)
                self.assertTrue(components.shape.mobject == _mobject(mesh))

    def test_name_property_that_raises_names_by_fn_set(self):
        class _NameRaises(Transform):
            NATIVE_NODE_TYPE = "mergeNameRaisesProbe"

            @property
            def name(self):
                raise AttributeError("no name")

        cmds.createNode("transform", name="w")
        cmds.addAttr("w", longName="name", dataType="string")
        node = object.__new__(_NameRaises)
        vars(node).update(vars(Node(_mobject("w"))))
        vars(node)["_attr_dict"] = {}
        for plug in (Node(node).tx, node.find_attr("tx"), Node(node).t[0]):
            with self.subTest(plug=type(plug).__name__):
                self.assertEqual(str(plug), "w.translateX")
                self.assertEqual(plug.full_name, "w.translateX")


class TestReviewFixes(MayaTestCase):
    """The fixes the review of the merge prototype (proto/node-merge) found,
    ported in round 3 with the prototype's test ids."""

    TEST_START_NEW_SCENE = True

    _FREED = (
        r"^Transform node \(freed by a new scene, a file open or a reference unload\) "
        r"already deleted!$"
    )

    def _held(self, names):
        held = []
        for name in names:
            node     = Node(name)
            compound = node.t
            compound.tx  # a cached child
            typed = Node(name)
            typed.tx  # a cached typed attr
            held.append(
                (node, node.tx, compound, node.worldMatrix[0], node.find_attr("ty"), typed)
            )
        return held

    def _assert_freed(self, held):
        persp = Node("persp")
        ops = {
            "node.tx (cached)":  lambda n, p, t, w, a, d: n.tx,
            "node.ty":           lambda n, p, t, w, a, d: n.ty,
            "str(node)":         lambda n, p, t, w, a, d: str(n),
            "repr(node)":        lambda n, p, t, w, a, d: repr(n),
            "node == node":      lambda n, p, t, w, a, d: n == n,
            "hash(node)":        lambda n, p, t, w, a, d: hash(n),
            "node.tx = 1":       lambda n, p, t, w, a, d: setattr(n, "tx", 1),
            "str(plug)":         lambda n, p, t, w, a, d: str(p),
            "plug.get()":        lambda n, p, t, w, a, d: p.get(),
            "plug.set(1)":       lambda n, p, t, w, a, d: p.set(1),
            "plug << 1":         lambda n, p, t, w, a, d: p << 1,
            "plug << None":      lambda n, p, t, w, a, d: p << None,
            "plug >> None":      lambda n, p, t, w, a, d: p >> None,
            "plug.alias":        lambda n, p, t, w, a, d: p.alias,
            "plug.is_connected": lambda n, p, t, w, a, d: p.is_connected,
            "plug.is_locked":    lambda n, p, t, w, a, d: p.is_locked,
            "plug.get_inputs()": lambda n, p, t, w, a, d: p.get_inputs(),
            "plug + 1":          lambda n, p, t, w, a, d: p + 1,
            "plug == plug":      lambda n, p, t, w, a, d: p == p,
            "plug.equals(plug)": lambda n, p, t, w, a, d: p.equals(t),
            "plug.equals(self)": lambda n, p, t, w, a, d: p.equals(p),
            "compound.tx":       lambda n, p, t, w, a, d: t.tx,
            "compound[0]":       lambda n, p, t, w, a, d: t[0],
            "compound.child(0)": lambda n, p, t, w, a, d: t.child(0),
            "element.get()":     lambda n, p, t, w, a, d: w.get(),
            "attribute.get()":   lambda n, p, t, w, a, d: a.get(),
            "attribute parent":  lambda n, p, t, w, a, d: a.get_parent(),
            "persp.tx << plug":  lambda n, p, t, w, a, d: persp.tx << p,
            "memo key":          lambda n, p, t, w, a, d: _attribute_key(p),
            "str(typed)":        lambda n, p, t, w, a, d: str(d),
            "str(typed.tx)":     lambda n, p, t, w, a, d: str(d.tx),
            "typed.tx.get()":    lambda n, p, t, w, a, d: d.tx.get(),
            "typed.rx":          lambda n, p, t, w, a, d: d.rx,
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
            self.assertFalse(hasattr(node, "__deepcopy__"))
            self.assertIs(plug.node, node)
            self.assertIs(Node(node), node)
            self.assertIs(type(node >> None), Transform)

    def test_held_nodes_and_plugs_across_a_new_scene_raise(self):
        # a freed node's fn sets and MPlugs point at freed memory: reading them
        # named another node or crashed Maya; nothing reads them now
        held = self._held([cmds.createNode("transform", name=f"held{i}") for i in range(3)])
        cmds.file(new=True, force=True)
        for _ in range(50):
            cmds.createNode("multiplyDivide")  # reuse the freed memory
        self._assert_freed(held)

    def test_held_nodes_and_plugs_across_a_reference_unload_raise(self):
        folder = tempfile.mkdtemp(prefix="rig_freed_ref_")
        path   = os.path.join(folder, "freed_ref.ma").replace("\\", "/")
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

    def test_held_nodes_and_plugs_across_a_file_open_raise(self):
        # the opened file brings nodes of the same names: a held node is freed,
        # it neither names nor retargets to them
        folder = tempfile.mkdtemp(prefix="rig_freed_open_")
        path   = os.path.join(folder, "freed_open.ma").replace("\\", "/")
        try:
            names = [cmds.createNode("transform", name=f"held{i}") for i in range(2)]
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
            held = self._held(names)
            cmds.file(path, open=True, force=True)
            self.assertTrue(all(cmds.objExists(name) for name in names))
            self._assert_freed(held)
            self.assertEqual(str(Node("held0").tx), "held0.translateX")
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)

    def test_deleted_node_in_the_undo_queue_keeps_its_name(self):
        cmds.undoInfo(state=True, infinity=True)
        node = Node(cmds.createNode("transform", name="gone"))
        plug = node.tx
        typed = node >> None
        cmds.delete("gone")
        for op in (
            lambda: str(plug),
            lambda: str(node.tx),
            lambda: node.ty,
            lambda: plug.get(),
            lambda: str(typed),
        ):
            with self.assertRaisesRegex(RuntimeError, "^gone already deleted!$"):
                op()
        cmds.undo()
        self.assertEqual(str(plug), "gone.translateX")

    def test_instanced_world_space_elements_connect_the_element_asked_for(self):
        _instanced_locator()
        for path, index, name in (
            ("|T2|S", 0, "T2|S.worldMatrix[0]"),
            ("|T2|S", 1, "T2|S.worldMatrix"),
            ("|T1|S", 0, "T1|S.worldMatrix"),
            ("|T1|S", 1, "T1|S.worldMatrix[1]"),
        ):
            with self.subTest(path=path, index=index):
                plug = Node(path).worldMatrix[index]
                self.assertEqual(str(plug), name)
                self.assertEqual(plug.get()[3][0], 7.0 if index else 0.0)
                dst = cmds.createNode("multMatrix")
                Node(dst).matrixIn[0] << plug
                self.assertEqual(_source_index(dst + ".matrixIn[0]"), index)
                self.assertEqual(cmds.getAttr(dst + ".matrixSum")[12], 7.0 if index else 0.0)
        # without an index: the element of the path's own instance, as in cmds
        dst = cmds.createNode("multMatrix")
        Node(dst).matrixIn[0] << Node("|T2|S").worldMatrix
        self.assertEqual(_source_index(dst + ".matrixIn[0]"), 1)
        # one memo key per element, whatever the path it is read through
        second, first = Node("|T2|S"), Node("|T1|S")
        self.assertNotEqual(
            _attribute_key(second.worldMatrix[0]), _attribute_key(second.worldMatrix[1])
        )
        self.assertEqual(
            _attribute_key(first.worldMatrix[1]), _attribute_key(second.worldMatrix[1])
        )
        # so a memoized function reads the element asked for
        from rig.matrix import decompose

        tx = [
            cmds.getAttr(str(decompose(second.worldMatrix[index])).split(".")[0] + ".outputTranslateX")
            for index in (0, 1)
        ]
        self.assertEqual(tx, [0.0, 7.0])
        # a node with one instance is named as before
        single = Node(cmds.createNode("transform", name="single"))
        self.assertEqual(str(single.worldMatrix[0]), "single.worldMatrix")

    def test_stale_path_read_through_mdagpath_first_and_undo(self):
        cmds.undoInfo(state=True, infinity=True)
        cmds.createNode("transform", name="T1")
        cmds.createNode("transform", name="T2")
        cmds.createNode("locator", name="S", parent="T1")
        cmds.parent("T1|S", "T2", add=True, shape=True, relative=True)
        node = Node("|T2|S")
        held = node.visibility
        cmds.parent("T2|S", removeObject=True, shape=True)
        self.assertTrue(node.mdagpath.isValid())
        self.assertEqual(node.mdagpath.fullPathName(), "|T1|S")
        self.assertEqual(str(node.visibility), "S.visibility")
        # undoing the removal: the node names the path it was taken through again
        cmds.undo()
        self.assertEqual(node.long_name, "|T2|S")
        self.assertEqual(str(node.visibility), "T2|S.visibility")
        self.assertEqual(str(held), "T2|S.visibility")
        self.assertEqual(node.mdagpath.fullPathName(), "|T2|S")
        # and a second removal re-resolves it again
        cmds.parent("T2|S", removeObject=True, shape=True)
        self.assertEqual(str(held), "S.visibility")
        cmds.undo()
        self.assertEqual(str(held), "T2|S.visibility")

    def test_rshift_between_plugs_says_how_to_connect(self):
        a = Node(cmds.createNode("transform", name="a"))
        b = Node(cmds.createNode("transform", name="b"))
        cmds.addAttr("b", longName="fresh", attributeType="double")
        for target in (b.ty, b.find_attr("ty"), b.fresh, Plug("b.tz")):
            with self.subTest(target=f"{type(target).__name__} {target}"):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^'>>' does not connect plugs: write {target} << a\.translateX, "
                    rf"or a\.translateX\.connect\({target}, force=True\)$",
                ):
                    a.tx >> target
        self.assertIsNone(cmds.listConnections("b", source=True, destination=False))
        self.assertEqual(
            sorted(cmds.listAttr("a", userDefined=True) or []), [],
        )
        # a plain name is still a clone target
        self.assertEqual(str(a.tx >> "txCopy"), "a.txCopy")

    def test_shape_attr_read_through_a_transform_follows_the_shape(self):
        xf   = cmds.polyCube(name="c", ch=False)[0]
        node = Node(xf)
        self.assertEqual(str(node.outMesh), "cShape.outMesh")
        self.assertNotIn("outMesh", node._attr_dict)
        cmds.delete("cShape")
        tmp = cmds.polySphere(name="tmp", ch=False)[0]
        cmds.parent(cmds.listRelatives(tmp, shapes=True)[0], xf, shape=True, relative=True)
        self.assertEqual(str(node.outMesh), "tmpShape.outMesh")
        self.assertEqual(node.outMesh.node.name, "tmpShape")
        # typed access too, and the transform's own attrs are still cached
        self.assertEqual(str((node >> None).find_attr("outMesh")), "tmpShape.outMesh")
        node.tx
        self.assertIn("translateX", node._attr_dict)


class TestInstancedPlugIdentity(MayaTestCase):
    """Decision D-B: a plug's identity follows the Maya plug (node, attribute and
    logical indices), whatever the instance path it is named through."""

    TEST_START_NEW_SCENE = True

    def test_one_plug_through_two_paths_is_one_key(self):
        _instanced_locator()
        first, second = Node("|T1|S"), Node("|T2|S")
        for label, get in (
            ("v", lambda n: n.v),
            ("lp[0]", lambda n: n.localPosition[0]),
            ("lpx", lambda n: n.localPositionX),
        ):
            with self.subTest(plug=label):
                a, b = get(first), get(second)
                # each is still named through the path it was read from
                self.assertTrue(str(a).startswith("T1|S."))
                self.assertTrue(str(b).startswith("T2|S."))
                self.assertEqual(hash(a), hash(b))
                self.assertEqual(len({a: 1, b: 2}), 1)
                self.assertEqual(len({a, b}), 1)
                self.assertIn(b, [a])
                self.assertTrue(a.equals(b))
                self.assertTrue(bool(a == b))
                self.assertFalse(bool(a != b))
                self.assertEqual(_attribute_key(a), _attribute_key(b))
        # a plug of another attribute or another node is still another key
        self.assertEqual(len({first.v: 1, second.lodVisibility: 2, Node("T1").v: 3}), 3)
        self.assertFalse(bool(first.v == Node("T1").v))

    def test_world_space_elements_of_different_instances_are_distinct(self):
        _instanced_locator()
        first, second = Node("|T1|S"), Node("|T2|S")
        pairs = {
            # the same element read through either path: one key
            "wm[1] via T1 and T2": (first.worldMatrix[1], second.worldMatrix[1], True),
            "wm[0] via T1 and T2": (first.worldMatrix[0], second.worldMatrix[0], True),
            # an unindexed world space array is its path's element, as in cmds
            "T2 wm and wm[1]": (second.worldMatrix, first.worldMatrix[1], True),
            "T1 wm and wm[0]": (first.worldMatrix, second.worldMatrix[0], True),
            "Plug('T2|S.worldMatrix') and wm[1]": (
                Plug("T2|S.worldMatrix"), second.worldMatrix[1], True,
            ),
            # elements of different instances are different plugs
            "wm[0] and wm[1]": (second.worldMatrix[0], second.worldMatrix[1], False),
            "T1 wm and T2 wm": (first.worldMatrix, second.worldMatrix, False),
            "wim[0] and wim[1]": (
                first.worldInverseMatrix[0], first.worldInverseMatrix[1], False,
            ),
            # instObjGroups is per instance too, and its children name the index
            "iog[1] via T1 and T2": (first.instObjGroups[1], second.instObjGroups[1], True),
            "iog[0] and iog[1]": (second.instObjGroups[0], second.instObjGroups[1], False),
            "iog[1].og via T1 and T2": (
                first.instObjGroups[1].objectGroups,
                second.instObjGroups[1].objectGroups,
                True,
            ),
        }
        for label, (a, b, same) in pairs.items():
            with self.subTest(pair=label):
                self.assertEqual(a.equals(b), same)
                self.assertEqual(hash(a) == hash(b), same)
                self.assertEqual(_attribute_key(a) == _attribute_key(b), same)
        # distinct elements are distinct set members (their hashes differ, so no
        # `==`, which cannot compare two matrices, is needed)
        self.assertEqual(len({first.worldMatrix, second.worldMatrix}), 2)
        # a memoized function builds one node per element
        from rig.matrix import decompose

        via_first  = decompose(first.worldMatrix)
        via_second = decompose(second.worldMatrix)
        self.assertNotEqual(str(via_first), str(via_second))
        self.assertEqual(str(decompose(second.worldMatrix[0])), str(via_first))
        self.assertEqual(str(decompose(first.worldMatrix[1])), str(via_second))
        tx = [cmds.getAttr(str(d).split(".")[0] + ".outputTranslateX") for d in (via_first, via_second)]
        self.assertEqual(tx, [0.0, 7.0])

    def test_single_instance_world_matrix_spellings_stay_one_key(self):
        # v2.0.0a2 names both 'a.worldMatrix', and cmds resolves that to [0]
        node = Node(cmds.createNode("transform", name="a"))
        whole, element = node.worldMatrix, node.worldMatrix[0]
        self.assertEqual(str(whole), str(element))
        self.assertEqual(hash(whole), hash(element))
        self.assertTrue(whole.equals(element))
        self.assertEqual(_attribute_key(whole), _attribute_key(element))

    def test_component_element_and_its_storage_are_one_key(self):
        plane = cmds.nurbsPlane(name="np", degree=3, patchesU=1, patchesV=1, ch=False)[0]
        shape = cmds.listRelatives(plane, shapes=True)[0]
        element = Node(shape).cv[1, 2]
        storage = Plug(f"{shape}.controlPoints[6]")
        self.assertEqual(str(element), f"{shape}.cv[1][2]")
        self.assertEqual(len({element: 1, storage: 2}), 1)
        self.assertIn(storage, [element])
        self.assertNotIn(Node(shape).cv[1, 3], [element])

    def test_plug_hash_follows_the_plug_through_a_stale_path(self):
        _instanced_locator()
        cmds.undoInfo(state=True, infinity=True)
        second = Node("|T2|S")
        held   = second.v
        key    = hash(Node("|T1|S").v)
        self.assertEqual(hash(held), key)
        cmds.parent("T2|S", removeObject=True, shape=True)
        # the held node re-resolves to the surviving path, the plug is unchanged
        self.assertEqual(str(held), "S.visibility")
        self.assertEqual(hash(held), hash(Node("S").v))

    # -- round 3 step S2: the rest of the plug world follows D-B -- #

    def test_typed_attributes_through_two_paths_are_one_key(self):
        _instanced_locator()
        first, second = Node("|T1|S"), Node("|T2|S")
        a, b = first.find_attr("v"), second.find_attr("v")
        self.assertEqual((str(a), str(b)), ("T1|S.visibility", "T2|S.visibility"))
        self.assertEqual(hash(a), hash(b))
        self.assertTrue(a == b)
        self.assertFalse(a != b)
        self.assertEqual(len({a: 1, b: 2}), 1)
        self.assertEqual(len({a, b}), 1)
        self.assertIn(b, [a])
        self.assertEqual([a].index(b), 0)
        # through a rig Node, and against the Plug of the same plug: equal (the
        # X1 fold) and one key, so a mixed dict or set lookup builds nothing
        self.assertTrue(Node("|T1|S").find_attr("v") == Node("|T2|S").find_attr("v"))
        plug = Node("|T2|S").v
        self.assertEqual(hash(plug), hash(a))
        self.assertTrue(plug.equals(a))
        before = sorted(cmds.ls())
        self.assertIn(plug, {a})
        self.assertIn(a, {plug: 1})
        self.assertEqual(len({a, plug}), 1)
        self.assertEqual(sorted(cmds.ls()), before)
        # another attribute, another node, another instance's element: another key
        self.assertFalse(a == second.find_attr("lodVisibility"))
        self.assertFalse(a == Node("T1").find_attr("v"))
        wm0, wm1 = first.find_attr("worldMatrix")[0], second.find_attr("worldMatrix")[1]
        self.assertEqual(len({wm0, wm1}), 2)
        self.assertFalse(wm0 == wm1)
        self.assertTrue(wm1 == first.find_attr("worldMatrix")[1])
        # a typed attr is not equal to its name
        self.assertFalse(a == "T1|S.visibility")

    def test_every_spelling_of_a_plug_is_one_key(self):
        # equals() and hash agree for every way of reaching one plug, typed
        # spellings included (see test_typed_attributes_through_two_paths_are_one_key)

        _instanced_locator()
        pma = cmds.createNode("plusMinusAverage", name="pma")
        spellings = [
            (Node("|T1|S").lpx, Node("|T2|S").localPosition[0], Plug("T2|S.localPositionX"),
             Node("|T1|S").lp.localPositionX, Node("|T2|S").find_attr("lpx")),
            (Node(pma).input3D[1].input3Dx, Plug(f"{pma}.input3D[1].input3Dx"),
             Node(pma).input3D[1][0], Node(pma).find_attr("input3D")[1].child(0)),
            (Node("|T2|S").worldMatrix, Node("|T1|S").worldMatrix[1],
             Plug("T2|S.worldMatrix"), Node("|T1|S").find_attr("worldMatrix")[1]),
        ]
        for group in spellings:
            with self.subTest(plug=str(group[0])):
                for other in group[1:]:
                    self.assertTrue(group[0].equals(other), str(other))
                    self.assertEqual(hash(other), hash(group[0]), str(other))
        # distinct plugs of the groups hash apart
        firsts = [group[0] for group in spellings] + [Node(pma).input3D[2].input3Dx]
        self.assertEqual(len({hash(plug) for plug in firsts}), len(firsts))

    def test_list_membership_follows_the_maya_plug(self):
        _instanced_locator()
        plane  = cmds.nurbsPlane(name="np", degree=3, patchesU=1, patchesV=1, ch=False)[0]
        shape  = cmds.listRelatives(plane, shapes=True)[0]
        first, second = Node("|T1|S"), Node("|T2|S")
        before = sorted(cmds.ls())
        pl = List([second.v, second.lodVisibility])
        self.assertIn(first.v, pl)
        self.assertEqual(pl.index(first.v), 0)
        self.assertEqual(List([second.v, first.v, first.lodv]).count(first.v), 2)
        pl.remove(first.v)
        self.assertEqual([str(p) for p in pl], ["T2|S.lodVisibility"])
        self.assertNotIn(Node("T1").v, List([first.v]))
        # world space elements of different instances are different plugs
        self.assertNotIn(first.worldMatrix, List([second.worldMatrix]))
        self.assertIn(first.worldMatrix[1], List([second.worldMatrix]))
        # a component element and the Plug of its storage are one plug too
        self.assertIn(Plug(f"{shape}.controlPoints[6]"), List([Node(shape).cv[1, 2]]))
        self.assertNotIn(Plug(f"{shape}.controlPoints[7]"), List([Node(shape).cv[1, 2]]))
        # a str is read as the Maya plug it names (decision S3 Q4): through
        # either instance path, by its short name too (identity, not the name)
        self.assertIn("T2|S.visibility", List([second.v]))
        self.assertIn("T1|S.visibility", List([second.v]))
        self.assertIn("T2|S.v", List([second.v]))
        # none of it built a node
        self.assertEqual(sorted(cmds.ls()), before)

    def test_plug_key_survives_rename_alias_and_delete(self):
        cmds.undoInfo(state=True, infinity=True)
        node  = Node(cmds.createNode("transform", name="a"))
        held  = node.tx
        typed = Node("a").find_attr("tx")
        table, members, typed_set = {held: "x"}, {held}, {typed}
        key, typed_key = hash(held), hash(typed)
        cmds.rename("a", "b")
        self.assertEqual(str(held), "b.translateX")
        self.assertEqual((hash(held), hash(typed)), (key, typed_key))
        self.assertEqual(table[held], "x")
        self.assertEqual(table[Node("b").tx], "x")
        self.assertIn(Node("b").find_attr("translateX"), typed_set)
        members.add(held)
        members.add(Node("b").tx)
        self.assertEqual(len(members), 1)
        # an alias names the plug anew; it is still the same key
        cmds.addAttr("b", longName="knob", attributeType="double")
        knob     = Node("b").knob
        knob_key = hash(knob)
        cmds.aliasAttr("dial", "b.knob")
        self.assertEqual(str(knob), "b.dial")
        self.assertEqual(hash(knob), knob_key)
        self.assertEqual(hash(Node("b").dial), knob_key)
        # deleted to the undo queue: the held key is still found, and hashing
        # does not raise, while naming it does
        cmds.delete("b")
        self.assertEqual(hash(held), key)
        self.assertEqual(table[held], "x")
        members.discard(held)
        self.assertEqual(len(members), 0)
        with self.assertRaisesRegex(RuntimeError, "^b already deleted!$"):
            str(held)
        # a new node of the same name is another plug, another key
        cmds.createNode("transform", name="b")
        fresh = Plug("b.tx")
        self.assertNotEqual(hash(fresh), key)
        self.assertNotIn(fresh, table)
        cmds.undo()
        cmds.undo()
        self.assertEqual(table[Node("b").tx], "x")

    def test_plug_key_survives_a_new_scene(self):
        node   = Node(cmds.createNode("transform", name="a"))
        held   = node.tx
        typed  = Node("a").find_attr("ty")
        keys   = (hash(held), hash(typed))
        table  = {held: 1, typed: 2}
        unseen = node.tz  # never hashed before the free
        cmds.file(new=True, force=True)
        self.assertEqual((hash(held), hash(typed)), keys)
        self.assertEqual((table[held], table[typed]), (1, 2))
        self.assertIsInstance(hash(unseen), int)
        del table[held]
        self.assertEqual(list(table.values()), [2])

    def test_a_plain_str_is_a_name_not_a_plug_key(self):
        node  = Node(cmds.createNode("transform", name="a"))
        typed = Node("a").find_attr("tx")
        before = sorted(cmds.ls())
        for key in (node.tx, typed):
            with self.subTest(key=type(key).__name__):
                self.assertNotEqual(hash(key), hash("a.translateX"))
                self.assertIsNone({key: 1}.get("a.translateX"))
                self.assertNotIn("a.translateX", {key})
        self.assertTrue(node.tx.equals("a.translateX"))
        self.assertFalse(node.tx.equals("a.tx"))
        self.assertEqual(sorted(cmds.ls()), before)

    def test_memoized_networks_are_shared_through_two_paths(self):
        from rig import functions, random as rrandom
        from rig.matrix import decompose

        _instanced_locator()
        cmds.createNode("transform", name="G1")
        cmds.createNode("transform", name="G2")
        cmds.createNode("transform", name="X", parent="G1")
        cmds.parent("|G1|X", "G2", add=True)
        x1, x2 = Node("|G1|X"), Node("|G2|X")
        s1, s2 = Node("|T1|S"), Node("|T2|S")
        pairs = {
            "decompose(matrix)": (decompose(x1.matrix), decompose(x2.matrix)),
            "abs(lpx)": (functions.abs(s1.lpx), functions.abs(s2.lpx)),
            "lpx + 1": (s1.lpx + 1, s2.lpx + 1),
            "random.value(seed)": (
                rrandom.value(s1.lpx, seed=3), rrandom.value(s2.lpx, seed=3),
            ),
        }
        for label, (a, b) in pairs.items():
            with self.subTest(call=label):
                self.assertEqual(str(a), str(b))
        # a world space matrix is per instance: one network each
        self.assertNotEqual(
            str(decompose(x1.worldMatrix)), str(decompose(x2.worldMatrix))
        )
        # the shared network survives a rename of the instanced node
        cmds.rename("|G1|X", "Y")
        self.assertEqual(
            str(decompose(Node("|G2|Y").matrix)), str(pairs["decompose(matrix)"][0])
        )

    def test_unindexed_world_space_array_is_its_paths_element(self):
        # a static world matrix and a container publish pick the element of the
        # path the array was read through, as a connection and cmds do
        import numpy as np

        cmds.createNode("transform", name="G1")
        cmds.createNode("transform", name="G2")
        cmds.setAttr("G2.tx", 10)
        cmds.createNode("transform", name="X", parent="G1")
        cmds.parent("|G1|X", "G2", add=True)
        world = np.eye(4)
        world[3, 0] = 15.0
        for label, get, tx in (
            ("G2 wm", lambda x: x.worldMatrix, 5.0),
            ("G2 wm[1]", lambda x: x.worldMatrix[1], 5.0),
            ("G2 wm[0]", lambda x: x.worldMatrix[0], 15.0),
            ("G1 wm", lambda x: Node("|G1|X").worldMatrix, 15.0),
        ):
            with self.subTest(plug=label):
                cmds.setAttr("|G1|X.tx", 0)
                get(Node("|G2|X")) << world
                self.assertAlmostEqual(cmds.getAttr("|G1|X.tx"), tx)
        _instanced_locator()
        with container("box"):
            published = container.publish_input(Node("|T2|S").worldMatrix, "wmIn")
            first     = container.publish_input(Node("|T1|S").worldMatrix, "wmFirst")
        sources = [
            cmds.listConnections(str(p), source=True, destination=False, plugs=True)
            for p in (published, first)
        ]
        self.assertEqual(sources, [["T2|S.worldMatrix"], ["T1|S.worldMatrix"]])
        self.assertEqual(_source_index(str(published)), 1)
        # a node with one instance, as before
        single = Node(cmds.createNode("transform", name="single"))
        with container("box2"):
            one = container.publish_input(single.worldMatrix, "wmSingle")
        self.assertEqual(_source_index(str(one)), 0)


# every name the removed DSL wrapper and its dead helpers used (round 4a M4, M8)
_REMOVED_NAMES = (
    "_wrapper_is_canonical",
    "_copy_wrapper",
    "_canonical_kind",
    "_CANONICAL_KIND",
    "_is_only_path",
    "_REUSABLE_WRAPPER_PARTS",
    "_NAMED_DG",
    "_NAMED_DAG",
    "_NAMED_DAG_PATH",
    "_NODE_WRAPPER_CLASS",
    "_NODE_WRAPPER_HOOK",
    "_unwrapped",
    "_dg_node",
    "_WRAPPER_STATE",
)


class TestNoWrapperLeft(MayaTestCase):
    """M8: the dead canonical-wrapper helpers and the ``_dg_node`` shim are gone.
    ``x._dg_node`` raises AttributeError through the ``_`` guard (on a deleted
    or freed node too, reading only its API 1.0 handle), no package module
    names a removed helper, and the package code that read the shim takes the
    node object itself (``_stack_values``, ``_node_is_transform``)."""

    TEST_START_NEW_SCENE = True

    def test_the_shim_is_gone(self):
        cmds.undoInfo(state=True, infinity=True)
        cube  = cmds.polyCube(name="cube", ch=False)[0]
        nodes = {
            "dg": Node(cmds.createNode("multiplyDivide", name="md")),
            "transform": Node(cube),
            "mesh": Node("cubeShape"),
            "container": Node(cmds.container(name="box")),
            "layer": Node(cmds.createDisplayLayer(name="lay", empty=True)),
            "set": Node(cmds.sets(name="set1", empty=True)),
        }
        for label, node in nodes.items():
            with self.subTest(node=label):
                self.assertIsInstance(node, DGNode)
                self.assertFalse(hasattr(node, "_dg_node"))
                with self.assertRaisesRegex(AttributeError, "^_dg_node$"):
                    node._dg_node
                self.assertFalse(any(hasattr(cls, "_dg_node") for cls in type(node).__mro__))
        # deleted to the undo queue, and undone
        md = nodes["dg"]
        cmds.delete("md")
        self.assertFalse(hasattr(md, "_dg_node"))
        cmds.undo()
        self.assertFalse(hasattr(md, "_dg_node"))
        # freed by a new scene: only the API 1.0 handle is read
        cmds.file(new=True, force=True)
        for label, node in nodes.items():
            with self.subTest(freed=label):
                probe = mock.Mock()
                vars(node)["_fn_set"] = probe
                self.assertFalse(hasattr(node, "_dg_node"))
                self.assertEqual(probe.mock_calls, [])

    def test_no_module_names_a_removed_helper(self):
        import rig

        root    = os.path.dirname(rig.__file__)
        pattern = re.compile(r"\b(" + "|".join(_REMOVED_NAMES) + r")\b")
        hits    = []
        for base, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in ("_tests", "__pycache__"))
            for name in sorted(f for f in files if f.endswith(".py")):
                path = os.path.join(base, name)
                with open(path, encoding="utf-8") as handle:
                    for number, line in enumerate(handle, 1):
                        if pattern.search(line):
                            hits.append(f"{os.path.relpath(path, root)}:{number}: {line.strip()}")
        self.assertEqual(hits, [])
        # nor does a module hold one at runtime
        from rig.nodetypes import _base, dg_node
        from rig._internal import container as container_module
        from rig._internal import introspect, members, memoize
        from rig._internal import plug as plug_module

        for module in (_base, dg_node, container_module, introspect, members, memoize, plug_module):
            for name in _REMOVED_NAMES:
                with self.subTest(module=module.__name__, name=name):
                    self.assertFalse(hasattr(module, name))
        # NodeMeta no longer hooks class attribute writes (the hooks only
        # cleared the deleted canonical-kind cache)
        self.assertNotIn("__setattr__", vars(_base.NodeMeta))
        self.assertNotIn("__delattr__", vars(_base.NodeMeta))

    def test_stack_values_treats_typed_nodes_as_nodes(self):
        from rig import List
        from rig._internal import introspect

        a     = Node(cmds.createNode("transform", name="a"))
        cube  = cmds.polyCube(name="cube", ch=False)[0]
        nodes = [
            a,
            Node(cmds.createNode("multiplyDivide", name="md")),
            Node(cube),
            Node("cubeShape"),
            Node(cmds.sets(name="set1", empty=True)),
            Node(cmds.container(name="box")),
        ]
        # a list of nodes is never handed to numpy
        stacks = mock.Mock(wraps=np.array)
        with mock.patch.object(introspect.np, "array", stacks):
            self.assertIs(introspect._stack_values(nodes), nodes)
            got = List(nodes) >> None
        self.assertEqual(stacks.call_count, 0)
        self.assertIs(type(got), list)
        self.assertEqual(len(got), len(nodes))
        for value, node in zip(got, nodes):
            self.assertIs(value, node)
        # a node among values is kept as it is; plain values still stack
        mixed = List([a.tx, a]) >> None
        self.assertEqual(mixed[0], 0.0)
        self.assertIs(mixed[1], a)
        stacked = List([a.tx, a.ty]) >> None
        self.assertIsInstance(stacked, np.ndarray)
        self.assertEqual(stacked.tolist(), [0.0, 0.0])

    def test_node_is_transform_takes_the_node(self):
        from rig._internal import decompose

        cmds.undoInfo(state=True, infinity=True)
        xf    = Node(cmds.createNode("transform", name="xf"))
        md    = Node(cmds.createNode("multiplyDivide", name="md"))
        jnt   = Node(cmds.createNode("joint", name="jnt"))
        casts = mock.Mock(side_effect=decompose._cast)
        with mock.patch.object(decompose, "_cast", casts):
            self.assertTrue(decompose._node_is_transform(xf))
            self.assertTrue(decompose._node_is_transform(jnt))
            self.assertFalse(decompose._node_is_transform(md))
            self.assertEqual(casts.call_count, 0)
            # a name is still cast
            self.assertTrue(decompose._node_is_transform("xf"))
            self.assertFalse(decompose._node_is_transform("md"))
            self.assertEqual(casts.call_count, 2)
        # a static or live matrix still drives a transform, and still names a
        # node that is not one
        matrix       = np.eye(4)
        matrix[3, 0] = 5.0
        self.assertIs(xf << matrix, xf)
        self.assertAlmostEqual(cmds.getAttr("xf.tx"), 5.0)
        src = Node(cmds.createNode("transform", name="src"))
        cmds.setAttr("src.ty", 2.0)
        self.assertIs(xf << src.worldMatrix[0], xf)
        self.assertAlmostEqual(cmds.getAttr("xf.ty"), 2.0)
        with self.assertRaisesRegex(TypeError, "^Cannot inject ndarray into a bare Node"):
            md << matrix
        # a deleted or freed node raises its "already deleted!" error, as naming
        # it did before the node was passed itself
        cmds.delete("xf")
        for label, other in (("static", matrix), ("plug", src.worldMatrix[0])):
            with self.subTest(deleted=label):
                with self.assertRaisesRegex(RuntimeError, "^xf already deleted!$"):
                    xf << other
        with self.assertRaisesRegex(RuntimeError, "^xf already deleted!$"):
            decompose._node_is_transform(xf)
        cmds.undo()
        cmds.file(new=True, force=True)
        freed = (
            r"^Transform node \(freed by a new scene, a file open or a reference "
            r"unload\) already deleted!$"
        )
        with self.assertRaisesRegex(RuntimeError, freed):
            xf << matrix
        with self.assertRaisesRegex(RuntimeError, freed):
            decompose._node_is_transform(xf)

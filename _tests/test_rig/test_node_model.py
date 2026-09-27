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
"""

import ast
import copy
import os
import shutil
import sys
import tempfile
from unittest import mock

from maya import cmds
from maya.api import OpenMaya
from rig import Node
from rig.nodetypes import DGNode, PyNode, Transform
from rig._internal.math_nodes import _decompose_matrix
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
        # hot: `DGNode.ensure_valid` (and its name of a deleted node), the owner
        # checks of a plug, the plug hash and node serial, the fn set of a
        # plug's node, the memo identity of a DG node, the wrapper's cache hit
        ("rig.nodetypes.dg_node", "DGNode.ensure_valid"),
        ("rig.nodetypes._base", "_ensure_owner_alive"),
        ("rig.nodetypes._base", "_node_serial"),
        ("rig.nodetypes._base", "_plug_hash"),
        ("rig.nodetypes._base", "_plug_node_fn_set"),
        ("rig._internal.plug", "_owner_alive"),
        ("rig._internal.memoize", "_named_dg_identity"),
        ("rig._internal.node", "Node.__getattr__"),
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

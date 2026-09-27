"""Tests for the node model: one node class hierarchy, the owner rule, typed
nodes speaking the DSL.

Each class names the round-4a step it belongs to:

* M1: the typed internals (``rig/nodetypes``) read Maya attributes through
  ``find_attr`` (an AST lint and a runtime tripwire), so they keep getting
  Attributes once typed nodes speak the DSL (``node.<attr>`` gives a Plug).
"""

import ast
import os
import sys
from unittest import mock

from maya import cmds
from rig.nodetypes import DGNode, PyNode
from rig._tests._base import MayaTestCase


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

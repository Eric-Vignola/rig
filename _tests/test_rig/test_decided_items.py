"""The items the user decided before the push (2026-10-17), each class failing on
a350721 (the negative control):

* `TestTypoRefusal`: rig's value objects refuse an undeclared public assignment, as
  cgmath does ("'Tag' has no attribute 'nmae'; public attributes must be declared"):
  Tag, the membership tokens, rig's errors, keys, specs, undo_chunk, NodeOp. Real
  fields keep working; an exception keeps Python's own behaviour.
* `TestAssignIsPrivate`: ShadingEngine.assign raises a TypeError naming x << sg (one way
  only); rig's membership runs the private helper.
* `TestErrorFields`: NodeNotFoundError.uuid is by_uuid, node_class is _node_class.
* `TestTypedDataNames`: TYPED_DATA_NAMES is a read-only mapping.
* `TestImportLeaks`: rig.nodetypes.material_node.shade_convert, rig.matrix.X / Y and the bridges' imports
  are gone; every public module declares __all__, so `from m import *` gives no
  typing name, no module and no import of another module.
"""

import importlib
import inspect
import pickle
import types

from maya import cmds

import rig
from rig import Layer, Node, Tag
from rig._internal.node_ops import NodeOp, _ALL_NODEOP_CACHES
from rig._internal.plug import InjectionError
from rig.nodetypes import Joint, ShadingEngine
from rig.nodetypes.errors import AmbiguousNodeError, NodeLookupError, NodeNotFoundError, NodeTypeError
from rig.nodetypes import Blinn
from rig.spec import Float, destroy, lock
from rig._tests._base import MayaTestCase

_TYPO = r"has no attribute 'nmae'; public attributes must be declared"


class TestTypoRefusal(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_tag(self):
        tag = Tag("cap")
        with self.assertRaisesRegex(
            AttributeError, r"^'Tag' has no attribute 'nmae'; public attributes must be declared$"
        ):
            tag.nmae = 1
        self.assertFalse(hasattr(tag, "nmae"))
        tag._private = 1  # private state is the class's business
        self.assertEqual(tag.name, "cap")

    def test_tokens(self):
        layer = Layer.define("L")
        red   = Blinn.define("red")
        for token, shown in ((-layer, "layer token"), (Layer(), "layer token"),
                             (-red, "material token"), (Blinn(), "material token"),
                             (-Tag("cap"), "Tag"), (Tag(), "Tag")):
            with self.subTest(token=repr(token)):
                with self.assertRaisesRegex(AttributeError, f"^'{shown}' {_TYPO}$"):
                    token.nmae = 1

    def test_errors(self):
        errors = (
            NodeLookupError("a*"), NodeNotFoundError("nope"), AmbiguousNodeError("a", ["|g|a", "|h|a"]),
            NodeTypeError("grp", "joint", "transform"), InjectionError("locked"),
        )
        for error in errors:
            with self.subTest(error=type(error).__name__):
                with self.assertRaisesRegex(AttributeError, f"^'{type(error).__name__}' {_TYPO}$"):
                    error.nmae = 1
                # Python's own fields
                error.args = ("x",)
                self.assertEqual(error.args, ("x",))
                error.add_note("a note")
                self.assertEqual(error.__notes__, ["a note"])
                try:
                    raise error from KeyError("k")
                except type(error) as caught:
                    self.assertIsNotNone(caught.__traceback__)
                    self.assertIsInstance(caught.__cause__, KeyError)
                self.assertIs(error.with_traceback(None), error)
        # the real fields
        error = NodeTypeError("grp", "joint", "transform")
        error.hint = "use Transform"
        self.assertEqual(str(error), "'grp' is a transform, not a joint; use Transform")
        found = NodeNotFoundError("nope")
        found.name = "other"
        self.assertEqual(found.name, "other")
        # a copy (pickle) sets its fields back through the same check
        copy = pickle.loads(pickle.dumps(AmbiguousNodeError("a", ["|g|a", "|h|a"])))
        self.assertEqual(copy.candidates, ["|g|a", "|h|a"])

    def test_keys_specs_chunk_nodeop(self):
        keys = rig.AnimCurve([0.0, 1.0])
        with self.assertRaisesRegex(AttributeError, f"^'AnimCurve keys' {_TYPO}$"):
            keys.nmae = 1
        spec = Float("speed")
        with self.assertRaisesRegex(AttributeError, f"^'Float' {_TYPO}$"):
            spec.nmae = 1
        spec.notes = "a note"  # a real field
        self.assertEqual(spec.notes, "a note")
        spec = destroy("speed")
        spec.strict = True
        self.assertIs(spec.strict, True)
        with self.assertRaisesRegex(AttributeError, _TYPO):
            spec.nmae = 1
        with self.assertRaisesRegex(AttributeError, _TYPO):
            lock.nmae = 1
        chunk = rig.undo_chunk("build")
        with self.assertRaisesRegex(AttributeError, f"^'undo_chunk' {_TYPO}$"):
            chunk.nmae = 1
        chunk.name = "build arm"
        self.assertEqual(repr(chunk), "undo_chunk('build arm')")
        op = NodeOp("probe")
        self.addCleanup(_ALL_NODEOP_CACHES.remove, op)
        with self.assertRaisesRegex(AttributeError, f"^'NodeOp' {_TYPO}$"):
            op.nmae = 1
        op.scalar_fn = abs
        self.assertIs(op.scalar_fn, abs)


class TestAssignIsPrivate(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_assign_names_the_operator(self):
        cube = Node(cmds.polyCube(name="cube", constructionHistory=False)[0])
        sg   = ShadingEngine.for_material(Blinn.define("red"))
        before = cmds.sets(str(sg), query=True)
        with self.assertRaisesRegex(
            TypeError, r'^sg\.assign is not rig\'s form: write cube\.f\[0:2\] << ShadingEngine\("redSG"\)'
        ):
            sg.assign(["cube.f[0:1]"])
        self.assertEqual(cmds.sets(str(sg), query=True), before)
        cube.f[0:2] << sg  # a Python slice: faces 0 and 1
        self.assertEqual(cmds.sets(str(sg), query=True), ["cube.f[0:1]"])


class TestErrorFields(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_by_uuid_and_private_node_class(self):
        with self.assertRaises(NodeNotFoundError) as caught:
            Joint("spnie_01")
        error = caught.exception
        self.assertIs(error.by_uuid, False)
        self.assertIs(error._node_class, Joint)
        for old in ("uuid", "node_class"):
            with self.subTest(old=old):
                self.assertFalse(hasattr(error, old))
        gone = cmds.createNode("transform", name="gone")
        uid  = cmds.ls(gone, uuid=True)[0]
        cmds.delete(gone)
        cmds.flushUndo()
        with self.assertRaises(NodeNotFoundError) as caught:
            Node(uid)
        self.assertIs(caught.exception.by_uuid, True)
        self.assertEqual(str(NodeNotFoundError(uid, by_uuid=True)), f"no node has the uuid {uid!r}")


class TestTypedDataNames(MayaTestCase):

    def test_read_only(self):
        from rig.nodetypes.transform import TYPED_DATA_NAMES

        self.assertIsInstance(TYPED_DATA_NAMES, types.MappingProxyType)
        with self.assertRaises(TypeError):
            TYPED_DATA_NAMES["x"] = "y"
        self.assertNotIn("x", TYPED_DATA_NAMES)
        self.assertIn("matrix", TYPED_DATA_NAMES.values())


_PUBLIC = (
    "rig.euler", "rig.functions", "rig.interpolate", "rig.matrix", "rig.membership",
    # re-pinned (user decision 2026-10-22, the shade module retired)
    "rig.quaternion", "rig.random", "rig.trigonometry", "rig.tween", "rig.utils",
    "rig.vector", "rig.nodetypes", "rig.nodetypes.anim_curve", "rig.nodetypes.blendshape",
    "rig.nodetypes.choice", "rig.nodetypes.dag_node", "rig.nodetypes.deformer",
    "rig.nodetypes.dg_node", "rig.nodetypes.display_layer", "rig.nodetypes.errors",
    "rig.nodetypes.follicle", "rig.nodetypes.geometry", "rig.nodetypes.joint",
    "rig.nodetypes.material_node", "rig.nodetypes.mesh", "rig.nodetypes.nurbs",
    "rig.nodetypes.object_set", "rig.nodetypes.reference", "rig.nodetypes.shading_engine",
    "rig.nodetypes.skincluster", "rig.nodetypes.transform", "rig.spec", "rig.spec.compound",
    "rig.spec.enum_attr", "rig.spec.modifiers", "rig.spec.numeric", "rig.spec.typed",
)


class TestImportLeaks(MayaTestCase):

    def test_the_named_leaks_are_gone(self):
        import rig.bridges.commands as rc
        import rig.bridges.nodes as rn
        import rig.matrix
        import rig.nodetypes.material_node

        # re-pinned (user decision 2026-10-22, the shade module retired): the
        # converter stays private to astype's module
        self.assertFalse(hasattr(rig.nodetypes.material_node, "shade_convert"))
        for name in ("X", "Y"):
            self.assertNotIn(name, vars(rig.matrix))
        for module, names in ((rc, ("Any", "Callable", "Node", "Attribute")),
                              (rn, ("Any", "Callable", "keyword"))):
            for name in names:
                with self.subTest(module=module.__name__, name=name):
                    self.assertNotIn(name, vars(module))
        # aim's defaults are still the unit axes
        parameters = inspect.signature(rig.matrix.aim).parameters
        self.assertEqual([parameters[p].default for p in ("up_vector", "aim_axis", "up_axis")],
                         [(0, 1, 0), (1, 0, 0), (0, 1, 0)])

    def test_every_public_module_declares_its_names(self):
        for name in _PUBLIC:
            module = importlib.import_module(name)
            with self.subTest(module=name):
                self.assertTrue(hasattr(module, "__all__"))
                namespace = {}
                exec(f"from {name} import *", namespace)
                namespace.pop("__builtins__")
                for key, value in namespace.items():
                    self.assertNotIsInstance(value, types.ModuleType, key)
                    self.assertNotEqual(getattr(value, "__module__", None), "typing", key)
                for key in ("Any", "cmds", "np", "OpenMaya", "re", "os"):
                    self.assertNotIn(key, namespace)

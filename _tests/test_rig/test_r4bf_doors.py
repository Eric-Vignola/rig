"""Round 4b follow-up F4: a class reference's miss names ``define``; a missing
tag reads no ids with a node on the left too.

* ``TestDefineDoor``: ``Cls("x")`` for a name no node has ends its
  NodeNotFoundError with "; Cls.define('x') finds or makes it", after the
  hints (did you mean, another namespace, the scope prefix), for every class
  whose ``define`` makes the node from the name alone, and that ``define``
  does make the node the reference then finds. A class that refuses
  ``define`` (``_DEFINE_REFUSED``: Mesh, SkinCluster ...), an abstract one
  (DGNode, DAGNode, Geometry, the generic Material, which takes ``type=``),
  ``Node("x")``, a uuid, a name ``define`` refuses as written (a path,
  ``1bad``, ``''``) and a key in a namespace that does not exist or belongs
  to a file reference keep the message they had. The door is read when the
  error is printed, as the hints are (a caller that discards the error never
  reads it). Only the message changes: the same error class and facts,
  nothing written, and the hit is untouched. A pickled copy keeps the message
  and the door as text, without the class (which may not pickle).
* ``TestGhostTag``: ``node >> Tag("ghost")`` (a tag the node does not have)
  answers an empty id array, as components on the left did, of the shape an
  empty tag of that node reads (``(0,)``, ``(0, 2)`` on a surface, ``(0, 3)``
  on a lattice); ``node in Tag("ghost")`` stays False; nothing is written; the
  other refusals are unchanged.
"""

import os
import pickle
import shutil
import tempfile
from unittest import mock

import numpy as np
from maya import cmds

from rig import container, Layer, List, Node, Tag
from rig.nodetypes import (
    BlendShape,
    Blinn,
    Choice,
    DAGNode,
    DGNode,
    DisplayLayer,
    Follicle,
    Geometry,
    Joint,
    Lambert,
    Material,
    Mesh,
    NurbsCurve,
    NurbsSurface,
    ObjectSet,
    OpenPBRSurface,
    Phong,
    PhongE,
    Reference,
    ShadingEngine,
    SkinCluster,
    StandardSurface,
    SurfaceShader,
    Transform,
)
from rig.nodetypes.deformer import Deformer
from rig.nodetypes.errors import NodeLookupError, NodeNotFoundError
from rig._internal.container import Container
from rig._tests._base import MayaTestCase


def _scene():
    return set(cmds.ls())


def _label(cls):
    from rig.nodetypes._base import _type_label

    return _type_label(cls)


class _DoorCtl(Transform):
    CUSTOM_NODE_TYPE = "r4bfDoorCtl"


class _DoorWrapper(Transform):
    pass


# the classes whose define makes a node from the name alone: their miss names it
DOOR_CLASSES = (
    Transform, Joint, Choice, DisplayLayer, ObjectSet, ShadingEngine, Deformer,
    Lambert, Blinn, Phong, PhongE, SurfaceShader, StandardSurface, OpenPBRSurface,
    _DoorCtl, _DoorWrapper,
)

# the classes whose define refuses the name alone: their miss is as it was
NO_DOOR_CLASSES = (
    DGNode, DAGNode, Geometry, Material,                                    # abstract / type=
    Mesh, NurbsCurve, NurbsSurface, SkinCluster, BlendShape, Reference, Follicle, Container,
)


class _Case(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def miss(self, call):
        """The NodeNotFoundError ``call()`` raises, having written nothing."""
        before = _scene()
        with self.assertRaises(NodeNotFoundError) as caught:
            call()
        self.assertIs(type(caught.exception), NodeNotFoundError)
        self.assertEqual(_scene(), before)
        return caught.exception


class TestDefineDoor(_Case):
    def test_a_class_that_defines_names_its_define(self):
        for cls in DOOR_CLASSES:
            with self.subTest(cls=cls.__name__):
                self.new_scene()
                error = self.miss(lambda: cls("nosuch_x"))
                self.assertEqual(
                    str(error),
                    f"no {_label(cls)} named 'nosuch_x'; {cls.__name__}.define('nosuch_x') finds or makes it",
                )
                self.assertEqual(error.door, f"{cls.__name__}.define('nosuch_x')")
                # the door works: define makes the node the reference then finds
                made = cls.define("nosuch_x")
                self.assertEqual(cls("nosuch_x"), made)
                self.assertIsInstance(made, cls)

    def test_a_class_that_refuses_define_keeps_its_message(self):
        for cls in NO_DOOR_CLASSES:
            with self.subTest(cls=cls.__name__):
                error = self.miss(lambda: cls("nosuch_x"))
                self.assertEqual(str(error), f"no {_label(cls)} named 'nosuch_x'")
                self.assertIsNone(error.door)
                # the define the door would name refuses the name alone
                before = _scene()
                with self.assertRaises(TypeError):
                    cls.define("nosuch_x")
                self.assertEqual(_scene(), before)

    def test_the_door_follows_the_hints(self):
        cmds.createNode("joint", name="spine_01")
        cmds.namespace(add="char")
        cmds.createNode("joint", name="char:root")
        cmds.createNode("joint", name="char:spine_01")
        cases = (
            (lambda: Joint("spnie_01"),
             "no joint named 'spnie_01' (did you mean 'spine_01'?); "
             "Joint.define('spnie_01') finds or makes it"),
            (lambda: Joint("spnie_01.tx"),
             "no joint named 'spnie_01' (did you mean 'spine_01'?); "
             "Joint.define('spnie_01') finds or makes it"),
            (lambda: Joint("root"),
             "no joint named 'root' ('char:root' exists); Joint.define('root') finds or makes it"),
            (lambda: Joint("char:spnie_01"),
             "no joint named 'char:spnie_01' (did you mean 'char:spine_01'?); "
             "Joint.define('char:spnie_01') finds or makes it"),
            (lambda: Transform("spnie_01"),
             "no transform named 'spnie_01' (did you mean 'spine_01'?); "
             "Transform.define('spnie_01') finds or makes it"),
            (lambda: Blinn("rde"), "no blinn named 'rde'; Blinn.define('rde') finds or makes it"),
            (lambda: Layer("ghost"),
             "no displayLayer named 'ghost'; DisplayLayer.define('ghost') finds or makes it"),
        )
        for call, text in cases:
            with self.subTest(text=text):
                self.assertEqual(str(self.miss(call)), text)
        # while another namespace is current the lookup reads both; define keys
        # the current one, and the reference then finds what it made
        cmds.namespace(set="char")
        try:
            error = self.miss(lambda: Joint("elbow"))
            self.assertEqual(str(error), "no joint named 'elbow'; Joint.define('elbow') finds or makes it")
            self.assertEqual(Joint.define("elbow").name, "char:elbow")
            self.assertEqual(Joint("elbow").name, "char:elbow")
        finally:
            cmds.namespace(set=":")

    def test_no_door_where_define_makes_no_namespace_or_file_node(self):
        """define never creates a namespace nor makes a node in a file
        reference's namespace; a miss there names no define. A namespace of the
        scene's own keeps the door."""
        folder = tempfile.mkdtemp(prefix="rig_r4bf_doors_")
        self.addCleanup(shutil.rmtree, folder, True)
        path = os.path.join(folder, "char.ma").replace("\\", "/")
        cmds.createNode("joint", name="spine_01")
        cmds.file(rename=path)
        cmds.file(save=True, type="mayaAscii", force=True)
        self.new_scene()
        cmds.file(path, reference=True, namespace="char")
        cmds.namespace(add="rig")
        for name, door in (
            ("char:spnie_01", None),        # a file reference's namespace
            ("nochar:spnie_01", None),      # no such namespace
            ("rig:spnie_01", "Joint.define('rig:spnie_01')"),
            (":spnie_01", "Joint.define(':spnie_01')"),
        ):
            with self.subTest(name=name):
                error = self.miss(lambda: Joint(name))
                self.assertEqual(error.door, door)
                self.assertTrue(str(error).startswith(f"no joint named {name!r}"))
                self.assertEqual(str(error).endswith(f"; {door} finds or makes it"), door is not None)
                if door is None:
                    before = _scene()
                    with self.assertRaisesRegex(ValueError, "define never"):
                        Joint.define(name)
                    self.assertEqual(_scene(), before)
        # a bare name while the reference's namespace is current: define would
        # key it there
        cmds.namespace(set=":char")
        try:
            self.assertIsNone(self.miss(lambda: Joint("elbow")).door)
        finally:
            cmds.namespace(set=":")

    def test_the_scope_prefix_hint_comes_first(self):
        with container("outer"):
            with container("inner"):
                Transform.create(name="k")
                error = self.miss(lambda: Transform("k"))
        self.assertEqual(
            str(error),
            "no transform named 'k' ('inner_k' exists (the scope prefix)); "
            "Transform.define('k') finds or makes it",
        )

    def test_no_door_for_node_a_uuid_or_a_name_define_refuses(self):
        cmds.createNode("transform", name="grp")
        error = self.miss(lambda: Node("nosuch_x"))
        self.assertEqual(str(error), "no node named 'nosuch_x'")
        self.assertIsNone(error.door)
        uuid = "12345678-1234-1234-1234-123456789ABC"
        error = self.miss(lambda: Joint(uuid))
        self.assertEqual(str(error), f"no node has the uuid '{uuid}'")
        self.assertIsNone(error.door)
        for name in ("|grp|nosuch_x", "grp|nosuch_x", "1bad", "", "a-b", "|", ":"):
            with self.subTest(name=name):
                error = self.miss(lambda: Joint(name))
                self.assertEqual(str(error), f"no joint named {name!r}")
                self.assertIsNone(error.door)
                # define refuses that name as written
                before = _scene()
                with self.assertRaises((TypeError, ValueError)):
                    Joint.define(name)
                self.assertEqual(_scene(), before)

    def test_only_the_message_changes(self):
        cmds.createNode("joint", name="spine_01")
        error = self.miss(lambda: Joint("spnie_01"))
        # the facts and the family are the reference's
        self.assertEqual((error.name, error.label, error.uuid), ("spnie_01", "joint", False))
        self.assertEqual(error.args, ("spnie_01",))
        self.assertEqual(repr(error), "NodeNotFoundError('spnie_01')")
        for family in (NodeLookupError, LookupError, TypeError, ValueError):
            self.assertIsInstance(error, family)
        self.assertIs(error.node_class, Joint)
        # the message is built once, then kept
        text = str(error)
        with mock.patch.object(cmds, "ls", wraps=cmds.ls) as ls:
            self.assertEqual(str(error), text)
        ls.assert_not_called()
        copy = pickle.loads(pickle.dumps(error))
        self.assertEqual((type(copy), copy.door, str(copy)), (NodeNotFoundError, error.door, text))
        # a held error printed after the scene changed drops the hints, not the door
        held = self.miss(lambda: Joint("spnie_01"))
        self.new_scene()
        self.assertEqual(str(held), "no joint named 'spnie_01'; Joint.define('spnie_01') finds or makes it")
        # an error made directly names no door
        self.assertEqual(str(NodeNotFoundError("nosuch_x", "joint")), "no joint named 'nosuch_x'")
        # the hit, exists and the plug-left spelling are untouched
        cmds.createNode("joint", name="spine_01")
        self.assertEqual(repr(Joint("spine_01")), 'Joint("spine_01")')
        self.assertFalse(Joint.exists("spnie_01"))
        self.assertTrue(Joint.exists("spine_01"))

    def test_the_door_is_read_when_printed(self):
        """Like the hints: a caller that discards the error never reads the
        door (a namespaced name's door reads the namespaces and references)."""
        cmds.namespace(add="char")
        with mock.patch.object(Joint, "_define_door", wraps=Joint._define_door) as door:
            for name in ("nosuch_x", "char:nosuch_x"):
                with self.subTest(name=name):
                    error = self.miss(lambda: Joint(name))
                    self.assertFalse(Joint.exists(name))
                    door.assert_not_called()
                    self.assertEqual(str(error), f"no joint named {name!r}; Joint.define({name!r}) finds or makes it")
                    door.assert_called_once_with(name)
                    str(error)
                    door.assert_called_once_with(name)
                    door.reset_mock()
            with self.assertRaisesRegex(RuntimeError, "^Missing joints found"):
                SkinCluster._sanitize_influences(["nosuch_x", "char:nosuch_x"])
            door.assert_not_called()
        # a scene that cannot be read drops the door, as it drops a hint
        error = self.miss(lambda: Joint("char:nosuch_x"))
        with mock.patch.object(Joint, "_define_door", side_effect=RuntimeError("the scene is gone")):
            self.assertEqual(str(error), "no joint named 'char:nosuch_x'")
            self.assertIsNone(error.door)

    def test_a_membership_reference_names_its_define(self):
        cube = Node(cmds.polyCube(name="cube", ch=False)[0])
        for call, text in (
            (lambda: cube << Blinn("rde"), "no blinn named 'rde'; Blinn.define('rde') finds or makes it"),
            (lambda: cube in Blinn("rde"), "no blinn named 'rde'; Blinn.define('rde') finds or makes it"),
            (lambda: cube << -Layer("bg"),
             "no displayLayer named 'bg'; DisplayLayer.define('bg') finds or makes it"),
            (lambda: cube << Mesh("shapeIn"), "no mesh named 'shapeIn'"),
            (lambda: cube << Material("x"), "no surface shader named 'x'"),
        ):
            with self.subTest(text=text):
                self.assertEqual(str(self.miss(call)), text)
        # a missing influence is still named by the skinCluster's own error
        with self.assertRaisesRegex(RuntimeError, r"^Missing joints found: \['nosuch_x'\]$"):
            SkinCluster._sanitize_influences(["nosuch_x"])

    def test_a_copy_keeps_the_message_and_the_door_as_text(self):
        """A pickled error keeps its facts, message and door, and leaves out
        the node class, which may not pickle (a class made in a function)."""

        class _LocalXform(Transform):
            pass

        cmds.createNode("transform", name="spine_01")
        for cls in (_LocalXform, Joint):
            with self.subTest(cls=cls.__name__):
                error = self.miss(lambda: cls("spnie_01"))
                copy  = pickle.loads(pickle.dumps(error))
                self.assertIs(type(copy), NodeNotFoundError)
                self.assertEqual((copy.args, copy.name, copy.label, copy.uuid),
                                 (error.args, error.name, error.label, error.uuid))
                self.assertEqual((str(copy), copy.door), (str(error), error.door))
                self.assertIn(f"; {cls.__name__}.define('spnie_01') finds or makes it", str(copy))
                self.assertIsNone(copy.node_class)
                self.assertIs(error.node_class, cls)
        # Node(x)'s error has no class: pickled as before, its message read when printed
        error = self.miss(lambda: Node("spnie_01"))
        copy  = pickle.loads(pickle.dumps(error))
        self.assertIsNone(copy._message)
        self.assertEqual(str(copy), str(error))


class TestGhostTag(_Case):
    def setUp(self):
        super().setUp()
        self.sph = Node(cmds.polySphere(name="sph", ch=False)[0])
        self.srf = Node(cmds.sphere(name="srf", ch=False)[0])
        self.crv = Node(cmds.curve(name="crv", degree=1, point=[(0, 0, 0), (1, 0, 0), (2, 0, 0)]))
        box      = cmds.polyCube(name="box", ch=False)[0]
        self.lat = Node(cmds.lattice(box, divisions=(2, 3, 4))[1])

    def test_a_node_reads_no_ids_from_a_missing_tag(self):
        shapes = {"sph": (0,), "srf": (0, 2), "crv": (0,), "lat": (0, 3)}
        for attr, shape in shapes.items():
            geo = getattr(self, attr)
            with self.subTest(geo=attr):
                before = _scene()
                with mock.patch.object(cmds, "componentTag", wraps=cmds.componentTag) as tag_cmd:
                    ids = geo >> Tag("ghost")
                tag_cmd.assert_not_called()
                self.assertEqual(_scene(), before)
                self.assertIsInstance(ids, np.ndarray)
                self.assertEqual(ids.shape, shape)
                # the answer an empty tag of that node gives
                geo << Tag("empty")
                empty = geo >> Tag("empty")
                self.assertEqual((ids.shape, ids.dtype), (empty.shape, empty.dtype))
                self.assertFalse(geo in Tag("ghost"))
                self.assertTrue(geo not in Tag("ghost"))

    def test_the_node_agrees_with_its_components(self):
        for lhs, comps in (
            (self.sph, self.sph.vtx[:3]),
            (self.srf, self.srf.cv[1:3, 2:4]),
            (self.lat, self.lat.pt[0, 0, :]),
        ):
            with self.subTest(lhs=str(lhs)):
                node_ids = lhs >> Tag("ghost")
                self.assertEqual(node_ids.shape, (comps >> Tag("ghost")).shape)
                # the dtype of the node's own reads (a component read keeps its
                # selection's dtype, as before)
                comps << Tag("real")
                self.assertEqual(node_ids.dtype, (lhs >> Tag("real")).dtype)
        # a transform stands for its shape, a List of one node for the node
        self.assertEqual((Node("sph") >> Tag("ghost")).shape, (0,))
        self.assertEqual((List([self.sph]) >> Tag("ghost")).shape, (0,))

    def test_a_deleted_tag_reads_no_ids(self):
        self.sph.vtx[:8] << Tag("cap")
        np.testing.assert_array_equal(self.sph >> Tag("cap"), np.arange(8))
        self.sph << -Tag("cap")
        self.assertEqual((self.sph >> Tag("cap")).shape, (0,))
        self.assertFalse(self.sph in Tag("cap"))

    def test_the_other_refusals_are_unchanged(self):
        joint  = Node(cmds.createNode("joint", name="jnt"))
        before = _scene()
        for error, pattern, call in (
            (ValueError, r"^no component tag 'ghost' on sphShape$", lambda: self.sph << -Tag("ghost")),
            (ValueError, r"^no component tag 'ghost' on sphShape$",
             lambda: self.sph.vtx[:2] << -Tag("ghost")),
            (ValueError, r"^no component tag 'ghost' on sphShape$", lambda: Tag("ghost").clear(self.sph)),
            (TypeError, "is a plug; membership takes the node", lambda: self.sph.tx >> Tag("ghost")),
            (TypeError, "one node at a time", lambda: List([self.sph, self.srf]) >> Tag("ghost")),
            (TypeError, "not geometry", lambda: joint >> Tag("ghost")),
            (TypeError, "at= places a tag", lambda: self.sph >> Tag("ghost", at=self.sph)),
        ):
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(error, pattern):
                    call()
        self.assertEqual(_scene(), before)

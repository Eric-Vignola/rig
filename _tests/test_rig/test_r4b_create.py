"""Round 4b, step NC3: ``Cls.create`` always makes a new node, with one signature.

* ``TestCreate``: ``name=`` and (a DAG class) ``parent=`` are keywords only; a
  class that makes just its node refuses a positional argument
  (``Transform.create("test")``: TypeError, nothing made), a DG class refuses
  ``parent=``, a class whose command names the node refuses ``name=``, and the
  base classes of an abstract Maya type make no node. Every keyword that is no
  flag of the class's command is an attribute of the new node: checked on the
  node type before anything is made (a typo, or a wrong enum field name,
  raises with a zero ``ls`` delta), then set with ``<<`` (a value, an enum
  field name, a plug that connects, a spec such as ``lock``). Inside a scope
  the sets are in the create's one undo step. ``parent=`` places the node
  under its parent even beside a world node of the same name (J2).
  ``Node.create`` of a type no class is registered for splits its keywords
  the same way (createNode's flags, else attributes).
* ``TestCreateFlags``: each class's ``_CREATE_FLAGS`` go to its command; a
  name that is both a flag and an attribute is the flag (a skinCluster's
  ``normalizeWeights``, a blendShape's ``envelope``).
* ``TestSharedRefused``: ``shared=`` on ``Cls.create``, ``Node.create`` and
  ``rn.<type>`` is a TypeError before anything is made (create always makes a
  new node; the message names ``define``); ``rn``'s ``s=`` is the attribute
  ``s``, a transform's scale.
"""

import tempfile
from pathlib import Path
from unittest import mock

from maya import cmds

from rig import container, lock, Node
from rig.bridges import nodes as rn
from rig.nodetypes import (
    BlendShape,
    Choice,
    DAGNode,
    DGNode,
    DisplayLayer,
    Geometry,
    Joint,
    Mesh,
    NurbsCurve,
    ObjectSet,
    Reference,
    ShadingEngine,
    SkinCluster,
    Transform,
)
from rig.nodetypes import _base, dg_node
from rig.nodetypes.deformer import Deformer
from rig._internal.plug import InjectionError
from rig._tests._base import MayaTestCase


def _scene():
    return set(cmds.ls())


class _SceneCase(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def assertNothingMade(self, before):
        self.assertEqual(_scene() - before, set())


class TestCreate(_SceneCase):
    """``create``'s signature and its attribute keywords."""

    def test_name_and_parent_are_keywords(self):
        grp = Transform.create(name="grp")
        t   = Transform.create(name="t", parent=grp)
        j   = Joint.create(name="j", parent="grp")
        self.assertEqual(t.long_name, "|grp|t")
        self.assertEqual(j.long_name, "|grp|j")
        self.assertIsInstance(j, Joint)
        # always new: Maya picks the name
        self.assertEqual(str(Transform.create(name="grp")), "grp1")

    def test_a_positional_name_is_refused_before_anything_is_made(self):
        cmds.createNode("transform", name="test")
        before = _scene()
        for cls, text in (
            (Transform, r"^Transform\.create\(\) takes name= and parent= as keywords \(got 'test'\)$"),
            (Joint,     r"^Joint\.create\(\) takes name= and parent= as keywords \(got 'test'\)$"),
            (Choice,    r"^Choice\.create\(\) takes name= as a keyword \(got 'test'\)$"),
            (ObjectSet, r"^ObjectSet\.create\(\) takes name= as a keyword \(got 'test'\)$"),
        ):
            with self.subTest(cls=cls.__name__):
                with self.assertRaisesRegex(TypeError, text):
                    cls.create("test")
                self.assertNothingMade(before)
        with self.assertRaisesRegex(TypeError, r"\(got 'a', 1\)"):
            Transform.create("a", 1, name="x")
        # inside a scope too: nothing made, nothing registered
        with container("box") as box:
            inside = _scene()
            with self.assertRaises(TypeError):
                Transform.create("test")
            self.assertNothingMade(inside)
        self.assertEqual(cmds.container(str(box), query=True, nodeList=True), None)
        # Node.create keeps its own message
        with self.assertRaisesRegex(TypeError, "takes keyword arguments only"):
            Node.create("transform", "test")
        # Mesh takes its name as a keyword too
        with self.assertRaises(TypeError):
            Mesh.create(None, None, "m")

    def test_a_dg_class_refuses_parent(self):
        cmds.createNode("transform", name="grp")
        before = _scene()
        for cls in (Choice, ObjectSet, DisplayLayer, ShadingEngine):
            with self.subTest(cls=cls.__name__):
                with self.assertRaisesRegex(TypeError, r"takes no parent=: a \w+ is a DG node"):
                    cls.create(name="x", parent="grp")
                self.assertNothingMade(before)
        # a DAG class whose command places the node (a mesh at the world)
        with self.assertRaisesRegex(TypeError, r"^Mesh\.create\(\) takes no parent=: its command places"):
            Mesh.create(None, name="m", parent="grp")
        self.assertNothingMade(before)
        # parent=None is no parent
        self.assertEqual(str(Choice.create(name="c", parent=None)), "c")

    def test_a_class_whose_command_names_the_node_refuses_name(self):
        before = _scene()
        with self.assertRaisesRegex(TypeError, r"^Reference\.create\(\) takes no name="):
            Reference.create("nowhere.ma", "ns", name="x")
        self.assertNothingMade(before)

    def test_abstract_base_classes_make_no_node(self):
        before = _scene()
        for cls, node_type in ((DGNode, "entity"), (DAGNode, "dagNode"), (Geometry, "geometryShape")):
            with self.subTest(cls=cls.__name__):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^{cls.__name__}\.create\(\) makes no node: '{node_type}' is an abstract "
                    r"Maya type; Node\.create\('<type>', name=\.\.\.\) makes a node of a type$",
                ):
                    cls.create(name="x")
                self.assertNothingMade(before)

        # a user wrapper of DGNode (no type of its own) is refused the same way
        class Meta(DGNode):
            pass

        with self.assertRaisesRegex(TypeError, r"^Meta\.create\(\) makes no node: 'entity'"):
            Meta.create(name="m")
        self.assertNothingMade(before)

    def test_the_abstract_types_are_exactly_the_abstract_node_classes(self):
        abstract = {
            name.split()[0] for name in cmds.allNodeTypes(includeAbstract=True) if "(abstract)" in name
        }
        native = {cls.NATIVE_NODE_TYPE for cls in _base._NODE_CLASS_DICT.values()}
        native |= {DGNode.NATIVE_NODE_TYPE, DAGNode.NATIVE_NODE_TYPE, Geometry.NATIVE_NODE_TYPE,
                   Deformer.NATIVE_NODE_TYPE}
        self.assertEqual(native & abstract, set(dg_node._ABSTRACT_TYPES))
        # geometryFilter is not abstract: Maya makes one
        self.assertEqual(cmds.nodeType(str(Deformer.create(name="gf"))), "geometryFilter")

    def test_attributes_on_create(self):
        drv = cmds.createNode("transform", name="drv")
        cmds.setAttr(f"{drv}.tx", 7)
        t = Transform.create(name="t", tx=1, rotateOrder="xzy", ty=Node("drv").tx, tz=lock)
        self.assertEqual(cmds.getAttr("t.tx"), 1)
        self.assertEqual(cmds.getAttr("t.rotateOrder"), 3)
        self.assertEqual(cmds.listConnections("t.ty", plugs=True), ["drv.translateX"])
        self.assertEqual(cmds.getAttr("t.ty"), 7)
        self.assertTrue(cmds.getAttr("t.tz", lock=True))
        self.assertIsInstance(t, Transform)
        # compounds, short names, a joint's own attributes
        j = Joint.create(name="j", t=[1, 2, 3], radius=2, ro="zxy")
        self.assertEqual(cmds.getAttr("j.t"), [(1.0, 2.0, 3.0)])
        self.assertEqual(cmds.getAttr("j.radius"), 2)
        self.assertEqual(cmds.getAttr("j.ro"), 2)
        # a DG class: an attribute, where DGNode's _create used to drop it silently
        c = Choice.create(name="c", selector=2)
        self.assertEqual(cmds.getAttr(f"{c}.selector"), 2)

    def test_a_typo_raises_before_the_node_exists(self):
        before = _scene()
        with self.assertRaisesRegex(
            AttributeError,
            r"^Transform\.create\(\): a transform has no attribute 'tyop', and no create flag is "
            r"named so \(n, name, p, parent, skipSelect, ss\); nothing was made$",
        ):
            Transform.create(name="t", tx=1, tyop=2)
        self.assertNothingMade(before)
        with self.assertRaisesRegex(TypeError, "'nope' is not one of its enum fields"):
            Transform.create(name="t", rotateOrder="nope")
        self.assertNothingMade(before)
        with self.assertRaises(AttributeError):
            Choice.create(name="c", foo=1)
        self.assertNothingMade(before)
        # a numeric attribute given a str raises before anything too
        with self.assertRaises(InjectionError):
            Transform.create(name="t", tx="abc")
        self.assertNothingMade(before)
        # inside a scope: nothing made, nothing registered
        with container("box") as box:
            inside = _scene()
            with self.assertRaises(AttributeError):
                Transform.create(name="t", tyop=1)
            self.assertNothingMade(inside)
        self.assertEqual(cmds.container(str(box), query=True, nodeList=True), None)

    def test_in_a_scope_one_undo_removes_the_node_and_its_sets(self):
        cmds.undoInfo(state=True, infinity=True)
        cmds.flushUndo()
        drv = cmds.createNode("transform", name="drv")
        with container("outer") as outer:
            with container("inner"):
                t = Transform.create(name="u", tx=3, ty=Node(drv).tx)
        self.assertEqual(str(t), "inner_u")
        self.assertIn("inner_u", cmds.container(str(outer), query=True, nodeList=True))
        self.assertEqual(cmds.getAttr("inner_u.tx"), 3)
        cmds.undo()
        self.assertFalse(cmds.objExists("inner_u"))
        self.assertTrue(cmds.objExists(str(outer)))
        cmds.redo()
        self.assertTrue(t.is_valid)
        self.assertEqual(cmds.getAttr("inner_u.tx"), 3)
        self.assertEqual(cmds.listConnections("inner_u.ty", plugs=True), ["drv.translateX"])

    def test_in_a_namespace(self):
        cmds.namespace(add="ns")
        cmds.namespace(set="ns")
        try:
            t = Transform.create(name="x", tx=2)
        finally:
            cmds.namespace(set=":")
        self.assertEqual(str(t), "ns:x")
        self.assertEqual(cmds.getAttr("ns:x.tx"), 2)

    def test_parent_beside_a_world_node_of_the_same_name(self):
        """J2 (fixed in 4a): a world ``ctl`` does not rename the new child."""
        cmds.createNode("transform", name="ctl")
        grp = Transform.create(name="grp")
        ctl = Transform.create(name="ctl", parent=grp)
        self.assertEqual(ctl.long_name, "|grp|ctl")
        self.assertTrue(cmds.objExists("|ctl"))

    def test_node_create_of_an_unregistered_type(self):
        md = Node.create("multiplyDivide", name="md", operation="divide", input1X=3)
        self.assertEqual(str(md), "md")
        self.assertEqual(cmds.getAttr("md.operation"), 2)
        self.assertEqual(cmds.getAttr("md.input1X"), 3)
        drv = cmds.createNode("transform", name="drv")
        md2 = Node.create("multiplyDivide", input2=Node(drv).t)
        self.assertEqual(cmds.listConnections(f"{md2}.input2", plugs=True), ["drv.translate"])
        before = _scene()
        with self.assertRaisesRegex(
            AttributeError,
            r"^Node\.create\('multiplyDivide', \.\.\.\): a multiplyDivide has no attribute 'tyop'",
        ):
            Node.create("multiplyDivide", name="x", tyop=1)
        self.assertNothingMade(before)
        with self.assertRaisesRegex(TypeError, "'nope' is not one of its enum fields"):
            Node.create("multiplyDivide", operation="nope")
        self.assertNothingMade(before)
        # a registered type: the typed create's rule
        t = Node.create("transform", name="nt", tx=2)
        self.assertIsInstance(t, Transform)
        self.assertEqual(cmds.getAttr("nt.tx"), 2)
        before = _scene()
        with self.assertRaises(AttributeError):
            Node.create("transform", name="nt", tyop=2)
        self.assertNothingMade(before)

    def test_node_create_attributes_in_a_scope(self):
        with container("outer") as outer:
            with container("inner"):
                md = Node.create("multiplyDivide", name="md", operation="power")
        self.assertEqual(str(md), "inner_md")
        self.assertEqual(cmds.getAttr("inner_md.operation"), 3)
        self.assertIn("inner_md", cmds.container(str(outer), query=True, nodeList=True))


class TestCreateFlags(_SceneCase):
    """Each class's ``_CREATE_FLAGS`` reach its command; other keywords are
    attributes; a name that is both is the flag."""

    def _rig(self):
        cube = cmds.polyCube(name="cube", ch=False)[0]
        j1   = cmds.createNode("joint", name="j1")
        j2   = cmds.createNode("joint", name="j2", parent=j1)
        cmds.setAttr(f"{j2}.ty", 1)
        cmds.select(clear=True)
        return cube, j1, j2

    def test_the_flag_tables(self):
        self.assertEqual(DGNode._CREATE_FLAGS, {"name", "n", "skipSelect", "ss"})
        self.assertEqual(DAGNode._CREATE_FLAGS, {"name", "n", "skipSelect", "ss", "parent", "p"})
        self.assertIs(Transform._CREATE_FLAGS, DAGNode._CREATE_FLAGS)
        self.assertIs(Choice._CREATE_FLAGS, DGNode._CREATE_FLAGS)
        self.assertIs(ObjectSet._CREATE_FLAGS, DGNode._CREATE_FLAGS)
        self.assertEqual(
            DisplayLayer._CREATE_FLAGS,
            {"name", "n", "empty", "e", "noRecurse", "nr", "number", "num", "makeCurrent", "mc"},
        )
        self.assertEqual(ShadingEngine._CREATE_FLAGS, {"name", "n"})
        self.assertEqual(Mesh._CREATE_FLAGS, {"name", "uv_data"})
        self.assertEqual(NurbsCurve._CREATE_FLAGS, {"name", "degree", "kv"})
        self.assertEqual(Reference._CREATE_FLAGS, {"file_path", "namespace"})
        for cls in (SkinCluster, BlendShape):
            with self.subTest(cls=cls.__name__):
                self.assertTrue({"name", "n", "frontOfChain", "foc"} <= cls._CREATE_FLAGS)
                self.assertFalse({"skipSelect", "parent", "p"} & cls._CREATE_FLAGS)

    def test_dg_and_dag_flags(self):
        cmds.select(clear=True)
        Choice.create(name="c", ss=False)
        self.assertEqual(cmds.ls(selection=True), ["c"])
        Transform.create(name="quiet", skipSelect=True)
        self.assertEqual(cmds.ls(selection=True), ["c"])
        grp = Transform.create(name="grp")
        self.assertEqual(Transform.create(name="k", p=grp).long_name, "|grp|k")

    def test_display_layer(self):
        cube = cmds.polyCube(name="cube", ch=False)[0]
        cmds.select(cube)
        empty = DisplayLayer.create(name="L", empty=True, displayType=2, visibility=False)
        self.assertIsNone(cmds.editDisplayLayerMembers(str(empty), query=True))
        self.assertEqual(cmds.getAttr("L.displayType"), 2)
        self.assertFalse(cmds.getAttr("L.visibility"))
        # an enum field name, read before the layer exists
        ref = DisplayLayer.create(name="R", displayType="reference")
        self.assertEqual(cmds.getAttr(f"{ref}.displayType"), 2)
        before = _scene()
        with self.assertRaises(TypeError):
            DisplayLayer.create(name="bad", displayType="nope")
        self.assertNothingMade(before)
        # objects, positional; noRecurse / number reach the command ('e' is an
        # attribute of a layer too: the flag, empty)
        filled = DisplayLayer.create(cube, name="F", noRecurse=True)
        self.assertIn(cube, cmds.editDisplayLayerMembers(str(filled), query=True))
        DisplayLayer.create(name="N", number=7, e=True)
        self.assertEqual(cmds.getAttr("N.identification"), 7)
        self.assertIsNone(cmds.editDisplayLayerMembers("N", query=True))
        # makeCurrent reaches the command (mocked: mayapy's layer manager
        # refuses it with "No display layer has number ...")
        with mock.patch("rig.nodetypes.display_layer.cmds.createDisplayLayer", return_value="N") as call:
            DisplayLayer.create(name="M", makeCurrent=True)
        self.assertEqual(call.call_args.kwargs, {"name": "M", "makeCurrent": True, "empty": True})

    def test_shading_engine(self):
        sg = ShadingEngine.create(name="sg", caching=True)
        self.assertEqual(str(sg), "sg")
        self.assertTrue(cmds.getAttr("sg.caching"))
        before = _scene()
        with self.assertRaisesRegex(AttributeError, "a shadingEngine has no attribute 'skipSelect'"):
            ShadingEngine.create(name="x", skipSelect=True)
        self.assertNothingMade(before)

    def test_skincluster_flag_that_is_an_attribute_too_is_the_flag(self):
        cube, j1, j2 = self._rig()
        real = cmds.skinCluster
        with mock.patch("rig.nodetypes.skincluster.cmds.skinCluster", side_effect=real) as call:
            skin = SkinCluster.create(
                cube, [j1, j2], name="sk", normalizeWeights=2, maximumInfluences=1, envelope=0.5
            )
        creates = [c for c in call.call_args_list if not c.kwargs.get("edit") and not c.kwargs.get("query")]
        self.assertEqual(len(creates), 1)
        kwargs = creates[0].kwargs
        self.assertEqual(kwargs["normalizeWeights"], 2)
        self.assertEqual(kwargs["maximumInfluences"], 1)
        self.assertNotIn("envelope", kwargs)  # an attribute, set after
        self.assertEqual(cmds.getAttr("sk.normalizeWeights"), 2)
        self.assertEqual(cmds.getAttr("sk.envelope"), 0.5)
        self.assertEqual(str(skin), "sk")
        # the inputs by keyword
        cmds.delete(str(skin))
        again = SkinCluster.create(geom=cube, influences=[j1, j2], name="sk2")
        self.assertEqual(str(again), "sk2")

    def test_blendshape_flags(self):
        base = cmds.polyCube(name="base", ch=False)[0]
        tgt  = cmds.polyCube(name="tgt", ch=False)[0]
        real = cmds.blendShape
        with mock.patch("rig.nodetypes.blendshape.cmds.blendShape", side_effect=real) as call:
            bs = BlendShape.create(tgt, base, name="bs", frontOfChain=False, envelope=0.25, caching=True)
        kwargs = call.call_args_list[0].kwargs
        self.assertIs(kwargs["frontOfChain"], False)
        self.assertEqual(kwargs["envelope"], 0.25)  # a flag and an attribute: the flag
        self.assertNotIn("caching", kwargs)
        self.assertEqual(str(bs), "bs")
        self.assertEqual(cmds.getAttr("bs.envelope"), 0.25)
        self.assertTrue(cmds.getAttr("bs.caching"))

    def test_mesh(self):
        cube = cmds.polyCube(name="src", ch=False)[0]
        data = Mesh(cmds.listRelatives(cube, shapes=True)[0]).serialize(include_uvs=False)
        cmds.delete(cube)
        mesh = Mesh.create(data, name="m", displayColors=True)
        self.assertEqual(str(mesh), "mShape")
        self.assertTrue(cmds.getAttr("mShape.displayColors"))
        before = _scene()
        with self.assertRaises(AttributeError):
            Mesh.create(data, name="m2", tyop=1)
        self.assertNothingMade(before)

    def test_reference_namespace_by_keyword(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cmds.createNode("transform", name="thing")
            path = Path(f"{temp_dir}/nc3_ref.ma").as_posix()
            cmds.file(rename=path)
            cmds.file(force=True, type="mayaAscii", save=True)
            self.new_scene()
            ref = Reference.create(path, namespace="hero")
            self.assertEqual(ref.namespace, "hero")
            self.assertTrue(cmds.objExists("hero:thing"))
            ref.delete()
            self.new_scene()


class TestSharedRefused(_SceneCase):
    """``shared=`` on any creator is a TypeError naming ``define``, with nothing
    made (CC-6); ``rn``'s ``s=`` is the scale attribute."""

    def test_typed_create(self):
        cmds.createNode("transform", name="s")
        before = _scene()
        for cls, name, text in (
            (Transform, "s", r"^Transform\.create\(shared=\.\.\.\): create always makes a new node; "
                             r"Transform\.define\('s'\) finds or makes it$"),
            (Joint, None, r"^Joint\.create\(shared=\.\.\.\): .* Joint\.define\('x'\) finds or makes it$"),
            (Choice, "c", r"^Choice\.create\(shared=\.\.\.\): .* Choice\.define\('c'\)"),
            (DisplayLayer, "L", r"^DisplayLayer\.create\(shared=\.\.\.\): .* DisplayLayer\.define\('L'\)"),
        ):
            for value in (True, False):
                with self.subTest(cls=cls.__name__, shared=value):
                    with self.assertRaisesRegex(TypeError, text):
                        cls.create(name=name, shared=value)
                    self.assertNothingMade(before)

        class Ctl(Transform):
            pass

        with self.assertRaisesRegex(TypeError, r"Ctl\.define\('k'\)"):
            Ctl.create(n="k", shared=True)
        self.assertNothingMade(before)

    def test_node_create(self):
        cmds.createNode("transform", name="s")
        before = _scene()
        with self.assertRaisesRegex(
            TypeError,
            r"^Node\.create\('transform', shared=\.\.\.\): create always makes a new node; "
            r"Transform\.define\('s'\) finds or makes it$",
        ):
            Node.create("transform", name="s", shared=True)
        self.assertNothingMade(before)
        with self.assertRaisesRegex(
            TypeError,
            r"^Node\.create\('multiplyDivide', shared=\.\.\.\): create always makes a new node; "
            r"Node\.define\('multiplyDivide', 'x'\) finds or makes it$",
        ):
            Node.create("multiplyDivide", shared=True)
        self.assertNothingMade(before)

    def test_rn_factories(self):
        cmds.createNode("transform", name="r")
        before = _scene()
        with self.assertRaisesRegex(
            TypeError,
            r"^rn\.transform\(shared=\.\.\.\): create always makes a new node; "
            r"Transform\.define\('r'\) finds or makes it$",
        ):
            rn.transform(name="r", shared=True)
        self.assertNothingMade(before)
        with self.assertRaisesRegex(TypeError, r"Node\.define\('multiplyDivide', 'm'\)"):
            rn.multiplyDivide(n="m", shared=False)
        self.assertNothingMade(before)
        # inside a scope: nothing made, nothing registered
        with container("box") as box:
            inside = _scene()
            with self.assertRaises(TypeError):
                rn.transform(name="r", shared=True)
            with self.assertRaises(TypeError):
                Transform.create(name="r", shared=True)
            with self.assertRaises(TypeError):
                Node.create("transform", name="r", shared=True)
            self.assertNothingMade(inside)
        self.assertEqual(cmds.container(str(box), query=True, nodeList=True), None)

    def test_rn_s_is_the_scale(self):
        a = rn.transform(name="a", s=[2, 2, 2])
        b = rn.transform(name="a", s=[1, 3, 1])
        self.assertEqual(cmds.getAttr(f"{a}.s"), [(2.0, 2.0, 2.0)])
        self.assertEqual(cmds.getAttr(f"{b}.s"), [(1.0, 3.0, 1.0)])
        self.assertEqual((str(a), str(b)), ("a", "a1"))  # two nodes: s is no longer shared
        # the typed create too: s is no create flag
        t = Transform.create(name="t", s=[4, 4, 4])
        self.assertEqual(cmds.getAttr(f"{t}.s"), [(4.0, 4.0, 4.0)])
        # the other short flags keep their meaning
        grp = rn.transform(n="grp")
        kid = rn.transform(n="kid", p=grp, ss=True)
        self.assertEqual(kid.long_name, "|grp|kid")
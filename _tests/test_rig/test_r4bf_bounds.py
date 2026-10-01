"""Round 4b follow-up F5: a bounded did-you-mean, and a cheaper first declaration.

* ``TestBoundedHint``: the did-you-mean of a printed NodeNotFoundError reads a
  bounded listing, never the whole scene: a class's reference compares the
  names of its node type (``cmds.ls(type=..., head=500)``, subtypes included:
  ``Transform`` takes joints), so ``Joint('hip_gpr')`` no longer suggests the
  transform ``hip_grp`` that ``Joint`` would refuse; ``Node(x)``, DGNode, the
  generic Material, Container, a geometry class (which takes a transform too)
  and an error made directly compare every node (``cmds.ls(head=1000)``). The
  other hints, the door and the message's facts are as they were.
* ``TestFirstDeclaration``: a new attribute's name is checked against the
  node's aliases and a container's published names with one API call on the
  fn set ``has_attr`` just read; the refusals keep their messages, and a free
  name never builds one. A spec's keywords are checked without building the
  set of unknown ones when there is none; the messages are as they were.
"""

from unittest import mock

from maya import cmds

from rig import List, Node
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
    Phong,
    Reference,
    ShadingEngine,
    SkinCluster,
    Transform,
)
from rig.nodetypes import errors as _errors
from rig.nodetypes.deformer import Deformer
from rig.nodetypes.errors import NodeNotFoundError
from rig._internal.container import Container
from rig.spec import Float, Vector
import rig.spec._base as _spec_base
from rig._tests._base import MayaTestCase


class _BoundsCtl(Transform):
    CUSTOM_NODE_TYPE = "r4bfBoundsCtl"


class _BoundsWrapper(Transform):
    pass


def _scene():
    return set(cmds.ls())


def _build():
    """A transform ``hip_grp``, a joint ``knee_jnt``, a blinn network
    ``skin_mat`` (with ``skin_matSG``), a display layer ``bg_lyr``, a cube
    ``cube``."""
    cmds.createNode("transform", name="hip_grp")
    cmds.createNode("joint", name="knee_jnt")
    cmds.select(clear=True)
    Blinn.define("skin_mat")
    DisplayLayer.define("bg_lyr")
    cmds.polyCube(name="cube", ch=False)
    cmds.select(clear=True)


class _Case(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def miss(self, call):
        """The str of the NodeNotFoundError ``call()`` raises, having written
        nothing."""
        before = _scene()
        with self.assertRaises(NodeNotFoundError) as caught:
            call()
        self.assertEqual(_scene(), before)
        return str(caught.exception)


class TestBoundedHint(_Case):
    def test_a_class_compares_the_names_of_its_type(self):
        _build()
        cases = (
            # another type's name is no longer suggested: the class would refuse it
            (lambda: Joint("hip_gpr"), "no joint named 'hip_gpr'; Joint.define('hip_gpr') finds or makes it"),
            (lambda: Blinn("hip_gpr"), "no blinn named 'hip_gpr'; Blinn.define('hip_gpr') finds or makes it"),
            (lambda: DisplayLayer("skin_mta"),
             "no displayLayer named 'skin_mta'; DisplayLayer.define('skin_mta') finds or makes it"),
            (lambda: DAGNode("skin_mta"), "no DAG node named 'skin_mta'"),
            # its own type's names are (the blinn, not its engine 'skin_matSG')
            (lambda: Joint("knee_jtn"),
             "no joint named 'knee_jtn' (did you mean 'knee_jnt'?); Joint.define('knee_jtn') finds or makes it"),
            (lambda: Blinn("skin_mta"),
             "no blinn named 'skin_mta' (did you mean 'skin_mat'?); Blinn.define('skin_mta') finds or makes it"),
            (lambda: DisplayLayer("bg_lry"),
             "no displayLayer named 'bg_lry' (did you mean 'bg_lyr'?); "
             "DisplayLayer.define('bg_lry') finds or makes it"),
            # a subtype is of the type: a joint is a transform and a DAG node
            (lambda: Transform("hip_gpr"),
             "no transform named 'hip_gpr' (did you mean 'hip_grp'?); Transform.define('hip_gpr') finds or makes it"),
            (lambda: Transform("knee_jtn"),
             "no transform named 'knee_jtn' (did you mean 'knee_jnt'?); Transform.define('knee_jtn') finds or makes it"),
            (lambda: DAGNode("knee_jtn"), "no DAG node named 'knee_jtn' (did you mean 'knee_jnt'?)"),
        )
        for call, text in cases:
            with self.subTest(text=text):
                self.assertEqual(self.miss(call), text)

    def test_an_untyped_miss_compares_every_node(self):
        _build()
        cases = (
            (lambda: Node("hip_gpr"), "no node named 'hip_gpr' (did you mean 'hip_grp'?)"),
            (lambda: Node("skin_mta"), "no node named 'skin_mta' (did you mean 'skin_mat' or 'skin_matSG'?)"),
            (lambda: DGNode("knee_jtn"), "no node named 'knee_jtn' (did you mean 'knee_jnt'?)"),
            (lambda: Material("hip_gpr"), "no surface shader named 'hip_gpr' (did you mean 'hip_grp'?)"),
            # a geometry class takes a transform too: Mesh('cube') is the cube's shape
            (lambda: Mesh("cbue"), "no mesh named 'cbue' (did you mean 'cube'?)"),
            (lambda: NurbsCurve("hip_gpr"), "no nurbsCurve named 'hip_gpr' (did you mean 'hip_grp'?)"),
        )
        for call, text in cases:
            with self.subTest(text=text):
                self.assertEqual(self.miss(call), text)
        self.assertEqual(repr(Mesh("cube")), 'Mesh("cubeShape")')
        # an error made directly has no class: every node
        self.assertEqual(str(NodeNotFoundError("hip_gpr", "joint")),
                         "no joint named 'hip_gpr' (did you mean 'hip_grp'?)")

    def test_each_class_names_its_hint_type(self):
        typed = {
            Transform: "transform", Joint: "joint", DAGNode: "dagNode", Choice: "choice",
            DisplayLayer: "displayLayer", ObjectSet: "objectSet", ShadingEngine: "shadingEngine",
            Deformer: "geometryFilter", SkinCluster: "skinCluster", BlendShape: "blendShape",
            Lambert: "lambert", Blinn: "blinn", Phong: "phong", Follicle: "follicle",
            Reference: "reference", _BoundsCtl: "transform", _BoundsWrapper: "transform",
        }
        for cls, node_type in typed.items():
            with self.subTest(cls=cls.__name__):
                self.assertEqual(cls._hint_type(), node_type)
        for cls in (DGNode, Material, Container, Geometry, Mesh, NurbsCurve, NurbsSurface):
            with self.subTest(cls=cls.__name__):
                self.assertIsNone(cls._hint_type())

    def test_the_listing_is_bounded(self):
        """One ``cmds.ls`` per hint, each with ``head``: a print never lists the
        whole scene."""
        _build()
        for call, listing in (
            (lambda: Joint("knee_jtn"), {"type": "joint", "head": _errors._HINT_HEAD_TYPED}),
            (lambda: Transform("knee_jtn"), {"type": "transform", "head": _errors._HINT_HEAD_TYPED}),
            (lambda: Node("knee_jtn"), {"head": _errors._HINT_HEAD}),
            (lambda: Mesh("knee_jtn"), {"head": _errors._HINT_HEAD}),
        ):
            with self.subTest(listing=listing):
                with self.assertRaises(NodeNotFoundError) as caught:
                    call()
                with mock.patch.object(cmds, "ls", wraps=cmds.ls) as ls:
                    str(caught.exception)
                calls = [(c.args, c.kwargs) for c in ls.call_args_list]
                self.assertIn(((), listing), calls)
                # the other call is the namespace hint's, one name across namespaces
                self.assertEqual(calls, [(("knee_jtn",), {"recursive": True}), ((), listing)])
        self.assertEqual((_errors._HINT_HEAD_TYPED, _errors._HINT_HEAD), (500, 1000))

    def test_a_big_scene_is_read_in_part(self):
        """In a scene of more names than the listing takes, the hint compares
        the names that listing gives (here cut to 2 to keep the test small)."""
        for i in range(3):
            cmds.createNode("joint", name=f"limb_{i}")
            cmds.select(clear=True)
        listed = cmds.ls(type="joint", head=2)
        self.assertEqual(len(listed), 2)
        unlisted = (set(cmds.ls(type="joint")) - set(listed)).pop()
        with mock.patch.object(_errors, "_HINT_HEAD_TYPED", 2):
            text = self.miss(lambda: Joint(unlisted + "x"))
            self.assertNotIn(f"'{unlisted}'", text)
            self.assertIn(f"'{listed[0]}'", self.miss(lambda: Joint(listed[0] + "x")))
        self.assertIn(f"did you mean '{unlisted}'", self.miss(lambda: Joint(unlisted + "x")))

    def test_the_cost_does_not_grow_with_the_scene(self):
        """A pin of the bound: at 3,000 transforms no listing the hints read
        returns more than the head it asks for."""
        cmds.createNode("joint", name="spine_01")
        for i in range(3000):
            cmds.createNode("transform", name=f"t{i:05d}")
        for call in (lambda: Joint("spnie_01"), lambda: Transform("spnie_01"), lambda: Node("spnie_01")):
            with self.assertRaises(NodeNotFoundError) as caught:
                call()
            returned = []

            def counting(*args, **kwargs):
                result = cmds_ls(*args, **kwargs) or []
                returned.append(len(result))
                return result
            cmds_ls = cmds.ls
            with mock.patch.object(cmds, "ls", side_effect=counting):
                str(caught.exception)
            self.assertTrue(returned)
            self.assertLessEqual(max(returned), _errors._HINT_HEAD)
        self.assertGreater(len(cmds.ls()), 3000)

    def test_the_other_hints_and_the_door_are_unchanged(self):
        cmds.createNode("joint", name="spine_01")
        cmds.namespace(add="char")
        cmds.createNode("joint", name="char:root")
        cmds.createNode("transform", name="char:grp")
        cases = (
            (lambda: Joint("root"), "no joint named 'root' ('char:root' exists); Joint.define('root') finds or makes it"),
            # the namespace hint names any node of the leaf, of any type
            (lambda: Joint("grp"), "no joint named 'grp' ('char:grp' exists); Joint.define('grp') finds or makes it"),
            (lambda: Node("chr:root"), "no node named 'chr:root' ('char:root' exists)"),
            (lambda: Joint("spnie_01"),
             "no joint named 'spnie_01' (did you mean 'spine_01'?); Joint.define('spnie_01') finds or makes it"),
            (lambda: Mesh("nosuch"), "no mesh named 'nosuch'"),
        )
        for call, text in cases:
            with self.subTest(text=text):
                self.assertEqual(self.miss(call), text)


class TestFirstDeclaration(_Case):
    def setUp(self):
        super().setUp()
        self.holder = Node.create("network", name="holder")
        self.holder << Float("dyn")
        cmds.aliasAttr("nick", "holder.dyn")

    def test_a_free_name_builds_no_refusal(self):
        with mock.patch.object(_spec_base, "_refuse_other_name", wraps=_spec_base._refuse_other_name) as refuse:
            plug = self.holder << Float("free", dv=2)
            self.holder << Vector("vec")
            List([self.holder]) << Float("listed")
        refuse.assert_not_called()
        self.assertEqual((str(plug), plug.node, cmds.getAttr("holder.free")), ("holder.free", self.holder, 2.0))
        self.assertLessEqual({"free", "vec", "vecX", "vecY", "vecZ", "listed"},
                             set(cmds.listAttr("holder", userDefined=True)))

    def test_an_alias_or_a_published_name_still_refuses(self):
        grp = Node.create("transform", name="grp")
        cmds.aliasAttr("lift", "grp.translateY")
        text = r"^'holder\.nick' is an alias of holder\.dyn; re-declare dyn, or pick another name$"
        for call in (
            lambda: self.holder << Float("nick"),
            lambda: self.holder << Vector("nick"),
            lambda: self.holder.dyn << Float("nick"),
            lambda: List([grp, self.holder]) << Float("nick"),
        ):
            with self.subTest(call=call):
                before = cmds.listAttr("holder", userDefined=True)
                with self.assertRaisesRegex(TypeError, text):
                    call()
                self.assertEqual(cmds.listAttr("holder", userDefined=True), before)
        with self.assertRaisesRegex(TypeError, r"^'grp\.lift' is an alias of grp\.translateY; "):
            grp << Float("lift")
        with self.assertRaisesRegex(TypeError, r"^'grp\.tx' is a static attribute of the transform"):
            grp << Float("tx")
        box = cmds.container(name="box")
        cmds.container(box, edit=True, addNode=["holder"])
        cmds.container(box, edit=True, publishName="pub")
        with self.assertRaisesRegex(TypeError, r"^'box\.pub' is a name the container publishes; "):
            Node("box") << Float("pub")

    def test_a_spec_keyword_check_keeps_its_messages(self):
        for kwargs, text in (
            ({"zeta": 1, "alpha": 2}, r"^Float\('zz'\): alpha=, zeta= are not addAttr flags; nothing was changed$"),
            ({"bogus": 1}, r"^Float\('zz'\): bogus= is not an addAttr flag; nothing was changed$"),
            ({"niceNmae": "Z"}, r"\(did you mean niceName=\?\); nothing was changed$"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(TypeError, text):
                    Float("zz", **kwargs)
        spec = Float("zz", k=False, at="double", min=0, size=2, overwrite=False)
        self.assertEqual(spec.kargs, {"attributeType": "double", "min": 0, "keyable": False, "longName": "zz"})

"""Removal methods, ``Transform.create(data)`` and three small fixes (user decisions
2026-10-16, "Membership direction", "List casting details + Transform.create(data)" and
"Objects outside nodes and plugs": fix regardless).

* `TestRemovalMethods`: ``red.remove(x)``, ``sg.remove(x)``, ``layer.remove(x)``,
  ``Tag('cap').remove(x)`` do exactly what ``x << -c`` does: the same writes, the same
  refusals (type and text, before any write), one undo step, and they return ``x``, as
  ``<<`` returns its left-hand side. ``x << -c`` stays. A token has no methods.
* `TestCreateFromData`: ``Transform.create(transform_data)`` / ``Joint.create(joint_data)``
  build the node the data describes (name, values, user attributes, the joint fields);
  ``name=`` / ``parent=`` / attribute keywords as ``create`` takes them; refusals before
  anything is made; one undo step; the selection kept; inside ``with container()`` as a
  typed create; ``create_hierarchy`` shares the per-node code.
* `TestRandom3DWithoutSeed`: ``value3D()`` / ``uniform3D()`` / ``randint3D()`` with no
  ``seed=`` build three fresh streams (they raised IndexError mid-build); a seed that is
  not three seeds is a TypeError before anything is made; seeded networks are unchanged.
* `TestMirrorMapTakesBothAxes`: ``mirror_map`` takes rig's ``Axis`` and cgmath's.

Everything new is reached by name inside the tests, so on a tree without the step each
test fails on its own (the negative control).
"""

from unittest import mock

import numpy as np
from maya import cmds

from rig import Layer, List, Node, Tag, container
from rig import random as R
from rig.nodetypes import Axis, DisplayLayer, Joint, NodeNotFoundError, ShadingEngine, Transform
from rig.shade import Blinn
from rig.spec import Float
from rig._tests._base import MayaTestCase


def _cube(name="cube", **kwargs):
    kwargs.setdefault("constructionHistory", False)
    return Node(cmds.polyCube(name=name, **kwargs)[0])


def _refusal(fn):
    """``(type, text)`` of the exception ``fn()`` raises (None when it raises none)."""
    try:
        fn()
    except Exception as error:  # noqa: BLE001 -- the refusal itself is the subject
        return type(error), str(error)
    return None


# --------------------------------------------------------------------- #
#  Removal methods
# --------------------------------------------------------------------- #


class TestRemovalMethods(MayaTestCase):
    """The method of every membership collection, which ``x << -c`` spells short."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.a, self.b = _cube("a"), _cube("b")
        self.red = Blinn.define("red")
        self.lay = Layer.define("lay")
        List([self.a, self.b]) << self.red << self.lay
        self.sph = Node(cmds.polySphere(name="sph", constructionHistory=False)[0])
        self.sph.vtx[:8] << Tag("cap")

    def _state(self):
        return (
            sorted(cmds.ls()),
            sorted(cmds.sets("redSG", query=True) or []),
            sorted(str(n) for n in self.lay.get_members()),
            (self.sph >> Tag("cap")).tolist(),
        )

    def test_each_kind_writes_what_the_operator_writes(self):
        sg = self.red.get_engine()
        rows = (
            # (the method, the operator, the read compared)
            (lambda: self.red.remove(self.a.f[:2]), lambda: self.b.f[:2] << -self.red,
             lambda n: (n >> self.red).tolist()),
            (lambda: sg.remove(self.a.f[2:4]), lambda: self.b.f[2:4] << -sg,
             lambda n: (n >> self.red).tolist()),
            (lambda: self.lay.remove(self.a), lambda: self.b << -self.lay,
             lambda n: n in self.lay),
        )
        for method, operator, read in rows:
            method()
            operator()
            with self.subTest(read=read(self.a)):
                self.assertEqual(read(self.a), read(self.b))
        self.assertEqual((self.a >> self.red).tolist(), [4, 5])
        self.assertIs(self.a in self.lay, False)
        # a tag: components leave it. Re-pinned (user decision 2026-10-16,
        # "Removal methods details": Tag.remove takes members only): a node is a
        # TypeError naming Tag('cap').delete(node), with nothing written; the
        # operator node << -Tag('cap') still deletes the tag
        Tag("cap").remove(self.sph.vtx[:2])
        self.assertEqual((self.sph >> Tag("cap")).tolist(), [2, 3, 4, 5, 6, 7])
        self.sph.vtx[2:4] << -Tag("cap")
        self.assertEqual((self.sph >> Tag("cap")).tolist(), [4, 5, 6, 7])
        with self.assertRaisesRegex(TypeError, r"Tag\('cap'\)\.remove takes members .*Tag\('cap'\)\.delete\("):
            Tag("cap").remove(self.sph)
        self.assertEqual((self.sph >> Tag("cap")).tolist(), [4, 5, 6, 7])
        Tag("cap").delete(self.sph)
        self.assertEqual(Tag.of(self.sph), [])

    def test_it_returns_the_left_hand_side(self):
        faces = self.a.f[:2]
        self.assertIs(self.red.remove(faces), faces)
        both = List([self.a, self.b])
        self.assertIs(self.lay.remove(both), both)
        self.assertIs(self.red.get_engine().remove(self.b), self.b)
        verts = self.sph.vtx[:2]
        self.assertIs(Tag("cap").remove(verts), verts)
        # a plain list is a left-hand side too (it cannot ride on the left of <<)
        pair = [self.a, self.b]
        self.assertIs(self.red.remove(pair), pair)
        self.assertEqual(cmds.sets("redSG", query=True), None)

    def test_the_same_refusals_before_any_write(self):
        poly = Node(cmds.polyCube(name="poly")[0])   # procedural face tags
        rows = (
            (lambda: self.red.remove(self.a.tx), lambda: self.a.tx << -self.red),
            (lambda: self.red.remove(self.a.vtx[:2]), lambda: self.a.vtx[:2] << -self.red),
            (lambda: self.lay.remove(self.a.f[:2]), lambda: self.a.f[:2] << -self.lay),
            (lambda: self.lay.remove(self.a.tx), lambda: self.a.tx << -self.lay),
            (lambda: Layer("defaultLayer").remove(self.a), lambda: self.a << -Layer("defaultLayer")),
            (lambda: self.red.get_engine().remove(self.a.e[:2]), lambda: self.a.e[:2] << -self.red.get_engine()),
            (lambda: Tag("ghost").remove(self.sph.vtx[:2]), lambda: self.sph.vtx[:2] << -Tag("ghost")),
            (lambda: Tag("top").remove(poly.f[:1]), lambda: poly.f[:1] << -Tag("top")),
            (lambda: Tag("cap").remove(self.sph.tx), lambda: self.sph.tx << -Tag("cap")),
        )
        before = self._state()
        for method, operator in rows:
            got, want = _refusal(method), _refusal(operator)
            with self.subTest(want=want):
                self.assertIsNotNone(want)
                self.assertEqual(got, want)
                self.assertEqual(self._state(), before)

    def test_a_token_has_no_method_and_the_kind_names_its_own_form(self):
        before = self._state()
        for token in (-self.lay, Layer(), -self.red, Blinn()):
            with self.subTest(token=repr(token)):
                with self.assertRaisesRegex(AttributeError, r"remove is the \w+ node's"):
                    token.remove(self.a)
        with self.assertRaisesRegex(TypeError, r"^Tag\(\)\.remove\(x\): a method names one tag; x << Tag\(\)"):
            Tag().remove(self.sph)
        with self.assertRaisesRegex(TypeError, r"^-Tag\('cap'\)\.remove\(\) is unassigned"):
            (-Tag("cap")).remove(self.sph.vtx[:2])
        self.assertEqual(self._state(), before)

    def test_a_referenced_tag_needs_force_as_the_operator_does(self):
        sph     = Node(cmds.polySphere(name="clustered", constructionHistory=False)[0])
        cluster = Node(cmds.cluster(str(sph))[0])   # the injection node is now the Orig shape
        sph.vtx[:4] << Tag("lid")
        cluster.input[0].componentTagExpression << Tag("lid")
        # re-pinned (user decision 2026-10-16, "Removal methods details":
        # Tag.remove takes members only, so a node is refused there, naming
        # delete): Tag('lid').delete(node) refuses as the operator does
        self.assertIs(_refusal(lambda: Tag("lid").remove(sph))[0], TypeError)
        got, want = _refusal(lambda: Tag("lid").delete(sph)), _refusal(lambda: sph << -Tag("lid"))
        self.assertEqual(got[0], want[0])
        self.assertIs(got[0], RuntimeError)
        self.assertTrue(sph in Tag("lid"))
        Tag("lid", force=True).delete(sph)
        self.assertEqual(Tag.of(sph), [])

    def test_one_undo_step(self):
        sg = self.red.get_engine()
        rows = (
            (lambda: self.red.remove(self.a.f[:2]), lambda: (self.a >> self.red).tolist(), "rig.material"),
            (lambda: sg.remove(self.b), lambda: (self.b >> self.red).tolist(), "rig.material"),
            (lambda: self.lay.remove(List([self.a, self.b])), lambda: (self.a in self.lay, self.b in self.lay),
             "rig.layer"),
            (lambda: Tag("cap").remove(self.sph.vtx[:3]), lambda: (self.sph >> Tag("cap")).tolist(), "rig.tag"),
        )
        for remove, read, name in rows:
            with self.subTest(name=name):
                start = read()
                cmds.flushUndo()
                remove()
                done = read()
                self.assertNotEqual(done, start)
                self.assertEqual(cmds.undoInfo(query=True, undoName=True), name)
                cmds.undo()
                self.assertEqual(read(), start)
                self.assertIs(cmds.undoInfo(query=True, undoQueueEmpty=True), True)
                cmds.redo()
                self.assertEqual(read(), done)

    def test_the_operator_stays(self):
        self.a << -self.red
        self.b << -self.lay
        self.assertEqual((self.a >> self.red).size, 0)
        self.assertIs(self.b in self.lay, False)

    def test_the_methods_are_statements(self):
        for fn in (Blinn.remove, ShadingEngine.remove, DisplayLayer.remove, Tag.remove):
            with self.subTest(fn=fn.__qualname__):
                self.assertIs(getattr(fn, "_rig_statement", False), True)


# --------------------------------------------------------------------- #
#  Transform.create(data)
# --------------------------------------------------------------------- #


def _scene():
    return set(cmds.ls(long=True))


class TestCreateFromData(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def _source(self):
        src = Transform.create(name="src")
        src.t << (1, 2, 3)
        src.r << (10, 20, 30)
        src.s << (1, 2, 3)
        src.ro << "zxy"
        src.ra << (5, 6, 7)
        src.v << False
        src << Float("weight", dv=0.25)
        src.weight << 0.75
        return src

    def test_a_transform_from_its_data(self):
        src  = self._source()
        made = Transform.create(src.serialize())
        self.assertIs(type(made), Transform)
        self.assertEqual(str(made), "src1")              # Maya renames a taken name
        for attr in ("t", "r", "s", "ra"):
            with self.subTest(attr=attr):
                np.testing.assert_allclose(getattr(made, attr).get(), getattr(src, attr).get())
        self.assertEqual(made.ro.get(), src.ro.get())
        self.assertIs(made.v.get(), False)
        self.assertAlmostEqual(made.weight.get(), 0.75)
        self.assertIsNone(made.get_parent())

    def test_a_joint_from_its_data(self):
        src = Joint.create(name="jnt")
        src.t << (1, 0, 0)
        src.jo << (10, 20, 30)
        src.radius << 2.5
        src.ssc << False
        data = src.serialize()
        for cls in (Joint, Transform):   # a joint's data on Transform makes a Joint
            with self.subTest(cls=cls.__name__):
                made = cls.create(data)
                self.assertIs(type(made), Joint)
                np.testing.assert_allclose(made.jo.get(), [10, 20, 30])
                self.assertEqual(made.radius.get(), 2.5)
                self.assertIs(made.ssc.get(), False)
                np.testing.assert_allclose(made.t.get(), [1, 0, 0])

    def test_a_locator_and_a_space_transform(self):
        from cgmath.hierarchy import TransformData

        loc  = Node(cmds.spaceLocator(name="loc")[0])
        loc.tx << 4
        made = Transform.create(loc.serialize())
        self.assertEqual(cmds.listRelatives(str(made), shapes=True, type="locator"), ["loc1Shape"])
        self.assertEqual(made.tx.get(), 4)
        box = Transform.create(TransformData("box", node_type="space_transform", translate=(0, 5, 0)))
        self.assertEqual(cmds.listRelatives(str(box), shapes=True, type="mesh"), ["boxShape"])
        self.assertEqual(box.ty.get(), 5)

    def test_name_parent_and_attribute_keywords(self):
        src  = self._source()
        grp  = Transform.create(name="grp")
        grp.t << (100, 0, 0)
        made = Transform.create(src.serialize(), name="copy", parent=grp, tx=9, weight=0.5)
        self.assertEqual(made.get_long_name(), "|grp|copy")
        # the data's values are local ones, the keywords set last
        np.testing.assert_allclose(made.t.get(), [9, 2, 3])
        self.assertAlmostEqual(made.weight.get(), 0.5)
        short = Transform.create(src.serialize(), n="short", p="grp")
        self.assertEqual(short.get_long_name(), "|grp|short")

    def test_refusals_before_anything_is_made(self):
        from cgmath.hierarchy import TransformData

        tdata = self._source().serialize()
        jdata = Joint.create(name="j").serialize()
        before = _scene()
        rows = (
            (TypeError, r"^Transform\.create\(\) takes name= and parent= as keywords, or one TransformData "
                        r"\(got 'src'\)$", lambda: Transform.create("src")),
            (TypeError, r"or one TransformData \(got ", lambda: Transform.create(tdata, tdata)),
            (TypeError, r"^Joint\.create\(\): the data describes a transform, which Joint does not make; "
                        r"Transform\.create\(data\) makes it$", lambda: Joint.create(tdata)),
            (TypeError, r"a TransformData of a 'ikHandle' cannot be built",
             lambda: Transform.create(TransformData("ik", node_type="ikHandle"))),
            (NodeNotFoundError, r"^no DAG node named 'nope'$", lambda: Transform.create(tdata, parent="nope")),
            (AttributeError, r"^Transform\.create\(\): a transform has no attribute 'txx'",
             lambda: Transform.create(tdata, txx=1)),
            (AttributeError, r"^Joint\.create\(\): a joint has no attribute 'nope'",
             lambda: Joint.create(jdata, nope=1)),
            (TypeError, r"takes name= or n=, not both", lambda: Transform.create(tdata, name="a", n="b")),
        )
        for error, text, call in rows:
            with self.subTest(text=text):
                with self.assertRaisesRegex(error, text):
                    call()
                self.assertEqual(_scene(), before)

    def test_a_value_that_fails_deletes_the_node(self):
        tdata  = self._source().serialize()
        before = _scene()
        with self.assertRaises(Exception) as caught:
            Transform.create(tdata, t=(1, 2))
        # the value's own error, raised once the node existed (not a refusal of the data)
        self.assertNotIn("TransformData", str(caught.exception))
        self.assertEqual(_scene(), before)

    def test_one_undo_step_and_the_selection_kept(self):
        tdata = self._source().serialize()
        cmds.select("src")
        before = _scene()
        cmds.flushUndo()
        made = Transform.create(tdata, tx=5)
        self.assertEqual(cmds.ls(selection=True), ["src"])
        self.assertEqual(cmds.undoInfo(query=True, undoName=True), "rig.Transform.create")
        made_names = _scene() - before
        cmds.undo()
        self.assertEqual(_scene(), before)
        self.assertIs(cmds.undoInfo(query=True, undoQueueEmpty=True), True)
        cmds.redo()
        self.assertEqual(_scene() - before, made_names)
        self.assertEqual(made.tx.get(), 5)

    def test_inside_a_container_as_a_typed_create(self):
        tdata = self._source().serialize()
        with container("outer") as outer:
            with container("inner"):
                plain = Transform.create(name="plain")
                kept  = Transform.create(tdata)
                named = Transform.create(tdata, name="named")
                loc   = Transform.create(Node(cmds.spaceLocator(name="lc")[0]).serialize())
        members = set(cmds.container(str(outer), query=True, nodeList=True) or [])
        made    = {str(plain), str(kept), str(named), str(loc), f"{loc}Shape"}
        self.assertTrue(made <= members, members)
        # an explicit name takes the scope's prefix as create(name=) does; the data's
        # name is kept (Mesh.create keeps its data's name)
        prefix = str(plain)[: -len("plain")]
        self.assertNotEqual(prefix, "")
        self.assertEqual(str(named), f"{prefix}named")
        self.assertEqual(str(kept), "src1")

    def test_create_hierarchy_shares_the_per_node_code(self):
        root = Joint.create(name="root")
        kid  = Joint.create(name="kid", parent=root)
        kid.tx << 3
        loc = Node(cmds.spaceLocator(name="tip")[0])
        cmds.parent(str(loc), str(kid))
        made = Transform.create_hierarchy(root.serialize_hierarchy())
        self.assertEqual([x.get_long_name() for x in made], ["|root1", "|root1|kid", "|root1|kid|tip"])
        self.assertEqual([type(x) for x in made], [Joint, Joint, Transform])
        self.assertEqual(made[1].tx.get(), 3)
        from rig.nodetypes import transform

        with mock.patch.object(transform, "_make_from_data", wraps=transform._make_from_data) as spy:
            Transform.create_hierarchy(root.serialize_hierarchy())
            Transform.create(root.serialize())
        self.assertEqual(spy.call_count, 4)

    def test_the_creator_is_a_statement(self):
        self.assertIs(getattr(Transform._create_from_data, "_rig_statement", False), True)


# --------------------------------------------------------------------- #
#  rig.random 3D without a seed
# --------------------------------------------------------------------- #


def _eval(plug):
    value = cmds.getAttr(str(plug))
    return list(value[0]) if isinstance(value, list) else value


class TestRandom3DWithoutSeed(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_no_seed_builds_three_fresh_streams(self):
        cmds.currentTime(1)
        for build in (
            lambda: R.value3D(),
            lambda: R.uniform3D([0, 0, 0], [10, 10, 10]),
            lambda: R.randint3D([0, 0, 0], [100, 100, 100]),
        ):
            before = len(cmds.ls("*CYCLE_SAFE_RANDOM_GENERATOR*"))
            out    = build()
            with self.subTest(out=str(out)):
                self.assertEqual(len(_eval(out)), 3)
                self.assertEqual(len(cmds.ls("*CYCLE_SAFE_RANDOM_GENERATOR*")) - before, 3)

    def test_the_fresh_seeds_are_three(self):
        cmds.currentTime(1)
        with mock.patch.object(R, "_python_randint", side_effect=[11, 22, 33]) as draw:
            fresh = R.value3D()
        self.assertEqual(draw.call_count, 3)
        np.testing.assert_allclose(_eval(fresh), _eval(R.value3D(seed=[11, 22, 33])))

    def test_a_seed_that_is_not_three_seeds_is_refused_before_anything_is_made(self):
        before = set(cmds.ls())
        for seed in (3, [1, 2], [1, 2, 3, 4], 2.5):
            for name, call in (
                ("value3D", lambda s: R.value3D(seed=s)),
                ("uniform3D", lambda s: R.uniform3D([0, 0, 0], [1, 1, 1], seed=s)),
                ("randint3D", lambda s: R.randint3D([0, 0, 0], [9, 9, 9], seed=s)),
            ):
                with self.subTest(name=name, seed=seed):
                    with self.assertRaisesRegex(TypeError, rf"^rig\.random\.{name}\(\) seed= takes three seeds"):
                        call(seed)
                    self.assertEqual(set(cmds.ls()), before)

    def test_a_seeded_network_is_unchanged(self):
        cmds.currentTime(1)
        three = _eval(R.value3D(seed=[1, 2, 3]))
        each  = [_eval(R.value(seed=s)) for s in (1, 2, 3)]
        np.testing.assert_allclose(three, each)


# --------------------------------------------------------------------- #
#  mirror_map takes both Axis enums
# --------------------------------------------------------------------- #


class TestMirrorMapTakesBothAxes(MayaTestCase):
    TEST_START_NEW_SCENE = True

    VALUES = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    def test_cgmath_axis_mirrors_as_rig_axis(self):
        from cgmath.geometry.mesh import Axis as CgmathAxis

        mesh = _cube().get_shape()
        mesh.add_map("mk", values=self.VALUES)
        for name in ("X", "Y", "Z"):
            for inverse in (False, True):
                with self.subTest(axis=name, inverse=inverse):
                    mesh.set_map_values("mk", self.VALUES)
                    mesh.mirror_map("mk", Axis[name], inverse)
                    want = mesh.get_map_values("mk")
                    mesh.set_map_values("mk", self.VALUES)
                    mesh.mirror_map("mk", CgmathAxis[name], inverse)
                    self.assertEqual(mesh.get_map_values("mk"), want)
                    self.assertNotEqual(want, self.VALUES)

    def test_anything_else_is_refused_before_the_map_is_read(self):
        mesh = _cube().get_shape()
        mesh.add_map("mk", values=self.VALUES)
        for axis in ("x", 0, None):
            with self.subTest(axis=axis):
                with self.assertRaisesRegex(TypeError, r"^a mirror axis is rig\.nodetypes\.Axis or cgmath"):
                    mesh.mirror_map("nosuchmap", axis)
        self.assertEqual(mesh.get_map_values("mk"), self.VALUES)

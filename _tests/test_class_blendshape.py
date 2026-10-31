import os
import shutil
import string
import tempfile

import numpy as np
from cgmath.geometry import MorphData, MorphList
from maya import cmds
from maya.api import OpenMaya
from rig.nodetypes import BlendShape, Node
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk

# the spelling the names review retired, read by name
_OLD_METHODS = ("set_targets_data",)


class TestBlendShape(MayaTestCase):
    """
    Test the BlendShape class
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()

        # make 2 cubes and a blend shape
        self.base        = Node(cmds.polyCube(name="base", ch=False)[0])
        self.target_name = "target"
        self.target      = Node(cmds.polyCube(name="target", ch=False)[0])
        self.bls         = Node.create("blendShape", self.target, self.base)

        # move a vert
        cmds.xform(f"{self.target}.vtx[3]", ws=True, t=(10, 2, 3))
        cmds.xform(f"{self.target}.vtx[4]", ws=True, t=(-1, 2, -3))
        self.indices      = np.array([3, 4])
        self.index_ranges = ["vtx[3:4]"]
        self.offsets      = np.array([(9.5, 1.5, 2.5), (-0.5, 1.5, -2.5)])

        # set weight
        self.weight = 0.5
        cmds.setAttr(f"{self.bls}.target", self.weight)

    def test_properties(self):
        self.assertEqual(self.bls.get_target_count(), 1)
        self.assertEqual(self.bls.get_targets(), [self.target_name])
        base_shape = self.base.get_children(type="mesh")[0]
        self.assertEqual(self.bls.get_geometries(), [base_shape])
        self.assertEqual(base_shape.get_deformers(node_type="blendShape"), [self.bls])
        self.assertEqual(self.bls.get_target_index(self.target_name), 0)
        self.assertEqual(self.bls.get_target_name(0), self.target_name)
        self.assertEqual(self.bls.get_target_weight(0), self.weight)

    def test_serialize(self):
        target_shape = self.target.get_children(type="mesh")[0]
        self.assertTrue(target_shape.get_output_attr(world_space=True).is_connected())
        org_data = self.bls.get_target_data(self.target_name)
        self.assertTrue(np.allclose(org_data.indices, self.indices))
        self.assertTrue(np.allclose(org_data.offsets, self.offsets))
        self.assertTrue(target_shape.get_output_attr(world_space=True).is_connected())

        new_target = "new_target"
        i          = self.bls.add_empty_target(new_target)
        self.assertEqual(self.bls.get_target_count(), 2)
        self.assertEqual(self.bls.get_targets(), [self.target_name, new_target])
        self.assertEqual(self.bls.get_target_index(new_target), 1)

        data = self.bls.get_target_data(new_target)
        self.assertEqual(data.indices.size, 0)
        self.assertEqual(data.offsets.size, 0)

        self.bls.set_target_data(i, org_data)
        data = self.bls.get_target_data(i)
        self.assertTrue(np.allclose(data.indices, self.indices))
        self.assertTrue(np.allclose(data.offsets, self.offsets))
        _, _, comp_attr = self.bls.get_target_data_attrs(i)
        self.assertEqual(comp_attr.get(), self.index_ranges)

        self.assertEqual(len(self.bls.serialize()), 2)
        self.assertEqual(len(self.bls.serialize(match_name="new_.*")), 1)

        # make sure we can serialize without an originalGeometry plug.
        new_bls = Node.create("blendShape", self.target, self.base)
        new_bls.originalGeometry[0].break_connections()
        self.assertEqual(len(new_bls.serialize()), 1)

    def test_create_from_morph_data(self):
        # rebuild a blendshape directly from a MorphList: targets are taken
        # from the data and offsets are applied during creation
        data = self.bls.serialize()
        self.bls.delete()

        bls = BlendShape.create(self.base, data)
        self.assertEqual(bls.get_targets(), [self.target_name])
        self.assertEqual(bls.serialize(), data)

        # a single MorphData is accepted as well
        bls.delete()
        bls = BlendShape.create(self.base, data[0])
        self.assertEqual(bls.get_targets(), [self.target_name])
        self.assertEqual(bls.serialize(), data)

    def test_rebuild(self):
        org_points   = self.target.get_children(type="mesh")[0].get_points()
        rebuilt_mesh = self.bls.rebuild_target(0)
        new_points   = rebuilt_mesh.get_children(type="mesh")[0].get_points()
        self.assertTrue(np.allclose(org_points, new_points, rtol=0.001))


class TestBlendShapeMovedBase(MayaTestCase):
    """``serialize`` reads the Orig mesh in its object space, as ``get_target_data`` reads a
    live target (round features s7, F2): on v2.0.0a5 a base moved to x=5 offset every target
    by -5 (runs\\features\\saveload\\probes\\p4_deformers.py)."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.base = Node(cmds.polyCube(name="base", ch=False)[0])
        self.base.t << [5, 0, 0]
        self.smile = Node(cmds.polyCube(name="smile", ch=False)[0])
        cmds.xform("smile.vtx[3]", objectSpace=True, translation=(1, 1, 1), relative=True)

    def test_a_moved_base_round_trips(self):
        bls = BlendShape.create(self.smile, self.base, name="face")
        cmds.delete("smile")  # the target's offsets, no live mesh
        data = bls.serialize()
        self.assertEqual(np.asarray(data[0].indices).tolist(), [3])
        np.testing.assert_allclose(data[0].offsets, [[1.0, 1.0, 1.0]], atol=1e-6)
        bls.delete()
        again = BlendShape.create(data, self.base)
        again.set_target_weight("smile", 1)
        np.testing.assert_allclose(
            np.array(self.base.get_shape().get_points(world_space=False))[3, :3], (1.5, 1.5, 1.5), atol=1e-6
        )

    def test_a_live_target_on_a_moved_base(self):
        bls = BlendShape.create(self.smile, self.base, name="face")
        self.assertTrue(self.smile.get_shape().get_output_attr(world_space=True).is_connected())
        data = bls.serialize()
        self.assertEqual(np.asarray(data[0].indices).tolist(), [3])
        np.testing.assert_allclose(data[0].offsets, [[1.0, 1.0, 1.0]], atol=1e-6)


class TestSetTargetDataTakesWhatSerializeReturns(UndoWalk, MayaTestCase):
    """``set_target_data`` is the inverse of ``get_target_data`` and ``serialize`` (user
    decision 2026-10-30): it takes a MorphData, a MorphList or a list of MorphData, each onto
    the target of its name, beside the name or index and its data. Every refusal comes before
    any edit, the caller's data is copied, and each call is one undo step. A test named
    ``guard`` holds on the tree before it too: what stays."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.base = Node(cmds.polyCube(name="base", ch=False)[0])
        self.bs   = BlendShape.create(Node(cmds.polyCube(name="smile", ch=False)[0]), self.base, name="face")
        cmds.delete("smile")  # its offsets stay, with no live mesh
        cmds.flushUndo()

    @staticmethod
    def morph(name, index=0, height=1.0):
        return MorphData(name=name, indices=np.array([index]), offsets=np.array([[0.0, height, 0.0]]))

    def read(self):
        """The targets' indices."""
        return {m.name: np.asarray(m.indices).tolist() for m in self.bs.serialize()}

    def assert_refused(self, error, pattern, call):
        """`call` raises `error` matching `pattern`, and nothing changed: the targets, the
        undo queue."""
        before = self.read()
        cmds.flushUndo()
        with self.assertRaisesRegex(error, pattern):
            call()
        self.assertEqual(self.read(), before)
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_a_morph_list_goes_onto_the_targets_of_its_names(self):
        self.bs.set_target_data(MorphList([self.morph("smile", 1), self.morph("brow", 2)]))
        self.assertEqual(self.bs.get_targets(), ["smile", "brow"])
        self.assertEqual(self.read(), {"smile": [1], "brow": [2]})

    def test_what_serialize_returned_goes_back(self):
        self.bs.set_target_data(MorphList([self.morph("smile", 1), self.morph("brow", 2, 3.0)]))
        written = self.bs.serialize()
        self.bs.set_target_data(MorphList([self.morph("brow", 7)]))
        self.assertNotEqual(self.bs.serialize(), written)
        self.bs.set_target_data(written)
        self.assertEqual(self.bs.serialize(), written)

    def test_a_morph_data_goes_onto_the_target_of_its_name(self):
        self.bs.set_target_data(self.morph("smile", 4))
        self.bs.set_target_data(self.morph("brow", 5))
        self.assertEqual(self.read(), {"smile": [4], "brow": [5]})
        self.bs.set_target_data(self.bs.get_target_data("brow"))
        self.assertEqual(self.read(), {"smile": [4], "brow": [5]})

    def test_a_plain_list_of_morph_data(self):
        self.bs.set_target_data([self.morph("smile", 4), self.morph("brow", 5)])
        self.assertEqual(self.read(), {"smile": [4], "brow": [5]})

    def test_tolerance_prunes_every_morph(self):
        self.bs.set_target_data(MorphList([self.morph("smile", 1, 0.5), self.morph("brow", 2, 2.0)]), tolerance=1.0)
        self.assertEqual(self.read(), {"smile": [], "brow": [2]})

    def test_guard_a_name_or_an_index_takes_its_data(self):
        self.bs.set_target_data("smile", self.morph("not_read", 1))
        self.assertEqual(self.read(), {"smile": [1]})
        self.bs.set_target_data(0, (np.array([2]), np.array([[0.0, 2.0, 0.0]])))
        self.assertEqual(self.read(), {"smile": [2]})
        self.bs.set_target_data(target="brow", data=(np.array([3]), np.array([[0.0, 3.0, 0.0]])), tolerance=1.0)
        self.assertEqual(self.read(), {"smile": [2], "brow": [3]})
        self.bs.set_target_data("brow", self.morph("not_read", 4), force=False)
        self.assertEqual(self.read(), {"smile": [2], "brow": [4]})

    def test_guard_force_false_refuses_a_name_no_target_has(self):
        with self.assertRaisesRegex(RuntimeError, "^Target face.missing not found.$"):
            self.bs.set_target_data("missing", self.morph("not_read"), force=False)
        self.assertEqual(self.bs.get_targets(), ["smile"])

    def test_tolerance_and_force_are_keywords(self):
        for args in ((0.5,), (None, True)):
            with self.subTest(args=args):
                self.assert_refused(
                    TypeError, "positional arguments", lambda: self.bs.set_target_data("smile", self.morph("x"), *args)
                )

    def test_data_beside_morphs_is_a_type_error(self):
        morph = self.morph("smile", 1)
        for first in (morph, MorphList([morph]), [morph]):
            with self.subTest(given=type(first).__name__):
                self.assert_refused(TypeError, "goes alone", lambda: self.bs.set_target_data(first, morph))

    def test_a_name_or_an_index_without_data_names_both_forms(self):
        for target in ("smile", "brow", 0):
            with self.subTest(target=target):
                self.assert_refused(
                    TypeError,
                    r"set_target_data\(target, data\).*set_target_data\(morphs\)",
                    lambda: self.bs.set_target_data(target),
                )

    def test_a_morph_with_no_name_is_a_value_error(self):
        good, nameless = self.morph("smile", 6), self.morph("", 7)
        for given in (nameless, MorphList([good, nameless]), [good, nameless]):
            with self.subTest(given=type(given).__name__):
                self.assert_refused(ValueError, "no name", lambda: self.bs.set_target_data(given))

    def test_a_list_of_anything_else_is_a_type_error(self):
        self.assert_refused(
            TypeError, "list of MorphData", lambda: self.bs.set_target_data([self.morph("smile", 6), "brow"])
        )

    def test_force_false_checks_every_name_before_the_first_edit(self):
        morphs = MorphList([self.morph("smile", 6), self.morph("missing", 7)])
        self.assert_refused(
            RuntimeError, "^Target face.missing not found.$", lambda: self.bs.set_target_data(morphs, force=False)
        )

    def test_force_false_sets_the_targets_that_are_there(self):
        self.bs.set_target_data(MorphList([self.morph("smile", 6)]), force=False)
        self.assertEqual(self.read(), {"smile": [6]})

    def test_data_it_cannot_read_adds_no_target(self):
        self.assert_refused(TypeError, "", lambda: self.bs.set_target_data("brow", 5))

    def test_a_name_with_morphs_for_data_is_a_type_error(self):
        morph = self.morph("smile", 1)
        for data in (MorphList([morph]), MorphList([morph, morph]), [morph], [morph, morph], (morph, morph)):
            for target in ("smile", 0):
                with self.subTest(given=type(data).__name__, count=len(data), target=target):
                    self.assert_refused(
                        TypeError,
                        r"goes alone: set_target_data\(morphs\)",
                        lambda: self.bs.set_target_data(target, data),
                    )

    def test_guard_a_plain_list_of_indices_and_offsets_is_data(self):
        self.bs.set_target_data("smile", [[1], [[0.0, 1.0, 0.0]]])
        self.assertEqual(self.read(), {"smile": [1]})

    def test_a_name_maya_refuses_is_refused_before_any_edit(self):
        for name in ("bad name", "1abc", "a.b", "a|b", "a[0]", "envelope", "en", "weight"):
            morphs = MorphList([self.morph("smile", 2), self.morph(name, 3)])
            with self.subTest(name):
                self.assert_refused(ValueError, "cannot be named", lambda: self.bs.set_target_data(morphs))

    def test_a_name_with_its_data_is_refused_too(self):
        morph = self.morph("not_read")
        self.assert_refused(ValueError, "named 'bad name'", lambda: self.bs.set_target_data("bad name", morph))

    def test_the_name_check_follows_maya(self):
        """Each ASCII symbol and digit, first and further in: a name Maya takes for an alias
        is a new target, and one it refuses is refused before any edit."""
        other = Node(cmds.polyCube(name="other", ch=False)[0])
        BlendShape.create(self.morph("probe"), other, name="probe_bs")
        cmds.setAttr("probe_bs.weight[1]", 0)

        def maya_takes(name):
            try:
                cmds.aliasAttr(name, "probe_bs.weight[1]")
            except RuntimeError:
                return False
            taken = name in cmds.aliasAttr("probe_bs", query=True)
            if taken:
                cmds.aliasAttr("probe_bs.weight[1]", remove=True)
            return taken

        quiet = cmds.scriptEditorInfo(query=True, suppressWarnings=True)
        cmds.scriptEditorInfo(suppressWarnings=True)  # no line per refused alias
        try:
            for char in " " + string.punctuation + string.digits:
                for name in (f"{char}ab", f"a{char}b"):
                    with self.subTest(name):
                        if maya_takes(name):
                            self.bs.set_target_data(self.morph(name, 1))
                            self.assertIn(name, self.bs.get_targets())
                        else:
                            morph = self.morph(name, 1)
                            self.assert_refused(ValueError, "cannot be named", lambda: self.bs.set_target_data(morph))
        finally:
            cmds.scriptEditorInfo(suppressWarnings=quiet)

    def test_guard_a_name_maya_takes_stays_a_name(self):
        names = ("a-b", "_x", "<ab", "\u00e9t\u00e9")
        for i, name in enumerate(names):
            self.bs.set_target_data(name, self.morph("not_read", i))
        self.assertEqual(self.read(), {"smile": [], **{name: [i] for i, name in enumerate(names)}})

    def test_guard_a_name_a_target_has_is_no_new_name(self):
        cmds.setAttr("face.weight[1]", 0)
        selection = OpenMaya.MSelectionList().add("face.weight[1]")
        depend_fn = OpenMaya.MFnDependencyNode(selection.getDependNode(0))
        depend_fn.setAlias("a#b", "weight[1]", selection.getPlug(0))  # the command refuses it, the API does not
        self.bs.set_target_data("a#b", self.morph("not_read", 3))
        self.assertEqual(np.asarray(self.bs.get_target_data(1).indices).tolist(), [3])

    def undone_target(self):
        """The undo of a new target keeps its weight element (an older behaviour, documented):
        no name, no offsets."""
        self.bs.set_target_data(self.morph("new", 2))
        cmds.undo()
        self.assertEqual(list(self.bs.get_target_indices()), [0, 1])

    def unnamed_with_offsets(self):
        """A weight element with no name that holds offsets: `serialize` names it ``weight[1]``."""
        self.undone_target()
        self.bs.set_target_data(1, self.morph("ignored", 2))
        morphs = self.bs.serialize()
        self.assertEqual([m.name for m in morphs], ["smile", "weight[1]"])
        cmds.flushUndo()
        return morphs

    def test_serialize_leaves_out_an_undone_target(self):
        self.undone_target()
        self.assertEqual([m.name for m in self.bs.serialize()], ["smile"])
        self.assertEqual(self.bs.get_target_count(),            2)
        self.assertEqual(list(self.bs.get_target_indices()),    [0, 1])

    def test_match_name_leaves_it_out_too(self):
        self.undone_target()
        self.assertEqual(len(self.bs.serialize(match_name=r"weight\[1\]")), 0)
        self.assertEqual([m.name for m in self.bs.serialize(match_name="smile")], ["smile"])

    def test_an_undone_target_round_trips(self):
        self.undone_target()
        before = self.read()
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        path = self.bs.save(os.path.join(folder, "face.npz"))
        for label, call in (
            ("set_target_data", lambda: self.bs.set_target_data(self.bs.serialize())),
            ("load the data",   lambda: self.bs.load(self.bs.serialize())),
            ("load the file",   lambda: self.bs.load(path)),
        ):
            with self.subTest(label):
                call()
                self.assertEqual(self.read(), before)

    def test_an_undone_target_loads_onto_another_mesh(self):
        self.undone_target()
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        path = self.bs.save(os.path.join(folder, "face.npz"))
        made = BlendShape.load(path, Node(cmds.polyCube(name="other", ch=False)[0]), name="other_bs")
        self.assertEqual(made.get_targets(), ["smile"])

    def test_a_weight_element_with_no_name_is_refused_before_any_edit(self):
        unnamed = self.unnamed_with_offsets()[1]
        morphs  = MorphList([self.morph("smile", 5), unnamed])
        self.assert_refused(ValueError, r"cannot be named 'weight\[1\]'", lambda: self.bs.set_target_data(morphs))

    def test_a_weight_element_with_no_name_is_refused_by_another_blendshape(self):
        morphs = self.unnamed_with_offsets()
        other  = BlendShape.create(self.morph("first"), Node(cmds.polyCube(name="other", ch=False)[0]), name="other_bs")
        cmds.flushUndo()
        with self.assertRaisesRegex(ValueError, r"cannot be named 'weight\[1\]'"):
            other.set_target_data(morphs)
        self.assertEqual(other.get_targets(), ["first"])
        self.assertEqual(np.asarray(other.get_target_data("first").indices).tolist(), [0])
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_the_callers_data_is_left_as_it_was(self):
        morph = MorphData(name="smile", indices=np.array([0, 1]), offsets=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))
        for label, stored, call in (
            ("a name",      1, lambda: self.bs.set_target_data("smile", morph)),
            ("a MorphData", 1, lambda: self.bs.set_target_data(morph)),
            ("a MorphList", 0, lambda: self.bs.set_target_data(MorphList([morph]), tolerance=2.0)),
            ("a list",      1, lambda: self.bs.set_target_data([morph])),
            ("create",      1, lambda: BlendShape.create(morph, self.base, name="made")),
        ):
            with self.subTest(label):
                call()
                bs = BlendShape("made") if label == "create" else self.bs
                # the array itself: get_target_data prunes what it reads
                self.assertEqual(len(bs.get_target_data_attrs(0)[1].get() or []), stored)
                np.testing.assert_array_equal(morph.indices, [0, 1])
                np.testing.assert_array_equal(morph.offsets, [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

    def test_three_new_targets_are_one_undo_step(self):
        before = self.read()
        self.bs.set_target_data(MorphList([self.morph(name, i) for i, name in enumerate("abc")]))
        after = self.read()
        self.assertEqual(after,            {"smile": [], "a": [0], "b": [1], "c": [2]})
        self.assertEqual(self.undo_name(), "rig.BlendShape.set_target_data")
        self.assertEqual(self.undo_all(),  1)
        self.assertEqual(self.read(),      before)
        self.assertEqual(self.redo_all(),  1)
        self.assertEqual(self.read(),      after)

    def test_a_name_is_one_undo_step(self):
        self.bs.set_target_data("smile", self.morph("not_read", 1))
        self.assertEqual(self.undo_name(), "rig.BlendShape.set_target_data")
        self.assertEqual(self.undo_all(),  1)
        self.assertEqual(self.read(),      {"smile": []})
        self.bs.set_target_data("brow", self.morph("not_read", 2))
        self.assertEqual(self.undo_all(), 1)
        self.assertEqual(self.read(), {"smile": []})

    def test_create_with_morphs_is_one_undo_step(self):
        other = Node(cmds.polyCube(name="other", ch=False)[0])
        cmds.flushUndo()
        before = cmds.ls()
        made   = BlendShape.create(MorphList([self.morph(name, i) for i, name in enumerate("abc")]), other, name="made")
        self.assertEqual(made.get_targets(), ["a", "b", "c"])
        self.assertEqual(self.undo_name(), "rig.BlendShape.create")
        self.assertEqual(self.undo_all(), 1)
        self.assertEqual(cmds.ls(), before)
        self.assertEqual(self.redo_all(), 1)
        self.assertEqual(BlendShape("made").get_targets(), ["a", "b", "c"])

    def test_a_plain_create_is_one_undo_step(self):
        other  = Node(cmds.polyCube(name="other", ch=False)[0])
        target = Node(cmds.polyCube(name="target", ch=False)[0])
        cmds.flushUndo()
        before = cmds.ls()
        BlendShape.create(target, other, name="plain")
        self.assertEqual(self.undo_name(), "rig.BlendShape.create")
        self.assertEqual(self.undo_all(),  1)
        self.assertEqual(cmds.ls(),        before)

    def test_create_refuses_a_morph_with_no_name_before_anything_is_made(self):
        other = Node(cmds.polyCube(name="other", ch=False)[0])
        cmds.flushUndo()
        before = cmds.ls()
        with self.assertRaisesRegex(ValueError, "^BlendShape.create: .*no name"):
            BlendShape.create(self.morph(""), other)
        self.assertEqual(cmds.ls(), before)

    def test_create_refuses_a_name_maya_refuses_before_anything_is_made(self):
        other = Node(cmds.polyCube(name="other", ch=False)[0])
        cmds.flushUndo()
        before = cmds.ls()
        for name in ("bad name", "envelope", "weight"):
            morphs = MorphList([self.morph("good"), self.morph(name)])
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, f"^BlendShape.create: a new target cannot be named '{name}'"):
                    BlendShape.create(morphs, other)
                with self.assertRaisesRegex(ValueError, "cannot be named"):
                    BlendShape.load(morphs, other)
                self.assertEqual(cmds.ls(), before)
                self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_a_load_refuses_a_name_maya_refuses_before_it_frees_a_live_target(self):
        target = Node(cmds.polyCube(name="live", ch=False)[0])
        bs     = BlendShape.create(target, Node(cmds.polyCube(name="other", ch=False)[0]), name="other_bs")
        source = target.get_shape().get_output_attr(world_space=True)
        cmds.flushUndo()
        with self.assertRaisesRegex(ValueError, r"^BlendShape\('other_bs'\).load: a new target cannot be named"):
            bs.load(MorphList([self.morph("live"), self.morph("bad name")]))
        self.assertTrue(source.is_connected())
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_a_load_refuses_a_morph_with_no_name_before_it_frees_a_live_target(self):
        target = Node(cmds.polyCube(name="live", ch=False)[0])
        bs     = BlendShape.create(target, Node(cmds.polyCube(name="other", ch=False)[0]), name="other_bs")
        source = target.get_shape().get_output_attr(world_space=True)
        self.assertTrue(source.is_connected())
        with self.assertRaisesRegex(ValueError, r"^BlendShape\('other_bs'\).load: .*no name"):
            bs.load(MorphList([self.morph("live"), self.morph("")]))
        self.assertTrue(source.is_connected())

    def test_guard_a_load_is_its_own_step(self):
        self.bs.load(MorphList([self.morph(name, i) for i, name in enumerate("abc")]))
        self.assertEqual(self.undo_name(), "rig.BlendShape.load")
        self.assertEqual(self.undo_all(), 1)

    def test_the_old_name_is_gone_naming_the_new_form(self):
        old = _OLD_METHODS[0]
        self.assertFalse(hasattr(BlendShape, old))
        with self.assertRaises(AttributeError) as caught:
            getattr(self.bs, old)(MorphList())
        self.assertIn("is now node.set_target_data(morphs)", str(caught.exception))

"""Review fix (features run): the curve classes' own routes follow the plug's rules.

* D3 (the user's answer): "updating the curve of a locked plug raises the same way". At
  8286cb8 ``AnimCurve.of(k.tx)[0].set_keys(...)`` and ``AnimCurveTL.define("k_translateX",
  ...)`` rewrote a locked plug's animation while ``k.tx << AnimCurve(...)`` refused. Both
  now raise the InjectionError ``<<`` raises, naming ``Plug('k.translateX').unlock()``, even
  with the keys the curve holds, and change nothing.
* The default out tangent of ``set_keys`` / ``define`` follows the plugs the curve keys: a
  step for a bool or an enum, as ``plug << AnimCurve(...)`` gives. At 8286cb8 they rewrote a
  keyed bool's or enum's curve to linear (an enum read 1 between 0 and 2), and keys equal to
  the curve's made an undo step that did nothing.
* ``List([a, a]).tx << AnimCurve(...)`` (one plug in two rows) made two curves and left the
  first unconnected; the last row now keys it, once, as the last of two values stays.
* The refusal of keys on an element of a per-instance array names the element
  (``a.worldMatrix[0]``), not the array.
"""

from maya import cmds

from rig import AnimCurve, AnimCurveTL, AnimCurveTU, Enum, InjectionError, List, Node, lock
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk


def _steps(statement):
    cmds.flushUndo()
    statement()
    count = 0
    while not cmds.undoInfo(query=True, undoQueueEmpty=True):
        cmds.undo()
        count += 1
    return count


class _Case(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True


class TestALockedPlugsCurve(_Case):
    def setUp(self):
        super().setUp()
        self.k = Node.create("transform", name="k")
        self.k.tx << AnimCurve([1, 2], times=[0, 10])
        self.k.tx << lock
        cmds.flushUndo()
        self.curve = AnimCurve.of(self.k.tx)[0]

    def check_refused(self, statement):
        state0 = self.scene_state()
        with self.assertRaises(InjectionError) as caught:
            statement()
        text = str(caught.exception)
        self.assertIn("'k.translateX'", text)
        self.assertIn("Plug('k.translateX').unlock()", text)
        self.assertIn("Nothing was changed", text)
        self.assert_state_equal(self.scene_state(), state0, "refused")
        self.assertEqual(self.curve.get_keys()._values.tolist(), [1.0, 2.0])
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_set_keys(self):
        self.check_refused(lambda: self.curve.set_keys([5, 6], times=[0, 10]))

    def test_set_keys_with_the_keys_it_holds(self):
        self.check_refused(lambda: self.curve.set_keys([1, 2], times=[0, 10]))

    def test_define(self):
        self.check_refused(lambda: AnimCurveTL.define("k_translateX", [7, 8], times=[0, 10]))

    def test_define_with_the_keys_it_holds(self):
        self.check_refused(lambda: AnimCurveTL.define("k_translateX", [1, 2], times=[0, 10]))

    def test_a_locked_compound(self):
        """A curve on ``k.ty`` with ``k.t`` locked: the child is locked through it."""
        self.k.ty << AnimCurve([0, 1])
        self.k.t << lock
        cmds.flushUndo()
        curve = AnimCurve.of(self.k.ty)[0]
        with self.assertRaises(InjectionError):
            curve.set_keys([3, 4])

    def test_unlocked_it_works(self):
        """guard: unlock, then set_keys and define update the curve."""
        self.k.tx.unlock()
        self.curve.set_keys([5, 6], times=[0, 10])
        self.assertEqual(self.curve.get_keys()._values.tolist(), [5.0, 6.0])
        AnimCurveTL.define("k_translateX", [7, 8], times=[0, 10])
        self.assertEqual(self.curve.get_keys()._values.tolist(), [7.0, 8.0])

    def test_an_unconnected_curve(self):
        """guard: a curve that keys nothing takes new keys."""
        free = AnimCurveTU.create([0, 1], name="free")
        free.set_keys([2, 3])
        self.assertEqual(free.get_num_keys(), 2)


class TestASteppedCurve(_Case):
    def setUp(self):
        super().setUp()
        self.a = Node.create("transform", name="a")
        self.a << Enum("mode", en="off:on:max")
        self.a.v << AnimCurve([1, 0, 1])
        self.a.mode << AnimCurve([0, 2, 0])
        cmds.flushUndo()

    def out_tangents(self, name):
        return cmds.keyTangent(name, query=True, outTangentType=True)

    def test_the_keys_it_holds_write_nothing(self):
        for curve, keys in ((AnimCurve.of(self.a.v)[0], [1, 0, 1]), (AnimCurve.of(self.a.mode)[0], [0, 2, 0])):
            with self.subTest(curve=curve.get_name()):
                self.assertEqual(_steps(lambda: curve.set_keys(keys)), 0)
                self.assertEqual(set(self.out_tangents(curve.get_name())), {"step"})

    def test_define_with_the_keys_it_holds(self):
        self.assertEqual(_steps(lambda: AnimCurveTU.define("a_mode", [0, 2, 0])), 0)
        self.assertEqual(set(self.out_tangents("a_mode")), {"step"})

    def test_new_keys_get_a_step(self):
        """set_keys and define with new keys keep the step: an enum reads 0 between 0 and 2."""
        AnimCurveTU.define("a_mode", [0, 2, 2])
        self.assertEqual(set(self.out_tangents("a_mode")), {"step"})
        cmds.currentTime(0.5)
        self.assertEqual(cmds.getAttr("a.mode"), 0)
        AnimCurve.of(self.a.v)[0].set_keys([0, 1, 0])
        self.assertEqual(set(self.out_tangents("a_visibility")), {"step"})

    def test_a_number_curve_stays_linear(self):
        """guard: a curve on tx keys linear; one that keys nothing too."""
        self.a.tx << AnimCurve([0, 1, 0])
        AnimCurve.of(self.a.tx)[0].set_keys([0, 2, 0])
        self.assertEqual(set(self.out_tangents("a_translateX")), {"linear"})
        free = AnimCurveTU.create([0, 1], name="free")
        free.set_keys([1, 0])
        self.assertEqual(set(self.out_tangents("free")), {"linear"})

    def test_a_curve_on_a_bool_and_a_number(self):
        """guard: a curve shared by a bool and a number keys linear (not every plug is a
        bool or an enum)."""
        curve = AnimCurve.of(self.a.v)[0]
        cmds.connectAttr(curve.get_name() + ".output", "a.tz")
        curve.set_keys([0, 1, 0])
        self.assertEqual(set(self.out_tangents(curve.get_name())), {"linear"})


class TestOnePlugInTwoRows(_Case):
    def test_a_node_twice(self):
        a = Node.create("transform", name="a")
        List([a, a]).tx << AnimCurve([0, 1])
        self.assertEqual(cmds.ls(type="animCurve"), ["a_translateX"])
        self.assertEqual(AnimCurve.of(a.tx)[0].get_keys()._values.tolist(), [0.0, 1.0])

    def test_the_last_row_wins(self):
        a = Node.create("transform", name="a")
        List([a.tx, a.tx]) << [AnimCurve([0, 1]), AnimCurve([0, 5])]
        self.assertEqual(cmds.ls(type="animCurve"), ["a_translateX"])
        self.assertEqual(AnimCurve.of(a.tx)[0].get_keys()._values.tolist(), [0.0, 5.0])
        # a re-run updates that curve in place, once
        List([a.tx, a.tx]) << [AnimCurve([0, 1]), AnimCurve([0, 7])]
        self.assertEqual(cmds.ls(type="animCurve"), ["a_translateX"])
        self.assertEqual(AnimCurve.of(a.tx)[0].get_keys()._values.tolist(), [0.0, 7.0])

    def test_one_step(self):
        """guard: one undo step takes the statement back."""
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        self.assertEqual(_steps(lambda: List([a, b, a]).tx << AnimCurve([0, 1])), 1)
        self.assertEqual(cmds.ls(type="animCurve"), [])


class TestTheElementIsNamed(_Case):
    def test_world_matrix_element(self):
        a = Node.create("transform", name="a")
        with self.assertRaises(TypeError) as caught:
            a.wm[0] << AnimCurve([0, 1])
        self.assertIn("a.worldMatrix[0] << AnimCurve(...)", str(caught.exception))
        self.assertIn("'a.worldMatrix[0]' is matrix", str(caught.exception))

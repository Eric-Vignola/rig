"""A function computes the same thing wherever it is called.

A rig function with a real container of its own (called at the top level, or
under ``flatten_containers=False``) publishes its list inputs on that
container. Until 2.0.0a15 every list was published as a ``double`` multi, so a
list of vectors handed on only each vector's first channel: ``max([A.t, B.t])``
at the top level was ``max(A.tx, B.tx)``, and ``interpolate.sequence`` with
vector values returned a scalar or raised. Lists are now published typed by
their elements (`rig._internal.container._publish_tokens`).

A generic plug that carries a vector (a ``choice`` node's output) has no child
plugs, so a per-channel consumer (``rig.lerp`` on vectors, ``abs``, ``clamp``,
``max``) raised "Plug is not a compound parent". Its channels are now read
through a ``plusMinusAverage`` (`rig._internal.types._generic_channels`).
"""

import numpy as np
from maya import cmds
from rig import container, elerp, functions as f, get_options, interpolate, lerp, List, Node
from rig import set_options, slerp
from rig._internal.types import _get_compound
from rig._tests._base import MayaTestCase

PLACES = ("top level", "inside a container")


def _value(out):
    """a plug's or a List's value as a float array"""
    return np.asarray(out.get(), dtype=float)


class _Options(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        options = dict(get_options())
        self.addCleanup(lambda: set_options(**options))

    def call(self, place, fn):
        if place == "top level":
            return fn()
        with container("user1"):
            return fn()

    def every_way(self):
        """(flatten, place) for both flatten settings and both places"""
        for flatten in (True, False):
            for place in PLACES:
                yield flatten, place


class TestSequenceOfVectors(_Options):

    def build(self, values, flatten, place, method):
        set_options(flatten_containers=flatten)
        nodes = []
        for name, value in zip(("A", "B", "C"), values):
            node = Node.create("transform", name=name)
            node.t << value
            nodes.append(node)
        w = Node.create("transform", name="W")
        w.tx << 0.25  # xp [0, 0.5, 1]: the first segment, its weight 0.5
        return self.call(place, lambda: interpolate.sequence(
            w.tx, [0, 0.5, 1], [n.t for n in nodes], method=method))

    def test_lerp_slerp_and_elerp_give_the_vector_wherever_they_are_called(self):
        cases = (
            ("lerp", lerp, ([1, 2, 4], [4, 2, 1], [2, 2, 2]), [2.5, 2.0, 2.5]),
            ("elerp", elerp, ([1, 2, 4], [4, 2, 1], [2, 2, 2]), [2.0, 2.0, 2.0]),
            ("slerp", slerp, ([1, 0, 0], [0, 1, 0], [0, 0, 1]), [0.5 ** 0.5, 0.5 ** 0.5, 0.0]),
        )
        for name, method, values, expected in cases:
            for flatten, place in self.every_way():
                with self.subTest(method=name, flatten=flatten, place=place):
                    cmds.file(new=True, force=True)
                    out = self.build(values, flatten, place, method)
                    np.testing.assert_allclose(_value(out), expected, atol=1e-6)

    def test_a_top_level_call_publishes_the_values_as_a_vector_multi(self):
        self.build(([1, 2, 4], [4, 2, 1], [2, 2, 2]), True, "top level", lerp)
        from rig._internal.container import _resolve_published

        target = next(c for c in cmds.ls(type="container") if c.startswith("sequence"))
        yp = _resolve_published(target, "yp")
        self.assertIsNotNone(yp)
        self.assertEqual(cmds.getAttr(f"{yp}[1]"), [(4.0, 2.0, 1.0)])

    def test_scalar_values_are_unchanged(self):
        for flatten, place in self.every_way():
            with self.subTest(flatten=flatten, place=place):
                cmds.file(new=True, force=True)
                set_options(flatten_containers=flatten)
                x = Node.create("transform", name="x_src")
                x.tx << 1.5
                out = self.call(place, lambda: interpolate.sequence(
                    x.tx, [0.0, 1.0, 2.0, 3.0], [10.0, 20.0, 30.0, 40.0]))
                self.assertAlmostEqual(float(_value(out)), 25.0, places=4)


class TestListFunctionsOfVectors(_Options):
    """per channel, as inside a container, at the top level too"""

    EXPECTED = {
        "max":    [3, 9, 7],
        "min":    [1, 2, 0],
        "argmax": [1, 2, 1],
        "argmin": [0, 1, 0],
        "all":    [1, 1, 0],
        "any":    [1, 1, 1],
        "cumsum": [[1, 5, 0], [4, 7, 7], [6, 16, 8]],
        "diff":   [[2, -3, 7], [-1, 7, -6]],
    }

    def test_every_list_function_works_per_channel_wherever_it_is_called(self):
        for name, expected in self.EXPECTED.items():
            for flatten, place in self.every_way():
                with self.subTest(function=name, flatten=flatten, place=place):
                    cmds.file(new=True, force=True)
                    set_options(flatten_containers=flatten)
                    tokens = []
                    for node_name, value in zip("ABC", ([1, 5, 0], [3, 2, 7], [2, 9, 1])):
                        node = Node.create("transform", name=node_name)
                        node.t << value
                        tokens.append(node.t)
                    out = self.call(place, lambda: getattr(f, name)(tokens))
                    np.testing.assert_allclose(_value(out), expected, atol=1e-6)

    def test_mixed_lists_compute_as_inside_a_container(self):
        # a literal vector, a scalar beside a vector, eulers: published as the
        # elements say, or not at all
        cases = {
            "a literal vector": (lambda a, b: f.max([a.t, [5, 5, 5]]), [5, 7, 5]),
            "a scalar and a vector": (lambda a, b: f.max([a.tx, b.t]), [4, 2, 8]),
            "eulers": (lambda a, b: f.max([a.r, b.r]), [40, 20, 60]),
            "a cumsum with a literal": (lambda a, b: f.cumsum([a.t, [1, 1, 1]]), [[1, 7, 3], [2, 8, 4]]),
            "a sequence with a literal": (lambda a, b: interpolate.sequence(a.tx, [0, 10], [a.t, [9, 9, 9]]),
                                          [1.8, 7.2, 3.6]),
        }
        for name, (fn, expected) in cases.items():
            for place in PLACES:
                with self.subTest(case=name, place=place):
                    cmds.file(new=True, force=True)
                    a, b = Node.create("transform", name="A"), Node.create("transform", name="B")
                    a.t << [1, 7, 3]
                    b.t << [4, 2, 8]
                    a.r << [10, 20, 30]
                    b.r << [40, 5, 60]
                    out = self.call(place, lambda: fn(a, b))
                    np.testing.assert_allclose(_value(out), expected, atol=1e-6)

    def test_a_list_of_scalars_is_still_published_as_a_scalar_multi(self):
        a, b = Node.create("transform", name="A"), Node.create("transform", name="B")
        a.tx << 1
        b.tx << 3
        out = f.cumsum([a.tx, b.tx])
        np.testing.assert_allclose(_value(out), [1, 4])
        from rig._internal.container import _resolve_published

        target = next(c for c in cmds.ls(type="container") if c.startswith("cumsum"))
        self.assertEqual(cmds.getAttr(f"{_resolve_published(target, 'input')}[1]"), 3.0)


class TestGenericVectorChannels(_Options):
    """a choice of vectors read per channel"""

    def choice_of_vectors(self):
        a, b, c = (Node.create("transform", name=n) for n in "ABC")
        a.t << [1, 2, 3]
        b.t << [4, 6, 8]
        c.t << [0, 0, 1]
        return a, b, f.choice([a.t, c.t], selector=0)

    def test_per_channel_consumers_read_the_choice(self):
        with container("user1"):
            a, b, ch = self.choice_of_vectors()
            cases = {
                "lerp":  (lerp(ch, b.t, 0.5), [2.5, 4.0, 5.5]),
                "abs":   (f.abs(ch), [1, 2, 3]),
                "clamp": (f.clamp(ch, 0, 2), [1, 2, 2]),
                "max":   (f.max([ch, b.t]), [4, 6, 8]),
            }
        for name, (out, expected) in cases.items():
            with self.subTest(consumer=name):
                np.testing.assert_allclose(_value(out), expected, atol=1e-6)

    def test_the_channels_come_from_one_plus_minus_average(self):
        _, _, ch = self.choice_of_vectors()
        before = set(cmds.ls(type="plusMinusAverage"))
        channels = _get_compound(ch)
        made = set(cmds.ls(type="plusMinusAverage")) - before
        self.assertEqual(len(made), 1)
        self.assertEqual([c.get() for c in channels], [1.0, 2.0, 3.0])
        cmds.setAttr("C.t", 7, 8, 9)
        cmds.setAttr(f"{ch.get_node()}.selector", 1)
        self.assertEqual([c.get() for c in channels], [7.0, 8.0, 9.0])

    def test_a_compound_plug_and_a_scalar_plug_make_no_node(self):
        a = Node.create("transform", name="A")
        before = len(cmds.ls())
        self.assertEqual([str(c) for c in _get_compound(a.t)], ["A.translateX", "A.translateY", "A.translateZ"])
        self.assertEqual([str(c) for c in _get_compound(a.tx)], ["A.translateX"])
        self.assertEqual(len(cmds.ls()), before)

    def test_a_choice_of_quaternions_raises_naming_the_plug(self):
        quats = []
        for name in ("A", "B"):
            node = cmds.createNode("transform", name=name)
            dm = cmds.createNode("decomposeMatrix", name=name + "_dm")
            cmds.connectAttr(node + ".worldMatrix[0]", dm + ".inputMatrix")
            quats.append(Node(dm).outputQuat)
        ch = f.choice(quats, selector=0)
        with self.assertRaises(TypeError) as caught:
            _get_compound(ch)
        self.assertIn(str(ch), str(caught.exception))


class TestRailSpineUnflattened(_Options):
    """rail_spine raised under flatten_containers=False (a vector sequence)"""

    def build(self, flatten):
        from rig.examples import rail_spine as rs

        cmds.file(new=True, force=True)
        set_options(create_containers=True, flatten_containers=flatten)
        pcs = List()
        for i in range(5):
            pcs.append(Node(cmds.spaceLocator()[0]))
            pcs[i].ty << i * 5
        rs.create_rail(pcs, 5, scale_controls=pcs, orient_controls=pcs, degree=3,
                       periodic=False, aim_axis=1, up_axis=0, debug=False)
        pcs[1].tx << 10
        pcs[2].rz << 30
        return {j: np.array(cmds.getAttr(j + ".worldMatrix[0]")) for j in cmds.ls(type="joint")}

    def test_it_builds_and_matches_the_flattened_build(self):
        flat = self.build(True)
        nested = self.build(False)
        self.assertEqual(sorted(nested), sorted(flat))
        for joint, world in flat.items():
            np.testing.assert_allclose(nested[joint], world, atol=1e-6, err_msg=joint)

"""Round U, fixes independent of the undo work.

Step U5 (user decision, round-4a deferred items): a lattice ``pt`` int key that
leaves two or more axes unspecified raises :class:`IndexError` with a hint,
before any scene query. ``lat.pt[0]`` would be a flattened slab, and a chained
``lat.pt[0][1]`` silently read ``pt[0][0][1]`` from it (round-4a repro
``case_f11_lattice``). The slab is spelled ``pt[0, :, :]`` and one point
``pt[s, t, u]``. A key leaving one axis (``pt[s, t]``, a line) is unchanged, as
is any key with a slice, a NURBS-surface ``cv[u]`` / ``cv[u][v]``, a
``Plug("shape.pt[...]")`` string, ``Components(...)`` and the bool
``TypeError``.
"""

from unittest import mock

from maya import cmds

from rig import Components, List, Node
from rig._internal.plug import ComponentPlug, Plug
from rig._tests._base import MayaTestCase


def _scene():
    """Every node in the scene, by long name."""
    return set(cmds.ls(long=True))


def _hint(shape, given, left_names="t, u", colons=":, :"):
    return (
        f"{shape}.pt[{given}] leaves 2 of 3 axes unspecified, and a chained index "
        "would pick from the flattened selection: write "
        f"pt[{given}, {left_names}] for one point or pt[{given}, {colons}] for the slab"
    )


class TestLatticePartialIndex(MayaTestCase):
    """``pt[s]`` (and chains on it) raises; every other indexing form holds."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cube = cmds.polyCube(name="ru_cube", ch=False)[0]
        lattice = cmds.lattice(cube, divisions=(3, 4, 5))[1]
        self.shape = cmds.listRelatives(lattice, shapes=True)[0]
        self.node = Node(self.shape)

    # -- the refusal -- #

    def test_bare_int_raises_with_hint_and_builds_nothing(self):
        for key, given in ((0, "0"), (-1, "-1"), (2, "2"), ((0,), "0")):
            with self.subTest(key=key):
                before = _scene()
                with self.assertRaises(IndexError) as caught:
                    self.node.pt[key]
                self.assertEqual(str(caught.exception), _hint(self.shape, given))
                self.assertEqual(_scene(), before)

    def test_chained_partial_raises_with_hint(self):
        # the round-4a repro: today pt[0][1] read pt[0][0][1]
        before = _scene()
        with self.assertRaises(IndexError) as caught:
            self.node.pt[0][1]
        self.assertEqual(str(caught.exception), _hint(self.shape, "0"))
        with self.assertRaises(IndexError):
            self.node.pt[1][2][3]
        self.assertEqual(_scene(), before)

    def test_held_handle_raises_and_names_the_live_node(self):
        handle  = self.node.pt
        renamed = cmds.rename(self.shape, "ru_lattice_renamed")
        with self.assertRaises(IndexError) as caught:
            handle[0]
        self.assertEqual(str(caught.exception), _hint(renamed, "0"))

    def test_raises_before_any_scene_query(self):
        # no axis size is read: the refusal needs only the key and the rank,
        # so an out-of-range partial key gets the same hint
        with mock.patch.object(
            ComponentPlug, "_axis_sizes", side_effect=AssertionError("queried")
        ):
            for key in (0, 99, -7):
                with self.subTest(key=key):
                    with self.assertRaisesRegex(IndexError, r"leaves 2 of 3 axes"):
                        self.node.pt[key]

    # -- unchanged forms -- #

    def test_line_is_unchanged(self):
        line = self.node.pt[0, 1]
        self.assertIsInstance(line, List)
        self.assertEqual(
            [str(p) for p in line], [f"{self.shape}.pt[0][1][{u}]" for u in range(5)]
        )
        self.assertEqual(str(self.node.pt[0, 1][2]), str(self.node.pt[0, 1, 2]))
        self.assertTrue(self.node.pt[0, 1][2].equals(self.node.pt[0, 1, 2]))
        self.assertEqual(str(self.node.pt[-1, -1][-1]), f"{self.shape}.pt[2][3][4]")

    def test_element_is_unchanged(self):
        elem = self.node.pt[0, 1, 2]
        self.assertIsInstance(elem, ComponentPlug)
        self.assertNotIsInstance(elem, List)
        self.assertEqual(str(elem), f"{self.shape}.pt[0][1][2]")
        self.assertEqual(elem.name, Plug(f"{self.shape}.pt[0][1][2]").name)

    def test_slices_are_unchanged(self):
        slab = self.node.pt[0, :, :]
        self.assertEqual(len(slab), 20)
        self.assertEqual(str(slab[0]), f"{self.shape}.pt[0][0][0]")
        self.assertEqual(str(slab[-1]), f"{self.shape}.pt[0][3][4]")
        # a slice pads the missing trailing axis, as numpy does
        self.assertEqual([str(p) for p in self.node.pt[0, :]], [str(p) for p in slab])
        plane = self.node.pt[:, :, 0]
        self.assertEqual(len(plane), 12)
        self.assertEqual(str(plane[-1]), f"{self.shape}.pt[2][3][0]")
        self.assertEqual(len(self.node.pt[:]), 60)
        self.assertEqual(len(self.node.pt[0, 1:3]), 10)
        self.assertEqual(len(self.node.pt[()]), 60)

    def test_other_refusals_are_unchanged(self):
        for key in (True, (0, True), (True, 0, 0), "a"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(TypeError, r"must be int or slice"):
                    self.node.pt[key]
        with self.assertRaisesRegex(IndexError, r"is 3-D but 4 indices given"):
            self.node.pt[0, 0, 0, 0]
        with self.assertRaisesRegex(IndexError, r"axis 1 index 9 out of range"):
            self.node.pt[0, 9]

    def test_surface_cv_is_unchanged(self):
        surface = cmds.listRelatives(cmds.nurbsPlane(name="ru_plane")[0], shapes=True)[0]
        cv = Node(surface).cv
        row = cv[1]
        self.assertIsInstance(row, List)
        self.assertEqual([str(p) for p in row], [f"{surface}.cv[1][{v}]" for v in range(4)])
        self.assertEqual(str(cv[1][2]), f"{surface}.cv[1][2]")
        self.assertTrue(cv[1][2].equals(cv[1, 2]))
        self.assertEqual(str(cv[-1][0]), f"{surface}.cv[3][0]")
        with self.assertRaisesRegex(TypeError, r"must be int or slice"):
            cv[True]

    def test_plug_strings_and_components_are_unchanged(self):
        self.assertEqual(
            Plug(f"{self.shape}.pt[0][1][2]").name, self.node.pt[0, 1, 2].name
        )
        points = Components(self.shape, "pt")
        self.assertEqual(points.count, 60)
        long_name = cmds.ls(self.shape, long=True)[0]
        self.assertEqual(points[0].names, [f"{long_name}.pt[0][0][0]"])
        self.assertEqual(
            Components(self.shape, "pt", [[0, 1, 2]]).names, [f"{long_name}.pt[0][1][2]"]
        )

"""Tests for v4.F.b multi-attribute improvements.

Covers:

- ``_inject_value`` bare-multi + sequence dispatch (per-index write,
  auto-creates indices). Backwards compat: ``multi << scalar`` still
  auto-appends.
- ``PlugList`` empty-slice + parent-multi fallback
  (``empty_multi[:] << values`` works).
- ``Plug.next_index`` property and ``Plug.append(value)`` helper
  for explicit auto-append (replaces the implicit ``multi << scalar``
  idiom for users who want the explicit form).
- ``_clone_attribute`` mirrors source multi indices on the new
  attribute (so ``node.multi >> dst`` is followed by working
  ``dst.multi[:]`` slicing without manual priming).
- End-to-end chain: ``node.multi >> dst << values`` works in one line.
"""

from maya import cmds
from rig import Node, Plug, PlugList
from rig.spec import Float, Vector
from rig._tests._base import MayaTestCase


# --------------------------------------------------------------------- #
#  _inject_value: bare-multi + sequence dispatch
# --------------------------------------------------------------------- #


class TestInjectMultiSequence(MayaTestCase):
    """``multi_root << sequence`` writes element-wise to indices
    0..len(seq)-1, auto-creating any missing indices."""

    TEST_START_NEW_SCENE = True

    def test_scalar_multi_from_value_list(self):
        n = Node.create("network", name="net")
        n << Float("weights", multi=True)

        n.weights << [10.0, 20.0, 30.0, 40.0]

        for i, expected in enumerate([10.0, 20.0, 30.0, 40.0]):
            self.assertAlmostEqual(cmds.getAttr(f"net.weights[{i}]"), expected)

    def test_compound_multi_from_list_of_vectors(self):
        n = Node.create("network", name="net")
        n << Vector("offsets", multi=True)

        n.offsets << [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        for i, expected in enumerate([[1, 2, 3], [4, 5, 6], [7, 8, 9]]):
            actual = cmds.getAttr(f"net.offsets[{i}]")[0]
            for a, e in zip(actual, expected):
                self.assertAlmostEqual(a, e)

    def test_compound_multi_from_pluglist_of_compounds(self):
        # PlugList of compound source plugs -> per-index connect.
        sources = [Node.create("transform", name=f"src{i}") for i in range(3)]
        for i, s in enumerate(sources):
            s.t << [(i + 1), (i + 1) * 2, (i + 1) * 3]

        n = Node.create("network", name="net")
        n << Vector("inputs", multi=True)

        n.inputs << PlugList([s.t for s in sources])

        for i, expected in enumerate([[1, 2, 3], [2, 4, 6], [3, 6, 9]]):
            actual = cmds.getAttr(f"net.inputs[{i}]")[0]
            for a, e in zip(actual, expected):
                self.assertAlmostEqual(a, e)

    def test_explicit_index_still_auto_creates(self):
        # multi[i] << v with i > existing max materializes intermediates
        # via Maya's create-on-write semantic. (Existing behavior, just
        # confirming it works alongside the new sequence dispatch.)
        n = Node.create("network", name="net")
        n      << Float("w", multi=True)
        n.w[5] << 99.0
        # w[5] populated; intermediates 0..4 exist as default.
        self.assertAlmostEqual(cmds.getAttr("net.w[5]"), 99.0)


class TestInjectMultiScalarBackwardsCompat(MayaTestCase):
    """``multi << scalar`` continues to auto-append (Eric Vignola's
    original idiom). Sequence dispatch only fires for sequence sources."""

    TEST_START_NEW_SCENE = True

    def test_scalar_appends_to_next_index(self):
        n = Node.create("network", name="net")
        n   << Float("w", multi=True)

        n.w << 1.0
        n.w << 2.0
        n.w << 3.0

        # Each call appended at the next available index.
        self.assertAlmostEqual(cmds.getAttr("net.w[0]"), 1.0)
        self.assertAlmostEqual(cmds.getAttr("net.w[1]"), 2.0)
        self.assertAlmostEqual(cmds.getAttr("net.w[2]"), 3.0)


# --------------------------------------------------------------------- #
#  PlugList empty-slice fallback via _parent_multi
# --------------------------------------------------------------------- #
class TestPlugListEmptySliceFallback(MayaTestCase):
    """v3.O: ``empty_multi[:] << values`` routes through the parent
    multi (via PlugList._parent_multi back-reference set by
    Plug.__getitem__'s slice tagging) and auto-creates indices to
    match the source length.

    The back-reference mechanism is set up at v3.O via
    ``Plug.__getitem__`` (which tags PlugList instances returned from
    multi slicing with ``_parent_multi=self``). Without the v3.O
    tagging, the v2-base PlugList.__lshift__ fallback wouldn't know
    where to route empty-slice writes.
    """

    TEST_START_NEW_SCENE = True

    def test_empty_slice_with_value_list(self):
        n = Node.create("network", name="net")
        n << Float("w", multi=True)

        # multi has 0 elements -- legacy behavior: dst[:] returns []
        # and the broadcast caps at empty (no writes). New behavior:
        # routes through parent, auto-creates indices.
        n.w[:] << [10.0, 20.0, 30.0]

        for i, expected in enumerate([10.0, 20.0, 30.0]):
            self.assertAlmostEqual(cmds.getAttr(f"net.w[{i}]"), expected)

    def test_empty_slice_with_pluglist(self):
        sources = [Node.create("transform", name=f"src{i}") for i in range(3)]
        for i, s in enumerate(sources):
            s.tx << float(i + 1)

        n = Node.create("network", name="net")
        n      << Float("w", multi=True)
        n.w[:] << PlugList([s.tx for s in sources])

        # Verify per-element connections.
        for i in range(3):
            sources_of_w_i = (
                cmds.listConnections(
                    f"net.w[{i}]", source=True, destination=False, plugs=True
                )
                or []
            )
            self.assertIn(f"src{i}.translateX", sources_of_w_i)

    def test_pluglist_back_reference_is_set_for_multi_slice(self):
        n = Node.create("network", name="net")
        n << Float("w", multi=True)
        empty_pl = n.w[:]
        # Should be a PlugList tagged with the parent multi.
        self.assertIsInstance(empty_pl, PlugList)
        self.assertEqual(len(empty_pl), 0)
        # Plugs are compared by full_name (each access yields a fresh
        # Plug instance, so use string equality not assertIs).
        self.assertIsNotNone(empty_pl._parent_multi)
        self.assertEqual(str(empty_pl._parent_multi), str(n.w))

    def test_pluglist_back_reference_is_none_for_compound_slice(self):
        n = Node.create("transform", name="cube")
        # transform.translate[:] is a compound non-multi slice -- no
        # parent-multi back-reference (slice goes through the compound
        # branch in Plug.__getitem__).
        pl = n.translate[:]
        self.assertIsInstance(pl, PlugList)
        self.assertEqual(len(pl), 3)
        self.assertIsNone(getattr(pl, "_parent_multi", None))

    def test_non_empty_multi_slice_uses_existing_broadcast(self):
        # When the multi has existing elements, dst[:] returns them and
        # asymmetric broadcast applies (not the parent fallback).
        n = Node.create("network", name="net")
        n << Float("w", multi=True)
        # Pre-populate 3 elements.
        n.w << [10.0, 20.0, 30.0]
        # Now overwrite via [:] with a list of the same length.
        n.w[:] << [100.0, 200.0, 300.0]
        for i, expected in enumerate([100.0, 200.0, 300.0]):
            self.assertAlmostEqual(cmds.getAttr(f"net.w[{i}]"), expected)


class TestPlugAppendAndNextIndex(MayaTestCase):
    """``Plug.next_index`` and ``Plug.append`` -- explicit form of
    the implicit ``multi << scalar`` auto-append."""

    TEST_START_NEW_SCENE = True

    def test_next_index_starts_at_zero(self):
        n = Node.create("network", name="net")
        n << Float("w", multi=True)
        self.assertEqual(n.w.next_index, 0)

    def test_next_index_advances_after_append(self):
        n = Node.create("network", name="net")
        n   << Float("w", multi=True)
        n.w << 10.0
        self.assertEqual(n.w.next_index, 1)
        n.w << 20.0
        self.assertEqual(n.w.next_index, 2)

    def test_append_writes_at_next_available_index(self):
        n = Node.create("network", name="net")
        n << Float("w", multi=True)

        elem0 = n.w.append(11.0)
        elem1 = n.w.append(22.0)
        elem2 = n.w.append(33.0)

        self.assertAlmostEqual(cmds.getAttr(str(elem0)), 11.0)
        self.assertAlmostEqual(cmds.getAttr(str(elem1)), 22.0)
        self.assertAlmostEqual(cmds.getAttr(str(elem2)), 33.0)

    def test_append_returns_element_plug(self):
        n = Node.create("network", name="net")
        n << Float("w", multi=True)
        elem = n.w.append(5.0)
        self.assertIsInstance(elem, Plug)
        self.assertTrue(str(elem).endswith("[0]"))

    def test_append_compound(self):
        n = Node.create("network", name="net")
        n << Vector("offsets", multi=True)
        n.offsets.append([1, 2, 3])
        n.offsets.append([4, 5, 6])

        for i, expected in enumerate([[1, 2, 3], [4, 5, 6]]):
            actual = cmds.getAttr(f"net.offsets[{i}]")[0]
            for a, e in zip(actual, expected):
                self.assertAlmostEqual(a, e)

    def test_next_index_raises_on_non_multi(self):
        n = Node.create("transform", name="cube")
        with self.assertRaises(ValueError):
            _ = n.tx.next_index

    def test_append_raises_on_non_multi(self):
        n = Node.create("transform", name="cube")
        with self.assertRaises(ValueError):
            n.tx.append(5.0)


# --------------------------------------------------------------------- #
#  _clone_attribute multi-index population
# --------------------------------------------------------------------- #


class TestCloneAttributePopulatesMultiIndices(MayaTestCase):
    """``node.multi >> dst`` (clone) now mirrors the source's populated
    multi indices on the new attribute. Without this, ``dst.multi[:]``
    would be empty after clone."""

    TEST_START_NEW_SCENE = True

    def test_clone_scalar_multi_mirrors_indices(self):
        src = Node.create("network", name="src")
        src   << Float("w", multi=True)
        src.w << [1.0, 2.0, 3.0]

        dst = Node.create("network", name="dst")
        src.w >> dst

        # dst.w now has 3 indices materialized (not just an empty multi).
        indices = cmds.getAttr("dst.w", multiIndices=True) or []
        self.assertEqual(sorted(indices), [0, 1, 2])

    def test_clone_compound_multi_mirrors_indices(self):
        src = Node.create("network", name="src")
        src         << Vector("offsets", multi=True)
        src.offsets << [[1, 2, 3], [4, 5, 6]]

        dst = Node.create("network", name="dst")
        src.offsets >> dst

        indices = cmds.getAttr("dst.offsets", multiIndices=True) or []
        self.assertEqual(sorted(indices), [0, 1])

    def test_clone_copies_source_values_as_side_effect(self):
        # Materializing a multi index requires a setAttr -- so values are
        # copied as a side effect. Document this behavior.
        src = Node.create("network", name="src")
        src   << Float("w", multi=True)
        src.w << [10.0, 20.0, 30.0]

        dst = Node.create("network", name="dst")
        src.w >> dst

        for i, expected in enumerate([10.0, 20.0, 30.0]):
            self.assertAlmostEqual(cmds.getAttr(f"dst.w[{i}]"), expected)


# --------------------------------------------------------------------- #
#  End-to-end: chained clone + set
# --------------------------------------------------------------------- #


class TestChainCloneAndSet(MayaTestCase):
    """The motivating use case: clone a multi attr's structure onto a
    target and then write fresh values in one chained expression."""

    TEST_START_NEW_SCENE = True

    def test_clone_then_set_via_slice(self):
        # Source multi.
        src = Node.create("network", name="src")
        src   << Float("w", multi=True)
        src.w << [1.0, 2.0, 3.0]

        dst = Node.create("network", name="dst")

        # Chained clone + value-set. This is the user's wishlist idiom.
        # The ``>>`` clones structure (now also materializes indices);
        # the ``<<`` writes new values via the bare-multi sequence path.
        src.w >> dst << [100.0, 200.0, 300.0]

        for i, expected in enumerate([100.0, 200.0, 300.0]):
            self.assertAlmostEqual(cmds.getAttr(f"dst.w[{i}]"), expected)

    def test_clone_then_set_via_get_snapshot(self):
        # v3.O: PlugList.get() on a multi slice returns numpy/list of
        # values; the bare-multi sequence path then writes element-wise.
        src = Node.create("network", name="src")
        src   << Float("w", multi=True)
        src.w << [1.0, 2.0, 3.0]

        dst = Node.create("network", name="dst")

        dst_w_clone = src.w >> dst
        dst_w_clone << src.w[:].get()

        for i, expected in enumerate([1.0, 2.0, 3.0]):
            self.assertAlmostEqual(cmds.getAttr(f"dst.w[{i}]"), expected)


class TestInjectMultiToMulti(MayaTestCase):
    """v4.F.b: ``multi_dst << multi_src`` connects per-element on the
    source's logical indices (handles sparse indices too).

    Without this branch, the chained ``node.multi >> dst << node.multi``
    pattern would create a spurious extra index on dst because the
    auto-index path would fire for the bare Attribute source.
    """

    TEST_START_NEW_SCENE = True

    def test_multi_to_multi_per_element_connect(self):
        # Source multi with 3 populated indices.
        src = Node.create("network", name="src")
        src   << Float("w", multi=True)
        src.w << [10.0, 20.0, 30.0]

        # Empty destination multi.
        dst = Node.create("network", name="dst")
        dst << Float("w", multi=True)

        # multi -> multi
        dst.w << src.w

        # 3 connections, src.w[i] -> dst.w[i].
        for i in range(3):
            srcs = (
                cmds.listConnections(
                    f"dst.w[{i}]", source=True, destination=False, plugs=True
                )
                or []
            )
            self.assertIn(f"src.w[{i}]", srcs)
            self.assertAlmostEqual(cmds.getAttr(f"dst.w[{i}]"), [10.0, 20.0, 30.0][i])

    def test_clone_then_connect_chain_no_extra_index(self):
        # The motivating bug: ``node.multi >> dst << node.multi`` should
        # produce dst with EXACTLY the same indices as src -- no
        # spurious "auto-append" index from falling through to the
        # auto-index path.
        src = Node.create("network", name="src")
        src        << Vector("output", multi=True)
        src.output << [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

        dst = Node.create("network", name="dst")
        src.output >> dst << src.output

        # No spurious 5th index.
        indices = cmds.getAttr("dst.output", multiIndices=True) or []
        self.assertEqual(sorted(indices), [0, 1, 2, 3])

        # Each dst index has a live connection from the matching src index.
        for i in range(4):
            srcs = (
                cmds.listConnections(
                    f"dst.output[{i}]",
                    source      = True,
                    destination = False,
                    plugs       = True,
                )
                or []
            )
            self.assertIn(f"src.output[{i}]", srcs)

    def test_multi_to_multi_compound_per_element_connect(self):
        # Same flow with compound multi (vectors).
        src = Node.create("network", name="src")
        src         << Vector("offsets", multi=True)
        src.offsets << [[1, 2, 3], [4, 5, 6]]

        dst = Node.create("network", name="dst")
        dst << Vector("offsets", multi=True)

        dst.offsets << src.offsets

        for i, expected in enumerate([[1, 2, 3], [4, 5, 6]]):
            actual = cmds.getAttr(f"dst.offsets[{i}]")[0]
            for a, e in zip(actual, expected):
                self.assertAlmostEqual(a, e)

    def test_multi_to_multi_sparse_src_indices(self):
        # Source has sparse indices [0, 2, 5]; dst should get connections
        # at exactly those indices (preserving sparseness).
        src = Node.create("network", name="src")
        src      << Float("w", multi=True)
        src.w[0] << 10.0
        src.w[2] << 30.0
        src.w[5] << 60.0

        dst = Node.create("network", name="dst")
        dst   << Float("w", multi=True)
        dst.w << src.w

        dst_indices = sorted(cmds.getAttr("dst.w", multiIndices=True) or [])
        self.assertEqual(dst_indices, [0, 2, 5])
        for i, expected in [(0, 10.0), (2, 30.0), (5, 60.0)]:
            self.assertAlmostEqual(cmds.getAttr(f"dst.w[{i}]"), expected)

    def test_multi_dst_with_non_multi_attr_src_still_appends(self):
        # Backwards-compat: a non-multi Attribute source still uses the
        # auto-index path (existing behavior, NOT the new multi-to-multi
        # branch).
        src = Node.create("transform", name="src")
        src.tx << 7.0

        dst = Node.create("network", name="dst")
        dst   << Float("w", multi=True)
        dst.w << src.tx

        # One connection at index 0 (auto-appended, not per-multi-element).
        indices = cmds.getAttr("dst.w", multiIndices=True) or []
        self.assertEqual(sorted(indices), [0])
        srcs = (
            cmds.listConnections("dst.w[0]", source=True, destination=False, plugs=True)
            or []
        )
        self.assertIn("src.translateX", srcs)


class TestMatrixMultiAutoAppend(MayaTestCase):
    """v4.F.b regression: matrix-flavored multi attrs (e.g.
    ``multMatrix.matrixIn``, ``addMatrix.matrixIn``) must use the
    legacy auto-append path so that ``functions/matrix.multiply``'s
    ``for obj in tokens: node.matrixIn << obj`` loop chains correctly.

    These attrs report ``data_type == "compound"`` (the multi container
    type) but ``attribute_type == "matrix"`` (the element type), so the
    bare-multi dispatch must check BOTH to exclude them.

    Without this exclusion, every iteration overwrites ``matrixIn[0]``
    via the multi-to-multi sub-branch (because both src plugs are
    worldMatrix multi roots whose only logical index is [0]).
    """

    TEST_START_NEW_SCENE = True

    def test_multi_matrix_chain_via_loop(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        a.t << [1, 0, 0]
        b.t << [0, 2, 0]

        mm = Node.create("multMatrix", name="mm")
        # Loop pattern used by matrix.multiply / functions:
        for src_plug in (a.wm, b.wm):
            mm.matrixIn << src_plug

        # Both indices should be populated (NOT both writing to [0]).
        srcs0 = (
            cmds.listConnections(
                "mm.matrixIn[0]", source=True, destination=False, plugs=True
            )
            or []
        )
        srcs1 = (
            cmds.listConnections(
                "mm.matrixIn[1]", source=True, destination=False, plugs=True
            )
            or []
        )
        # Maya may store the connection as the indexed form
        # (`a.worldMatrix[0]`) or the multi root (`a.worldMatrix`).
        self.assertTrue(
            any(s.startswith("a.worldMatrix") for s in srcs0),
            f"matrixIn[0] should source from a.worldMatrix, got {srcs0}",
        )
        self.assertTrue(
            any(s.startswith("b.worldMatrix") for s in srcs1),
            f"matrixIn[1] should source from b.worldMatrix, got {srcs1}",
        )

    def test_addMatrix_chain_via_loop(self):
        # Same regression for addMatrix.matrixIn.
        a  = Node.create("transform", name="a")
        b  = Node.create("transform", name="b")

        am = Node.create("addMatrix", name="am")
        for src_plug in (a.wm, b.wm):
            am.matrixIn << src_plug

        # Maya's addMatrix.matrixIn is also indexed; both should be
        # populated.
        indices = cmds.getAttr("am.matrixIn", multiIndices=True) or []
        self.assertEqual(sorted(indices), [0, 1])
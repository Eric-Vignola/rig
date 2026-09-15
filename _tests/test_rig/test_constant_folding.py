"""Tests for the global constant-folding switch and the fold-before-cache
contract.

The DSL's math helpers normally *constant-fold*: given only literal Python
numbers they return a Python value instead of building a Maya node
(``abs(-5.0)`` -> ``5.0``). Setting ``constant_folding=False`` (or entering a
``with force_nodes():`` block) forces EVERY such call to materialize its node
network instead -- a debug / demo aid for inspecting the graph a literal
expression would build.

Folds are computed *before* -- and kept out of -- the ``@memoize`` / ``NodeOp``
caches. That is what makes a runtime flag flip correct:

  * a folded scalar and a force-mode node can never collide under one cache
    key (see :class:`TestFoldBeforeCacheTransitions`), and
  * node-building calls still dedupe in BOTH modes (see
    :class:`TestDedupePreservedInForceMode`).
"""

from unittest import mock

from maya import cmds
from rig import (
    force_nodes,
    functions as f,
    get_options,
    interpolate as interp,
    Node,
    Plug,
    set_options,
    trigonometry as trig,
    vector as vec,
)
from rig._internal.container import ContainerOptions
from rig._internal.memoize import _fold_eligible, memoize
from rig._internal.node_ops import NodeOp, SCOPE_SCALAR
from rig._tests._base import MayaTestCase


def _node_of(plug):
    """Return the node name owning ``plug`` (``'node.attr'`` -> ``'node'``)."""
    return str(plug).split(".")[0]


class _FoldingTestBase(MayaTestCase):
    """Save / restore the process-global ``constant_folding`` flag around every
    test.

    The flag is a process-global on :class:`ContainerOptions`; a force-mode
    test that leaked ``False`` would corrupt every later test in the suite
    (all of which assume the default ``True``). The save/restore here makes a
    leak impossible even when a test body raises.
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self._saved_folding = ContainerOptions.constant_folding

    def tearDown(self):
        ContainerOptions.constant_folding = self._saved_folding
        super().tearDown()


class TestConstantFoldingOption(_FoldingTestBase):
    """``constant_folding`` participates in the ``set_options`` / ``get_options``
    surface like every other global toggle."""

    def test_constant_folding_in_get_options(self):
        self.assertIn("constant_folding", get_options())

    def test_default_is_true(self):
        # Fresh suite default: folding on. (Restored by tearDown if a prior
        # test changed it -- so the saved value seen at setUp is the default.)
        self.assertTrue(self._saved_folding)

    def test_set_false_then_get_reflects(self):
        set_options(constant_folding=False)
        self.assertFalse(ContainerOptions.constant_folding)
        self.assertFalse(get_options()["constant_folding"])

    def test_set_true_then_get_reflects(self):
        set_options(constant_folding=False)
        set_options(constant_folding=True)
        self.assertTrue(ContainerOptions.constant_folding)
        self.assertTrue(get_options()["constant_folding"])

    def test_none_leaves_flag_unchanged(self):
        # Passing the param as None (the default) must NOT touch the flag --
        # so an unrelated set_options call can't accidentally re-enable it.
        set_options(constant_folding=False)
        set_options(use_shorthand=get_options()["use_shorthand"])
        self.assertFalse(ContainerOptions.constant_folding)

    def test_flip_does_not_clear_caches(self):
        # Unlike maya_version, flipping constant_folding must NOT clear the
        # memoize / NodeOp caches -- folds are never cached, so there is
        # nothing version-stale to invalidate.
        with mock.patch(
            "rig._internal.memoize._clear_all_caches"
        ) as mock_clear:
            set_options(constant_folding=False)
            set_options(constant_folding=True)
        mock_clear.assert_not_called()


class TestForceNodesContextManager(_FoldingTestBase):
    """``force_nodes()`` flips the flag off for the duration of a ``with``
    block and restores the prior value on exit -- including on exception and
    when nested."""

    def test_returns_a_context_manager(self):
        cm = force_nodes()
        self.assertTrue(hasattr(cm, "__enter__"))
        self.assertTrue(hasattr(cm, "__exit__"))

    def test_flag_false_inside_restored_after(self):
        prior = ContainerOptions.constant_folding
        with force_nodes():
            self.assertFalse(ContainerOptions.constant_folding)
        self.assertEqual(ContainerOptions.constant_folding, prior)

    def test_nested_blocks_restore_to_prior(self):
        prior = ContainerOptions.constant_folding
        with force_nodes():
            self.assertFalse(ContainerOptions.constant_folding)
            with force_nodes():
                self.assertFalse(ContainerOptions.constant_folding)
            # Inner exit restores to the OUTER block's value (still False),
            # not a hard True.
            self.assertFalse(ContainerOptions.constant_folding)
        self.assertEqual(ContainerOptions.constant_folding, prior)

    def test_restores_on_exception(self):
        prior = ContainerOptions.constant_folding
        with self.assertRaises(ValueError):
            with force_nodes():
                self.assertFalse(ContainerOptions.constant_folding)
                raise ValueError("boom")
        # __exit__ ran (restored) AND did not suppress the exception.
        self.assertEqual(ContainerOptions.constant_folding, prior)

    def test_composes_with_sticky_false(self):
        # An outer sticky set_options(False) must be the value restored on
        # block exit -- not the global default.
        set_options(constant_folding=False)
        with force_nodes():
            self.assertFalse(ContainerOptions.constant_folding)
        self.assertFalse(ContainerOptions.constant_folding)


class TestFoldingModeReturnsScalars(_FoldingTestBase):
    """Characterization: in the default mode, all-literal calls fold to plain
    Python values across every foldable submodule."""

    def test_functions_scalar_folds(self):
        self.assertEqual(f.abs(-5.0),            5.0)
        self.assertEqual(f.clamp(5.0, 0.0, 1.0), 1.0)
        self.assertEqual(f.pow(2.0, 3.0),        8.0)
        self.assertEqual(f.sqrt(9.0),            3.0)
        for result in (f.abs(-5.0), f.clamp(5.0, 0.0, 1.0), f.pow(2.0, 3.0)):
            self.assertNotIsInstance(result, Plug)

    def test_functions_reduce_folds(self):
        self.assertEqual(f.sum([1.0, 2.0, 3.0]), 6.0)
        self.assertEqual(f.avg([2.0, 4.0]),      3.0)
        self.assertEqual(f.max([1.0, 5.0, 3.0]), 5.0)
        self.assertEqual(f.min([1.0, 5.0, 3.0]), 1.0)

    def test_trig_folds(self):
        self.assertAlmostEqual(trig.sind(90.0),      1.0)
        self.assertAlmostEqual(trig.cosd(0.0),       1.0)
        self.assertAlmostEqual(trig.atan2(1.0, 1.0), 0.7853981633974483)

    def test_interpolate_folds(self):
        self.assertEqual(interp.smoothstep(0.0, 1.0, 0.5), 0.5)
        self.assertEqual(interp.inverse_lerp(0.0, 10.0, 5.0), 0.5)

    def test_vector_folds(self):
        self.assertEqual(vec.lerp(0.0, 10.0, 0.5), 5.0)
        self.assertAlmostEqual(vec.elerp(2.0, 8.0, 0.5), 4.0)
        self.assertEqual(vec.length([3.0, 4.0, 0.0]), 5.0)


class TestForceModeBuildsNodes(_FoldingTestBase):
    """In force mode, the same all-literal calls materialize a Maya node and
    return a :class:`Plug` instead of a Python number."""

    def test_functions_scalar_build_nodes(self):
        with force_nodes():
            self.assertIsInstance(f.abs(-5.0),            Plug)
            self.assertIsInstance(f.clamp(5.0, 0.0, 1.0), Plug)
            self.assertIsInstance(f.pow(2.0, 3.0),        Plug)
            self.assertIsInstance(f.sqrt(9.0),            Plug)

    def test_functions_reduce_build_nodes(self):
        with force_nodes():
            self.assertIsInstance(f.sum([1.0, 2.0, 3.0]), Plug)
            self.assertIsInstance(f.max([1.0, 5.0, 3.0]), Plug)
            self.assertIsInstance(f.min([1.0, 5.0, 3.0]), Plug)

    def test_trig_build_nodes(self):
        with force_nodes():
            self.assertIsInstance(trig.sind(90.0), Plug)
            self.assertIsInstance(trig.atan2(1.0, 1.0), Plug)

    def test_interpolate_build_nodes(self):
        with force_nodes():
            self.assertIsInstance(interp.smoothstep(0.0, 1.0, 0.5), Plug)
            self.assertIsInstance(interp.inverse_lerp(0.0, 10.0, 5.0), Plug)

    def test_vector_build_nodes(self):
        with force_nodes():
            self.assertIsInstance(vec.lerp(0.0, 10.0, 0.5),    Plug)
            self.assertIsInstance(vec.elerp(2.0, 8.0, 0.5),    Plug)
            self.assertIsInstance(vec.length([3.0, 4.0, 0.0]), Plug)

    def test_sticky_set_options_also_builds(self):
        # The non-context-manager form drives the same gate.
        set_options(constant_folding=False)
        self.assertIsInstance(f.abs(-5.0), Plug)


class TestDedupePreservedInForceMode(_FoldingTestBase):
    """The memoize contract holds in force mode: identical calls return the
    SAME node network rather than building duplicates."""

    def test_scalar_identical_calls_dedupe(self):
        with force_nodes():
            p1 = f.abs(-5.0)
            p2 = f.abs(-5.0)
        self.assertIsInstance(p1, Plug)
        self.assertEqual(_node_of(p1), _node_of(p2))

    def test_node_count_proves_single_build(self):
        before = len(cmds.ls(type="absolute") or [])
        with force_nodes():
            f.abs(-5.0)
            f.abs(-5.0)
            f.abs(-5.0)
        after = len(cmds.ls(type="absolute") or [])
        self.assertEqual(after - before, 1)

    def test_reduce_identical_calls_dedupe(self):
        with force_nodes():
            p1 = f.sum([1.0, 2.0, 3.0])
            p2 = f.sum([1.0, 2.0, 3.0])
        self.assertEqual(_node_of(p1), _node_of(p2))

    def test_length_identical_calls_dedupe(self):
        with force_nodes():
            p1 = vec.length([3.0, 4.0, 0.0])
            p2 = vec.length([3.0, 4.0, 0.0])
        self.assertEqual(_node_of(p1), _node_of(p2))


class TestFoldBeforeCacheTransitions(_FoldingTestBase):
    """Folding before the cache lookup is what makes a runtime flag flip safe:
    a fold result never reads (or poisons) a node cached under the same key."""

    def test_fold_then_force_yields_a_node(self):
        # Folding-mode scalar must NOT shadow a later force-mode build.
        scalar = f.abs(-5.0)
        self.assertEqual(scalar, 5.0)
        self.assertNotIsInstance(scalar, Plug)

        with force_nodes():
            plug = f.abs(-5.0)
        self.assertIsInstance(plug, Plug)

    def test_force_then_fold_yields_a_scalar(self):
        # A force-mode node must NOT shadow a later fold of the same inputs.
        with force_nodes():
            plug = f.abs(-5.0)
            self.assertIsInstance(plug, Plug)

        scalar = f.abs(-5.0)
        self.assertEqual(scalar, 5.0)
        self.assertNotIsInstance(scalar, Plug)

    def test_reduce_round_trips_both_directions(self):
        self.assertEqual(f.sum([1.0, 2.0, 3.0]), 6.0)
        with force_nodes():
            self.assertIsInstance(f.sum([1.0, 2.0, 3.0]), Plug)
        self.assertEqual(f.sum([1.0, 2.0, 3.0]), 6.0)


class TestFoldEligible(_FoldingTestBase):
    """Unit coverage of the ``_fold_eligible`` predicate dispatcher."""

    def test_scalar_marker(self):
        self.assertTrue(_fold_eligible("scalar", (1, 2.0, 3), {}))
        self.assertTrue(_fold_eligible("scalar", (5,), {"y": 2.0}))
        self.assertFalse(_fold_eligible("scalar", ([1, 2, 3],), {}))
        self.assertFalse(_fold_eligible("scalar", ("str",), {}))
        self.assertFalse(_fold_eligible("scalar", (1.0,), {"k": "x"}))

    def test_reduce_marker(self):
        self.assertTrue(_fold_eligible("reduce", ([1.0, 2.0, 3.0],), {}))
        self.assertFalse(_fold_eligible("reduce", ([1.0, "x"],), {}))
        self.assertFalse(_fold_eligible("reduce", (1.0, 2.0), {}))  # 2 args
        self.assertFalse(_fold_eligible("reduce", (5.0,), {}))      # not a sequence
        self.assertFalse(_fold_eligible("reduce", ([1.0],), {"k": "x"}))

    def test_callable_marker(self):
        pred = lambda a, k: a[0] == 42
        self.assertTrue(_fold_eligible(pred, (42,), {}))
        self.assertFalse(_fold_eligible(pred, (7,), {}))

    def test_unknown_marker_is_false(self):
        self.assertFalse(_fold_eligible(False, (1.0,), {}))
        self.assertFalse(_fold_eligible("bogus", (1.0,), {}))


class TestMemoizeFoldableMechanics(_FoldingTestBase):
    """White-box tests of the ``@memoize(foldable=...)`` fold-before-cache
    wrapper using synthetic functions (no Maya nodes needed)."""

    def test_scalar_fold_recomputes_and_never_caches(self):
        calls = {"n": 0}

        @memoize(foldable="scalar")
        def fn(x):
            calls["n"] += 1
            return x * 2

        self.assertEqual(fn(5), 10)
        self.assertEqual(fn(5), 10)
        # Folding mode: recomputed every call, cache stays empty.
        self.assertEqual(calls["n"], 2)
        self.assertEqual(len(fn._cache), 0)

    def test_scalar_cached_in_force_mode(self):
        calls = {"n": 0}

        @memoize(foldable="scalar")
        def fn(x):
            calls["n"] += 1
            return x * 2

        with force_nodes():
            self.assertEqual(fn(5), 10)
            self.assertEqual(fn(5), 10)
        # Force mode: the fold bypass is skipped, so the result is cached.
        self.assertEqual(calls["n"], 1)
        self.assertEqual(len(fn._cache), 1)

    def test_reduce_fold_recomputes_and_never_caches(self):
        calls = {"n": 0}

        @memoize(foldable="reduce")
        def fn(xs):
            calls["n"] += 1
            return sum(xs)

        self.assertEqual(fn([1, 2, 3]),  6)
        self.assertEqual(fn([1, 2, 3]),  6)
        self.assertEqual(calls["n"],     2)
        self.assertEqual(len(fn._cache), 0)

    def test_callable_predicate_selects_fold_vs_cache(self):
        calls = {"n": 0}

        @memoize(foldable=lambda a, k: isinstance(a[0], float))
        def fn(x):
            calls["n"] += 1
            return x

        # float -> predicate True -> folded (uncached), recompute each call.
        fn(1.0)
        fn(1.0)
        self.assertEqual(calls["n"], 2)

        # int -> predicate False -> cached.
        calls["n"] = 0
        fn(2)
        fn(2)
        self.assertEqual(calls["n"], 1)

    def test_non_foldable_caches_even_with_literals(self):
        calls = {"n": 0}

        @memoize  # bare -> foldable defaults False
        def fn(x):
            calls["n"] += 1
            return x * 2

        self.assertEqual(fn(5),        10)
        self.assertEqual(fn(5),        10)
        self.assertEqual(calls["n"],   1)
        self.assertEqual(fn._foldable, False)

    def test_foldable_attribute_exposed(self):
        @memoize(foldable="scalar")
        def fn(x):
            return x

        self.assertEqual(fn._foldable, "scalar")

    def test_both_decorator_syntaxes(self):
        @memoize
        def a(x):
            return x

        @memoize()
        def b(x):
            return x

        @memoize(foldable="scalar")
        def c(x):
            return x

        self.assertEqual(a._foldable, False)
        self.assertEqual(b._foldable, False)
        self.assertEqual(c._foldable, "scalar")


class TestNodeOpForceMode(_FoldingTestBase):
    """The ``NodeOp`` all-numeric short-circuit is gated on the same flag so a
    ``scalar_fn`` op also materializes nodes under ``force_nodes()``."""

    def test_scalar_fn_folds_in_default_mode(self):
        op = NodeOp("foldtest_default", scalar_fn=lambda a, b: a + b)

        @op.impl(since=0, scope=SCOPE_SCALAR)
        def _impl(a, b):
            return "node-built"

        self.assertEqual(op(2, 3), 5)

    def test_scalar_fn_skipped_in_force_mode(self):
        op = NodeOp("foldtest_force", scalar_fn=lambda a, b: a + b)

        @op.impl(since=0, scope=SCOPE_SCALAR)
        def _impl(a, b):
            return "node-built"

        with force_nodes():
            self.assertEqual(op(2, 3), "node-built")

    def test_real_op_via_plug_unaffected(self):
        # Sanity: a NodeOp-backed operator with a genuine Plug input always
        # builds, regardless of the flag.
        node   = Node.create("transform", name="cube1")
        result = node.tx + node.ty
        self.assertIsInstance(result, Plug)
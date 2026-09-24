"""Tests for lock / connection-aware injection semantics of ``<<``.

Covers P0-1: ``<<`` asserts the destination's desired state.

* A **locked** attribute is an inviolable barrier -- setting a value or
  connecting into it raises :class:`InjectionError`, and a failed connect
  must NOT destroy an existing connection.
* An unlocked but **connected** attribute is overwritten: setting a value
  disconnects the incoming connection first, then sets (no raise).
* ``<< None`` disconnects -- allowed even on a locked attribute.
* Compound sets are **all-or-nothing**: if any targeted child is locked,
  the whole set raises and no channel is modified.
* Genuine failures (type mismatch, ...) raise instead of silently no-op.
"""

from unittest import mock

from maya import cmds
from rig.nodetypes import Attribute
from rig import InjectionError, Node
from rig._internal import plug as plugmod
from rig._tests._base import MayaTestCase


def _incoming(plug_name):
    return cmds.listConnections(plug_name, s=True, d=False, plugs=True) or []


def _setattr_raising_on_lock(lock_value):
    """A ``cmds.setAttr`` stand-in that raises when called with
    ``lock=<lock_value>`` and delegates to the real one otherwise.

    The real ``cmds.setAttr`` is captured here, at build time -- the caller
    invokes this builder before installing the fake via ``mock.patch``, so the
    delegation does not recurse through the patched ``plug.cmds.setAttr``
    (``plug.cmds`` and this module's ``cmds`` are the same module object).
    Capturing lazily (rather than at module import) also keeps import safe
    before ``maya.standalone`` is initialized during test discovery.
    """

    real_setattr = cmds.setAttr

    def _fake(*args, **kwargs):
        if kwargs.get("lock") is lock_value:
            raise RuntimeError(f"simulated setAttr(lock={lock_value}) failure")
        return real_setattr(*args, **kwargs)

    return _fake


class TestInjectLockedRaises(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_set_value_on_locked_scalar_raises(self):
        n = Node.create("transform", name="c1")
        n.tx << 5
        cmds.setAttr("c1.translateX", lock=True)
        with self.assertRaises(InjectionError):
            n.tx << 10

    def test_set_value_on_locked_leaves_value_unchanged(self):
        n = Node.create("transform", name="c1")
        n.tx << 5
        cmds.setAttr("c1.translateX", lock=True)
        try:
            n.tx << 10
        except InjectionError:
            pass
        self.assertEqual(cmds.getAttr("c1.translateX"), 5.0)

    def test_connect_into_locked_raises(self):
        n   = Node.create("transform", name="c1")
        drv = Node.create("transform", name="drv")
        cmds.setAttr("c1.translateX", lock=True)
        with self.assertRaises(InjectionError):
            n.tx << drv.tx

    def test_connect_into_locked_preserves_existing_connection(self):
        # Regression: the old code disconnected the existing driver *first*,
        # then failed to connect the new (locked) one -> left the attr with
        # neither. Lock must be checked before any mutation.
        n     = Node.create("transform", name="c1")
        drv   = Node.create("transform", name="drv")
        other = Node.create("transform", name="other")
        n.tx << drv.tx
        cmds.setAttr("c1.translateX", lock=True)
        try:
            n.tx << other.tx
        except InjectionError:
            pass
        self.assertEqual(_incoming("c1.translateX"), ["drv.translateX"])


class TestInjectConnectedOverwrites(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_set_value_on_connected_disconnects_and_sets(self):
        n   = Node.create("transform", name="c1")
        drv = Node.create("transform", name="drv")
        n.tx << drv.tx
        self.assertEqual(_incoming("c1.translateX"), ["drv.translateX"])
        n.tx << 7
        self.assertEqual(cmds.getAttr("c1.translateX"), 7.0)
        self.assertEqual(_incoming("c1.translateX"), [])

    def test_connect_new_source_replaces_existing(self):
        n = Node.create("transform", name="c1")
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        n.tx << a.tx
        n.tx << b.tx
        self.assertEqual(_incoming("c1.translateX"), ["b.translateX"])


class TestInjectDisconnect(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_disconnect_none_removes_connection(self):
        n   = Node.create("transform", name="c1")
        drv = Node.create("transform", name="drv")
        n.tx << drv.tx
        n.tx << None
        self.assertEqual(_incoming("c1.translateX"), [])

    def test_disconnect_none_allowed_on_locked(self):
        # Decision: lock blocks set/connect, but NOT disconnect.
        n   = Node.create("transform", name="c1")
        drv = Node.create("transform", name="drv")
        n.tx << drv.tx
        cmds.setAttr("c1.translateX", lock=True)
        n.tx << None  # must not raise
        self.assertEqual(_incoming("c1.translateX"), [])


class TestInjectCompoundAllOrNothing(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_compound_set_with_one_locked_child_raises(self):
        n = Node.create("transform", name="c1")
        n.t << [0, 0, 0]
        cmds.setAttr("c1.translateY", lock=True)
        with self.assertRaises(InjectionError):
            n.t << [1, 2, 3]

    def test_compound_set_with_one_locked_child_leaves_all_unchanged(self):
        n = Node.create("transform", name="c1")
        n.t << [0, 0, 0]
        cmds.setAttr("c1.translateY", lock=True)
        try:
            n.t << [1, 2, 3]
        except InjectionError:
            pass
        self.assertEqual(list(cmds.getAttr("c1.translate")[0]), [0.0, 0.0, 0.0])


class TestInjectInheritedLockDisconnect(MayaTestCase):
    """``<< None`` disconnects an effective-locked plug regardless of WHERE the
    lock lives -- on the leaf's own flag or inherited from a locked ancestor
    compound -- honoring the "disconnect is allowed even on a locked attribute"
    contract uniformly. The full ancestor chain is unlocked for the removal and
    then restored exactly: an inherited lock is never converted into an explicit
    own-lock, at any level.
    """

    TEST_START_NEW_SCENE = True

    def test_disconnect_under_locked_parent_succeeds_without_leak(self):
        # The lock is INHERITED from the locked parent compound (the child has
        # no own-lock). ``<< None`` must still disconnect -- the parent is
        # unlocked for the removal and re-locked -- and the child must NOT pick
        # up an explicit own-lock (no topology leak). This mirrors the own-lock
        # case: the caller never has to care where the lock physically sits.
        n   = Node.create("transform", name="c1")
        drv = Node.create("transform", name="drv")
        n.tx << drv.tx
        cmds.setAttr("c1.translate", lock=True)  # lock the PARENT compound only
        n.tx << None  # must SUCCEED, not raise
        self.assertEqual(_incoming("c1.translateX"), [])  # driver removed
        self.assertTrue(
            cmds.getAttr("c1.translate", lock=True),
            "parent compound lock must be restored after the disconnect",
        )
        # No leak: with the parent unlocked, translateX carries no own lock.
        cmds.setAttr("c1.translate", lock=False)
        self.assertFalse(
            cmds.getAttr("c1.translateX", lock=True),
            "inherited parent-lock must not be converted into a child own-lock",
        )

    def test_disconnect_under_locked_grandparent_succeeds_without_leak(self):
        # Multi-level: the lock lives on a GRANDPARENT compound. The whole
        # ancestor chain (leaf -> par -> grand) is unlocked for the removal and
        # restored, so the disconnect succeeds with no own-lock leak anywhere.
        host = Node.create("transform", name="host")
        drv  = Node.create("transform", name="drv")
        cmds.addAttr("host", ln="grand", at="compound", nc=1)
        cmds.addAttr("host", ln="par", at="compound", nc=1, p="grand")
        cmds.addAttr("host", ln="leaf", at="double", p="par")
        cmds.connectAttr("drv.translateX", "host.leaf")
        cmds.setAttr("host.grand", lock=True)  # lock the GRANDPARENT
        host.leaf << None  # must SUCCEED
        self.assertEqual(_incoming("host.leaf"), [])
        self.assertTrue(
            cmds.getAttr("host.grand", lock=True), "grandparent lock restored"
        )
        cmds.setAttr("host.grand", lock=False)
        self.assertFalse(cmds.getAttr("host.leaf", lock=True), "no leak on leaf")
        self.assertFalse(cmds.getAttr("host.par", lock=True), "no leak mid-chain")

    def test_disconnect_with_own_and_inherited_lock_restores_both(self):
        # Mixed: the leaf has its OWN lock AND the parent compound is locked.
        # Both are unlocked for the removal; afterward the leaf's own lock is
        # restored (it was the user's intent) and no NEW own-lock leaks onto an
        # ancestor.
        n   = Node.create("transform", name="c1")
        drv = Node.create("transform", name="drv")
        n.tx << drv.tx
        cmds.setAttr("c1.translateX", lock=True)  # OWN lock on the leaf
        cmds.setAttr("c1.translate", lock=True)   # AND lock the parent
        n.tx << None  # must SUCCEED
        self.assertEqual(_incoming("c1.translateX"), [])
        self.assertTrue(cmds.getAttr("c1.translate", lock=True), "parent restored")
        cmds.setAttr("c1.translate", lock=False)
        self.assertTrue(
            cmds.getAttr("c1.translateX", lock=True),
            "the leaf's OWN lock must be preserved across the disconnect",
        )


class TestInjectGenuineErrorRaises(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_set_incompatible_type_raises(self):
        n = Node.create("transform", name="c1")
        with self.assertRaises(InjectionError):
            n.tx << "not_a_number"


class TestInjectHappyPathUnchanged(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_plain_set_unlocked_unconnected(self):
        n = Node.create("transform", name="c1")
        n.tx << 3
        self.assertEqual(cmds.getAttr("c1.translateX"), 3.0)

    def test_plain_connect_unlocked(self):
        n   = Node.create("transform", name="c1")
        drv = Node.create("transform", name="drv")
        n.tx << drv.tx
        self.assertEqual(_incoming("c1.translateX"), ["drv.translateX"])

    def test_compound_set_unlocked(self):
        n = Node.create("transform", name="c1")
        n.t << [1, 2, 3]
        self.assertEqual(list(cmds.getAttr("c1.translate")[0]), [1.0, 2.0, 3.0])


class TestInjectLockDefensiveCoverage(MayaTestCase):
    """Coverage for the defensive / error branches of the lock-aware
    disconnect and set helpers in ``plug.py`` -- the paths that real Maya
    almost never triggers and so are exercised here with mocks or direct
    helper calls."""

    TEST_START_NEW_SCENE = True

    def test_disconnect_own_locked_multi_element_succeeds(self):
        # The lock-chain walk handles array ELEMENTS (``input1D[0]``) via the
        # ``isElement -> array()`` branch: an own-locked multi element is
        # unlocked, disconnected, and re-locked.
        pma = Node.create("plusMinusAverage", name="pma")
        drv = Node.create("transform", name="drv")
        pma.input1D[0] << drv.tx
        cmds.setAttr("pma.input1D[0]", lock=True)  # OWN lock on the element
        pma.input1D[0] << None  # disconnect must succeed
        self.assertEqual(_incoming("pma.input1D[0]"), [])
        self.assertTrue(
            cmds.getAttr("pma.input1D[0]", lock=True),
            "the element's own lock must be restored after the disconnect",
        )

    def test_disconnect_source_failure_raises_injection_error(self):
        # _disconnect_sources_respecting_lock: a driver-removal that fails for a
        # non-lock reason surfaces as InjectionError (the ``src_attr.disconnect``
        # ``except RuntimeError`` branch).
        n   = Node.create("transform", name="c1")
        drv = Node.create("transform", name="drv")
        n.tx << drv.tx
        with mock.patch.object(
            Attribute, "disconnect", side_effect=RuntimeError("boom")
        ):
            with self.assertRaises(InjectionError):
                n.tx << None

    def test_disconnect_incoming_wraps_inspect_failure(self):
        # _disconnect_incoming: get_connected_attrs raising -> InjectionError.
        n   = Node.create("transform", name="c1")
        drv = Node.create("transform", name="drv")
        n.tx << drv.tx
        with mock.patch.object(
            Attribute, "get_connected_attrs", side_effect=RuntimeError("boom")
        ):
            with self.assertRaises(InjectionError):
                plugmod._disconnect_incoming(n.tx)

    def test_unlock_failure_raises_injection_error(self):
        # _disconnect_sources_respecting_lock: a guarded unlock that fails
        # surfaces as InjectionError ('Cannot unlock ...').
        n   = Node.create("transform", name="c1")
        drv = Node.create("transform", name="drv")
        n.tx << drv.tx
        cmds.setAttr("c1.translateX", lock=True)  # OWN lock
        with mock.patch.object(
            plugmod.cmds, "setAttr", side_effect=_setattr_raising_on_lock(False)
        ):
            with self.assertRaises(InjectionError):
                n.tx << None

    def test_relock_failure_is_logged_not_raised(self):
        # _disconnect_sources_respecting_lock: a re-lock that fails in the
        # finally is logged (LOGGER.error) and must NOT raise or mask -- the
        # disconnect already succeeded.
        n   = Node.create("transform", name="c1")
        drv = Node.create("transform", name="drv")
        n.tx << drv.tx
        cmds.setAttr("c1.translateX", lock=True)  # OWN lock
        with mock.patch.object(
            plugmod.cmds, "setAttr", side_effect=_setattr_raising_on_lock(True)
        ):
            n.tx << None  # must NOT raise despite the re-lock failure
        self.assertEqual(_incoming("c1.translateX"), [])

    def test_locked_channels_non_attribute_returns_empty(self):
        # _locked_channels: a non-Attribute dst -> [].
        self.assertEqual(plugmod._locked_channels("not_an_attribute"), [])

    def test_locked_channels_skips_uncoercible_leaf(self):
        # _locked_channels: a compound leaf that isn't an Attribute and can't
        # be coerced is skipped (``except Exception: continue``).
        import rig._internal.types as typesmod

        n = Node.create("transform", name="c1")
        with mock.patch.object(typesmod, "_get_compound", return_value=["bogus.nope"]):
            self.assertEqual(plugmod._locked_channels(n.translate), [])

    def test_locked_channels_swallows_is_locked_error(self):
        # _locked_channels: a leaf whose is_locked query raises is skipped
        # (``except Exception: pass``).
        n = Node.create("transform", name="c1")
        with mock.patch.object(
            type(n.translateX),
            "is_locked",
            new_callable = mock.PropertyMock,
            side_effect  = RuntimeError("boom"),
        ):
            self.assertEqual(plugmod._locked_channels(n.translate), [])

    def test_do_set_swallows_connection_probe_error(self):
        # _do_set: when the post-failure connection probe itself raises, it is
        # swallowed (incoming=[]) and the original set error is reported.
        n = Node.create("transform", name="c1")
        with mock.patch.object(
            Attribute, "get_connected_attrs", side_effect=RuntimeError("boom")
        ):
            with self.assertRaises(InjectionError):
                n.tx << "not_a_number"

    def test_do_set_second_set_failure_raises(self):
        # _do_set: a set that still fails AFTER breaking an incoming connection
        # raises InjectionError (the retry ``except ... as second`` branch).
        n   = Node.create("transform", name="c1")
        drv = Node.create("transform", name="drv")
        n.tx << drv.tx  # establish an incoming connection
        with self.assertRaises(InjectionError):
            n.tx << "not_a_number"  # first set fails (connected), retry also fails
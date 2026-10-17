"""Round 5 D2: connects through the API (`rig._internal.api_edits.api_connect`).

A connect into a free destination is an ``MDGModifier``, journaled: a statement's connects
are one undo step. Every other connect, and every connect whose guard fails, takes the cmds
route, which raises today's errors. The approved differences (user, 2026-10-02): one undo
step per statement for its API edits, the DG order in the .ma after undo-all then redo-all,
and one extra Maya ``Error:`` line on an incompatible-type connect.

* `TestTheRoute`: which connects take the API (V10: no ``cmds.connectAttr`` for
  ``a.tx << b.ty``), and that every guard case gives the cmds route's outcome (the API on and
  off in the same process: exception, chain, connections, values).
* `TestStatementGranularity` (V6, with the CR-13 dangling check after every undo / redo step):
  a statement's connects are one step; the steps and states between statements, user cmds
  interleaved, chunks, a raising List connect, CR-5 with a real connect, queue eviction.
* `TestReferenceEdits`: a connect on a referenced node records the same reference edits on
  both routes, and its undo removes them.
* `TestPluginLoading` (user decision U1, U2): rig's plug-in loads at the first API connect
  (by full path, else by name); when it cannot load, connects take cmds with one warning per
  session and the edits with no other undoable route raise a clear error; an unload is refused
  until flushUndo.
* `TestTheBuild` (V10): rail_spine's registrations per build.
"""

import contextlib
import gc
import os
import shutil
import sys
import tempfile
import weakref
from unittest import mock

from maya import cmds
from maya.api import OpenMaya

import rig
from rig import List, Node, get_options, set_options
from rig._internal import api_edits, journal
from rig._internal.plug import Plug
from rig.nodetypes import _base, plugins
from rig.nodetypes._base import Attribute
from rig.nodetypes.dg_node import DGNode
from rig.nodetypes.material_node import Lambert
from rig.nodetypes.mesh import Mesh
from rig.nodetypes.plugins import UNDOABLE_API_COMMAND, bundled_plugin_path
from rig.nodetypes.transform import Transform
from rig.spec._base import _AttrSpec
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk

PLUGIN = "undoable_api_command"
PLUGIN_PATH = bundled_plugin_path(PLUGIN)


class _ApiCase(UndoWalk, MayaTestCase):
    """A new scene, the undo queue on, infinite and flushed, rig's plug-in loaded; the
    transforms ``a``, ``b``, ``c`` and the multiplyDivides ``m``, ``k``. The journal is
    closed and empty after every test."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        plugins._ensure_loaded(PLUGIN, True)
        self.defaults = set(cmds.ls())
        self.make_nodes()
        cmds.flushUndo()

    def make_nodes(self):
        for name in ("a", "b", "c"):
            cmds.createNode("transform", name=name)
        for name in ("m", "k"):
            cmds.createNode("multiplyDivide", name=name)

    def tearDown(self):
        try:
            self.assertEqual((journal._S.open, journal._S.pending), (False, []))
        finally:
            journal._S.open, journal._S.pending = False, []
            cmds.flushUndo()
            plugins._ensure_loaded(PLUGIN, True)
            super().tearDown()

    def queue_empty(self):
        return cmds.undoInfo(query=True, undoQueueEmpty=True)

    def count_steps(self):
        """Undoes the whole queue (the dangling check after each step), redoes it, and
        returns the number of steps."""
        steps = 0
        while not self.queue_empty():
            self.undo_steps(1)
            steps += 1
            self.assertLess(steps, 1000)
        self.redo_all()
        return steps

    def connections(self):
        """The (src, dst) connections into the nodes a test made (not a new scene's)."""
        found = []
        for node in cmds.ls():
            if node in self.defaults:
                continue
            found.extend(cmds.listConnections(node, connections=True, plugs=True, source=True,
                                              destination=False) or [])
        return sorted(zip(found[1::2], found[::2]))


def _brief(error, depth=0):
    """An exception and its chain, as comparable data (type, text, __suppress_context__)."""
    if error is None or depth > 5:
        return None
    return (type(error).__name__, str(error), error.__suppress_context__,
            _brief(error.__context__, depth + 1), _brief(error.__cause__, depth + 1))


class _UserPlug(Plug):
    """A user's Plug subclass: not an exact rig class, so the cmds route."""


class _UserNamed(Transform):
    """A user's node class with its own naming: the cmds route. Re-pinned (user
    decision 2026-10-16, the property ban): a node class declares no property, so
    its own naming is its ``__str__``."""

    def __str__(self):
        return self.get_fn_set().partialPathName()


# (label, setup(), the connect, the route the API_EDITS=True run takes: "api" or "cmds")
def _guard_cases():
    def deleted_dst():
        p = Node("b").ty
        cmds.delete("b")
        Node("a").tx.connect(p)

    def deleted_src():
        p = Node("m").outputX
        cmds.delete("m")
        p.connect(Node("k").input1X)

    def freed_dst():
        p = Node("k").input1X
        cmds.file(new=True, force=True)
        cmds.createNode("multiplyDivide", name="m")
        Node("m").outputX.connect(p)

    def deleted_dynamic():
        cmds.addAttr("k", longName="dyn", attributeType="double")
        p = Node("k").dyn
        cmds.deleteAttr("k.dyn")
        Node("m").outputX.connect(p)

    def non_readable_src():
        cmds.addAttr("m", longName="nr", attributeType="double", readable=False)
        Node("m").nr.connect(Node("k").input1X)

    def non_writable_dst():
        cmds.addAttr("k", longName="nw", attributeType="double", writable=False)
        Node("m").outputX.connect(Node("k").nw)

    def locked_array_root_element():
        cmds.createNode("plusMinusAverage", name="p")
        cmds.setAttr("p.input1D", lock=True)
        Node("m").outputX.connect(Node("p").input1D[0])

    def missing_element():
        cmds.createNode("plusMinusAverage", name="p")
        Node("m").outputX.connect(Node("p").input1D[7])

    def lock_node(name):
        def case():
            cmds.lockNode(name, lock=True)
            try:
                Node("a").tx.connect(Node("b").ty)
            finally:
                cmds.lockNode(name, lock=False)
        return case

    def connected_dst():
        Node("a").tx.connect(Node("b").ty)
        Node("a").tz.connect(Node("b").ty)

    def force_connected_dst():
        Node("a").tx.connect(Node("b").ty)
        Node("a").tz.connect(Node("b").ty, force=True)

    def connected_parent():
        cmds.connectAttr("a.t", "b.t")
        Node("m").outputX.connect(Node("b").tx)

    def connected_child():
        cmds.connectAttr("a.tx", "b.ty")
        Node("a").translate.connect(Node("b").translate)

    def locked_parent():
        cmds.setAttr("b.t", lock=True)
        Node("a").tx.connect(Node("b").ty)

    def array_root():
        cmds.createNode("plusMinusAverage", name="p")
        cmds.createNode("plusMinusAverage", name="q")
        Node("p").input1D.connect(Node("q").input1D)

    def user_plug():
        _UserPlug("a.tx").connect(Node("b").ty)

    def user_naming():
        _UserNamed("a").tx.connect(Node("b").ty)

    def alias_dst():
        cmds.aliasAttr("hey", "k.input1X")
        Node("m").outputX.connect(Node("k").hey)

    def message():
        cmds.addAttr("k", longName="msgIn", attributeType="message")
        Node("m").message.connect(Node("k").msgIn)

    def unit_conversion():
        Node("m").outputX.connect(Node("a").rx)

    def converted_material():
        red = Lambert.create(name="red")
        p = red.diffuse
        red.astype("phong")
        Node("m").outputX.connect(p)

    return [
        ("free", lambda: Node("a").tx.connect(Node("b").ty), "api"),
        ("compound", lambda: Node("a").t.connect(Node("b").t), "api"),
        ("rshift", lambda: Node("a").tx >> Node("b").ty, "api"),
        ("missing element", missing_element, "api"),
        ("alias dst", alias_dst, "api"),
        ("message", message, "api"),
        ("unit conversion", unit_conversion, "api"),
        ("lockNode on the dst's node", lock_node("b"), "api"),
        ("lockNode on the src's node", lock_node("a"), "api"),
        ("a str dst", lambda: Node("a").tx.connect("b.ty"), "cmds"),
        ("self", lambda: Node("a").tx.connect(Node("a").tx), "cmds"),
        ("child into its parent", lambda: Node("a").tx.connect(Node("a").translate), "cmds"),
        ("same node", lambda: Node("a").tx.connect(Node("a").ty), "cmds"),
        ("deleted dst", deleted_dst, "cmds"),
        ("deleted src", deleted_src, "cmds"),
        ("freed dst", freed_dst, "cmds"),
        ("deleted dynamic attr", deleted_dynamic, "cmds"),
        ("non-readable src", non_readable_src, "cmds"),
        ("non-writable dst", non_writable_dst, "cmds"),
        ("an output dst", lambda: Node("m").outputX.connect(Node("k").outputY), "cmds"),
        ("locked dst", lambda: (cmds.setAttr("b.ty", lock=True), Node("a").tx.connect(Node("b").ty)), "cmds"),
        ("locked parent", locked_parent, "cmds"),
        ("element under a locked array root", locked_array_root_element, "cmds"),
        ("connected dst", connected_dst, "cmds"),
        ("connected dst, force", force_connected_dst, "cmds"),
        ("connected parent", connected_parent, "cmds"),
        ("connected child", connected_child, "cmds"),
        ("array root", array_root, "cmds"),
        ("incompatible types", lambda: Node("a").worldMatrix[0].connect(Node("b").tx), "cmds"),
        ("a user's Plug class", user_plug, "cmds"),
        ("a user's node naming", user_naming, "cmds"),
        ("converted material", converted_material, "cmds"),
    ]


class TestTheRoute(_ApiCase):
    """Which connects take the API, and that the guard cases give the cmds route's outcome."""

    def test_a_connect_runs_no_cmds_connect(self):
        with mock.patch.object(cmds, "connectAttr", wraps=cmds.connectAttr) as connect:
            Node("a").tx << Node("b").ty
        self.assertEqual(connect.call_count, 0)
        self.assertTrue(cmds.isConnected("b.ty", "a.tx"))
        self.assertEqual(self.count_steps(), 1)
        self.assertTrue(cmds.isConnected("b.ty", "a.tx"))

    def test_the_switch_off_takes_cmds(self):
        with mock.patch.object(journal, "API_EDITS", False), \
                mock.patch.object(cmds, "connectAttr", wraps=cmds.connectAttr) as connect:
            Node("a").tx << Node("b").ty
        self.assertEqual(connect.call_count, 1)
        self.assertTrue(cmds.isConnected("b.ty", "a.tx"))

    def outcome(self, case, api):
        cmds.file(new=True, force=True)
        self.make_nodes()
        cmds.flushUndo()
        calls = []
        real  = api_edits.api_connect

        def spy(src, dst):
            calls.append(real(src, dst))
            return calls[-1]

        printed = []

        def on_output(message, kind, *_):
            if kind == OpenMaya.MCommandMessage.kError:
                printed.append(message.strip())

        callback = OpenMaya.MCommandMessage.addCommandOutputCallback(on_output)
        try:
            with mock.patch.object(journal, "API_EDITS", api), mock.patch.object(_base, "_api_connect", spy):
                try:
                    case()
                    error = None
                except Exception as exc:  # noqa: BLE001 (the error is what is compared)
                    error = _brief(exc)
        finally:
            OpenMaya.MMessage.removeCallback(callback)
        state = (error, self.connections(), sorted(cmds.ls()))
        undone = [self.undo_all(), self.connections(), sorted(cmds.ls())][1:]
        return state, undone, calls, printed

    def test_every_guard_case_gives_the_cmds_outcome(self):
        for label, case, route in _guard_cases():
            with self.subTest(case=label):
                on, on_undone, calls, printed_on = self.outcome(case, True)
                off, off_undone, _, printed_off = self.outcome(case, False)
                self.assertEqual(on, off)
                self.assertEqual(on_undone, off_undone)
                self.assertEqual(calls[-1:], [route == "api"] if calls else [])
                if route == "api" or label != "incompatible types":
                    self.assertEqual(printed_on, printed_off)
                else:
                    # the approved difference: the API connect's doIt prints its Error: line
                    self.assertEqual(len(printed_on), len(printed_off) + 1)
                    self.assertIn("not compatible", printed_on[0])

    def test_a_failed_api_connect_chains_no_api_error(self):
        # CR-6: the incompatible types fail the API's doIt; its except block ends inside the
        # helper, so the cmds route's error and chain are today's
        for api in (True, False):
            with self.subTest(api=api), mock.patch.object(journal, "API_EDITS", api):
                with self.assertRaisesRegex(RuntimeError, "not compatible") as raised:
                    Node("a").worldMatrix[0].connect(Node("b").tx)
                self.assertIsNone(raised.exception.__context__)
                self.assertIsNone(raised.exception.__cause__)
        self.assertEqual(self.connections(), [])


class TestStatementGranularity(_ApiCase):
    """V6: a statement's connects are one undo step, with the dangling check after each."""

    def test_one_connect_is_one_step(self):
        Node("a").tx << Node("b").ty
        self.assertEqual(self.count_steps(), 1)

    def fanout(self, api):
        nodes = [Node(cmds.createNode("transform", name=f"n{i}")) for i in range(50)]
        cmds.flushUndo()
        with mock.patch.object(journal, "API_EDITS", api):
            List(nodes).tx << Node("a").ty
        self.assertEqual(sorted(cmds.listConnections("a.ty", destination=True, source=False)),
                         sorted(f"n{i}" for i in range(50)))
        return self.count_steps()

    def test_a_list_fanout_of_fifty_is_one_step(self):
        self.assertEqual(self.fanout(api=False), 50)
        cmds.file(new=True, force=True)
        self.make_nodes()
        self.assertEqual(self.fanout(api=True), 1)
        self.undo_steps(1)
        self.assertEqual(cmds.listConnections("a.ty", destination=True, source=False), None)
        self.redo_steps(1)
        self.assertEqual(len(cmds.listConnections("a.ty", destination=True, source=False)), 50)

    def test_a_compound_connect_is_one_step(self):
        Node("b").t << Node("a").t
        self.assertTrue(cmds.isConnected("a.t", "b.t"))
        self.assertEqual(self.count_steps(), 1)

    def test_value_sets_stay_cmds_steps(self):
        # with the API off, node.t << (1, 2, 3) is three cmds steps (R5 D3 journals them:
        # test_r5_api_sets.TestStatementGranularity, one step)
        with mock.patch.object(journal, "API_EDITS", False):
            Node("a").t << (1, 2, 3)
        self.assertEqual(self.count_steps(), 3)

    def network(self, api):
        cmds.flushUndo()
        with mock.patch.object(journal, "API_EDITS", api):
            Node("b").ty << (Node("a").tx + Node("b").tx) * 0.5
        made = sorted(set(cmds.ls()) - {"a", "b", "c", "m", "k"} - set(self.default_nodes))
        return made, self.connections()

    def test_an_operator_network(self):
        self.default_nodes = cmds.ls()
        made_off, wired_off = self.network(api=False)
        steps_off = self.count_steps()
        cmds.file(new=True, force=True)
        self.make_nodes()
        self.default_nodes = cmds.ls()
        made_on, wired_on = self.network(api=True)
        self.assertEqual((made_on, wired_on), (made_off, wired_off))
        # undo the last journal only: b.ty is unwired, the network's nodes still exist
        self.undo_steps(1)
        self.assertEqual(cmds.listConnections("b.ty", source=True, destination=False), None)
        self.assertTrue(all(cmds.objExists(n) for n in made_on))
        self.redo_steps(1)
        steps_on = self.count_steps()
        self.assertLess(steps_on, steps_off)
        self.assertEqual(self.connections(), wired_off)

    def interleaved(self, api):
        states = []

        def state():
            states.append((self.connections(), cmds.getAttr("c.tz")))

        state()
        with mock.patch.object(journal, "API_EDITS", api):
            Node("a").tx << Node("b").ty
            state()
            cmds.setAttr("c.tz", 3.0)          # a user's cmds edit
            state()
            Node("c").ty << Node("b").tx
            state()
            cmds.connectAttr("b.tz", "a.tz")   # a user's cmds connect
            state()
            List([Node("m").input1X, Node("m").input1Y]) << Node("k").outputX
            state()
        walked = []
        while not self.queue_empty():
            self.undo_steps(1)
            walked.append((self.connections(), cmds.getAttr("c.tz")))
        return states, walked

    def test_user_cmds_interleaved(self):
        states_on, walked_on = self.interleaved(True)
        cmds.file(new=True, force=True)
        self.make_nodes()
        cmds.flushUndo()
        states_off, walked_off = self.interleaved(False)
        self.assertEqual(states_on, states_off)
        # one step per statement with the API: the states between statements, step by step
        self.assertEqual(walked_on, list(reversed(states_on[:-1])))
        # the cmds route: the List connect's two steps, then the same
        self.assertEqual(walked_off[1:], walked_on)

    def test_a_raising_list_connect(self):
        def run(api):
            cmds.file(new=True, force=True)
            self.make_nodes()
            cmds.setAttr("k.input1Y", lock=True)
            cmds.flushUndo()
            with mock.patch.object(journal, "API_EDITS", api):
                try:
                    List([Node("k").input1X, Node("k").input1Y, Node("k").input1Z]) << Node("m").outputX
                    error = None
                except Exception as exc:  # noqa: BLE001
                    error = _brief(exc)
            after = self.connections()
            self.undo_all()
            return error, after, self.connections()

        on, off = run(True), run(False)
        self.assertIsNotNone(on[0])
        self.assertEqual(on, off)

    def test_chunks(self):
        with rig.undo_chunk("x"):
            Node("a").tx << Node("b").ty
            Node("c").tx << Node("b").ty
        self.assertEqual(cmds.undoInfo(query=True, undoName=True), "x")
        cmds.undoInfo(openChunk=True, chunkName="u")
        try:
            Node("a").ty << Node("b").tx
            cmds.setAttr("c.tz", 2.0)
            Node("c").ty << Node("b").tx
        finally:
            cmds.undoInfo(closeChunk=True)
        self.assertEqual(self.count_steps(), 2)

    def test_a_post_create_disconnect_of_a_journaled_connect(self):
        # CR-5 with a real connect: the journal, then the user's disconnect
        class UsersTransform(Transform):
            @classmethod
            def post_create(cls, new_node_name, *args, **kwargs):
                cmds.disconnectAttr("m.outputX", "k.input1X")
                return super().post_create(new_node_name, *args, **kwargs)

        def body():
            Node("k").input1X << Node("m").outputX
            UsersTransform.create(name="x")

        def connected():
            return cmds.isConnected("m.outputX", "k.input1X")

        journal.run_statement(body, (), {})
        self.assertFalse(connected())
        found = []
        for _ in range(3):
            self.undo_steps(1)
            found.append((connected(), cmds.objExists("x")))
        self.assertTrue(self.queue_empty())
        self.redo_all()
        self.assertEqual(found, [(True, True), (False, True), (False, False)])
        self.assertEqual((connected(), cmds.objExists("x")), (False, True))

    def test_a_specs_apply_after_a_journaled_connect(self):
        class UsersSpec(_AttrSpec):
            def apply(self, target):
                cmds.connectAttr("k.outputY", "m.input1X", force=True)
                return target

        List([Node("m").input1X, Node("m").input1Y]) << [Node("k").outputX, UsersSpec()]
        self.assertTrue(cmds.isConnected("k.outputY", "m.input1X"))
        self.undo_steps(1)  # the user's connect: the journal's is back
        self.assertTrue(cmds.isConnected("k.outputX", "m.input1X"))
        self.undo_steps(1)
        self.assertEqual(self.connections(), [])
        self.assertTrue(self.queue_empty())

    def test_queue_eviction_frees_the_journals(self):
        for i in range(50):
            cmds.addAttr("k", longName=f"d{i}", attributeType="double")
        cmds.undoInfo(state=True, infinity=False, length=10)
        kept = []

        class Tracked(journal._Journal):
            __slots__ = ("__weakref__",)

            def __init__(self, ops):
                super().__init__(ops)
                kept.append(weakref.ref(self))

        with mock.patch.object(journal, "_Journal", Tracked):
            k = Node("k")
            for i in range(50):
                getattr(k, f"d{i}") << Node("m").outputX
        gc.collect()
        alive = sum(ref() is not None for ref in kept)
        self.assertEqual(len(kept), 50)
        self.assertLessEqual(alive, 10)
        self.assertGreaterEqual(alive, 1)
        cmds.flushUndo()
        gc.collect()
        self.assertEqual(sum(ref() is not None for ref in kept), 0)


class TestReferenceEdits(_ApiCase):
    """A connect on a referenced node records the same reference edits on both routes."""

    def setUp(self):
        super().setUp()
        self.folder = tempfile.mkdtemp(prefix="rig_d2_ref_")
        self.path = os.path.join(self.folder, "d2_ref.ma").replace("\\", "/")
        cmds.file(new=True, force=True)
        cmds.createNode("transform", name="rl")
        cmds.file(rename=self.path)
        cmds.file(save=True, force=True, type="mayaAscii")

    def tearDown(self):
        try:
            super().tearDown()
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(self.folder, ignore_errors=True)

    def edits(self, api):
        cmds.file(new=True, force=True)
        self.make_nodes()
        cmds.file(self.path, reference=True, namespace="R")
        cmds.flushUndo()
        with mock.patch.object(journal, "API_EDITS", api):
            Node("R:rl").tx << Node("a").ty
            Node("a").tz << Node("R:rl").rx
        node  = cmds.referenceQuery(self.path, referenceNode=True)
        after = sorted(cmds.referenceQuery(node, editStrings=True))
        self.undo_all()
        undone = sorted(cmds.referenceQuery(node, editStrings=True))
        self.redo_all()
        return after, undone, sorted(cmds.referenceQuery(node, editStrings=True))

    def test_the_same_edits_on_both_routes(self):
        on, off = self.edits(True), self.edits(False)
        self.assertEqual(on, off)
        self.assertEqual(len([e for e in on[0] if e.startswith("connectAttr")]), 2)
        self.assertEqual([e for e in on[1] if e.startswith("connectAttr")], [])
        self.assertEqual(on[2], on[0])


class TestPluginLoading(_ApiCase):
    """U1: rig's plug-in loads at the first API connect, by full path, else by name; when it
    cannot load, connects take cmds (one warning per session) and the edits with no other
    undoable route raise a clear error. U2: an unload is refused while journals are queued."""

    NOT_FOUND = 'Plug-in, "undoable_api_command", was not found on MAYA_PLUG_IN_PATH.'

    def unload(self):
        cmds.flushUndo()
        if cmds.pluginInfo(PLUGIN, query=True, loaded=True):
            cmds.unloadPlugin(PLUGIN)
        self.assertFalse(hasattr(cmds, UNDOABLE_API_COMMAND))

    @contextlib.contextmanager
    def missing(self):
        """The plug-in in neither place: not in rig's folder, not found by name (yields the
        loadPlugin mock)."""
        self.unload()
        load = mock.MagicMock(side_effect=RuntimeError(self.NOT_FOUND))
        with mock.patch.object(plugins, "bundled_plugin_path", return_value=None),                 mock.patch.object(cmds, "loadPlugin", load),                 mock.patch.object(plugins, "_WARNED", False),                 mock.patch.object(plugins, "_LOAD_FAILED", None):
            yield load

    def test_the_first_connect_loads_it_by_full_path(self):
        self.unload()
        with mock.patch.object(cmds, "loadPlugin", wraps=cmds.loadPlugin) as load:
            Node("a").tx << Node("b").ty
            Node("a").ty << Node("b").tx
        self.assertEqual(load.call_count, 1)
        self.assertEqual(os.path.normcase(os.path.abspath(load.call_args[0][0])),
                         os.path.normcase(os.path.abspath(PLUGIN_PATH)))
        self.assertEqual(self.count_steps(), 2)

    def test_not_in_rigs_folder_it_loads_by_name(self):
        self.unload()
        real = cmds.loadPlugin

        def by_name(name, **kwargs):
            self.assertEqual(name, PLUGIN)  # what MAYA_PLUG_IN_PATH would find
            return real(PLUGIN_PATH, **kwargs)

        with mock.patch.object(plugins, "bundled_plugin_path", return_value=None), \
                mock.patch.object(cmds, "loadPlugin", side_effect=by_name) as load, \
                mock.patch.object(cmds, "connectAttr", wraps=cmds.connectAttr) as connect:
            Node("a").tx << Node("b").ty
        self.assertEqual(load.call_count, 1)
        self.assertEqual(connect.call_count, 0)
        self.assertTrue(cmds.isConnected("b.ty", "a.tx"))

    def test_missing_connects_take_cmds_with_one_warning(self):
        with self.missing() as load, mock.patch.object(cmds, "warning") as warning, \
                mock.patch.object(cmds, "connectAttr", wraps=cmds.connectAttr) as connect:
            Node("a").tx << Node("b").ty
            Node("a").ty << Node("b").tx
            self.assertEqual(load.call_count, 1)  # not tried again in this kind epoch
            _base._clear_static_data_type()      # what a plug-in load or unload does
            Node("a").tz << Node("b").tx
        self.assertEqual(load.call_count, 2)
        self.assertEqual(connect.call_count, 3)
        self.assertEqual(warning.call_count, 1)
        text = warning.call_args[0][0]
        for part in ("undoable_api_command", "could not be loaded", "MAYA_PLUG_IN_PATH", self.NOT_FOUND,
                     "cmds route", "slower"):
            self.assertIn(part, text)
        self.assertEqual(self.connections(), [("b.translateX", "a.translateY"), ("b.translateX", "a.translateZ"),
                                              ("b.translateY", "a.translateX")])
        self.assertEqual(self.count_steps(), 3)

    def test_missing_the_edits_with_no_other_route_raise(self):
        cmds.polyCube(name="box", constructionHistory=False)
        cmds.flushUndo()
        box    = Mesh("boxShape")
        before = cmds.xform("box.vtx[0]", query=True, translation=True)
        points = OpenMaya.MPointArray(box.get_fn_set().getPoints())
        points[0] = OpenMaya.MPoint(9, 9, 9)
        with self.missing(), self.assertRaises(RuntimeError) as raised:
            box.set_points(points, world_space=False)
        text = str(raised.exception)
        for part in ("rig needs its undo plug-in 'undoable_api_command' for Mesh.set_points",
                     "MAYA_PLUG_IN_PATH", self.NOT_FOUND):
            self.assertIn(part, text)
        self.assertIsNone(raised.exception.__cause__)
        self.assertEqual(cmds.xform("box.vtx[0]", query=True, translation=True), before)
        self.assertTrue(self.queue_empty())

        class Unnamed:
            def doIt(self):  # noqa: N802
                raise AssertionError("not run")

        with self.missing(), self.assertRaisesRegex(RuntimeError, "for Unnamed, and it could not be loaded"):
            plugins._run_undoable(Unnamed())

    def test_the_bundled_file_names_where_it_was_loaded_from(self):
        self.assertIn(os.path.normcase(PLUGIN_PATH), os.path.normcase(plugins._plugin_whereabouts()))
        with mock.patch.object(plugins, "bundled_plugin_path", return_value=None):
            self.assertIn("MAYA_PLUG_IN_PATH", plugins._plugin_whereabouts())

    def test_an_unload_is_refused_until_flush_undo(self):
        # U2: the build's journals use the plug-in
        Node("a").tx << Node("b").ty
        with self.assertRaisesRegex(RuntimeError, "still in use"):
            cmds.unloadPlugin(PLUGIN)
        cmds.flushUndo()
        cmds.unloadPlugin(PLUGIN)
        # the next connect loads it again
        with mock.patch.object(cmds, "loadPlugin", wraps=cmds.loadPlugin) as load:
            Node("a").ty << Node("b").tx
        self.assertEqual(load.call_count, 1)
        self.assertEqual(self.count_steps(), 1)

    def test_not_ready_during_undo_and_redo(self):
        seen = []

        class Spy:
            def doIt(self):  # noqa: N802
                seen.append(plugins._fast_path_ready())

            undoIt = redoIt = doIt  # noqa: N815

        plugins._run_undoable(Spy())
        cmds.undo()
        cmds.redo()
        self.assertEqual(seen, [True, False, False])


class TestTheBuild(_ApiCase):
    """V10: rail_spine's registrations and connects per build."""

    def test_rail_spine(self):
        from rig.examples import rail_spine as rs

        options = dict(get_options())
        self.addCleanup(lambda: set_options(**options))
        set_options(create_containers=True)
        pcs = List()
        for i in range(5):
            pcs.append(Node(cmds.spaceLocator()[0]))
            pcs[i].ty << i * 5
        cmds.flushUndo()
        counts = {"register": 0, "api": 0, "set barriers": 0}
        register, connect = journal._register, api_edits.api_connect

        def counted_register(ops):
            counts["register"] += 1
            caller = sys._getframe(2)  # _flush's caller
            if caller.f_code.co_name == "command" and caller.f_locals.get("name") == "setAttr":
                counts["set barriers"] += 1
            return register(ops)

        def counted_connect(src, dst):
            made = connect(src, dst)
            counts["api"] += made
            return made

        u = [x / 19 for x in range(20)]
        with mock.patch.object(journal, "_register", counted_register), \
                mock.patch.object(_base, "_api_connect", counted_connect), \
                mock.patch.object(cmds, "connectAttr", wraps=cmds.connectAttr) as cmds_connect:
            rs.create_rail(pcs, u, scale_controls=pcs, orient_controls=pcs, degree=3, periodic=False,
                           aim_axis=1, up_axis=0, debug=True)
        # plan_D V10: at most 860 (R5 D3 journals the value sets, so few set barriers are left)
        self.assertLessEqual(counts["register"], 860)
        self.assertLessEqual(counts["register"] - counts["set barriers"], 860)
        self.assertGreater(counts["api"], 4500)
        self.assertLess(cmds_connect.call_count, 100)

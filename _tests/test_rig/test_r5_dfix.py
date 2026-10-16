"""Round 5, DFIX: the regressions of the phase D review (D2 / D3: the API connects and value
sets, journaled per statement).

Each test runs its case with the API fast paths on and off (``journal.API_EDITS``, R5M's
``cmds`` route) in the same process, and compares what the user sees: the scene after the
edit, after every undo step (with the dangling check, CR-13), after undo-all and redo-all,
and Maya's printed ``Error:`` / ``Warning:`` lines. The undo step counts may differ only by
the approved batching (one step per statement for its API edits).

* `TestUserCallbacks`: a user's callback that deletes a node the statement journaled, or
  unloads a reference, inside the statement (`journal._before_removal`).
* `TestUndoOffEdits`: an edit made with undo recording off (``-stateWithoutFlush off``)
  replaces or removes what a journal made: its undo is silent, as the ``cmds`` route's is
  (`journal._walk`).
* `TestTheRedoBranch`: after an undo, a failed registration of a statement's earlier edits
  propagates (`api_edits._apply`), never silently dropping them.
* `TestARefusedPlugin`: Maya refuses to initialise rig's plug-in file (U1): Maya's error is
  printed once, not at every later plug-in load, and the U1 error gives Maya's reason.
* `TestJournalUndo`: a journal's undo and redo make no ``undoInfo`` toggle; a user's cmds
  edit in a callback during that undo lands as on the ``cmds`` route.
* `TestASwallowedConnect`: ``container.publish_output``'s external output connect, refused
  by Maya, prints nothing (rig passes over it).
* `TestAnErrorMidBuild`: the user's requirement of 2026-10-15. rail_spine (3 riders) with an
  error injected between statements, inside one, at the first and at the last, bare and
  inside ``rig.undo_chunk``: undone step by step to the start and redone, with no error
  from the undo queue, the setup scene after undo-all, 0 dangling connections after every
  step, the partial scene after redo-all, and the same scenes and error as the ``cmds``
  route.
"""

import contextlib
import inspect
import os
import re
import shutil
import tempfile
import types
from unittest import mock

from maya import cmds
from maya.api import OpenMaya

import rig
from rig import List, Node, get_options, set_options
from rig._internal import api_edits, journal
from rig._internal.container import container
from rig.nodetypes import _base, plugins
from rig.nodetypes.mesh import Mesh
from rig.nodetypes.plugins import UNDOABLE_API_COMMAND
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk

PLUGIN = "undoable_api_command"


def _brief(error, depth=0):
    """An exception and its chain, as comparable data (type, text, __suppress_context__)."""
    if error is None or depth > 5:
        return None
    return (type(error).__name__, str(error), error.__suppress_context__,
            _brief(error.__context__, depth + 1), _brief(error.__cause__, depth + 1))


def _plug(name):
    sel = OpenMaya.MSelectionList()
    sel.add(name)
    return sel.getPlug(0)


def _value_modifier(name, value):
    modifier = OpenMaya.MDGModifier()
    modifier.newPlugValueDouble(_plug(name), value)
    return modifier


def _dangling():
    """The connections with an endpoint whose node does not exist (fast: one listConnections
    over every node, ``objExists`` only for a node ``ls`` does not list)."""
    nodes = cmds.ls() or []
    names = set(nodes)
    pairs = (cmds.listConnections(nodes, connections=True, plugs=True) if nodes else None) or []
    bad, known = [], {}
    for i in range(0, len(pairs) - 1, 2):
        for plug in (pairs[i], pairs[i + 1]):
            node = plug.split(".", 1)[0]
            if node in names:
                continue
            missing = known.get(node)
            if missing is None:
                missing = known[node] = not cmds.objExists(node)
            if missing:
                bad.append((pairs[i], pairs[i + 1]))
                break
    return bad


class _Printed:
    """The ``Error:`` and ``Warning:`` lines Maya prints while active (``MCommandMessage``):
    ``lines`` is a list of ``("E" | "W", text)``."""

    def __enter__(self):
        self.lines = []
        self._callback = OpenMaya.MCommandMessage.addCommandOutputCallback(self._on_output)
        return self

    def _on_output(self, message, kind, *_args):
        if kind == OpenMaya.MCommandMessage.kError:
            self.lines.append(("E", message.strip()))
        elif kind == OpenMaya.MCommandMessage.kWarning:
            self.lines.append(("W", message.strip()))

    def __exit__(self, *exc):
        OpenMaya.MMessage.removeCallback(self._callback)
        return False


class _DFixCase(UndoWalk, MayaTestCase):
    """A new scene, the undo queue on, infinite and flushed, rig's plug-in loaded. The
    journal is closed and empty after every test."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        plugins._ensure_loaded(PLUGIN, True)
        cmds.flushUndo()

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

    @staticmethod
    def fresh_scene():
        cmds.file(new=True, force=True)
        cmds.flushUndo()

    @contextlib.contextmanager
    def route(self, api):
        """The API fast paths on (``api``) or off (the cmds route) in the block."""
        with mock.patch.object(journal, "API_EDITS", api):
            yield

    def both(self, case):
        """``case()`` (returns a record) with the API on, then off, each in a new scene."""
        records = {}
        for api in (True, False):
            self.fresh_scene()
            with self.route(api):
                records[api] = case()
        return records[True], records[False]

    def state(self, nodes):
        """``{node: (its incoming connections, its input2X)}`` of the ``nodes`` that exist."""
        out = {}
        for name in nodes:
            if cmds.objExists(name):
                incoming = cmds.listConnections(name, connections=True, plugs=True, source=True,
                                                destination=False) or []
                value = (round(cmds.getAttr(name + ".input2X"), 6)
                         if cmds.attributeQuery("input2X", node=name, exists=True) else None)
                out[name] = (sorted(zip(incoming[1::2], incoming[::2])), value)
        return out

    def walk(self, nodes, cycles=0):
        """Undo all (the state and the dangling check after every step), redo all, then
        ``cycles`` more undo-all / redo-all rounds; the record, with what Maya printed."""
        record = {"after": self.state(nodes), "undo": [], "dangling": []}
        with _Printed() as printed:
            for _ in range(1 + cycles):
                record["undo"] = []
                n = 0
                while not self.queue_empty():
                    self.assertLess(n, 200)
                    cmds.undo()
                    n += 1
                    record["undo"].append(self.state(nodes))
                    record["dangling"].extend(_dangling())
                while not cmds.undoInfo(query=True, redoQueueEmpty=True):
                    cmds.redo()
                    record["dangling"].extend(_dangling())
                record["redo"] = self.state(nodes)
        record["printed"] = printed.lines
        return record


# --------------------------------------------------------------------------------------- #
#  A user's callback inside a statement
# --------------------------------------------------------------------------------------- #


class TestUserCallbacks(_DFixCase):
    """A user's attribute-changed callback that deletes, or unloads, inside a rig statement:
    the deletion goes on the undo queue after the edits the statement made before it, as on
    the cmds route (the review: undo-all left a deleted node's connection, redo-all a
    dangling one; the unload's journal printed an Error at every undo)."""

    NODES = ("a", "b", "c", "src", "ref:r")

    def callback_case(self, action, target="a"):
        cmds.createNode("multiplyDivide", name="src")
        cmds.createNode("multiplyDivide", name="b")
        cmds.createNode("multiplyDivide", name="c")
        cmds.flushUndo()
        busy = []

        def on_change(message, plug, _other, _data):
            if busy or OpenMaya.MGlobal.isUndoing() or OpenMaya.MGlobal.isRedoing():
                return
            if not (message & OpenMaya.MNodeMessage.kConnectionMade
                    and message & OpenMaya.MNodeMessage.kIncomingDirection):
                return
            if plug.partialName() not in ("i1x", "input1X"):
                return
            busy.append(True)
            try:
                action()
            finally:
                busy.pop()

        sel = OpenMaya.MSelectionList()
        sel.add("b")
        callback = OpenMaya.MNodeMessage.addAttributeChangedCallback(sel.getDependNode(0), on_change)
        try:
            List([Node(target).input1X, Node("b").input1X]) << Node("src").outputX
        finally:
            OpenMaya.MMessage.removeCallback(callback)
        self.assertEqual((journal._S.open, journal._S.pending), (False, []))
        return self.walk(self.NODES, cycles=2)

    def test_a_callback_that_deletes_a_node_the_statement_journaled(self):
        def case():
            cmds.createNode("multiplyDivide", name="a")
            return self.callback_case(lambda: cmds.delete("a"))

        on, off = self.both(case)
        self.assertEqual(on["dangling"], [])
        self.assertEqual(on["printed"], [])
        self.assertNotIn("a", on["after"])
        self.assertEqual(on["undo"][-1]["a"], ([], 1.0))  # a back, unconnected
        # the deletion splits the statement's journal: the same steps as the cmds route
        self.assertEqual(on, off)

    def test_a_callback_that_deletes_another_node(self):
        def case():
            cmds.createNode("multiplyDivide", name="a")
            return self.callback_case(lambda: cmds.delete("c"))

        on, off = self.both(case)
        self.assertEqual(on["dangling"], [])
        self.assertEqual(on["printed"], [])
        for key in ("after", "redo", "printed", "dangling"):
            self.assertEqual(on[key], off[key], key)
        self.assertEqual(on["undo"][-1], off["undo"][-1])

    def test_a_callback_that_unloads_a_reference(self):
        folder = tempfile.mkdtemp(prefix="rig_r5_dfix_")
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        path = os.path.join(folder, "ref_src.ma").replace("\\", "/")

        def case():
            cmds.createNode("multiplyDivide", name="r")
            cmds.file(rename=path)
            cmds.file(save=True, force=True, type="mayaAscii")
            cmds.file(new=True, force=True)
            cmds.file(path, reference=True, namespace="ref")
            node = cmds.referenceQuery("ref:r", referenceNode=True)
            return self.callback_case(lambda: cmds.file(unloadReference=node), target="ref:r")

        on, off = self.both(case)
        self.assertEqual(on["dangling"], [])
        self.assertEqual(on["printed"], [])  # was an Error from the journal at every undo
        self.assertEqual(on, off)


# --------------------------------------------------------------------------------------- #
#  Undo-off edits
# --------------------------------------------------------------------------------------- #


class TestUndoOffEdits(_DFixCase):
    """An edit made with ``undoInfo -stateWithoutFlush off`` after journaled edits: undo-all
    and redo-all print nothing and leave the cmds route's scenes."""

    NODES = ("a", "b", "src")

    def case(self, edit):
        def run():
            for name in self.NODES:
                cmds.createNode("multiplyDivide", name=name)
            cmds.flushUndo()
            a, b, src = Node("a"), Node("b"), Node("src")
            List([a.input1X, b.input1X]) << src.outputX
            List([a.input2X, b.input2X]) << 3.0
            cmds.undoInfo(stateWithoutFlush=False)
            try:
                edit()
            finally:
                cmds.undoInfo(stateWithoutFlush=True)
            return self.walk(self.NODES)
        return run

    def check(self, edit):
        on, off = self.both(self.case(edit))
        self.assertEqual(on["printed"], [])
        self.assertEqual(on["dangling"], [])
        for key in ("after", "redo", "printed", "dangling"):
            self.assertEqual(on[key], off[key], key)
        self.assertEqual(on["undo"][-1], off["undo"][-1])
        self.assertLess(len(on["undo"]), len(off["undo"]))  # the approved batching

    def test_a_journaled_connection_replaced(self):
        # the review: undo printed "Error: (kFailure): Unexpected Internal Failure"
        def edit():
            cmds.disconnectAttr("src.outputX", "a.input1X")
            cmds.connectAttr("b.outputX", "a.input1X")

        self.check(edit)

    def test_a_journaled_connection_removed(self):
        self.check(lambda: cmds.disconnectAttr("src.outputX", "a.input1X"))

    def test_a_journaled_value_set_again(self):
        self.check(lambda: cmds.setAttr("a.input2X", 9))

    def test_a_failing_op_that_is_no_modifier_still_raises(self):
        # H20 (b) is kept for any other op: the walk ends, then the first error propagates
        class Failing:
            def undoIt(self):  # noqa: N802
                raise ValueError("not a modifier")

        cmds.createNode("multiplyDivide", name="m")
        modifier = _value_modifier("m.input1X", 4.0)
        modifier.doIt()
        entry = journal._Journal([modifier, Failing()])
        with self.assertRaisesRegex(ValueError, "^not a modifier$"):
            entry.undoIt()
        self.assertEqual(cmds.getAttr("m.input1X"), 0.0)  # the modifier was undone too


# --------------------------------------------------------------------------------------- #
#  The redo branch of journal.add
# --------------------------------------------------------------------------------------- #


class TestTheRedoBranch(_DFixCase):
    """After an undo, ``journal.add`` registers what is pending before the new edit; when that
    registration fails, its error propagates out of the statement (the edits reverted), and
    the statement does not go on through cmds without them."""

    def test_a_failed_registration_of_earlier_edits_propagates(self):
        cmds.createNode("multiplyDivide", name="m")
        cmds.flushUndo()
        failure = RuntimeError("rig's command could not be had")

        @journal.statement
        def edit():
            self.assertIs(api_edits._apply(_value_modifier("m.input1X", 5.0)), True)
            self.assertEqual(len(journal._S.pending), 1)
            # an undo ran inside the statement (a user's callback), and Maya's redo queue is
            # not empty: the next edit registers what is pending first ...
            journal._S.redo = True
            with mock.patch.object(journal._cmds, "undoInfo", return_value=False), \
                    mock.patch.object(journal, "_register", side_effect=failure):
                # ... which fails: _apply undoes its own edit and re-raises
                api_edits._apply(_value_modifier("m.input1Y", 6.0))
            raise AssertionError("not reached")

        with self.assertRaises(RuntimeError) as caught:
            edit()
        text = str(caught.exception)
        self.assertIn("1 edit(s) of this statement could not be put on the undo queue", text)
        self.assertIn("rig's command could not be had", text)
        self.assertTrue(text.endswith("they were reverted"), text)
        self.assertTrue(caught.exception.__suppress_context__)
        self.assertEqual((cmds.getAttr("m.input1X"), cmds.getAttr("m.input1Y")), (0.0, 0.0))
        self.assertTrue(self.queue_empty())

    def test_a_failed_registration_of_the_edit_alone_takes_cmds(self):
        # unchanged: a registration of the one new edit that fails sends it to cmds
        cmds.createNode("multiplyDivide", name="m")
        cmds.flushUndo()
        failure = RuntimeError("rig's command could not be had")
        with mock.patch.object(journal, "_register", side_effect=failure):
            self.assertIs(api_edits._apply(_value_modifier("m.input1X", 5.0)), False)
        self.assertEqual(cmds.getAttr("m.input1X"), 0.0)
        self.assertTrue(self.queue_empty())


# --------------------------------------------------------------------------------------- #
#  A plug-in file Maya refuses
# --------------------------------------------------------------------------------------- #

_REFUSING = '''"""rig's test: a plug-in that refuses to load."""


def maya_useNewAPI():
    pass


def initializePlugin(obj):
    raise RuntimeError("rig test: this plug-in refuses to load")


def uninitializePlugin(obj):
    pass
'''


class TestARefusedPlugin(_DFixCase):
    """U1 with a real plug-in file that Maya refuses to initialise (no mocked loadPlugin):
    Maya's error is printed at the first edit that loads it, once; the connects take cmds
    with rig's one warning; later plug-in loads do not make rig try it again; the edits with
    no other route raise an error that gives Maya's reason."""

    REASON = "rig test: this plug-in refuses to load"

    def setUp(self):
        super().setUp()
        folder = tempfile.mkdtemp(prefix="rig_r5_refused_")
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        self.path = os.path.join(folder, PLUGIN + ".py")
        with open(self.path, "w") as handle:
            handle.write(_REFUSING)
        cmds.flushUndo()
        if cmds.pluginInfo(PLUGIN, query=True, loaded=True):
            cmds.unloadPlugin(PLUGIN)
        self.assertFalse(hasattr(cmds, UNDOABLE_API_COMMAND))
        real = plugins.bundled_plugin_path
        self.patches = [
            mock.patch.object(plugins, "bundled_plugin_path",
                              side_effect=lambda name: self.path if name == PLUGIN else real(name)),
            mock.patch.object(plugins, "_WARNED", False),
            mock.patch.object(plugins, "_LOAD_FAILED", None),
        ]
        for patch in self.patches:
            patch.start()
        for name in ("a", "b"):
            cmds.createNode("transform", name=name)
        cmds.flushUndo()

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()
        super().tearDown()  # loads rig's own plug-in again

    def test_maya_s_refusal_is_printed_once(self):
        with _Printed() as printed:
            Node("b").tx << Node("a").tx
            first = list(printed.lines)
            del printed.lines[:]
            Node("b").ty << Node("a").ty
            # a plug-in load or unload: a new kind epoch (a real load too, when one is left)
            _base._clear_static_data_type()
            Node("b").tz << Node("a").tz
            for other in ("lookdevKit", "matrixNodes", "quatNodes"):
                if not cmds.pluginInfo(other, query=True, loaded=True):
                    cmds.loadPlugin(other, quiet=True)
                    break
            _base._clear_static_data_type()
            Node("b").rx << Node("a").rx
            Node("b").sx << 2.0
            later = list(printed.lines)
        errors = [text for kind, text in first if kind == "E"]
        self.assertTrue(any(self.REASON in text for text in errors), first)
        warnings = [text for kind, text in first if kind == "W" and text.startswith("rig:")]
        self.assertEqual(len(warnings), 1, first)
        for part in ("could not be loaded", self.path, "Maya refused to initialise it", self.REASON,
                     "cmds route", "slower"):
            self.assertIn(part, warnings[0])
        self.assertNotIn("unloadPlugin", warnings[0])
        self.assertEqual(later, [])
        self.assertFalse(hasattr(cmds, UNDOABLE_API_COMMAND))
        for src, dst in (("a.tx", "b.tx"), ("a.ty", "b.ty"), ("a.tz", "b.tz"), ("a.rx", "b.rx")):
            self.assertTrue(cmds.isConnected(src, dst))

    def test_the_edits_with_no_other_route_give_maya_s_reason(self):
        cmds.polyCube(name="box", constructionHistory=False)
        cmds.flushUndo()
        box    = Mesh("boxShape")
        before = cmds.xform("box.vtx[0]", query=True, translation=True)
        points = OpenMaya.MPointArray(box.fn_set.getPoints())
        points[0] = OpenMaya.MPoint(9, 9, 9)
        with _Printed() as printed, self.assertRaises(RuntimeError) as raised:
            box.set_points(points, world_space=False)
        text = str(raised.exception)
        for part in ("rig needs its undo plug-in 'undoable_api_command' for Mesh.set_points",
                     "could not be loaded", self.path, "Maya refused to initialise it", self.REASON):
            self.assertIn(part, text)
        self.assertNotIn("unloadPlugin", text)  # it is not loaded: nothing to unload
        self.assertIsNone(raised.exception.__cause__)
        self.assertTrue(any(kind == "E" and self.REASON in line for kind, line in printed.lines))
        self.assertEqual(cmds.xform("box.vtx[0]", query=True, translation=True), before)
        self.assertTrue(self.queue_empty())

    def test_a_forced_unload_keeps_the_recovery_text(self):
        # the other "did not register" state: loaded, no maya.cmds command (a forced unload)
        for patch in reversed(self.patches):
            patch.stop()
        self.patches = []
        plugins._ensure_loaded(PLUGIN, True)
        wrapper = getattr(cmds, UNDOABLE_API_COMMAND)
        delattr(cmds, UNDOABLE_API_COMMAND)
        try:
            with self.assertRaises(RuntimeError) as raised:
                plugins._ensure_loaded(PLUGIN, True)
        finally:
            setattr(cmds, UNDOABLE_API_COMMAND, wrapper)
        self.assertIn("cmds.unloadPlugin('undoable_api_command')", str(raised.exception))
        self.assertIsNone(plugins._refusal(raised.exception))


# --------------------------------------------------------------------------------------- #
#  A journal's undo and redo
# --------------------------------------------------------------------------------------- #


class TestJournalUndo(_DFixCase):
    """rig's command calls a journal's undoIt / redoIt directly (``RECORDS_NO_CMDS``), with
    no ``undoInfo`` query and toggle around them (about 18 us of each step); any other
    object keeps the toggle."""

    def setUp(self):
        super().setUp()
        for name in ("a", "b", "c"):
            cmds.createNode("multiplyDivide", name=name)
        cmds.flushUndo()

    def test_no_undo_info_call(self):
        List([Node("a").input1X, Node("b").input1X]) << Node("c").outputX
        with mock.patch.object(cmds, "undoInfo", wraps=cmds.undoInfo) as info:
            cmds.undo()
            self.assertFalse(cmds.isConnected("c.outputX", "a.input1X"))
            cmds.redo()
        self.assertEqual(info.call_count, 0)
        self.assertTrue(cmds.isConnected("c.outputX", "b.input1X"))
        self.assertTrue(journal._Journal.RECORDS_NO_CMDS)

    def test_another_object_keeps_the_toggle(self):
        class Edit:
            def doIt(self):  # noqa: N802
                pass

            undoIt = redoIt = doIt  # noqa: N815

        plugins._run_undoable(Edit())
        with mock.patch.object(cmds, "undoInfo", wraps=cmds.undoInfo) as info:
            cmds.undo()
        self.assertEqual(info.call_count, 3)  # the state query, off, on

    def test_a_callback_s_cmds_edit_during_the_undo(self):
        # a user's callback that edits through cmds while the journal is undone, without
        # checking MGlobal.isUndoing: the same outcome as the cmds route's undo
        def case():
            for name in ("a", "b", "c"):
                cmds.createNode("multiplyDivide", name=name)
            cmds.flushUndo()

            def on_change(message, plug, _other, _data):
                if message & OpenMaya.MNodeMessage.kConnectionBroken:
                    cmds.setAttr("c.input2X", cmds.getAttr("c.input2X") + 1)

            sel = OpenMaya.MSelectionList()
            sel.add("b")
            callback = OpenMaya.MNodeMessage.addAttributeChangedCallback(sel.getDependNode(0), on_change)
            try:
                List([Node("a").input1X, Node("b").input1X]) << Node("c").outputX
                while not self.queue_empty():
                    cmds.undo()
                undone = (self.state(("a", "b", "c")), cmds.undoInfo(query=True, redoQueueEmpty=True))
                while not cmds.undoInfo(query=True, redoQueueEmpty=True):
                    cmds.redo()
                redone = self.state(("a", "b", "c"))
            finally:
                OpenMaya.MMessage.removeCallback(callback)
            return undone, redone

        on, off = self.both(case)
        self.assertEqual(on, off)


# --------------------------------------------------------------------------------------- #
#  A connect rig passes over
# --------------------------------------------------------------------------------------- #


class TestASwallowedConnect(_DFixCase):
    """``container.publish_output`` of an external plug into an attribute Maya cannot connect
    it to: rig passes over the failed connect, and nothing is printed (the API route printed
    Maya's ``Error: Connection not made`` for a call that succeeds)."""

    def test_publish_output_of_an_incompatible_external_plug(self):
        options = dict(get_options())
        self.addCleanup(lambda: set_options(**options))
        set_options(create_containers=True)

        def case():
            cmds.createNode("transform", name="ext")
            cmds.addAttr("ext", longName="label", dataType="string")
            cmds.createNode("multiplyDivide", name="md")
            with _Printed() as printed:
                with container("grp"):
                    Node.create("transform", name="inner")
                    container.publish_output(Node("ext").worldMatrix[0], "outW", at="double")
                    container.publish_output(Node("ext").label, "outL", at="double")
                    container.publish_output(Node("md").outputX, "outX", at="double")
            return (printed.lines, cmds.listConnections("ext", source=False, destination=True) or [],
                    cmds.isConnected("md.outputX", "grp_attributes.outX"))

        on, off = self.both(case)
        self.assertEqual(on, off)
        self.assertEqual(on, ([], [], True))


# --------------------------------------------------------------------------------------- #
#  The user's requirement: an error mid-build, then undo
# --------------------------------------------------------------------------------------- #

# (anchor, text inserted before it) into rig.examples.rail_spine's source: the injection hooks
_HOOKS = (
    ('    with container("railNode1"):\n', '    _inj("start", locals())\n'),
    ('        # Prep orient/scale vector modulation.\n', '        _inj("after_knobs", locals())\n'),
    ('            # Orient driver setup.\n', '            _inj("after_cumsum", locals())\n'),
    ('            # Modulate the u translation: stretch + scale + shift + pivot.\n',
     '            _inj("loop_mid", locals())\n'),
    ('            rider << rider_matrix_out  # SRT+shear injection\n', '            _inj("loop_end", locals())\n'),
    ('    return rail\n', '        _inj("end", locals())\n\n'),
)

_HOOK_HEAD = '''
_RIDER = [-1]
INJECT = {"fn": None}


def _inj(label, loc):
    fn = INJECT["fn"]
    if fn is not None:
        fn(label, _RIDER[0], loc)

'''

_LINK_COUNTS = re.compile(r'^\s*setAttr -s \d+ "\.(lnk|slnk|shlk|sslk|ptlk|ptsl)";\s*$')


def _injected_rail_spine():
    """rig.examples.rail_spine with the injection hooks, as a module of the examples (code a
    user wrote, for the journal's rig_code)."""
    from rig.examples import rail_spine

    source = inspect.getsource(rail_spine)
    for anchor, text in _HOOKS:
        assert source.count(anchor) == 1, anchor
        source = source.replace(anchor, text + anchor)
    anchor = "        for u_default in u:\n"
    assert source.count(anchor) == 1, anchor
    source = source.replace(anchor, anchor + "            _RIDER[0] += 1\n")
    source = source.replace("def make_curve(", _HOOK_HEAD.lstrip("\n") + "\n" + "def make_curve(", 1)
    module = types.ModuleType("rig.examples._rail_spine_injected")
    module.__file__ = rail_spine.__file__
    exec(compile(source, rail_spine.__file__ + " (injected)", "exec"), module.__dict__)  # noqa: S102
    return module


def _order_free(path):
    """A saved .ma file's lines, without the header, the node UUIDs and the light-link counts,
    each createNode block one unit, sorted: the scene, whatever its node order (layer 2)."""
    with open(path, encoding="utf-8", errors="surrogateescape") as handle:
        lines = handle.read().splitlines()
    units, block = [], None
    for line in lines:
        if (line.startswith(("//", "fileInfo ", "requires ", "currentUnit ")) or "rename -uid" in line
                or _LINK_COUNTS.match(line)):
            continue
        if line.startswith("createNode "):
            if block is not None:
                units.append("\n".join(block))
            block = [line]
        elif line[:1] in ("\t", " ") and block is not None:
            block.append(line)
        else:
            if block is not None:
                units.append("\n".join(block))
                block = None
            units.append(line)
    if block is not None:
        units.append("\n".join(block))
    return sorted(units)


def _injections():
    """{case: (setup(pcs) or None, action(label, rider, locals))}: an error between two
    statements, inside one (a bad value mid-List, a locked middle element, a container
    scope, a math op, a typed create with a bad attribute, a connect into a locked plug, an
    SRT injection that fails midway), at the first statement and at the last."""
    from rig import functions as rf
    from rig.bridges import nodes as rn

    def at(label, rider=None):
        def decorate(fn):
            def action(here, r, loc):
                if here == label and (rider is None or r == rider):
                    fn(loc)
            return action
        return decorate

    def lock_pc2_tz(pcs):
        cmds.setAttr(str(pcs[2]) + ".tz", lock=True)

    @at("start")
    def first_typo(loc):
        eval("positon_controls", {}, dict(loc))  # noqa: S307 (a NameError)

    @at("start")
    def first_statement_locked_mid(loc):
        List(loc["position_controls"]).tz << 0.5

    @at("after_knobs")
    def between_attr(loc):
        loc["rail"].stretchh << 1

    @at("after_cumsum")
    def list_bad_value(loc):
        loc["position_controls"].tx << [1.0, 2.0, (1, 2), 4.0, 5.0]

    @at("after_cumsum")
    def list_locked_mid(loc):
        loc["position_controls"].tz << 0.25

    @at("loop_mid", 1)
    def container_scope(loc):
        with container("innerCtn"):
            node = rn.plusMinusAverage()
            node.input1D[0] << loc["rail"].pivot
            node.input1D[1] << 2.0
            loc["rider_path"].uValue << 0.3
            raise ValueError("boom inside a container scope")

    @at("loop_mid", 1)
    def math_bad(loc):
        (loc["u_default"] * loc["stretch_ratio"] + loc["rail"].pivot) * "notAnAttr"

    @at("loop_mid", 1)
    def math_bad_list(loc):
        rf.clamp(loc["u_default"] * 2.0, 0.0, List([loc["rail"].pivot, "zz"]))

    @at("loop_mid", 1)
    def typed_create(loc):
        rn.motionPath(fractionMode=1, uValue=0.5, notAnAttr=3)

    @at("loop_mid", 1)
    def locked_connect(loc):
        cmds.setAttr(str(loc["rider_path"]) + ".uValue", lock=True)  # the build's own connect fails

    @at("loop_end", 1)
    def srt_mid(loc):
        cmds.setAttr(str(loc["rider"]) + ".ry", lock=True)

    @at("loop_end", 2)
    def last_statement(loc):
        cmds.setAttr(str(loc["rider"]) + ".sy", lock=True)

    @at("end")
    def end_typo(loc):
        eval("rial", {}, dict(loc))  # noqa: S307

    return {
        "first_typo":                 (None, first_typo),
        "first_statement_locked_mid": (lock_pc2_tz, first_statement_locked_mid),
        "between_attr":               (None, between_attr),
        "list_bad_value":             (None, list_bad_value),
        "list_locked_mid":            (lock_pc2_tz, list_locked_mid),
        "container_scope":            (None, container_scope),
        "math_bad":                   (None, math_bad),
        "math_bad_list":              (None, math_bad_list),
        "typed_create":               (None, typed_create),
        "locked_connect":             (None, locked_connect),
        "srt_mid":                    (None, srt_mid),
        "last_statement":             (None, last_statement),
        "end_typo":                   (None, end_typo),
    }


class TestAnErrorMidBuild(_DFixCase):
    """The user's requirement (2026-10-15): "if rail_spine.py is executed with a typo in the
    code, undo should work". rail_spine with 3 riders and an error injected at each point of
    `_injections`, bare (as the Script Editor runs it) and inside ``rig.undo_chunk``; on the
    API route, undo step by step (0 dangling connections after every step, nothing printed),
    undo-all gives the setup scene, redo-all the partial scene, and a second undo-all the
    setup again; the error and the partial scene equal the cmds route's (R5M)."""

    U = (0.0, 0.5, 1.0)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rail = _injected_rail_spine()

    def setUp(self):
        super().setUp()
        self.folder = tempfile.mkdtemp(prefix="rig_r5_inject_")
        self.addCleanup(shutil.rmtree, self.folder, ignore_errors=True)
        options = dict(get_options())
        self.addCleanup(lambda: set_options(**options))

    def save(self, stem):
        path = os.path.join(self.folder, stem + ".ma").replace("\\", "/")
        cmds.file(rename=path)
        cmds.file(save=True, force=True, type="mayaAscii")
        return _order_free(path)

    def walk_all(self, undo, every):
        """Undo (or redo) until the queue is empty: the steps, the dangling connections found
        (after every ``every``-th step, and at the end)."""
        step, bad = 0, []
        empty = "undoQueueEmpty" if undo else "redoQueueEmpty"
        while not cmds.undoInfo(query=True, **{empty: True}):
            self.assertLess(step, 50000)
            cmds.undo() if undo else cmds.redo()
            step += 1
            if every and step % every == 0:
                bad.extend(_dangling())
        bad.extend(_dangling())
        return step, bad

    def run_case(self, name, chunk, api):
        setup, action = _injections()[name]
        self.fresh_scene()
        set_options(create_containers=True)
        pcs = List()
        for i in range(5):
            pcs.append(Node(cmds.spaceLocator()[0]))
            pcs[i].ty << i * 5
        if setup is not None:
            setup(pcs)
        cmds.flushUndo()
        record = {"S": self.save(name + "_S")}

        def build():
            self.rail._RIDER[0] = -1
            self.rail.INJECT["fn"] = action
            try:
                rail = self.rail.create_rail(pcs, list(self.U), scale_controls=pcs, orient_controls=pcs,
                                             degree=3, periodic=False, aim_axis=1, up_axis=0, debug=True)
                rail.stretch << 0
                pcs[1].tx << 10
            finally:
                self.rail.INJECT["fn"] = None

        error = None
        with self.route(api):
            try:
                if chunk:
                    with rig.undo_chunk("inject"):
                        build()
                else:
                    build()
            except Exception as raised:  # noqa: BLE001
                error = raised
            self.assertEqual((journal._S.open, journal._S.pending), (False, []))
            record["error"] = _brief(error)
            record["P"] = self.save(name + "_P")
            with _Printed() as printed:
                record["undo"] = self.walk_all(True, 1 if api else 0)
                record["E"] = self.save(name + "_E")
                record["redo"] = self.walk_all(False, 1 if api else 0)
                record["R"] = self.save(name + "_R")
                record["undo2"] = self.walk_all(True, 0)
                record["E2"] = self.save(name + "_E2")
            record["printed"] = [line for line in printed.lines if line[0] == "E"]
        return record

    def check(self, name, chunk):
        on  = self.run_case(name, chunk, True)
        off = self.run_case(name, chunk, False)
        what = f"{name} ({'chunk' if chunk else 'bare'})"
        if name != "none":
            self.assertIsNotNone(on["error"], what)
        self.assertEqual(on["error"], off["error"], what)
        self.assertEqual(on["printed"], [], what)
        self.assertEqual(on["undo"][1], [], what + ": dangling after an undo step")
        self.assertEqual(on["redo"][1], [], what + ": dangling after a redo step")
        self.assertTrue(on["E"] == on["S"], what + ": undo-all is not the setup scene")
        self.assertTrue(on["R"] == on["P"], what + ": redo-all is not the partial scene")
        self.assertTrue(on["E2"] == on["S"], what + ": the second undo-all is not the setup scene")
        self.assertTrue(on["P"] == off["P"], what + ": the partial scene is not the cmds route's")
        self.assertTrue(off["E"] == off["S"] and off["R"] == off["P"], what + ": the cmds route's walk")
        self.assertEqual(on["undo"][0], on["redo"][0], what)
        self.assertLessEqual(on["undo"][0], off["undo"][0], what)  # the approved batching

    def test_bare(self):
        for name in _injections():
            with self.subTest(name):
                self.check(name, chunk=False)

    def test_inside_an_undo_chunk(self):
        for name in _injections():
            with self.subTest(name):
                self.check(name, chunk=True)

    def test_the_injections_fire(self):
        # every hook is reached by the build (a renamed anchor would leave a case untested)
        seen = set()
        self.fresh_scene()
        set_options(create_containers=True)
        pcs = List()
        for i in range(5):
            pcs.append(Node(cmds.spaceLocator()[0]))
            pcs[i].ty << i * 5
        self.rail._RIDER[0] = -1
        self.rail.INJECT["fn"] = lambda label, rider, loc: seen.add((label, rider))
        try:
            self.rail.create_rail(pcs, list(self.U), scale_controls=pcs, orient_controls=pcs, degree=3,
                                  periodic=False, aim_axis=1, up_axis=0, debug=True)
        finally:
            self.rail.INJECT["fn"] = None
        for hook in (("start", -1), ("after_knobs", -1), ("after_cumsum", -1), ("loop_mid", 1),
                     ("loop_end", 1), ("loop_end", 2), ("end", 2)):
            self.assertIn(hook, seen)

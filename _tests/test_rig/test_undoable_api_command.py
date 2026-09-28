"""rig's undoable API command (``rig/nodetypes/plugins/undoable_api_command.py``) and
the plug-in helpers of ``rig.nodetypes.plugins`` (``load_plugin``, ``_run_undoable``).

The command is ``rigUndoableAPICommand`` (a name of rig's own); the object is handed
over per call and nothing keeps it after its undo entry; ``cmds`` calls in undoIt /
redoIt are not recorded; the command is wrapped once across a reload or a module
import; rig's plug-in loads by full path, once per session. Undo tests check the scene
(``rig._tests._undo``), never ``cmds.undo()``'s return value.
"""

import gc
import importlib
import os
import shutil
import sys
import tempfile
import time
import weakref
from unittest import mock

from maya import cmds, mel
from maya.api import OpenMaya

from rig.nodetypes import plugins
from rig.nodetypes.plugins import (
    UNDOABLE_API_COMMAND,
    _run_undoable,
    bundled_plugin_path,
    load_plugin,
)
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk

PLUGIN      = "undoable_api_command"
PLUGIN_PATH = bundled_plugin_path(PLUGIN)
MODULE      = "rig.nodetypes.plugins.undoable_api_command"

# a scratch plug-in that registers runUndoableAPICommand the way MPyNode's mpynode_api2
# does (its registration block, reduced): the name rig used to register too
_CLASH_SOURCE = '''
import functools
import maya.api.OpenMaya as om2
from maya import cmds

NAME = "runUndoableAPICommand"


def maya_useNewAPI():
    pass


class UndoableAPICommand(om2.MPxCommand):
    call_class = None

    def isUndoable(self):
        return True

    def doIt(self, args):
        self.py_class = self.call_class
        self.py_class.doIt()

    def redoIt(self):
        self.py_class.redoIt()

    def undoIt(self):
        state = cmds.undoInfo(query=True, state=True)
        if state:
            cmds.undoInfo(stateWithoutFlush=False)
        try:
            self.py_class.undoIt()
        finally:
            if state:
                cmds.undoInfo(stateWithoutFlush=True)


def initializePlugin(plugin):
    fn = om2.MFnPlugin(plugin, "clash", "1.0", "Any")
    try:
        fn.registerCommand(NAME, UndoableAPICommand)
    except Exception:
        pass
    cmd_func = getattr(cmds, NAME)

    @functools.wraps(cmd_func)
    def wrapped(py_class):
        UndoableAPICommand.call_class = py_class
        cmds.undoInfo(openChunk=True, chunkName=NAME)
        try:
            return cmd_func()
        finally:
            cmds.undoInfo(closeChunk=True)

    setattr(cmds, NAME, wrapped)


def uninitializePlugin(plugin):
    om2.MFnPlugin(plugin).deregisterCommand(NAME)
'''


def _plug(name):
    sel = OpenMaya.MSelectionList()
    sel.add(name)
    return sel.getPlug(0)


def _same_path(a, b):
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def command():
    """``cmds.rigUndoableAPICommand`` (the wrapper), fetched now."""
    return getattr(cmds, UNDOABLE_API_COMMAND)


def registered_class():
    """The command class the loaded plug-in registered (the one its wrapper hands to)."""
    for cell in command().run.__closure__:
        if isinstance(cell.cell_contents, type):
            return cell.cell_contents
    raise AssertionError("the wrapper holds no command class")


class SetDouble:
    """An API-only edit of one plug. With ``counter``, undoIt / redoIt also step that
    plug through ``cmds.setAttr`` (-1 / +1)."""

    runs = 0

    def __init__(self, name, value, counter=None, result=None, fail=False):
        self.plug, self.value, self.old = _plug(name), value, None
        self.counter, self.result, self.fail = counter, result, fail

    def doIt(self):
        SetDouble.runs += 1
        self.old = self.plug.asDouble()
        self.plug.setDouble(self.value)
        if self.fail:
            raise ValueError("doIt failed (test)")
        return self.result

    def undoIt(self):
        self.plug.setDouble(self.old)
        if self.counter:
            cmds.setAttr(self.counter, cmds.getAttr(self.counter) - 1)

    def redoIt(self):
        self.plug.setDouble(self.value)
        if self.counter:
            cmds.setAttr(self.counter, cmds.getAttr(self.counter) + 1)


class _CommandCase(UndoWalk, MayaTestCase):
    """A new scene with a transform ``t``; rig's plug-in loaded; the queue on,
    infinite and flushed."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        plugins._ensure_loaded(PLUGIN, True)
        cmds.createNode("transform", name="t")
        cmds.flushUndo()

    def tearDown(self):
        try:
            cmds.flushUndo()
        finally:
            super().tearDown()

    @staticmethod
    def values(attrs=("tx", "ty", "tz"), node="t"):
        return tuple(cmds.getAttr(f"{node}.{a}") for a in attrs)

    def unload(self):
        """Flush the queue and unload rig's plug-in (its command leaves maya.cmds)."""
        cmds.flushUndo()
        cmds.unloadPlugin(PLUGIN)
        self.assertFalse(hasattr(cmds, UNDOABLE_API_COMMAND))


class TestUndoableAPICommand(_CommandCase):
    """The command's name, its two call routes and the steps they make."""

    def test_registers_rigs_own_command_name(self):
        self.assertEqual(UNDOABLE_API_COMMAND, "rigUndoableAPICommand")
        self.assertEqual(cmds.pluginInfo(PLUGIN, query=True, command=True), [UNDOABLE_API_COMMAND])
        self.assertTrue(_same_path(cmds.pluginInfo(PLUGIN, query=True, path=True), PLUGIN_PATH))
        # the wrapper wraps Maya's builtin once
        self.assertTrue(hasattr(command(), "__wrapped__"))
        self.assertFalse(hasattr(command().__wrapped__, "__wrapped__"))

    def test_wrapped_call_is_one_named_step_with_the_cmds_of_doit(self):
        class WithCmds(SetDouble):
            def doIt(self):
                cmds.setAttr("t.sx", 2.0)
                return super().doIt()

        command()(WithCmds("t.tx", 1.0))
        self.assertEqual(self.values(("tx", "sx")), (1.0, 2.0))
        self.assertEqual(self.undo_name(), UNDOABLE_API_COMMAND)
        self.undo_steps(1)
        self.assertEqual(self.values(("tx", "sx")), (0.0, 1.0))
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
        self.redo_steps(1)
        self.assertEqual(self.values(("tx", "sx")), (1.0, 2.0))

    def test_run_undoable_is_one_unnamed_step(self):
        cmds.setAttr("t.sx", 2.0)
        _run_undoable(SetDouble("t.tx", 1.0))
        self.assertEqual(self.undo_name(), "")
        self.undo_steps(1)
        self.assertEqual(self.values(("tx", "sx")), (0.0, 2.0))
        self.undo_steps(1)
        self.assertEqual(self.values(("tx", "sx")), (0.0, 1.0))
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
        self.redo_steps(2)
        self.assertEqual(self.values(("tx", "sx")), (1.0, 2.0))

    def test_both_routes_return_the_result_of_doit(self):
        self.assertEqual(command()(SetDouble("t.tx", 1.0, result="abc")), ["abc"])
        self.assertEqual(_run_undoable(SetDouble("t.ty", 2.0, result="def")), ["def"])
        self.assertIsNone(_run_undoable(SetDouble("t.tz", 3.0)))

    def test_nested_hand_off(self):
        class Outer(SetDouble):
            def doIt(self):
                result = super().doIt()
                _run_undoable(SetDouble("t.ty", 2.0))
                return result

        _run_undoable(Outer("t.tx", 1.0))
        command()(Outer("t.tz", 3.0))
        self.assertEqual(self.values(), (1.0, 2.0, 3.0))
        self.assertIsNone(registered_class().call_class)
        self.undo_all()
        self.assertEqual(self.values(), (0.0, 0.0, 0.0))
        self.redo_all()
        self.assertEqual(self.values(), (1.0, 2.0, 3.0))

    def test_cmds_in_undoit_and_redoit_are_not_recorded(self):
        # recorded, the cmds of a redoIt flushed the redo queue: redo stopped after one step
        cmds.addAttr("t", longName="counter", attributeType="double")
        cmds.flushUndo()
        _run_undoable(SetDouble("t.tx", 1.0, counter="t.counter"))
        command()(SetDouble("t.ty", 2.0, counter="t.counter"))
        for cycle in range(2):
            self.undo_steps(2)
            self.assertEqual(self.values(("tx", "ty", "counter")), (0.0, 0.0, -2.0), f"cycle {cycle}")
            self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
            self.redo_steps(2)
            self.assertEqual(self.values(("tx", "ty", "counter")), (1.0, 2.0, 0.0), f"cycle {cycle}")
            self.assertTrue(cmds.undoInfo(query=True, redoQueueEmpty=True))
        self.assertTrue(cmds.undoInfo(query=True, state=True))


class TestUndoableAPICommandHandOff(_CommandCase):
    """The object is handed over for one call: nothing keeps it, a bare call refuses."""

    def test_slot_is_cleared_after_each_route(self):
        command()(SetDouble("t.tx", 1.0))
        self.assertIsNone(registered_class().call_class)
        _run_undoable(SetDouble("t.ty", 2.0))
        self.assertIsNone(registered_class().call_class)

    def test_the_object_lives_while_queued_and_no_longer(self):
        refs = []
        for route in (command(), _run_undoable):
            obj = SetDouble("t.tx", len(refs) + 1.0)
            refs.append(weakref.ref(obj))
            route(obj)
            del obj
        gc.collect()
        self.assertTrue(all(ref() is not None for ref in refs))  # the queued commands hold them
        cmds.flushUndo()
        gc.collect()
        self.assertEqual([ref() for ref in refs], [None, None])

    def test_a_new_scene_releases_the_last_object(self):
        obj = SetDouble("t.tx", 1.0)
        ref = weakref.ref(obj)
        _run_undoable(obj)
        del obj
        cmds.file(new=True, force=True)
        gc.collect()
        self.assertIsNone(ref())

    def test_undo_off_applies_and_keeps_nothing(self):
        cmds.undoInfo(state=False)
        obj = SetDouble("t.tx", 1.0)
        ref = weakref.ref(obj)
        _run_undoable(obj)
        del obj
        gc.collect()
        self.assertEqual(cmds.getAttr("t.tx"), 1.0)
        self.assertIsNone(ref())
        self.assertIsNone(registered_class().call_class)

    def test_a_bare_call_refuses_and_never_reruns_the_last_object(self):
        SetDouble.runs = 0
        _run_undoable(SetDouble("t.tx", 1.0))
        cmds.setAttr("t.tx", 5.0)
        with self.assertRaisesRegex(RuntimeError, "no object to run"):
            command().__wrapped__()
        with self.assertRaises(RuntimeError):
            mel.eval(UNDOABLE_API_COMMAND)
        self.assertEqual(SetDouble.runs, 1)
        self.assertEqual(cmds.getAttr("t.tx"), 5.0)
        # the refused calls queued nothing: setAttr, then the edit
        self.undo_steps(1)
        self.assertEqual(cmds.getAttr("t.tx"), 1.0)
        self.undo_steps(1)
        self.assertEqual(cmds.getAttr("t.tx"), 0.0)
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_a_failing_doit_queues_nothing_and_keeps_nothing(self):
        cmds.setAttr("t.sx", 2.0)
        obj = SetDouble("t.ty", 2.0, fail=True)
        ref = weakref.ref(obj)
        with self.assertRaises(RuntimeError):
            _run_undoable(obj)
        del obj
        gc.collect()
        self.assertIsNone(ref())
        self.assertIsNone(registered_class().call_class)
        self.undo_steps(1)  # the setAttr: the failed call left no entry
        self.assertEqual(cmds.getAttr("t.sx"), 1.0)
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))


class TestUndoableAPICommandLifecycle(_CommandCase):
    """Unload / reload, a wrapper kept across a reload, a module import of the plug-in
    file, a re-import of rig.nodetypes.plugins."""

    def test_unload_is_refused_while_an_entry_is_queued(self):
        _run_undoable(SetDouble("t.tx", 1.0))
        with self.assertRaisesRegex(RuntimeError, "in use"):
            cmds.unloadPlugin(PLUGIN)
        self.assertTrue(hasattr(cmds, UNDOABLE_API_COMMAND))
        self.undo_steps(1)
        self.assertEqual(cmds.getAttr("t.tx"), 0.0)
        self.redo_steps(1)
        self.assertEqual(cmds.getAttr("t.tx"), 1.0)

    def test_reload_by_path_wraps_once(self):
        for cycle in range(2):
            before = command()
            self.unload()
            _run_undoable(SetDouble("t.tx", cycle + 1.0))
            self.assertIsNot(command(), before)
            self.assertTrue(_same_path(cmds.pluginInfo(PLUGIN, query=True, path=True), PLUGIN_PATH))
            self.assertFalse(hasattr(command().__wrapped__, "__wrapped__"))
            self.undo_steps(1)
            self.assertEqual(cmds.getAttr("t.tx"), float(cycle))
            self.redo_steps(1)
            self.assertEqual(cmds.getAttr("t.tx"), cycle + 1.0)

    def test_a_wrapper_kept_across_a_reload_hands_over(self):
        # its own builtin is gone: calling it would crash Maya
        kept = command()
        self.unload()
        _run_undoable(SetDouble("t.tx", 1.0))
        kept.run(SetDouble("t.ty", 2.0))
        kept(SetDouble("t.tz", 3.0))
        self.assertEqual(self.values(), (1.0, 2.0, 3.0))
        self.assertIsNone(registered_class().call_class)
        self.undo_steps(3)
        self.assertEqual(self.values(), (0.0, 0.0, 0.0))
        self.redo_steps(3)
        self.assertEqual(self.values(), (1.0, 2.0, 3.0))
        self.unload()
        with self.assertRaisesRegex(RuntimeError, "not loaded"):
            kept.run(SetDouble("t.sx", 5.0))
        self.assertEqual(cmds.getAttr("t.sx"), 1.0)

    def test_the_plugin_file_imported_as_a_module_never_wraps_twice(self):
        package = sys.modules["rig.nodetypes.plugins"]
        had = sys.modules.pop(MODULE, None)
        wrapper = command()
        try:
            module = importlib.import_module(MODULE)
            self.assertIsNot(module.UndoableAPICommand, registered_class())
            self.assertEqual(module.COMMAND_NAME, UNDOABLE_API_COMMAND)
            module.UndoableAPICommand.wrap_command()
            self.assertIs(command(), wrapper)
            _run_undoable(SetDouble("t.tx", 1.0))
            command()(SetDouble("t.ty", 2.0))
            self.assertEqual(self.values(("tx", "ty")), (1.0, 2.0))
            self.assertIsNone(module.UndoableAPICommand.call_class)
            self.undo_steps(2)
            self.assertEqual(self.values(("tx", "ty")), (0.0, 0.0))
        finally:
            sys.modules.pop(MODULE, None)
            if had is not None:
                sys.modules[MODULE] = had
            elif hasattr(package, "undoable_api_command"):
                delattr(package, "undoable_api_command")

    def test_a_reimport_of_the_plugins_package_uses_the_loaded_command(self):
        old = sys.modules["rig.nodetypes.plugins"]
        parent = sys.modules["rig.nodetypes"]
        wrapper = command()
        try:
            del sys.modules["rig.nodetypes.plugins"]
            new = importlib.import_module("rig.nodetypes.plugins")
            self.assertIsNot(new, old)
            new._run_undoable(SetDouble("t.tx", 1.0))
            old._run_undoable(SetDouble("t.ty", 2.0))
            self.assertIs(command(), wrapper)  # no reload
            self.assertEqual(self.values(("tx", "ty")), (1.0, 2.0))
            self.undo_steps(2)
            self.assertEqual(self.values(("tx", "ty")), (0.0, 0.0))
        finally:
            sys.modules["rig.nodetypes.plugins"] = old
            parent.plugins = old


class TestUndoableAPICommandNameClash(_CommandCase):
    """A plug-in that registers runUndoableAPICommand (MPyNode's name) loaded before or
    after rig's: both commands keep working."""

    def setUp(self):
        super().setUp()
        if hasattr(cmds, "runUndoableAPICommand"):
            self.skipTest("runUndoableAPICommand is registered already (MPyNode loaded?)")
        self.folder = tempfile.mkdtemp(prefix="rig_clash_")
        self.clash_path = os.path.join(self.folder, "rig_test_clash.py")
        with open(self.clash_path, "w", encoding="utf-8") as fh:
            fh.write(_CLASH_SOURCE)

    def tearDown(self):
        try:
            cmds.flushUndo()
            if cmds.pluginInfo("rig_test_clash", query=True, loaded=True):
                cmds.unloadPlugin("rig_test_clash")
            shutil.rmtree(self.folder, ignore_errors=True)
        finally:
            super().tearDown()

    def check_both(self):
        _run_undoable(SetDouble("t.tx", 1.0))
        command()(SetDouble("t.ty", 2.0))
        cmds.runUndoableAPICommand(SetDouble("t.tz", 3.0))
        self.assertEqual(self.values(), (1.0, 2.0, 3.0))
        self.assertEqual(cmds.pluginInfo(PLUGIN, query=True, command=True), [UNDOABLE_API_COMMAND])
        self.assertEqual(
            cmds.pluginInfo("rig_test_clash", query=True, command=True), ["runUndoableAPICommand"]
        )
        self.undo_steps(3)
        self.assertEqual(self.values(), (0.0, 0.0, 0.0))
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
        self.redo_steps(3)
        self.assertEqual(self.values(), (1.0, 2.0, 3.0))

    def test_other_plugin_loaded_first(self):
        self.unload()
        cmds.loadPlugin(self.clash_path, quiet=True)
        self.check_both()

    def test_rig_loaded_first(self):
        cmds.loadPlugin(self.clash_path, quiet=True)
        self.check_both()


class TestLoadPlugin(_CommandCase):
    """load_plugin: rig's plug-in by full path and no query while it is loaded;
    unload_on_exit; a load that registers nothing raises."""

    def test_loaded_rig_plugin_is_neither_queried_nor_loaded(self):
        with mock.patch.object(cmds, "pluginInfo") as info, mock.patch.object(cmds, "loadPlugin") as load:
            with load_plugin(PLUGIN):
                pass
            _run_undoable(SetDouble("t.tx", 1.0))
        info.assert_not_called()
        load.assert_not_called()

    def test_rig_plugin_loads_by_full_path(self):
        self.unload()
        with mock.patch.object(cmds, "loadPlugin", wraps=cmds.loadPlugin) as load:
            with load_plugin(PLUGIN):
                self.assertTrue(hasattr(cmds, UNDOABLE_API_COMMAND))
        self.assertEqual(load.call_count, 1)
        self.assertTrue(_same_path(load.call_args[0][0], PLUGIN_PATH))

    def test_a_load_that_registers_nothing_raises(self):
        self.unload()
        with mock.patch.object(cmds, "loadPlugin", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "did not register rigUndoableAPICommand"):
                _run_undoable(SetDouble("t.tx", 1.0))
            with self.assertRaisesRegex(RuntimeError, "did not register rigUndoableAPICommand"):
                with load_plugin(PLUGIN):
                    pass
        self.assertEqual(cmds.getAttr("t.tx"), 0.0)

    def test_an_unknown_name_raises_mayas_error(self):
        with self.assertRaisesRegex(RuntimeError, "noSuchPlugin"):
            with load_plugin("noSuchPlugin"):
                pass

    def test_unload_on_exit_unloads_only_what_it_loaded(self):
        # loaded before the block: stays loaded
        with load_plugin(PLUGIN, unload_on_exit=True):
            pass
        self.assertTrue(cmds.pluginInfo(PLUGIN, query=True, loaded=True))
        # loaded by the block: unloaded after it (it raised TypeError: no quiet flag)
        self.unload()
        with load_plugin(PLUGIN, unload_on_exit=True):
            self.assertTrue(cmds.pluginInfo(PLUGIN, query=True, loaded=True))
        self.assertFalse(cmds.pluginInfo(PLUGIN, query=True, loaded=True))
        self.assertFalse(hasattr(cmds, UNDOABLE_API_COMMAND))

    def test_unload_on_exit_refused_is_a_warning(self):
        self.unload()
        with mock.patch.object(cmds, "warning") as warning:
            with load_plugin(PLUGIN, unload_on_exit=True):
                _run_undoable(SetDouble("t.tx", 1.0))
        self.assertEqual(warning.call_count, 1)
        self.assertIn("could not unload", warning.call_args[0][0])
        self.assertTrue(cmds.pluginInfo(PLUGIN, query=True, loaded=True))
        self.undo_steps(1)
        self.assertEqual(cmds.getAttr("t.tx"), 0.0)


class TestUndoableAPICommandCost(_CommandCase):
    """The lean route rig's edits take costs a fraction of the chunked one."""

    REPS, CALLS = 5, 300

    def best(self, call):
        best = float("inf")
        for _ in range(self.REPS):
            objs = [SetDouble("t.tx", 1.0) for _ in range(self.CALLS)]
            cmds.flushUndo()
            start = time.perf_counter()
            for obj in objs:
                call(obj)
            best = min(best, (time.perf_counter() - start) / self.CALLS * 1e6)
        return best

    def test_lean_call_costs_less_than_half_the_chunked_call(self):
        def chunked(obj):
            with load_plugin(PLUGIN):
                command()(obj)

        lean  = self.best(_run_undoable)
        wrapped = self.best(chunked)
        self.assertLess(lean, 0.5 * wrapped, f"lean {lean:.2f} us, chunked {wrapped:.2f} us per call")

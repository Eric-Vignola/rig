"""Round 3, decision D-A: the memo caches follow the scene.

Every new scene and file open clears the ``@memoize`` / ``NodeOp`` / seed
caches, a reference unload, reload or remove prunes the entries whose nodes it
freed, and the callbacks that do it are registered once per Maya session --
never twice after a re-import, and not at all before Maya is initialised
(see ``rig._internal.callbacks``).
"""

import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from unittest import mock

from maya import cmds
from maya.api import OpenMaya as om
from rig import Node, Plug, constant
from rig._internal import callbacks
from rig._internal import memoize as memoize_module
from rig._internal.memoize import memoize
from rig.nodetypes import _base
from rig._tests._base import MayaTestCase, is_standalone

MEMO  = memoize_module.__name__
BASE  = _base.__name__
TOP   = MEMO.split(".")[0]


def _holders(module=memoize_module):
    return module._ALL_MEMOIZED + module._ALL_NODEOP_CACHES


def _entries(module=memoize_module):
    """Every memo entry (of every @memoize, NodeOp and seed cache) of `module`."""
    return [
        entry
        for holder in _holders(module)
        for entry in (getattr(holder, "_cache", None) or {}).values()
    ]


def _holds(value):
    """True if some memo entry returns exactly `value`."""
    return any(entry.value is value for entry in _entries())


def _unloaded_plugin():
    """A plug-in that is not loaded (to load and unload), or None."""
    for plugin in ("invertShape", "curveWarp", "quatNodes"):
        try:
            if not cmds.pluginInfo(plugin, query=True, loaded=True):
                return plugin
        except RuntimeError:
            continue
    return None


def _save_scene(path, *nodes):
    cmds.file(new=True, force=True)
    for node in nodes:
        cmds.createNode("transform", name=node)
    cmds.file(rename=path)
    cmds.file(save=True, type="mayaAscii", force=True)


class _UserMemo:
    """User @memoize functions made for one test, taken out of the registry after."""

    def __init__(self, case):
        self.calls = []
        count      = len(memoize_module._ALL_MEMOIZED)

        @memoize
        def label(x):
            # a plain str result: no node, so no handle to prune it by
            self.calls.append(("label", x))
            return f"ctrl_{x}"

        @memoize
        def clock():
            # the only handle is time1, a default node a new scene keeps
            self.calls.append(("clock",))
            return Node("time1").outTime

        @memoize
        def passthrough(plug):
            self.calls.append(("passthrough", str(plug)))
            return plug

        self.label, self.clock, self.passthrough = label, clock, passthrough
        added = memoize_module._ALL_MEMOIZED[count:]
        case.addCleanup(
            lambda: [memoize_module._ALL_MEMOIZED.remove(w) for w in added
                     if w in memoize_module._ALL_MEMOIZED]
        )


class TestNewSceneClearsTheCaches(MayaTestCase):
    """kBeforeNew / kBeforeOpen clear every cache: all of the scene's nodes are
    about to be freed, so no entry can stay valid."""

    TEST_START_NEW_SCENE = True

    def _build(self):
        memo = _UserMemo(self)
        a    = Node(cmds.createNode("transform", name="a"))
        b    = Node(cmds.createNode("transform", name="b"))
        built = [constant(42.0), a.tx + b.tx, memo.label("hand"), memo.clock()]
        self.assertTrue(all(_holds(value) for value in built))
        memo.kept = [e for e in _entries() if e.value is built[2] or e.value is built[3]]
        return memo

    def test_new_scene_clears_every_cache(self):
        memo = self._build()
        cmds.file(new=True, force=True)
        self.assertEqual(_entries(), [])
        # including the entries a prune keeps: a str result, which has no handle,
        # and a result whose only handle is time1, which a new scene keeps
        self.assertEqual([len(e.handles) for e in memo.kept], [0, 1])
        self.assertTrue(all(h.isAlive() and h.isValid() for e in memo.kept for h in e.handles))
        # so the functions run again in the new scene
        memo.label("hand")
        memo.clock()
        self.assertEqual(memo.calls, [("label", "hand"), ("clock",)] * 2)

    def test_file_open_clears_every_cache(self):
        folder = tempfile.mkdtemp(prefix="rig_clear_open_")
        path   = os.path.join(folder, "clear_open.ma").replace("\\", "/")
        try:
            _save_scene(path, "kept")
            memo = self._build()
            cmds.file(path, open=True, force=True)
            self.assertEqual(_entries(), [])
            memo.label("hand")
            self.assertEqual(memo.calls.count(("label", "hand")), 2)
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)

    def test_a_new_scene_or_open_that_does_not_happen_keeps_the_caches(self):
        folder = tempfile.mkdtemp(prefix="rig_keep_")
        path   = os.path.join(folder, "keep.ma").replace("\\", "/")
        try:
            _save_scene(path, "kept")
            cmds.file(new=True, force=True)
            a    = Node(cmds.createNode("transform", name="a"))
            b    = Node(cmds.createNode("transform", name="b"))
            summ = a.tx + b.tx
            self.assertTrue(cmds.file(query=True, modified=True))

            def refused_new():
                cmds.file(new=True)                       # "Unsaved changes"

            def refused_open():
                cmds.file(path, open=True)                # "Unsaved changes"

            def missing_open():
                cmds.file(os.path.join(folder, "missing.ma"), open=True, force=True)

            def aborted(check, add, operation):
                def run():
                    callback_id = add(check, lambda *args: False)
                    try:
                        operation()
                    finally:
                        om.MMessage.removeCallback(callback_id)
                return run

            cases = {
                "new, unsaved changes":  (refused_new, RuntimeError),
                "open, unsaved changes": (refused_open, RuntimeError),
                "open, missing file":    (missing_open, RuntimeError),
                "new, check aborts": (
                    aborted(om.MSceneMessage.kBeforeNewCheck,
                            om.MSceneMessage.addCheckCallback,
                            lambda: cmds.file(new=True, force=True)),
                    None,
                ),
                "open, check aborts": (
                    aborted(om.MSceneMessage.kBeforeOpenCheck,
                            om.MSceneMessage.addCheckFileCallback,
                            lambda: cmds.file(path, open=True, force=True)),
                    None,
                ),
            }
            for case, (operation, error) in cases.items():
                with self.subTest(case=case):
                    if error is None:
                        operation()
                    else:
                        self.assertRaises(error, operation)
                    self.assertTrue(cmds.objExists("a"))
                    self.assertTrue(_holds(summ))
                    nodes = set(cmds.ls())
                    self.assertIs(a.tx + b.tx, summ)       # still deduped
                    self.assertEqual(set(cmds.ls()), nodes)
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)

    def test_each_message_runs_its_callback_once(self):
        folder = tempfile.mkdtemp(prefix="rig_once_")
        path   = os.path.join(folder, "once.ma").replace("\\", "/")
        ref    = os.path.join(folder, "once_ref.ma").replace("\\", "/")
        try:
            _save_scene(ref, "refT")
            _save_scene(path, "kept")
            cmds.file(new=True, force=True)

            def reference_node():
                return cmds.referenceQuery(ref, referenceNode=True)

            with mock.patch.object(
                memoize_module, "_clear_all_caches", wraps=memoize_module._clear_all_caches
            ) as clear, mock.patch.object(
                memoize_module, "prune_memoize_caches", wraps=memoize_module.prune_memoize_caches
            ) as prune:
                steps = [
                    ("new",       lambda: cmds.file(new=True, force=True),       1, 1),
                    ("open",      lambda: cmds.file(path, open=True, force=True), 1, 1),
                    ("reference", lambda: cmds.file(ref, reference=True, namespace="ref"), 0, 0),
                    ("unload",    lambda: cmds.file(unloadReference=reference_node()), 0, 1),
                    ("load",      lambda: cmds.file(loadReference=reference_node()),   0, 0),
                    ("reload",    lambda: cmds.file(loadReference=reference_node()),   0, 1),
                    ("remove",    lambda: cmds.file(ref, removeReference=True),        0, 1),
                    ("import",    lambda: cmds.file(ref, i=True, namespace="imp"),     0, 0),
                ]
                for step, operation, clears, prunes in steps:
                    with self.subTest(step=step):
                        clear.reset_mock()
                        prune.reset_mock()
                        operation()
                        self.assertEqual((clear.call_count, prune.call_count), (clears, prunes))
            self.assertEqual(len(callbacks.registered_ids(MEMO)), 6)
            self.assertEqual(len(callbacks.registered_ids(BASE)), 2)
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)


class TestReferenceUnloadPrunes(MayaTestCase):
    """An unload, reload or remove frees the reference's nodes only: the entries
    that hold one are dropped, every other entry keeps deduping."""

    def test_unload_prunes_only_the_freed_entries(self):
        folder = tempfile.mkdtemp(prefix="rig_unload_prune_")
        path   = os.path.join(folder, "unload_prune.ma").replace("\\", "/")
        memo   = _UserMemo(self)
        try:
            _save_scene(path, "refT")
            operations = {
                "unload": lambda: cmds.file(unloadReference="refRN"),
                "reload": lambda: cmds.file(loadReference="refRN"),
                "remove": lambda: cmds.file(path, removeReference=True),
            }
            for case, operation in operations.items():
                with self.subTest(case=case):
                    cmds.file(new=True, force=True)
                    cmds.file(path, reference=True, namespace="ref")
                    loc   = Node(cmds.createNode("transform", name="loc"))
                    local = loc.tx + loc.ty
                    fed   = Node("ref:refT").tx + 1        # a local node fed by the reference
                    held  = memo.passthrough(Node("ref:refT").tx)
                    label = memo.label("loc")
                    self.assertTrue(all(_holds(v) for v in (local, fed, held, label)))
                    operation()
                    # the entry that returned the reference's plug is gone ...
                    self.assertFalse(_holds(held))
                    # ... and every other one is kept, and still dedupes
                    self.assertTrue(all(_holds(v) for v in (local, fed, label)))
                    self.assertEqual(
                        [e for e in _entries()
                         if not all(h.isAlive() and h.isValid() for h in e.handles)], []
                    )
                    nodes = set(cmds.ls())
                    self.assertIs(loc.tx + loc.ty, local)
                    self.assertEqual(set(cmds.ls()), nodes)
                    self.assertEqual(memo.label("loc"), "ctrl_loc")
            self.assertEqual(memo.calls.count(("label", "loc")), 3)
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)


class TestCallbackRegistry(MayaTestCase):
    """`rig._internal.callbacks`: one registration per module name, kept on `sys`."""

    OWNER = "rig._tests.scene_callbacks_probe"

    def setUp(self):
        super().setUp()
        self.addCleanup(self._drop)

    def _drop(self):
        entry = callbacks.registry().pop(self.OWNER, None)
        if entry:
            callbacks._remove(entry["ids"])

    def _counting(self, calls, label):
        return [(om.MSceneMessage.addCallback, om.MSceneMessage.kBeforeNew,
                 lambda *args: calls.append(label))]

    def test_register_replaces_and_releases_a_purged_copy(self):
        calls, released = [], []
        first, second = object(), object()   # stand for two imports of one module
        self.assertTrue(callbacks.register(
            self.OWNER, first, self._counting(calls, "first"), lambda: released.append("first")))
        # the registry lives on sys, which a purge of the rig modules keeps
        self.assertIs(getattr(sys, callbacks.REGISTRY_ATTR), callbacks.registry())
        self.assertEqual(len(callbacks.registered_ids(self.OWNER)), 1)
        cmds.file(new=True, force=True)
        self.assertEqual(calls, ["first"])

        # a pending registration never takes over one that is made
        self.assertTrue(callbacks.ensure(self.OWNER, second, self._counting(calls, "stale")))
        cmds.file(new=True, force=True)
        self.assertEqual(calls, ["first"] * 2)

        # another copy replaces the callbacks and releases the first one's caches
        self.assertTrue(callbacks.register(
            self.OWNER, second, self._counting(calls, "second"), lambda: released.append("second")))
        self.assertEqual(released, ["first"])
        cmds.file(new=True, force=True)
        self.assertEqual(calls, ["first"] * 2 + ["second"])

        # the same module again (an in-place reload) replaces without a release
        self.assertTrue(callbacks.register(
            self.OWNER, second, self._counting(calls, "reloaded"), lambda: released.append("x")))
        self.assertEqual(released, ["first"])
        cmds.file(new=True, force=True)
        self.assertEqual(calls, ["first"] * 2 + ["second", "reloaded"])
        self.assertEqual(len(callbacks.registered_ids(self.OWNER)), 1)

    def test_nothing_is_registered_before_maya_is_initialised(self):
        calls = []
        with mock.patch.object(callbacks, "maya_is_initialised", return_value=False):
            self.assertFalse(callbacks.register(self.OWNER, object(), self._counting(calls, "x")))
            self.assertFalse(callbacks.ensure(self.OWNER, object(), self._counting(calls, "x")))
        self.assertNotIn(self.OWNER, callbacks.registry())
        cmds.file(new=True, force=True)
        self.assertEqual(calls, [])

    def test_a_failed_registration_leaves_no_callback(self):
        calls = []

        def broken(message, callback):
            raise RuntimeError("cannot add")

        specs = self._counting(calls, "added") + [(broken, None, None)]
        self.assertRaises(RuntimeError, callbacks.register, self.OWNER, object(), specs)
        self.assertNotIn(self.OWNER, callbacks.registry())
        # a pending registration, made from a cache fill, stays pending instead
        self.assertFalse(callbacks.ensure(self.OWNER, object(), specs))
        self.assertNotIn(self.OWNER, callbacks.registry())
        cmds.file(new=True, force=True)
        self.assertEqual(calls, [])


class TestPendingRegistration(MayaTestCase):
    """rig imported before Maya was initialised: nothing is registered, and the
    first cache fill registers what the import could not."""

    TEST_START_NEW_SCENE = True

    def _unregister(self, owner, reregister):
        entry = callbacks.registry().pop(owner)
        callbacks._remove(entry["ids"])

        def restore():
            if owner not in callbacks.registry():
                reregister()
        self.addCleanup(restore)

    def test_memo_callbacks_are_registered_on_the_first_entry(self):
        self._unregister(MEMO, memoize_module._register_scene_callbacks)
        with mock.patch.object(callbacks, "maya_is_initialised", return_value=False):
            self.assertFalse(memoize_module._register_scene_callbacks())
            memoize_module._CacheEntry(value=None, handles=[])
            self.assertFalse(memoize_module._SCENE_CALLBACKS_READY)
        self.assertEqual(callbacks.registered_ids(MEMO), [])
        a    = Node(cmds.createNode("transform", name="a"))
        summ = a.tx + a.ty
        self.assertTrue(memoize_module._SCENE_CALLBACKS_READY)
        self.assertEqual(len(callbacks.registered_ids(MEMO)), 6)
        self.assertTrue(_holds(summ))
        cmds.file(new=True, force=True)
        self.assertEqual(_entries(), [])

    def test_static_data_types_wait_for_the_plugin_callbacks(self):
        self._unregister(BASE, _base._register_plugin_callbacks)
        cmds.createNode("multiplyDivide", name="md")
        expected = cmds.getAttr("md.input1X", type=True)
        _base._STATIC_DATA_TYPE.clear()
        with mock.patch.object(callbacks, "maya_is_initialised", return_value=False):
            self.assertFalse(_base._register_plugin_callbacks())
            # answered, but not shared, while no plug-in callback can clear it
            self.assertEqual(Plug("md.input1X").data_type, expected)
            self.assertEqual(_base._STATIC_DATA_TYPE, {})
        self.assertEqual(Plug("md.input1X").data_type, expected)
        self.assertTrue(_base._PLUGIN_CALLBACKS_READY)
        self.assertEqual(len(callbacks.registered_ids(BASE)), 2)
        self.assertTrue(_base._STATIC_DATA_TYPE)

    def test_import_without_maya_initialised(self):
        # d6ad8b2 crashed mayapy here: adding a scene callback before
        # maya.standalone.initialize() is an access violation
        if not is_standalone():
            self.skipTest("needs mayapy to start an uninitialised interpreter")
        mayapy = sys.executable
        code = (
            "import json, sys\n"
            "try:\n"
            f"    import {TOP}\n"
            "except ImportError as err:\n"
            "    print('RESULT ' + json.dumps({'import_error': repr(err)})); sys.exit(0)\n"
            f"from {TOP}._internal import memoize\n"
            f"from {TOP}.nodetypes import _base\n"
            "from maya import cmds\n"
            "print('RESULT ' + json.dumps({\n"
            f"    'file': {TOP}.__file__,\n"
            "    'initialised': hasattr(cmds, 'ls'),\n"
            "    'registry': sorted(getattr(sys, '_rig_scene_callbacks', {}) or {}),\n"
            "    'ready': [memoize._SCENE_CALLBACKS_READY, _base._PLUGIN_CALLBACKS_READY],\n"
            "}))\n"
            "sys.stdout.flush()\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        cwd = tempfile.mkdtemp(prefix="rig_noinit_")
        try:
            proc = subprocess.run([mayapy, "-c", code], cwd=cwd, env=env, timeout=300,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        finally:
            shutil.rmtree(cwd, ignore_errors=True)
        output = proc.stdout.decode("utf-8", "replace")
        self.assertEqual(proc.returncode, 0, output[-2000:])
        lines = [line for line in output.splitlines() if line.startswith("RESULT ")]
        self.assertEqual(len(lines), 1, output[-2000:])
        result = json.loads(lines[0][len("RESULT "):])
        if "import_error" in result:
            self.skipTest(f"the uninitialised interpreter cannot import rig: {result}")
        self.assertEqual(
            os.path.normcase(os.path.dirname(result["file"])),
            os.path.normcase(os.path.dirname(sys.modules[TOP].__file__)),
        )
        self.assertEqual(
            {k: result[k] for k in ("initialised", "registry", "ready")},
            {"initialised": False, "registry": [], "ready": [False, False]},
        )


class TestReimport(MayaTestCase):
    """A re-import replaces the callbacks, whether in place or after a purge of
    ``sys.modules['rig*']``: never two copies, and the purged copy's caches are
    cleared so they stop holding the scene's plugs."""

    TEST_START_NEW_SCENE = True

    def _counters(self, module):
        clear = mock.patch.object(module, "_clear_all_caches", wraps=module._clear_all_caches)
        self.addCleanup(clear.stop)
        return clear.start()

    def test_reload_in_place_replaces_the_callbacks(self):
        lists = (memoize_module._ALL_MEMOIZED, memoize_module._ALL_NODEOP_CACHES)
        ids   = callbacks.registered_ids(MEMO)
        a     = Node(cmds.createNode("transform", name="a"))
        summ  = a.tx + a.ty
        self.assertIs(importlib.reload(memoize_module), memoize_module)
        # the registries and their entries are kept: the wrappers are still in use
        self.assertIs(memoize_module._ALL_MEMOIZED, lists[0])
        self.assertIs(memoize_module._ALL_NODEOP_CACHES, lists[1])
        self.assertTrue(_holds(summ))
        self.assertIs(a.tx + a.ty, summ)
        # the old callbacks were removed (Maya may give their ids to the new ones)
        self.assertEqual(len(ids), 6)
        self.assertEqual(len(callbacks.registered_ids(MEMO)), 6)
        clear = self._counters(memoize_module)
        cmds.file(new=True, force=True)
        self.assertEqual(clear.call_count, 1)
        self.assertEqual(_entries(), [])

    def test_reimport_after_a_purge_replaces_the_callbacks(self):
        saved = {k: m for k, m in sys.modules.items() if k == TOP or k.startswith(TOP + ".")}
        a     = Node(cmds.createNode("transform", name="a"))
        summ  = a.tx + a.ty
        self.assertTrue(_holds(summ))
        plugin   = _unloaded_plugin()
        counters = [self._counters(memoize_module)]
        copies   = [memoize_module]
        bases    = [_base]
        try:
            for attempt in range(2):
                for name in [k for k in sys.modules if k == TOP or k.startswith(TOP + ".")]:
                    del sys.modules[name]
                importlib.import_module(TOP)
                copies.append(sys.modules[MEMO])
                bases.append(sys.modules[BASE])
                with self.subTest(attempt=attempt):
                    self.assertIsNot(copies[-1], copies[-2])
                    registry = callbacks.registry()
                    self.assertIs(registry[MEMO]["module"], copies[-1])
                    self.assertIs(registry[BASE]["module"], bases[-1])
                    self.assertEqual(len(registry[MEMO]["ids"]), 6)
                    self.assertEqual(len(registry[BASE]["ids"]), 2)
                    # the replaced copy's caches were cleared
                    self.assertEqual(_entries(copies[-2]), [])
                    # only the new copy's callbacks run
                    counters.append(self._counters(copies[-1]))
                    for counter in counters:
                        counter.reset_mock()
                    cmds.file(new=True, force=True)
                    self.assertEqual([c.call_count for c in counters], [0] * attempt + [0, 1])
                    # and only the new copy's plug-in callbacks
                    if plugin:
                        for base in bases:
                            base._STATIC_DATA_TYPE["sentinel"] = "kept"
                        cmds.loadPlugin(plugin, quiet=True)
                        cmds.unloadPlugin(plugin)
                        self.assertEqual(
                            ["sentinel" in base._STATIC_DATA_TYPE for base in bases],
                            [True] * (attempt + 1) + [False],
                        )
                        for base in bases:
                            base._STATIC_DATA_TYPE.pop("sentinel", None)
        finally:
            for name in [k for k in sys.modules if k == TOP or k.startswith(TOP + ".")]:
                del sys.modules[name]
            sys.modules.update(saved)
            _base._register_plugin_callbacks()
            memoize_module._register_scene_callbacks()
        registry = callbacks.registry()
        self.assertIs(registry[MEMO]["module"], memoize_module)
        self.assertIs(registry[BASE]["module"], _base)
        # the restored copy clears its caches again, and has cleared the last copy's
        for counter in counters:
            counter.reset_mock()
        Node(cmds.createNode("transform", name="b")).tx + 1
        self.assertTrue(_entries())
        cmds.file(new=True, force=True)
        self.assertEqual([c.call_count for c in counters], [1, 0, 0])
        self.assertEqual(_entries(copies[-1]), [])
        self.assertEqual(_entries(), [])

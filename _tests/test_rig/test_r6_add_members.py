"""Round 6 S0: every addNode rig sends goes through ``container._add_members``.

Maya 2025 crashes (TdnContainerBase::canBeAddedToContainer) on one
``container -edit -addNode [...] -force`` call whose list holds the sole member
of another container followed by more nodes (the orders 'of' and 'fof'; 'oo',
'ffo', 'ofo' and a single node do not). rig sent such lists on two routes:
``container.add([...])`` (and a typed create's ``container.add(created)``) and
the material engine's adoption (``materials._adopt``: ``[engine, *materialInfo]``).
``_add_members`` sends the nodes already in a container (their ``.message``
feeds a hyperLayout) one call each, after one call for the others; a list that
raises falls back to one call per node, each failure a debug line.

The crash cases run in a subprocess: in-process they would take the test run down.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from unittest import mock

from maya import cmds
from rig import container
from rig._internal import container as container_module
from rig._internal.container import _add_members, _in_a_container
from rig._tests._base import MayaTestCase, is_standalone

TOP = container_module.__name__.split(".")[0]

# one case per process: the 'raw_*' cases send today's list with maya.cmds (Maya 2025
# crashes), the 'rig_*' cases take rig's route for the same list
_CASE = r'''
import json, os, sys
import maya.standalone
maya.standalone.initialize(name="python")
from maya import cmds
from {top} import container
from {top}._internal import materials
from {top}.bridges import nodes as rn
from {top}.nodetypes import ShadingEngine
case = sys.argv[1]
cmds.file(new=True, force=True)
def members(ctn):
    return cmds.container(ctn, query=True, nodeList=True) or []
if case.endswith("_add"):
    a = cmds.container(name="A")
    cmds.createNode("multiplyDivide", name="o", skipSelect=True)
    cmds.container(a, edit=True, addNode=["o"], force=True)
    cmds.createNode("multiplyDivide", name="f", skipSelect=True)
    if case == "raw_add":
        b = cmds.container(name="B")
        cmds.container(b, edit=True, addNode=["o", "f"], force=True)
    else:
        with container("B"):
            container.add(["o", "f"])
    result = {{"B": sorted(members("B")), "A": members("A")}}
else:
    # the engine is found in another container (X, its sole member) and its
    # materialInfo is free: the material's container adopts both
    rn.blinn(name="bare", container=False)
    cmds.container(name="asset")
    cmds.container("asset", edit=True, addNode=["bare"], force=True)
    real = ShadingEngine.for_material.__func__
    def found_in_x(cls, material, create=True):
        engine = real(cls, material, create=create)
        if not cmds.objExists("X"):
            cmds.container(name="X")
        cmds.container("X", edit=True, addNode=[str(engine)], force=True)
        return engine
    ShadingEngine.for_material = classmethod(found_in_x)
    if case == "raw_adopt":
        engine = ShadingEngine.for_material("bare", create=True)
        cmds.container("asset", edit=True, force=True,
                       addNode=[str(engine), *engine.get_material_info()])
    else:
        materials._adopt("bare")
    result = {{"asset": sorted(members("asset")), "X": members("X")}}
print("RESULT " + json.dumps(result))
sys.stdout.flush()
os._exit(0)
'''


class TestAddMembersCrash(MayaTestCase):
    """The crash before (today's list through maya.cmds) and after (rig's route)."""

    def _run(self, cases):
        mayapy = sys.executable
        cwd = tempfile.mkdtemp(prefix="rig_r6_s0_")
        try:
            script = os.path.join(cwd, "case.py")
            with open(script, "w") as fh:
                fh.write(_CASE.format(top=TOP))
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
            env["MAYA_DISABLE_CER"] = "1"  # no crash-report window for the raw cases
            env["TEMP"] = env["TMP"] = cwd  # the crash log and the recovered scene land here
            procs = {
                case: subprocess.Popen([mayapy, "-u", script, case], cwd=cwd, env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                for case in cases
            }
            out = {}
            for case, proc in procs.items():
                text = proc.communicate(timeout=600)[0].decode("utf-8", "replace")
                lines = [line for line in text.splitlines() if line.startswith("RESULT ")]
                out[case] = (proc.returncode, json.loads(lines[0][7:]) if lines else None, text[-1500:])
            return out
        finally:
            shutil.rmtree(cwd, ignore_errors=True)

    def test_list_moving_the_last_member_of_another_container(self):
        if not is_standalone():
            self.skipTest("needs mayapy to start a child interpreter")
        out = self._run(["raw_add", "rig_add", "raw_adopt", "rig_adopt"])
        expected = {
            "add": {"B": ["f", "o"], "A": []},
            "adopt": {"asset": ["bare", "bareSG", "materialInfo1"], "X": []},
        }
        maya_2025 = cmds.about(apiVersion=True) // 10000 == 2025
        for route, result in expected.items():
            with self.subTest(route=route, side="rig"):
                code, found, tail = out["rig_" + route]
                self.assertEqual((code, found), (0, result), tail)
            with self.subTest(route=route, side="raw"):
                code, found, tail = out["raw_" + route]
                if maya_2025:
                    # before S0: the list rig sent crashes Maya. A Maya that no
                    # longer crashes here makes the guard in _add_members optional
                    self.assertIsNone(found, tail)
                    self.assertNotEqual(code, 0, tail)
                elif found is not None:
                    self.assertEqual(found, result, tail)


class TestAddMembers(MayaTestCase):
    """The helper's calls: one per node already in a container, after one for the
    others; a single node unchecked; the per-node fallback on a RuntimeError."""

    def setUp(self):
        super().setUp()
        cmds.file(new=True, force=True)
        self.calls = []
        edits, calls = container_module.cmds_edit, self.calls

        class _Edits:
            def __getattr__(self, name):
                return getattr(edits, name)

            @staticmethod
            def container(*args, **kwargs):
                if kwargs.get("addNode"):
                    calls.append(list(kwargs["addNode"]))
                return edits.container(*args, **kwargs)

        patcher = mock.patch.object(container_module, "cmds_edit", _Edits())
        patcher.start()
        self.addCleanup(patcher.stop)

    def _nodes(self, *names):
        return [cmds.createNode("multiplyDivide", name=n, skipSelect=True) for n in names]

    def _members(self, ctn):
        return sorted(cmds.container(ctn, query=True, nodeList=True) or [])

    def test_member_of_another_container_is_sent_alone_after_the_others(self):
        self._nodes("o1", "o2", "f1", "f2")
        cmds.container(name="A", addNode=["o1", "o2"], force=True)  # o2 is never the last member
        cmds.container(name="B")
        self.assertTrue(_in_a_container("o1"))
        self.assertFalse(_in_a_container("f1"))
        self.assertFalse(_in_a_container("gone"))
        _add_members("B", ["o1", "f1", "o2", "f2"])
        self.assertEqual(self.calls, [["f1", "f2"], ["o1"], ["o2"]])
        self.assertEqual(self._members("B"), ["f1", "f2", "o1", "o2"])
        self.assertEqual(cmds.container("A", query=True, nodeList=True), None)

    def test_single_node_is_sent_unchecked(self):
        self._nodes("o1", "o2")
        cmds.container(name="A", addNode=["o1", "o2"], force=True)
        cmds.container(name="B")
        with mock.patch.object(container_module, "_in_a_container") as check:
            _add_members("B", ["o1"])
            _add_members("B", [])
        check.assert_not_called()
        self.assertEqual(self.calls, [["o1"]])
        self.assertEqual(self._members("B"), ["o1"])

    def test_fallback_one_call_per_node_when_a_name_is_gone(self):
        self._nodes("a", "b")
        cmds.container(name="B")
        with self.assertLogs(container_module.LOGGER, logging.DEBUG) as logs:
            _add_members("B", ["a", "gone", "b"])
        # the list raises (Maya drops it whole), then one call per node
        self.assertEqual(self.calls, [["a", "gone", "b"], ["a"], ["gone"], ["b"]])
        self.assertEqual(self._members("B"), ["a", "b"])
        self.assertEqual(len(logs.records), 1)
        self.assertIn("Failed to add ['gone'] to container B", logs.output[0])

    def test_single_node_failure_is_a_debug_line(self):
        cmds.container(name="B")
        with self.assertLogs(container_module.LOGGER, logging.DEBUG) as logs:
            _add_members("B", ["gone"])
        self.assertEqual(self.calls, [["gone"]])
        self.assertIn("Failed to add ['gone'] to container B", logs.output[0])

    def test_freed_leaf_is_a_debug_line(self):
        # a container freed under its block (a new scene, a file open, its hyperLayout
        # deleted): its name raises, and the add is skipped as before S0
        from rig import Node
        self._nodes("a", "b")
        leaf = Node(cmds.container(name="B"))
        cmds.delete("B")
        with self.assertLogs(container_module.LOGGER, logging.DEBUG) as logs:
            _add_members(leaf, ["a", "b"])
        self.assertEqual(self.calls, [])
        self.assertIn("Failed to add ['a', 'b'] to container", logs.output[0])
        with self.assertRaises(RuntimeError):
            _add_members(leaf, ["a"], strict=True)

    def test_strict_raises(self):
        self._nodes("a")
        cmds.container(name="B")
        with self.assertRaises(RuntimeError):
            _add_members("B", ["a", "gone"], strict=True)
        self.assertEqual(self.calls, [["a", "gone"]])

    def test_container_add_list_goes_through_the_helper(self):
        self._nodes("o1", "o2", "f1")
        cmds.container(name="A", addNode=["o1", "o2"], force=True)
        with container("B"):
            container.add(["o1", "f1"])
        self.assertIn(["f1"], self.calls)
        self.assertIn(["o1"], self.calls)
        self.assertEqual(self._members("B"), ["f1", "o1"])
        self.assertEqual(self._members("A"), ["o2"])

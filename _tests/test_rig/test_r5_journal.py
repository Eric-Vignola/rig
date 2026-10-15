"""Round 5, D1: the statement journal (``rig._internal.journal``).

`TestBarrierLint` keeps the barrier rule true: every ``maya.cmds`` edit rig makes goes
through ``cmds_edit.<command>`` (or ``mel_eval``), which registers the pending journal
first; only ``createNode``, ``loadPlugin`` and the member add of a node a container scope
just made stay raw.

The journal tests drive it with test-only edits (an ``MDGModifier`` applied, then
``journal.add``): nothing in rig journals an edit yet (D2 / D3 will). `TestJournal` is the
mechanism (nesting, exceptions, generators, call-outs, registration at depth 0, a failed
registration, undo off, chunks, the new / open discard); `TestBarriers` the barrier sites;
`TestStatementEntries` that the library's entry points are statements; `TestUserCodeOrder`
the call-outs (CR-5): code a user wrote runs after what the statement journaled before it,
in the undo queue too.
"""

import ast
import gc
import os
import tempfile
import shutil
import unittest
import weakref
from unittest import mock

from maya import cmds, mel
from maya.api import OpenMaya

import rig
from rig import container, List, Node
from rig._internal import callbacks, journal
from rig._internal import plug as plug_module
from rig._internal.node_ops import NodeOp
from rig._internal.plug import Plug
from rig.bridges import commands as rc
from rig.bridges import nodes as rn
from rig.nodetypes import dg_node as dg_node_module
from rig.nodetypes import plugins, Transform
from rig.nodetypes._base import Attribute
from rig.nodetypes.dg_node import DGNode
from rig.nodetypes.material_node import Lambert, Material
from rig.spec._base import _AttrSpec
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk

# --------------------------------------------------------------------------------------- #
#  The barrier lint
# --------------------------------------------------------------------------------------- #

# Every maya.cmds name rig calls is in exactly one of QUERY, DUAL and EDIT; an unknown name
# fails the lint ("classify it").
#   QUERY  never edits the scene or the undo queue: always raw ``cmds.<name>``
#   DUAL   edits unless called with a literal true query flag (q / query) or one of its
#          query-only flags (QUERY_ONLY_FLAGS): raw only for a query
#   EDIT   always an edit: ``cmds_edit.<name>``, except NON_BARRIER names and RAW_SITES
QUERY = frozenset({
    "about", "attributeName", "attributeQuery", "connectionInfo", "deformableShape",
    "error", "exactWorldBoundingBox", "filterExpand", "geometryAttrInfo", "getAttr", "getClassification",
    "help", "isConnected", "listAttr", "listConnections", "listHistory", "listNodeTypes", "listRelatives",
    "listSets", "ls", "namespaceInfo", "nodeType", "nodeTypeHierarchy", "objExists", "objectType",
    "pluginInfo", "pointPosition", "polyEvaluate", "polyListComponentConversion", "referenceQuery",
    "warning", "whatIs",
})

DUAL = frozenset({
    "addAttr", "aliasAttr", "componentTag", "container", "currentTime", "currentUnit", "dagPose",
    "deformer", "editDisplayLayerMembers", "file", "lockNode", "namespace", "optionVar",
    "playbackOptions", "sets", "skinCluster", "skinPercent", "undoInfo", "xform",
})

# the flags that make a DUAL command a query whatever their value (``namespace -exists x``)
QUERY_ONLY_FLAGS = {
    "namespace":    frozenset({"exists", "ex"}),
    "componentTag": frozenset({"queryEdit", "qe"}),
}

EDIT = frozenset({
    "bakeDeformer", "blendShape", "connectAttr", "copyAttr", "createDisplayLayer", "createNode", "curve",
    "delete", "deleteAttr", "disconnectAttr", "displaySmoothness", "duplicate", "flushUndo", "keyTangent",
    "loadPlugin", "makeIdentity", "makePaintable", "orientConstraint", "parent", "pointConstraint",
    "polyCube", "polyDelEdge", "redo", "removeMultiInstance", "rename", "renameAttr", "select", "setAttr",
    "setKeyframe", "shadingNode", "spaceLocator", "undo", "unloadPlugin",
})

# EDIT names that commute with the pending journal (a node made after it; not on the undo
# queue at all): raw everywhere
NON_BARRIER = frozenset({"createNode", "loadPlugin"})

# (file, enclosing function, command) allowed to call an EDIT / DUAL command raw: the member
# add of a node container.createNode just made (`_add` registers first for any other node);
# an undo chunk's close, which runs after its own barrier() whatever that barrier does
RAW_SITES = frozenset({
    ("_internal/container.py", "_ContainerStack._add", "container"),
    ("_internal/undo.py", "_Chunk.__exit__", "undoInfo"),
})

# mel.eval sites that only query (raw); every other one is journal.mel_eval. None: the last,
# DAGNode.get_deformers (findRelatedDeformer, a recorded command), walks the graph by the API
MEL_QUERY_SITES = frozenset()

# the names rig binds maya.cmds to
CMDS_NAMES = frozenset({"cmds", "_mc", "mc"})

# not linted: the journal (the proxy itself) and the undo command (its undoInfo calls run
# during undo / redo or around a registration, never with a pending journal)
EXCLUDED_FILES = frozenset({"_internal/journal.py", "nodetypes/plugins/undoable_api_command.py"})

# where getattr(cmds, ...) may look a command up by name
GETATTR_ALLOWED = ("bridges/commands.py", "nodetypes/plugins/")

# (file, enclosing function) where a .doIt may apply an API edit: the journaled modifiers (B0's
# helper and D2's connect), and the mesh edits' bodies, which run inside the undo command
# (anywhere else an edit would be applied outside the journal and the undo command)
DOIT_ALLOWED = frozenset({
    ("nodetypes/plugins/__init__.py", "_register_api_edit"),
    ("_internal/api_edits.py", "_apply"),
    ("nodetypes/mesh.py", "_free_elements"),
    ("nodetypes/mesh.py", "_relink"),
    ("nodetypes/mesh.py", "_MeshSetsEdit.redoIt"),
    ("nodetypes/mesh.py", "_MeshRenameUVSetCommand.redoIt"),
})

# where the classes live, for the "classify it" message
_TABLES = "QUERY / DUAL / EDIT in rig/_tests/test_rig/test_r5_journal.py"


def _query_flag(call, name):
    """True when the DUAL command ``name`` is called as a query: a literal true q / query, or
    one of its QUERY_ONLY_FLAGS (a q / query that is not a literal counts as an edit)."""
    only = QUERY_ONLY_FLAGS.get(name, ())
    for kw in call.keywords:
        if kw.arg in only:
            return True
        if kw.arg in ("q", "query") and isinstance(kw.value, ast.Constant) and kw.value.value:
            return True
    return False


def _is_cmds(node):
    """True for a name rig binds maya.cmds to (CMDS_NAMES)."""
    return isinstance(node, ast.Name) and node.id in CMDS_NAMES


def _enclosing(parents, node):
    names = []
    while node in parents:
        node = parents[node]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
    return ".".join(reversed(names))


def lint_source(rel, source):
    """The barrier-rule violations of the module ``source`` (``rel``: its path in the
    package, with forward slashes): ``[(line, message), ...]``."""
    tree    = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    out     = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decorators = [ast.unparse(d) for d in node.decorator_list]
            if any(d.split("(")[0].endswith("statement") for d in decorators) and (
                isinstance(node, ast.AsyncFunctionDef)
                or any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(node))
            ):
                out.append((node.lineno, f"statement on a generator or coroutine {node.name}"))
            continue
        if isinstance(node, ast.ImportFrom) and node.module in ("maya.cmds", "maya.mel"):
            out.append((node.lineno, f"from {node.module} import ...: call it through the module"))
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                full = alias.name if isinstance(node, ast.Import) else f"{node.module}.{alias.name}"
                bound = alias.asname or alias.name
                if full == "maya.cmds" and isinstance(node, ast.ImportFrom) and bound not in CMDS_NAMES:
                    out.append((node.lineno, f"maya.cmds bound to {bound!r}: use one of {sorted(CMDS_NAMES)}"))
                elif full == "maya.cmds" and isinstance(node, ast.Import) and alias.asname not in CMDS_NAMES:
                    out.append((node.lineno, f"maya.cmds bound to {bound!r}: use one of {sorted(CMDS_NAMES)}"))
                elif full == "maya.mel" and bound != "mel":
                    out.append((node.lineno, f"maya.mel bound to {bound!r}: bind it as mel"))
            continue
        if isinstance(node, ast.Attribute):
            # a command used other than by a call (an alias, a partial, map), maya.cmds /
            # maya.mel reached through ``maya``, the module's namespace, a MEL / Python string
            # run by the API, a doIt reached other than where it may be
            where  = _enclosing(parents, node)
            called = isinstance(parents.get(node), ast.Call) and parents[node].func is node
            if isinstance(node.value, ast.Name) and node.value.id == "maya" and node.attr in ("cmds", "mel"):
                out.append((node.lineno, f"maya.{node.attr} reached through maya in {where}: bind it"))
            elif _is_cmds(node.value) and node.attr.startswith("_"):
                out.append((node.lineno, f"cmds.{node.attr} in {where}: a dynamic command"))
            elif _is_cmds(node.value) and not called and node.attr in (DUAL | EDIT) - NON_BARRIER:
                out.append((node.lineno, f"uncalled cmds.{node.attr} in {where}: call it (cmds_edit.{node.attr})"))
            elif node.attr.startswith(("executeCommand", "executePythonCommand")):
                out.append((node.lineno, f"{node.attr} in {where}: a command run by the API"))
            elif node.attr == "doIt" and (rel, where) not in DOIT_ALLOWED:
                out.append((node.lineno, f".doIt in {where}: an API edit outside the journal"))
            continue
        if not isinstance(node, ast.Call):
            continue
        func_node = node.func
        where     = _enclosing(parents, node)
        if isinstance(func_node, ast.Attribute) and isinstance(func_node.value, ast.Name) \
                and func_node.value.id in CMDS_NAMES:
            name = func_node.attr
            if name in QUERY or name.startswith("_"):
                continue
            if name in DUAL:
                if _query_flag(node, name):
                    continue
            elif name not in EDIT:
                out.append((node.lineno, f"unclassified cmds.{name} in {where}: classify it ({_TABLES})"))
                continue
            if name in NON_BARRIER or (rel, where, name) in RAW_SITES:
                continue
            out.append((node.lineno, f"raw edit cmds.{name} in {where} (use cmds_edit.{name})"))
        elif isinstance(func_node, ast.Attribute) and isinstance(func_node.value, ast.Name) \
                and func_node.value.id == "mel" and func_node.attr == "eval":
            if (rel, where) not in MEL_QUERY_SITES:
                out.append((node.lineno, f"mel.eval in {where} (use mel_eval)"))
        elif isinstance(func_node, ast.Name) and func_node.id in ("getattr", "vars") and node.args \
                and isinstance(node.args[0], ast.Name) and node.args[0].id in CMDS_NAMES:
            if not rel.startswith(GETATTR_ALLOWED):
                out.append((node.lineno, f"{func_node.id}(cmds, ...) in {where}: a dynamic command"))
    return out


def package_files():
    """``[(rel, path), ...]`` of the package's modules the lint reads: not the tests, the
    examples or EXCLUDED_FILES."""
    root  = os.path.dirname(os.path.abspath(rig.__file__))
    found = []
    for folder, dirs, files in os.walk(root):
        rel_dir = os.path.relpath(folder, root).replace("\\", "/")
        if rel_dir.split("/")[0] in ("_tests", "examples") or "__pycache__" in rel_dir:
            dirs[:] = []
            continue
        for name in files:
            if name.endswith(".py"):
                path = os.path.join(folder, name)
                rel  = os.path.relpath(path, root).replace("\\", "/")
                if rel not in EXCLUDED_FILES:
                    found.append((rel, path))
    return sorted(found)


class TestBarrierLint(unittest.TestCase):
    """The package keeps the barrier rule; the lint sees each kind of violation."""

    def test_the_package_keeps_the_rule(self):
        violations = []
        for rel, path in package_files():
            with open(path, encoding="utf-8") as handle:
                violations += [f"{rel}:{line}: {msg}" for line, msg in lint_source(rel, handle.read())]
        self.assertEqual(violations, [])

    def test_the_lint_reads_the_package(self):
        rels = [rel for rel, _ in package_files()]
        self.assertIn("_internal/plug.py", rels)
        self.assertIn("nodetypes/_base.py", rels)
        self.assertNotIn("_internal/journal.py", rels)
        self.assertFalse([rel for rel in rels if rel.startswith(("_tests/", "examples/"))])

    def test_the_classes_are_disjoint(self):
        self.assertEqual(QUERY & DUAL, frozenset())
        self.assertEqual(QUERY & EDIT, frozenset())
        self.assertEqual(DUAL & EDIT, frozenset())
        self.assertLessEqual(NON_BARRIER, EDIT)

    def lint(self, source, rel="shade.py"):
        return [msg for _, msg in lint_source(rel, source)]

    def test_each_violation_is_seen(self):
        cases = {
            "def f():\n    cmds.setAttr('a.tx', 1)\n": "raw edit cmds.setAttr in f",
            "def f():\n    cmds.sets('a', edit=True, forceElement='s')\n": "raw edit cmds.sets in f",
            "def f():\n    cmds.fancyNewCommand('a')\n": "unclassified cmds.fancyNewCommand in f: classify it (QUERY / DUAL / EDIT in rig/_tests/test_rig/test_r5_journal.py)",
            "def f():\n    mel.eval('x;')\n": "mel.eval in f",
            "def f():\n    getattr(cmds, 'setAttr')('a.tx', 1)\n": "getattr(cmds, ...) in f",
            "def f(m):\n    m.doIt()\n": ".doIt in f",
            # D1FIX: the forms the lint missed
            "def f():\n    s = cmds.setAttr\n    s('a.tx', 1)\n": "uncalled cmds.setAttr in f",
            "def f():\n    functools.partial(cmds.delete, 'a')()\n": "uncalled cmds.delete in f",
            "def f(nodes):\n    list(map(cmds.delete, nodes))\n": "uncalled cmds.delete in f",
            "import maya\ndef f():\n    maya.cmds.setAttr('a.tx', 1)\n": "maya.cmds reached through maya in f",
            "import maya\ndef f():\n    maya.mel.eval('x')\n": "maya.mel reached through maya in f",
            "def f(m):\n    OpenMaya.MDGModifier.doIt(m)\n": ".doIt in f",
            "def f(m):\n    g = m.doIt\n    g()\n": ".doIt in f",
            "def f():\n    OpenMaya.MGlobal.executeCommand('setAttr a.tx 1')\n": "executeCommand in f",
            "def f(q):\n    cmds.sets('s', q=q)\n": "raw edit cmds.sets in f",
            "def f(n):\n    vars(cmds)[n]('a')\n": "vars(cmds, ...) in f",
            "def f(n):\n    cmds.__dict__[n]('a')\n": "cmds.__dict__ in f",
            "@statement\ndef f():\n    yield 1\n": "statement on a generator or coroutine f",
            "@journal.statement\nasync def f():\n    pass\n": "statement on a generator or coroutine f",
            "from maya.cmds import setAttr\n": "from maya.cmds import",
            "from maya import cmds as c\n": "maya.cmds bound to 'c'",
            "import maya.cmds as c\n": "maya.cmds bound to 'c'",
            "import maya.cmds\n": "maya.cmds bound to 'maya.cmds'",
            "from maya import mel as m\n": "maya.mel bound to 'm'",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                found = self.lint(source)
                self.assertEqual(len(found), 1, found)
                self.assertIn(expected, found[0])

    def test_what_is_allowed(self):
        for source in (
            "def f():\n    cmds_edit.setAttr('a.tx', 1)\n",
            "def f():\n    cmds.getAttr('a.tx')\n",
            "def f():\n    cmds.sets('s', query=True)\n",
            "def f():\n    cmds.xform('a', q=True, matrix=True)\n",
            "def f():\n    cmds.createNode('transform')\n",
            "def f():\n    cmds.loadPlugin('quatNodes', quiet=True)\n",
            "def f():\n    mel_eval('x;')\n",
            "from maya import cmds\nimport maya.cmds as cmds\nfrom maya import cmds as _mc\n",
            "@statement\ndef f():\n    return 1\n",
            # the query-only flags of DUAL commands, an uncalled NON_BARRIER command
            "def f(n):\n    cmds.namespace(exists=n)\n",
            "def f(p):\n    cmds.componentTag(p, queryEdit=True, tagName='t')\n",
            "def f():\n    make = cmds.createNode\n",
        ):
            with self.subTest(source=source):
                self.assertEqual(self.lint(source), [])
        # a literal query=False is an edit
        self.assertEqual(len(self.lint("def f():\n    cmds.sets('s', query=False)\n")), 1)
        # the allowed places
        self.assertEqual(self.lint("class _ContainerStack:\n    def _add(self):\n"
                                   "        cmds.container('c', edit=True, addNode=['n'])\n",
                                   rel="_internal/container.py"), [])
        self.assertEqual(self.lint("def f():\n    getattr(cmds, 'x')\n", rel="bridges/commands.py"), [])
        self.assertEqual(self.lint("def _register_api_edit(m):\n    m.doIt()\n",
                                   rel="nodetypes/plugins/__init__.py"), [])
        self.assertEqual(self.lint("def _relink(m):\n    m.doIt()\n", rel="nodetypes/mesh.py"), [])
        self.assertEqual(self.lint("class _ContainerStack:\n    def _add(self):\n"
                                   "        cmds.container('c', edit=True, addNode=['n'])\n",
                                   rel="_internal/container.py"), [])
        self.assertEqual(self.lint("class _Chunk:\n    def __exit__(self, *exc):\n"
                                   "        cmds.undoInfo(closeChunk=True)\n", rel="_internal/undo.py"), [])
        # a mel.eval in get_deformers is no query site any more; a doIt elsewhere in those files is not
        self.assertEqual(len(self.lint("class DAGNode:\n    def get_deformers(self):\n        mel.eval('x')\n",
                                       rel="nodetypes/dag_node.py")), 1)
        self.assertEqual(len(self.lint("def f(m):\n    m.doIt()\n", rel="nodetypes/plugins/__init__.py")), 1)
        self.assertEqual(len(self.lint("def f(m):\n    m.doIt()\n", rel="nodetypes/mesh.py")), 1)


# --------------------------------------------------------------------------------------- #
#  The journal
# --------------------------------------------------------------------------------------- #

PLUGIN = "undoable_api_command"


def _plug(name):
    sel = OpenMaya.MSelectionList()
    sel.add(name)
    return sel.getPlug(0)


def _set_op(name, value):
    """A test-only journaled edit: the double ``name`` set to ``value`` by an applied
    MDGModifier, journaled (pending in a statement, else its own step)."""
    modifier = OpenMaya.MDGModifier()
    modifier.newPlugValueDouble(_plug(name), value)
    modifier.doIt()
    journal.add(modifier)
    return modifier


def _connect_op(src, dst):
    """A test-only journaled connect (an applied MDGModifier, journaled)."""
    modifier = OpenMaya.MDGModifier()
    modifier.connect(_plug(src), _plug(dst))
    modifier.doIt()
    journal.add(modifier)
    return modifier


class _Spy:
    """A journal op recording its calls into ``log``; ``fail`` raises from both."""

    def __init__(self, name, log, fail=False):
        self.name, self.log, self.fail = name, log, fail

    def doIt(self):  # noqa: N802
        self.log.append(f"{self.name}.do")
        if self.fail:
            raise ValueError(f"{self.name} do")

    def undoIt(self):  # noqa: N802
        self.log.append(f"{self.name}.undo")
        if self.fail:
            raise ValueError(f"{self.name} undo")


class _Tracked:
    """``with _Tracked() as kept:`` replaces the journal object with one weakly listed in
    ``kept`` (Maya keeps the journal it registers: with undo off it keeps none)."""

    def __enter__(self):
        self.kept = kept = []

        class Tracked(journal._Journal):
            __slots__ = ("__weakref__",)

            def __init__(self, ops):
                super().__init__(ops)
                kept.append(weakref.ref(self))

        self._patch = mock.patch.object(journal, "_Journal", Tracked)
        self._patch.start()
        return kept

    def __exit__(self, *exc):
        self._patch.stop()
        return False


class _JournalCase(UndoWalk, MayaTestCase):
    """A new scene, the undo queue on, infinite and flushed, rig's plug-in loaded; the
    node ``a`` (a transform) with the doubles ``d1``, ``d2``; the multiplyDivides ``m``
    and ``k``. The journal is closed and empty before and after every test."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        plugins._ensure_loaded(PLUGIN, True)
        cmds.createNode("transform", name="a")
        for attr in ("d1", "d2"):
            cmds.addAttr("a", longName=attr, attributeType="double")
        cmds.createNode("multiplyDivide", name="m")
        cmds.createNode("multiplyDivide", name="k")
        cmds.flushUndo()
        self.assert_closed()

    def tearDown(self):
        try:
            self.assert_closed()
        finally:
            journal._S.open, journal._S.pending = False, []
            cmds.flushUndo()
            super().tearDown()

    def assert_closed(self):
        self.assertEqual((journal._S.open, journal._S.pending), (False, []))

    def journaled_float_sets(self):
        """``Attribute.set`` of one float through an applied MDGModifier, journaled (the
        shape of a fast path, test-only); returns the list of plugs it set."""
        real, done = Attribute.set, []

        def api_set(attr, *args, **kwargs):
            if len(args) == 1 and not kwargs and type(args[0]) is float:
                _set_op(str(attr), args[0])
                done.append(str(attr))
                return None
            return real(attr, *args, **kwargs)

        patch = mock.patch.object(Attribute, "set", api_set)
        patch.start()
        self.addCleanup(patch.stop)
        return done

    @staticmethod
    def values():
        return cmds.getAttr("a.d1"), cmds.getAttr("a.d2")

    @staticmethod
    def queue_empty():
        return cmds.undoInfo(query=True, undoQueueEmpty=True)


class TestJournal(_JournalCase):
    """The mechanism (`rig._internal.journal`)."""

    def test_add_outside_a_statement_is_one_step(self):
        _set_op("a.d1", 5.0)
        self.assertEqual(self.values(), (5.0, 0.0))
        self.assertEqual(self.undo_name(), "")
        self.undo_steps(1)
        self.assertEqual(self.values(), (0.0, 0.0))
        self.assertTrue(self.queue_empty())
        self.redo_steps(1)
        self.assertEqual(self.values(), (5.0, 0.0))

    def test_a_statement_is_one_step(self):
        @journal.statement
        def edit():
            _set_op("a.d1", 5.0)
            _set_op("a.d2", 6.0)
            _set_op("a.d1", 7.0)
            return "done"

        self.assertEqual(edit(), "done")
        self.assertEqual(self.values(), (7.0, 6.0))
        self.undo_steps(1)
        self.assertEqual(self.values(), (0.0, 0.0))  # undone backwards: d1 7 -> 5 -> 0
        self.assertTrue(self.queue_empty())
        self.redo_steps(1)
        self.assertEqual(self.values(), (7.0, 6.0))  # redone forwards

    def test_nesting(self):
        seen = []

        @journal.statement
        def inner():
            seen.append(("inner", journal._S.open))
            _set_op("a.d2", 6.0)

        @journal.statement
        def outer():
            _set_op("a.d1", 5.0)
            inner()
            seen.append(("after inner", journal._S.open, len(journal._S.pending)))
            journal.run_statement(inner, (), {})
            return len(journal._S.pending)

        self.assertEqual(outer(), 3)
        self.assertEqual(seen, [("inner", True), ("after inner", True, 2), ("inner", True)])
        self.undo_steps(1)
        self.assertEqual(self.values(), (0.0, 0.0))
        self.assertTrue(self.queue_empty())

    def test_an_exception_registers_then_propagates_unchanged(self):
        error = ValueError("the statement's own")

        @journal.statement
        def edit():
            _set_op("a.d1", 5.0)
            raise error

        with self.assertRaises(ValueError) as caught:
            edit()
        self.assertIs(caught.exception, error)
        self.assertIsNone(error.__context__)
        self.assertIsNone(error.__cause__)
        self.assertFalse(error.__suppress_context__)
        self.assertEqual(self.values(), (5.0, 0.0))  # what it applied is kept ...
        self.undo_steps(1)                          # ... as one step
        self.assertEqual(self.values(), (0.0, 0.0))
        self.assertTrue(self.queue_empty())

    def test_an_exception_inside_an_except_keeps_its_context(self):
        @journal.statement
        def edit():
            try:
                raise KeyError("first")
            except KeyError:
                _set_op("a.d1", 5.0)
                raise ValueError("second")

        with self.assertRaises(ValueError) as caught:
            edit()
        self.assertIsInstance(caught.exception.__context__, KeyError)
        self.assertIsNone(caught.exception.__cause__)
        self.undo_steps(1)
        self.assertTrue(self.queue_empty())

    def test_a_generator_or_a_coroutine_is_refused(self):
        def gen():
            yield 1

        async def coro():
            pass

        async def agen():
            yield 1

        for fn in (gen, coro, agen):
            with self.subTest(fn.__name__):
                with self.assertRaisesRegex(TypeError, "a generator or a coroutine"):
                    journal.statement(fn)

    def test_a_statement_is_marked_and_wraps(self):
        def edit(x, y=2):
            """doc"""
            return x + y

        entry = journal.statement(edit)
        self.assertIs(entry._rig_statement, True)
        self.assertIs(entry.__wrapped__, edit)
        self.assertEqual((entry.__name__, entry.__doc__), ("edit", "doc"))
        self.assertEqual(entry(1, y=3), 4)

    def test_statement_and_run_statement_share_one_implementation(self):
        # fix (c): the decorator's outermost call is run_statement; a nested one only tests
        # the flag
        @journal.statement
        def edit(x):
            return x

        @journal.statement
        def outer():
            return edit(2)

        with mock.patch.object(journal, "run_statement", wraps=journal.run_statement) as run:
            self.assertEqual(edit(1), 1)
            run.assert_called_once_with(edit.__wrapped__, (1,), {})
            run.reset_mock()
            self.assertEqual(outer(), 2)
            run.assert_called_once_with(outer.__wrapped__, (), {})

    def test_user_call_and_outside(self):
        seen = []

        def users():
            # the statement's journal is on the queue already; the scope is closed
            seen.append(("user", journal._S.open, list(journal._S.pending), self.queue_empty()))
            cmds.setAttr("a.d2", 9.0)
            return "user"

        def rigs():
            seen.append(("rig", journal._S.open, len(journal._S.pending)))
            return "rig"

        rigs.__module__ = "rig.functions"  # as code of rig's own: run inside the scope

        @journal.statement
        def edit():
            _set_op("a.d1", 5.0)
            self.assertEqual(journal.user_call(rigs), "rig")
            self.assertEqual(journal.user_call(users), "user")
            seen.append(("back", journal._S.open, len(journal._S.pending)))
            _set_op("a.d1", 7.0)
            with journal.outside():
                seen.append(("outside", journal._S.open, len(journal._S.pending)))
            seen.append(("back", journal._S.open))

        edit()
        self.assertEqual(seen, [
            ("rig", True, 1),
            ("user", False, [], False),
            ("back", True, 0),
            ("outside", False, 0),
            ("back", True),
        ])
        # the queue: [d1 = 5], [the user's d2 = 9], [d1 = 7]
        self.assertEqual(self.values(), (7.0, 9.0))
        for expected in ((5.0, 9.0), (5.0, 0.0), (0.0, 0.0)):
            self.undo_steps(1)
            self.assertEqual(self.values(), expected)
        self.assertTrue(self.queue_empty())

    def test_outside_reopens_after_an_exception(self):
        @journal.statement
        def edit():
            with self.assertRaises(KeyError):
                with journal.outside():
                    raise KeyError("x")
            return journal._S.open

        self.assertIs(edit(), True)
        with journal.outside():  # outside any statement: nothing to close
            self.assertFalse(journal._S.open)
        self.assertFalse(journal._S.open)

    def test_user_call_outside_any_statement_just_calls(self):
        self.assertEqual(journal.user_call(lambda x, y=0: x + y, 1, y=2), 3)
        self.assertFalse(journal._S.open)

    def test_rig_code(self):
        def users():
            pass

        def examples():
            pass

        examples.__module__ = "rig.examples.rail_spine"

        class Mine(Transform):
            pass

        self.assertTrue(journal.rig_code(rig.functions.clamp))
        self.assertTrue(journal.rig_code(Transform))
        self.assertTrue(journal.rig_code(Transform.create))       # a bound classmethod
        self.assertTrue(journal.rig_code(DGNode.post_create))
        self.assertTrue(journal.rig_code(Node("a").get_parent))  # a bound method
        self.assertFalse(journal.rig_code(users))                # rig._tests: a user's
        self.assertFalse(journal.rig_code(examples))
        self.assertFalse(journal.rig_code(Mine))
        self.assertTrue(journal.rig_code(Mine.create))            # inherited: rig's
        self.assertFalse(journal.rig_code(len))                   # not weakly referenceable
        self.assertFalse(journal.rig_code(lambda: None))

    def test_rig_code_keeps_no_function_alive(self):
        # fix (a): the cache holds module names only (D1FIX: it was a weak dict)
        def users():
            pass

        self.assertFalse(journal.rig_code(users))
        self.assertIs(journal._RIG_MODULES[users.__module__], False)
        self.assertTrue(all(type(key) is str for key in journal._RIG_MODULES))
        held = weakref.ref(users)
        del users
        gc.collect()
        self.assertIsNone(held())

    def test_a_journal_walks_every_op_when_one_raises(self):
        # fix (b): never half undone or redone silently
        log = []
        ops = [_Spy("A", log), _Spy("B", log, fail=True), _Spy("C", log), _Spy("D", log, fail=True)]
        entry = journal._Journal(ops)
        with self.assertRaisesRegex(ValueError, "^D undo$"):
            entry.undoIt()
        self.assertEqual(log, ["D.undo", "C.undo", "B.undo", "A.undo"])
        del log[:]
        with self.assertRaisesRegex(ValueError, "^B do$"):
            entry.redoIt()
        self.assertEqual(log, ["A.do", "B.do", "C.do", "D.do"])
        del log[:]
        entry.doIt()  # applied already: does nothing
        self.assertEqual(log, [])

    def forced_unload_state(self):
        """The state a forced unload leaves: no maya.cmds command, and a load that
        fails (`_ensure_loaded` raises then)."""
        wrapper = getattr(cmds, plugins.UNDOABLE_API_COMMAND)
        delattr(cmds, plugins.UNDOABLE_API_COMMAND)
        self.addCleanup(setattr, cmds, plugins.UNDOABLE_API_COMMAND, wrapper)
        failure = RuntimeError("a plug-in named 'undoable_api_command' is loaded, but maya.cmds has no ...")
        patch = mock.patch.object(plugins, "_ensure_loaded", side_effect=failure)
        patch.start()
        self.addCleanup(patch.stop)

    def test_a_failed_registration_reverts_and_says_so(self):
        # fix (d), the forced-unload flush
        self.forced_unload_state()

        @journal.statement
        def edit():
            _set_op("a.d1", 5.0)
            _set_op("a.d2", 6.0)
            _set_op("a.d1", 7.0)

        with self.assertRaises(RuntimeError) as caught:
            edit()
        text = str(caught.exception)
        self.assertIn("3 edit(s) of this statement could not be put on the undo queue", text)
        self.assertTrue(text.endswith("they were reverted"), text)
        self.assertIn("maya.cmds has no", text)
        self.assertIsNone(caught.exception.__cause__)
        self.assertTrue(caught.exception.__suppress_context__)
        self.assertEqual(self.values(), (0.0, 0.0))  # reverted, backwards
        self.assertTrue(self.queue_empty())

    def test_a_failed_registration_at_a_barrier_reverts_and_says_so(self):
        self.forced_unload_state()

        @journal.statement
        def edit():
            _set_op("a.d1", 5.0)
            journal.cmds_edit.setAttr("a.d2", 6.0)  # the barrier raises: the set never runs

        with self.assertRaisesRegex(RuntimeError, "they were reverted"):
            edit()
        self.assertEqual(self.values(), (0.0, 0.0))
        self.assertTrue(self.queue_empty())

    def test_a_failed_registration_at_depth_0_leaves_the_op_to_the_caller(self):
        self.forced_unload_state()
        modifier = OpenMaya.MDGModifier()
        modifier.newPlugValueDouble(_plug("a.d1"), 5.0)
        modifier.doIt()
        with self.assertRaisesRegex(RuntimeError, "maya.cmds has no"):
            journal.add(modifier)
        self.assertEqual(self.values(), (5.0, 0.0))
        modifier.undoIt()  # the caller reverts (and takes its cmds route)
        self.assertEqual(self.values(), (0.0, 0.0))
        self.assertTrue(self.queue_empty())

    def test_a_failed_registration_keeps_the_statements_own_error(self):
        # D1FIX: the body raised, then the end-of-statement registration failed: the body's
        # error propagates unchanged (the same object), the failure a note of it
        self.forced_unload_state()
        error = ValueError("the user's error")

        @journal.statement
        def edit():
            _set_op("a.d1", 5.0)
            raise error

        with self.assertRaises(ValueError) as caught:
            edit()
        self.assertIs(caught.exception, error)
        self.assertIsNone(error.__context__)
        self.assertIsNone(error.__cause__)
        self.assertEqual(len(error.__notes__), 1)
        self.assertIn("1 edit(s) of this statement could not be put on the undo queue", error.__notes__[0])
        self.assertIn("they were reverted", error.__notes__[0])
        self.assertEqual(self.values(), (0.0, 0.0))
        self.assertTrue(self.queue_empty())

    def chunk_is_closed(self):
        """No chunk is left open: two sets after it are two undo steps."""
        self.assertEqual(cmds.undoInfo(query=True, chunkName=True), "")
        cmds.setAttr("a.d1", 1.0)
        cmds.setAttr("a.d2", 2.0)
        self.undo_steps(1)
        self.assertEqual(self.values(), (1.0, 0.0))

    def test_a_failed_registration_at_a_chunks_close_closes_it(self):
        # D1FIX: the close runs whatever the chunk's barrier does
        self.forced_unload_state()

        @journal.statement
        def edit():
            with rig.undo_chunk("x"):
                _set_op("a.d1", 5.0)

        with self.assertRaisesRegex(RuntimeError, "they were reverted"):
            edit()
        self.assertEqual(self.values(), (0.0, 0.0))
        self.chunk_is_closed()

    def test_a_failed_registration_at_a_chunks_close_keeps_the_blocks_error(self):
        self.forced_unload_state()
        error = KeyError("the block's error")

        @journal.statement
        def edit():
            with rig.undo_chunk("x"):
                _set_op("a.d1", 5.0)
                raise error

        with self.assertRaises(KeyError) as caught:
            edit()
        self.assertIs(caught.exception, error)
        self.assertEqual(len(error.__notes__), 1)
        self.assertIn("could not be put on the undo queue", error.__notes__[0])
        self.assertEqual(self.values(), (0.0, 0.0))
        self.chunk_is_closed()

    def test_a_failed_registration_in_a_create_deletes_what_it_made(self):
        # D1FIX, D3's shape: the create's values journaled and their registration failing:
        # reverted, the node deleted (a create that raises leaves nothing), the chunk closed
        done = self.journaled_float_sets()
        self.forced_unload_state()
        for label, create in (
            ("Transform.create", lambda: Transform.create(name="t", tx=1.0)),
            ("Node.create", lambda: Node.create("transform", name="t", tx=1.0)),
            ("Transform.define", lambda: Transform.define("t", tx=1.0)),
            ("Node.define", lambda: Node.define("transform", "t", tx=1.0)),
        ):
            with self.subTest(label):
                del done[:]
                with self.assertRaisesRegex(RuntimeError, "they were reverted"):
                    create()
                self.assertEqual(len(done), 1)  # the value was journaled
                self.assertFalse(cmds.objExists("t"))
                self.assertEqual(cmds.undoInfo(query=True, chunkName=True), "")

    def test_a_keyword_named_fn_reaches_the_callee(self):
        # D1FIX: user_call's own parameter is positional-only
        def users(fn=None):
            return ("users", fn)

        self.assertEqual(journal.user_call(users, fn=1), ("users", 1))
        self.assertEqual(journal.run_statement(lambda: journal.user_call(users, fn=2), (), {}), ("users", 2))

        op = NodeOp("r5_journal_fn_op")

        @op.impl()
        def _impl(x, fn=None):
            return fn(x)

        t = Transform.create(name="t")
        cmds.setAttr("t.tx", 3.0)
        self.assertEqual(cmds.getAttr(str(op(t.tx, fn=lambda plug: plug * 2))), 6.0)

        seen = []

        class _Rec(DGNode):
            NATIVE_NODE_TYPE = "network"
            CUSTOM_NODE_TYPE = "r5JournalRec"
            _CREATE_FLAGS    = DGNode._CREATE_FLAGS | {"fn"}

            @classmethod
            def _create(cls, *args, **kwargs):
                seen.append(kwargs["fn"])
                return cmds.createNode("network", name=kwargs["name"], skipSelect=True)

        self.assertEqual(str(_Rec.create(name="r", fn=1)), "r")
        self.assertEqual(str(_Rec.define("r2", fn=2)), "r2")
        self.assertEqual(seen, [1, 2])

    def test_the_first_journaled_edit_registers_the_scene_callbacks(self):
        # D1FIX: an import before Maya was initialised registers them before the pending
        # list exists that they would drop
        seen = []

        def ensure():
            seen.append((journal._S.open, list(journal._S.pending)))
            return True

        with mock.patch.object(journal, "_SCENE_CALLBACKS_READY", False), \
                mock.patch.object(journal, "_ensure_scene_callbacks", ensure):
            journal.run_statement(_set_op, ("a.d1", 5.0), {})
        self.assertEqual(seen, [(True, [])])  # in the statement, before the first op joins it
        self.undo_steps(1)
        self.assertEqual(self.values(), (0.0, 0.0))

    def test_a_list_is_made_without_a_statement(self):
        # D1FIX: List() stores its private state directly (no statement entry)
        with mock.patch.object(journal, "run_statement", wraps=journal.run_statement) as run:
            items = List([1.0, 2.0])
        run.assert_not_called()
        self.assertIsNone(items._parent_multi)

    def test_undo_off_keeps_nothing(self):
        @journal.statement
        def edit():
            _set_op("a.d1", 5.0)
            _set_op("a.d2", 6.0)

        cmds.undoInfo(state=False)
        try:
            with _Tracked() as kept:
                edit()
                _set_op("a.d1", 7.0)  # at depth 0 too
        finally:
            cmds.undoInfo(state=True)
        gc.collect()
        self.assertEqual(self.values(), (7.0, 6.0))
        self.assertEqual(len(kept), 2)
        self.assertEqual([ref() for ref in kept], [None, None])  # Maya kept no journal
        self.assertTrue(self.queue_empty())

    def test_a_statement_inside_a_chunk_joins_it(self):
        @journal.statement
        def edit():
            _set_op("a.d1", 5.0)

        cmds.undoInfo(openChunk=True)
        try:
            cmds.setAttr("a.d2", 3.0)
            edit()
            _set_op("a.d1", 7.0)
        finally:
            cmds.undoInfo(closeChunk=True)
        self.undo_steps(1)
        self.assertEqual(self.values(), (0.0, 0.0))
        self.assertTrue(self.queue_empty())

    def test_rigs_undo_chunk_is_a_barrier(self):
        # opening registers what came before it, outside the chunk; closing registers
        # what was journaled inside it, inside the chunk
        @journal.statement
        def edit():
            _set_op("a.d1", 5.0)
            with rig.undo_chunk("inner"):
                self.assertEqual(journal._S.pending, [])
                _set_op("a.d2", 6.0)
            self.assertEqual(journal._S.pending, [])
            _set_op("a.d1", 7.0)

        edit()
        self.assertEqual(self.values(), (7.0, 6.0))
        self.undo_steps(1)
        self.assertEqual(self.values(), (5.0, 6.0))
        self.assertEqual(self.undo_name(), "inner")
        self.undo_steps(1)
        self.assertEqual(self.values(), (5.0, 0.0))
        self.undo_steps(1)
        self.assertEqual(self.values(), (0.0, 0.0))
        self.assertTrue(self.queue_empty())

    def test_cmds_edit_and_mel_eval_are_barriers(self):
        @journal.statement
        def edit():
            _set_op("a.d1", 5.0)
            journal.cmds_edit.setAttr("a.d1", 6.0)
            self.assertEqual(journal._S.pending, [])
            _set_op("a.d1", 7.0)
            journal.mel_eval("setAttr a.d1 8;")
            self.assertEqual(journal._S.pending, [])

        edit()
        for expected in (7.0, 6.0, 5.0, 0.0):
            self.undo_steps(1)
            self.assertEqual(self.values()[0], expected)
        self.assertTrue(self.queue_empty())

    def test_cmds_edit_reads_the_command_on_every_call(self):
        with mock.patch.object(cmds, "setAttr") as fake:
            journal.cmds_edit.setAttr("a.d1", 1.0, lock=True)
            fake.assert_called_once_with("a.d1", 1.0, lock=True)
            journal.cmds_edit.setAttr("a.d1", 2.0)  # no keywords: a positional call
            fake.assert_called_with("a.d1", 2.0)
        self.assertEqual(journal.cmds_edit.getAttr("a.d1"), 0.0)
        self.assertIs(journal.cmds_edit.setAttr, journal.cmds_edit.setAttr)  # built once
        self.assertEqual(journal.cmds_edit.setAttr.__name__, "setAttr")
        with self.assertRaises(AttributeError):
            journal.cmds_edit._private  # noqa: B018

    def test_a_new_scene_or_an_open_discards_the_pending_list(self):
        # reachable only from code that opens a file inside a statement without a barrier
        # (a user's MMessage callback); here the statement's body itself does
        self.assertTrue(journal._register_scene_callbacks())
        # kBeforeNew, kBeforeOpen, and (R5 D3) the Undo and Redo events (the redo queue, journal.add)
        self.assertEqual(len(callbacks.registered_ids(journal.__name__)), 4)
        folder = tempfile.mkdtemp(prefix="rig_r5_journal_")
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        path = os.path.join(folder, "scene.ma").replace("\\", "/")
        cmds.file(rename=path)
        cmds.file(save=True, type="mayaAscii", force=True)

        def raw_new():
            cmds.file(new=True, force=True)

        def raw_open():
            cmds.file(path, open=True, force=True)

        for opens in (raw_new, raw_open):
            with self.subTest(opens.__name__):
                cmds.flushUndo()

                @journal.statement
                def edit():
                    _set_op("a.d1", 5.0)
                    opens()
                    return list(journal._S.pending)

                self.assertEqual(edit(), [])
                self.assertTrue(self.queue_empty())
                if opens is raw_new:
                    cmds.file(path, open=True, force=True)

    def test_register_api_edit_joins_the_statement(self):
        @journal.statement
        def edit():
            for name, value in (("a.d1", 5.0), ("a.d2", 6.0)):
                modifier = OpenMaya.MDGModifier()
                modifier.newPlugValueDouble(_plug(name), value)
                plugins._register_api_edit(modifier)
            return len(journal._S.pending)

        self.assertEqual(edit(), 2)
        self.undo_steps(1)
        self.assertEqual(self.values(), (0.0, 0.0))
        self.assertTrue(self.queue_empty())

    def test_the_api_edits_switch_is_on(self):
        self.assertIs(journal.API_EDITS, True)


class TestBarriers(_JournalCase):
    """rig's own barrier sites register the pending journal first."""

    def pending_after(self, call):
        """``call()`` inside a statement that journaled ``a.d1 = 5`` first; the number of
        edits still pending after it."""
        @journal.statement
        def edit():
            _set_op("a.d1", 5.0)
            call()
            return len(journal._S.pending)

        return edit()

    def test_the_rc_bridge(self):
        self.assertEqual(self.pending_after(lambda: rc.setAttr("a.d1", 6.0)), 0)
        self.undo_steps(1)
        self.assertEqual(self.values()[0], 5.0)

    def test_run_undoable(self):
        class SetD2:
            def doIt(self):  # noqa: N802
                _plug("a.d2").setDouble(6.0)

            def undoIt(self):  # noqa: N802
                _plug("a.d2").setDouble(0.0)

            def redoIt(self):  # noqa: N802
                self.doIt()

        self.assertEqual(self.pending_after(lambda: plugins._run_undoable(SetD2())), 0)
        self.undo_steps(1)
        self.assertEqual(self.values(), (5.0, 0.0))
        self.undo_steps(1)
        self.assertEqual(self.values(), (0.0, 0.0))

    def test_rigs_cmds_edits(self):
        # re-pinned (round 5 D2): a connect into a free dst is journaled now (2 pending:
        # the set and the connect); its cmds route, with the fast paths off, is a barrier
        self.assertEqual(self.pending_after(lambda: Node("a").d2 << Node("m").outputX), 2)
        self.undo_steps(1)
        with mock.patch.object(journal, "API_EDITS", False):
            self.assertEqual(self.pending_after(lambda: Node("a").d2 << Node("m").outputX), 0)
        self.assertEqual(self.pending_after(lambda: Node("a").rename("b")), 0)

    def test_the_member_add_of_a_node_just_made_is_none(self):
        counts = []

        def made():
            container.createNode("transform", name="t")
            counts.append(len(journal._S.pending))

        def existing():
            container.add(Node("k"))
            counts.append(len(journal._S.pending))

        with container("c"):
            self.pending_after(made)
            self.pending_after(existing)
        self.assertEqual(counts, [1, 0])

    def test_the_with_body_of_a_scope_is_at_depth_0(self):
        with container("c"):
            self.assertFalse(journal._S.open)
        self.assertFalse(journal._S.open)


# the public functions of these modules that make no node and edit nothing
_EDITS_NOTHING = frozenset({"rig.functions.inf"})
_LIBRARIES = (
    "functions", "matrix", "vector", "interpolate", "trigonometry", "quaternion", "euler",
    "tween", "random", "_dispatch",
)


class TestStatementEntries(MayaTestCase):
    """Every entry point of a user's call into rig is a journal statement: a missed one
    costs registrations (each edit its own step), never order."""

    def test_every_public_library_function(self):
        import importlib

        missing, seen = [], 0
        for short in _LIBRARIES:
            module = importlib.import_module(f"rig.{short}")
            for name, value in sorted(vars(module).items()):
                if name.startswith("_") or not callable(value) or isinstance(value, type):
                    continue
                if getattr(value, "__module__", None) != module.__name__:
                    continue  # imported: checked where it is defined
                seen += 1
                qualified = f"{module.__name__}.{name}"
                if not getattr(value, "_rig_statement", False) and qualified not in _EDITS_NOTHING:
                    missing.append(qualified)
        self.assertEqual(missing, [])
        self.assertGreater(seen, 150)

    def test_the_top_level_and_the_factories(self):
        for fn in (rig.condition, rig.constant, rn.transform, rn.multiplyDivide):
            with self.subTest(getattr(fn, "__name__", repr(fn))):
                self.assertIs(getattr(fn, "_rig_statement", False), True)

    def test_the_operators_and_the_typed_creators(self):
        from rig._internal.container import _ContainerStack
        from rig.nodetypes.follicle import Follicle

        entries = {
            "Plug": [Plug.__dict__[n] for n in ("__lshift__", "__rshift__", "__setattr__")],
            "List": [List.__dict__[n] for n in ("__lshift__", "__rshift__", "__setattr__", "__add__",
                                                "__rmul__", "__neg__", "__eq__", "__lt__")],
            "DGNode": [DGNode.__dict__[n] for n in ("__lshift__", "__rshift__", "__setattr__")],
            "NodeOp": [NodeOp.__call__],
            "container": [getattr(_ContainerStack, n) for n in
                          ("__enter__", "__exit__", "createNode", "publish_input", "publish_output")],
            "create": [DGNode.create.__func__, DGNode.define.__func__, Node.__dict__["create"].__func__,
                       Node.__dict__["define"].__func__, Material.create.__func__, Material.define.__func__,
                       Material.astype, Lambert.create.__func__, Transform.create_hierarchy.__func__,
                       Follicle.create_on_mesh.__func__],
        }
        for kind, functions in entries.items():
            for fn in functions:
                with self.subTest(f"{kind} {fn.__name__}"):
                    self.assertIs(getattr(fn, "_rig_statement", False), True)

    def test_a_users_memoized_function_is_no_statement(self):
        @rig.memoize
        def users(x):
            return x

        @rig.vectorize
        def vectorized(x):
            return x

        self.assertFalse(hasattr(users, "_rig_statement"))
        self.assertFalse(hasattr(vectorized, "_rig_statement"))

    def test_a_plug_operator_runs_as_a_statement(self):
        cmds.file(new=True, force=True)
        a = Node(cmds.createNode("transform", name="a"))
        seen = []
        real = journal.run_statement

        def spy(func, args, kwargs):
            seen.append(func.__name__)
            return real(func, args, kwargs)

        with mock.patch.object(plug_module, "run_statement", spy):
            a.tx + a.ty
            -a.tx
        self.assertEqual(seen, ["__add__", "__neg__"])


class TestUserCodeOrder(_JournalCase):
    """CR-5: code a user wrote that rig calls inside a statement (a spec's apply, a
    post_create, ...) runs after what the statement journaled before it: one undo reverts
    the user's edit, the next the statement's journal."""

    def spec_case(self, plain_call=False):
        class UsersSpec(_AttrSpec):
            def apply(self, target):
                cmds.setAttr("a.d1", 9.0)
                return target

        done = self.journaled_float_sets()
        a = Node("a")
        if plain_call:
            # the negative control: the call-out taken away
            with mock.patch.object(plug_module, "user_call", lambda fn, *x, **k: fn(*x, **k)):
                List([a.d1, a.d2]) << [5.0, UsersSpec()]
        else:
            List([a.d1, a.d2]) << [5.0, UsersSpec()]
        self.assertEqual(done, ["a.d1"])
        self.assertEqual(self.values(), (9.0, 0.0))
        found = []
        for _ in range(2):
            self.undo_steps(1)
            found.append(self.values()[0])
        self.assertTrue(self.queue_empty())
        return found

    def test_a_specs_apply_runs_after_the_journal(self):
        self.assertEqual(self.spec_case(), [5.0, 0.0])

    def test_without_the_call_out_the_order_is_wrong(self):
        # what the test above would see without journal.user_call: the user's set is
        # recorded first, and undo leaves 5
        self.assertEqual(self.spec_case(plain_call=True), [0.0, 5.0])

    def output_spec_case(self, make, plain_call=False):
        # D1FIX: ``node >> spec`` (and ``List([node]) >> spec``) runs a user's spec's apply
        # through journal.user_call too (_apply_spec_as_output)
        from rig._internal import node as node_module

        class UsersSpec(_AttrSpec):
            def apply(self, target):
                Node("a").d1 << 5.0          # a rig statement: journaled
                cmds.setAttr("a.d1", 9.0)    # then the user's raw edit
                return target

        done = self.journaled_float_sets()
        if plain_call:
            with mock.patch.object(node_module, "user_call", lambda fn, *x, **k: fn(*x, **k)):
                make(UsersSpec())
        else:
            make(UsersSpec())
        self.assertEqual(done, ["a.d1"])
        self.assertEqual(self.values()[0], 9.0)
        found = []
        for _ in range(2):
            self.undo_steps(1)
            found.append(self.values()[0])
        self.assertTrue(self.queue_empty())
        return found

    def test_a_specs_apply_as_an_output_runs_outside_the_statement(self):
        for label, make in (("node", lambda spec: Node("a") >> spec),
                            ("List", lambda spec: List([Node("a")]) >> spec)):
            with self.subTest(label):
                self.assertEqual(self.output_spec_case(make), [5.0, 0.0])

    def test_without_the_output_spec_call_out_the_order_is_wrong(self):
        self.assertEqual(self.output_spec_case(lambda spec: Node("a") >> spec, plain_call=True), [0.0, 5.0])

    def post_create_case(self, plain_call=False):
        class UsersTransform(Transform):
            @classmethod
            def post_create(cls, new_node_name, *args, **kwargs):
                cmds.disconnectAttr("m.outputX", "k.input1X")
                return super().post_create(new_node_name, *args, **kwargs)

        def body():
            _connect_op("m.outputX", "k.input1X")
            UsersTransform.create(name="x")

        def connected():
            return cmds.isConnected("m.outputX", "k.input1X")

        if plain_call:
            with mock.patch.object(dg_node_module, "user_call", lambda fn, *x, **k: fn(*x, **k)):
                journal.run_statement(body, (), {})
        else:
            journal.run_statement(body, (), {})
        self.assertFalse(connected())
        found = []
        for _ in range(3):
            self.undo_steps(1)  # with the dangling check after each step
            found.append((connected(), cmds.objExists("x")))
        self.assertTrue(self.queue_empty())
        self.redo_all()
        return found, (connected(), cmds.objExists("x"))

    def test_a_post_create_runs_after_the_journal(self):
        # the queue: createNode x (made before the call-out: it commutes), the journal, the
        # user's disconnect
        found, redone = self.post_create_case()
        self.assertEqual(found, [(True, True), (False, True), (False, False)])
        self.assertEqual(redone, (False, True))

    def test_without_the_post_create_call_out_the_order_is_wrong(self):
        found, redone = self.post_create_case(plain_call=True)
        self.assertEqual(found[1], (True, True))  # the journal undone first: left connected
        self.assertEqual(redone, (True, True))    # and redone last

    def test_undo_off_around_a_statement(self):
        done = self.journaled_float_sets()
        a = Node("a")
        cmds.undoInfo(state=False)
        try:
            with _Tracked() as kept:
                List([a.d1, a.d2]) << [5.0, 6.0]
        finally:
            cmds.undoInfo(state=True)
        gc.collect()
        self.assertEqual(done, ["a.d1", "a.d2"])
        self.assertEqual(self.values(), (5.0, 6.0))
        self.assertEqual([ref() for ref in kept], [None])
        self.assertTrue(self.queue_empty())

    def undo_toggled_case(self, start, inside):
        class Toggles(Transform):
            @classmethod
            def post_create(cls, new_node_name, *args, **kwargs):
                cmds.undoInfo(state=inside)
                return super().post_create(new_node_name, *args, **kwargs)

        def body():
            _set_op("a.d1", 5.0)  # under the start state: registered before the call-out
            Toggles.create(name="x")
            _set_op("a.d2", 6.0)  # under the state the post_create left

        cmds.undoInfo(state=start)
        try:
            with _Tracked() as kept:
                journal.run_statement(body, (), {})
        finally:
            cmds.undoInfo(state=True)
        gc.collect()
        self.assertEqual(self.values(), (5.0, 6.0))
        return kept

    def test_undo_turned_off_inside_a_post_create(self):
        kept = self.undo_toggled_case(start=True, inside=False)
        # turning undo off flushed the queue, the first journal with it; the second was
        # registered with undo off: Maya keeps neither
        self.assertEqual([ref() for ref in kept], [None, None])
        self.assertTrue(self.queue_empty())

    def test_undo_turned_on_inside_a_post_create(self):
        kept = self.undo_toggled_case(start=False, inside=True)
        # the first journal was registered with undo off (kept nothing), the second with
        # undo on: one step, reverting d2 only
        self.assertEqual(len(kept), 2)
        self.assertIsNone(kept[0]())
        self.undo_steps(1)
        self.assertEqual(self.values(), (5.0, 0.0))
        self.assertTrue(self.queue_empty())

    def test_an_iterator_operand_is_read_outside_the_statement(self):
        seen = []

        def values():
            seen.append(journal._S.open)
            cmds.setAttr("a.d2", 9.0)
            yield 1.0

        @journal.statement
        def edit():
            _set_op("a.d1", 5.0)
            return rig.functions.sum(values())

        edit()
        self.assertEqual(seen, [False])
        self.undo_steps(1)
        self.assertEqual(self.values(), (5.0, 0.0))  # the user's set: after the journal
        self.undo_steps(1)
        self.assertEqual(self.values(), (0.0, 0.0))

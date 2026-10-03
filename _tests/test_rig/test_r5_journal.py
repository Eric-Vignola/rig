"""Round 5, D1: the statement journal (``rig._internal.journal``).

`TestBarrierLint` keeps the barrier rule true: every ``maya.cmds`` edit rig makes goes
through ``cmds_edit.<command>`` (or ``mel_eval``), which registers the pending journal
first; only ``createNode``, ``loadPlugin`` and the member add of a node a container scope
just made stay raw.
"""

import ast
import os
import unittest

import rig

# --------------------------------------------------------------------------------------- #
#  The barrier lint
# --------------------------------------------------------------------------------------- #

# Every maya.cmds name rig calls is in exactly one of QUERY, DUAL and EDIT; an unknown name
# fails the lint ("classify it").
#   QUERY  never edits the scene or the undo queue: always raw ``cmds.<name>``
#   DUAL   edits unless called with a literal query flag (q / query): raw only for a query
#   EDIT   always an edit: ``cmds_edit.<name>``, except NON_BARRIER names and RAW_SITES
QUERY = frozenset({
    "about", "attributeName", "attributeQuery", "connectionInfo", "currentUnit_q", "deformableShape",
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
# add of a node container.createNode just made (`_add` registers first for any other node)
RAW_SITES = frozenset({
    ("_internal/container.py", "_ContainerStack._add", "container"),
})

# mel.eval sites that only query (raw); every other one is journal.mel_eval
MEL_QUERY_SITES = frozenset({("nodetypes/dag_node.py", "DAGNode.get_deformers")})

# the names rig binds maya.cmds to
CMDS_NAMES = frozenset({"cmds", "_mc", "mc"})

# not linted: the journal (the proxy itself) and the undo command (its undoInfo calls run
# during undo / redo or around a registration, never with a pending journal)
EXCLUDED_FILES = frozenset({"_internal/journal.py", "nodetypes/plugins/undoable_api_command.py"})

# where getattr(cmds, ...) may look a command up by name
GETATTR_ALLOWED = ("bridges/commands.py", "nodetypes/plugins/")

# where an argument-less .doIt() may apply an API edit (anywhere else it would be applied
# outside the journal and the undo command)
DOIT_ALLOWED = ("nodetypes/plugins/", "nodetypes/mesh.py")


def _query_flag(call):
    for kw in call.keywords:
        if kw.arg in ("q", "query") and not (isinstance(kw.value, ast.Constant) and not kw.value.value):
            return True
    return False


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
        if not isinstance(node, ast.Call):
            continue
        func_node = node.func
        where     = _enclosing(parents, node)
        if isinstance(func_node, ast.Attribute) and isinstance(func_node.value, ast.Name) \
                and func_node.value.id in CMDS_NAMES:
            name = func_node.attr
            if name in QUERY:
                continue
            if name in DUAL:
                if _query_flag(node):
                    continue
            elif name not in EDIT:
                out.append((node.lineno, f"unclassified cmds.{name} in {where}: classify it"))
                continue
            if name in NON_BARRIER or (rel, where, name) in RAW_SITES:
                continue
            out.append((node.lineno, f"raw edit cmds.{name} in {where} (use cmds_edit.{name})"))
        elif isinstance(func_node, ast.Attribute) and isinstance(func_node.value, ast.Name) \
                and func_node.value.id == "mel" and func_node.attr == "eval":
            if (rel, where) not in MEL_QUERY_SITES:
                out.append((node.lineno, f"mel.eval in {where} (use mel_eval)"))
        elif isinstance(func_node, ast.Name) and func_node.id == "getattr" and node.args \
                and isinstance(node.args[0], ast.Name) and node.args[0].id in CMDS_NAMES:
            if not rel.startswith(GETATTR_ALLOWED):
                out.append((node.lineno, f"getattr(cmds, ...) in {where}: a dynamic command"))
        elif isinstance(func_node, ast.Attribute) and func_node.attr == "doIt" and not node.args \
                and not node.keywords:
            if not rel.startswith(DOIT_ALLOWED):
                out.append((node.lineno, f".doIt() in {where}: an API edit outside the journal"))
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
            "def f():\n    cmds.fancyNewCommand('a')\n": "unclassified cmds.fancyNewCommand in f",
            "def f():\n    mel.eval('x;')\n": "mel.eval in f",
            "def f():\n    getattr(cmds, 'setAttr')('a.tx', 1)\n": "getattr(cmds, ...) in f",
            "def f(m):\n    m.doIt()\n": ".doIt() in f",
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
        ):
            with self.subTest(source=source):
                self.assertEqual(self.lint(source), [])
        # a literal query=False is an edit
        self.assertEqual(len(self.lint("def f():\n    cmds.sets('s', query=False)\n")), 1)
        # the allowed places
        self.assertEqual(self.lint("class _ContainerStack:\n    def _add(self):\n"
                                   "        cmds.container('c', edit=True, addNode=['n'])\n",
                                   rel="_internal/container.py"), [])
        self.assertEqual(self.lint("class DAGNode:\n    def get_deformers(self):\n        mel.eval('x')\n",
                                   rel="nodetypes/dag_node.py"), [])
        self.assertEqual(self.lint("def f():\n    getattr(cmds, 'x')\n", rel="bridges/commands.py"), [])
        self.assertEqual(self.lint("def f(m):\n    m.doIt()\n", rel="nodetypes/plugins/__init__.py"), [])
        self.assertEqual(self.lint("def f(m):\n    m.doIt()\n", rel="nodetypes/mesh.py"), [])

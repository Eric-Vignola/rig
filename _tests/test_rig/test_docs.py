"""The ``.md`` pages run as documented.

* The node-class pages (``rig/nodetypes/CHEATSHEET.md`` and ``README.md``): every
  ``python`` block of a page runs top to bottom in one namespace, as the
  cheatsheet says (a block after a ``<!-- notest -->`` marker is skipped). A
  block must not raise, must not build a DSL network (``==``, ``sorted()`` or
  ``//`` on a DSL ``Plug`` build nodes, or raise, where the typed ``Attribute``
  spelling does not), and each ``print(...)  # expected`` comment must match the
  line it printed, in blocks where every printed line is annotated.
* Every page: each name a block imports from ``rig`` exists, and no page names
  ``PlugList`` or ``PyNode``, which round 4a removed (``List`` and ``Node`` are
  the names).
"""

import ast
import contextlib
import importlib
import io
import os
import re
from unittest import mock

import maya.standalone
from maya import cmds

import rig
from rig._tests._base import MayaTestCase

_FENCE = re.compile(r"(<!-- notest -->\s*)?^```python[^\n]*\n(.*?)^```", re.S | re.M)

_PAGES = (
    "README.md",
    "CHEATSHEET.md",
    os.path.join("examples", "README.md"),
    os.path.join("nodetypes", "README.md"),
    os.path.join("nodetypes", "CHEATSHEET.md"),
    os.path.join("spec", "README.md"),
    os.path.join("spec", "CHEATSHEET.md"),
    os.path.join("bridges", "README.md"),
    os.path.join("bridges", "CHEATSHEET.md"),
)

# node types only a DSL operator network creates (NodeOps run in containers)
_DSL_NETWORK_TYPES = (
    "condition", "equal", "lessThan", "greaterThan", "divide", "subtract", "floor",
    "modulo", "container",
)


def _text(relpath):
    path = os.path.join(os.path.dirname(rig.__file__), relpath)
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _blocks(relpath, with_notest=False):
    """The (index, source) of the python blocks of `relpath` (the run ones, or all)."""
    return [(i, m.group(2)) for i, m in enumerate(_FENCE.finditer(_text(relpath)))
            if with_notest or not m.group(1)]


def _expected(src):
    """The ``# expected`` comment of every annotated ``print(`` line, without the
    ``-- ...`` explanation."""
    expected = []
    for line in src.splitlines():
        if line.strip().startswith("print(") and "  # " in line:
            expected.append(line.split("  # ", 1)[1].strip().split(" -- ")[0].strip())
    return expected


class TestTypedDocsRun(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        for plugin in ("matrixNodes", "quatNodes"):
            cmds.loadPlugin(plugin, quiet=True)
        self._options = dict(rig.get_options())

    def tearDown(self):
        rig.set_options(**self._options)
        cmds.file(new=True, force=True)
        super().tearDown()

    def _run(self, relpath):
        namespace = {"__name__": "__doc__"}
        with mock.patch.object(maya.standalone, "initialize", lambda *a, **k: None):
            for i, src in _blocks(relpath):
                first = src.strip().splitlines()[0][:60] if src.strip() else ""
                first = first.encode("ascii", "replace").decode()  # the runner prints in cp1252
                with self.subTest(block=i, first=first):
                    out = io.StringIO()
                    try:
                        with contextlib.redirect_stdout(out):
                            exec(compile(src, f"{relpath}#{i}", "exec"), namespace)
                    except Exception as err:  # noqa: BLE001 -- reported ASCII-safe (see `first`)
                        raise self.failureException(
                            f"block raised {type(err).__name__}: {ascii(str(err))}") from None
                    built = {t: cmds.ls(type=t) for t in _DSL_NETWORK_TYPES}
                    self.assertEqual({t: n for t, n in built.items() if n}, {})
                    expected = _expected(src)
                    printed  = [line.rstrip() for line in out.getvalue().splitlines()]
                    if len(expected) == len(printed):
                        for want, got in zip(expected, printed):
                            if not (got.startswith(want) or want.startswith(got)):
                                self.assertEqual(ascii(got), ascii(want))

    def test_nodetypes_cheatsheet(self):
        self._run(os.path.join("nodetypes", "CHEATSHEET.md"))

    def test_nodetypes_readme(self):
        self._run(os.path.join("nodetypes", "README.md"))


class TestDocsNameLiveNames(MayaTestCase):
    def test_every_imported_name_exists(self):
        """``from rig... import X`` in any block (notest ones too) names a live
        module and attribute; ``import rig.x`` a live module."""
        for relpath in _PAGES:
            for i, src in _blocks(relpath, with_notest=True):
                tree = ast.parse(src)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "rig":
                        module = importlib.import_module(node.module)
                        for alias in node.names:
                            with self.subTest(page=relpath, block=i, name=f"{node.module}.{alias.name}"):
                                if not hasattr(module, alias.name):
                                    importlib.import_module(f"{node.module}.{alias.name}")
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name.split(".")[0] == "rig":
                                with self.subTest(page=relpath, block=i, name=alias.name):
                                    importlib.import_module(alias.name)

    def test_no_page_names_a_removed_name(self):
        for relpath in _PAGES:
            text = _text(relpath)
            for name in ("PlugList", "PyNode"):
                with self.subTest(page=relpath, name=name):
                    lines = [n for n, line in enumerate(text.splitlines(), 1) if name in line]
                    self.assertEqual(lines, [], f"{relpath} names {name} on these lines")

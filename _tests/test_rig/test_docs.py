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


# The property ban (user decision 2026-10-16, "Name clashes + property ban") and the
# per-object rule ("Objects outside nodes and plugs"): the members rig renamed, by the
# spelling they had up to v2.0.0a5. A predicate kept its name (called now), so only a
# dotted mention of one is checked.
_RENAMED_NODE = (
    "name long_name short_name clean_name uuid namespace node_type is_locked is_valid "
    "mobject fn_set mdagpath is_shape local_shape_attr world_shape_attr injection_node "
    "component_tags num_vertices num_polygons num_weight_points num_cvs uv_sets "
    "current_uv_set paintable_maps engine is_default num_targets num_keys file_path "
    "file_path_with_copy_number").split()
_RENAMED_PLUG = (
    "plug full_name node alias attribute_type data_type default_value enums is_dynamic "
    "is_keyable is_channel_box is_connected is_free_to_change num_children is_typed "
    "is_multi num_elements next_index").split()
_RENAMED_HANDLE = "indices count names kind shape geometry representation is_all stack containers".split()
_RENAMED = sorted(set(_RENAMED_NODE + _RENAMED_PLUG + _RENAMED_HANDLE), key=len, reverse=True)
# a bare `name` in prose is checked only where the word is not also plain English
_BARE_CHECKED = {
    n for n in _RENAMED
    if not n.startswith("is_") and n not in {
        "name", "node", "plug", "alias", "shape", "kind", "count", "names", "indices",
        "geometry", "stack", "containers", "engine", "namespace", "uuid", "enums",
        "representation"}}
_PROSE_FENCE = re.compile(r"^```.*?^```", re.S | re.M)
_PROSE_SPAN = re.compile(r"`([^`]+)`")
_PROSE_DOT = re.compile(r"([\w\)\]\"'.]+)\.(" + "|".join(_RENAMED) + r")\b(?!\s*\()(?!\w)")
# receivers whose fields are a value's (the per-object rule) or no rig member at all
_VALUE_RECEIVER = re.compile(r"^(Tag\(.*\)|data|uv|keys|err|spec|cgmath.*|rig\._internal|[\"'].*)$")
# a span that shows an old spelling on purpose: the guard, or a Maya child named so
_SHOWN_ON_PURPOSE = {
    "if a.tx.is_locked:", "if ctl.is_locked:", "if attr.is_locked:", "attr.is_locked = x",
    "nodes.is_locked", "renderLayer.adjustments[0].plug", 'node.namespace = "ns"',
    # the guard covers a comparison too (fix round 2026-10-16)
    "a.tx.is_locked == False", "ctl.is_valid == True",
}


def _prose_spans(relpath):
    """The (line, span) of every inline code span outside the code blocks of a
    page, skipping the rows of a ``| Was | Now |`` table (the renames, listed)."""
    text = _PROSE_FENCE.sub(lambda m: "\n" * m.group(0).count("\n"), _text(relpath))
    lines = text.split("\n")
    in_table = False
    for n, line in enumerate(lines):
        if line.startswith("| Was | Now"):
            in_table = True
        elif in_table and not line.startswith("|"):
            in_table = False
        if in_table:
            lines[n] = ""
    text = "\n".join(lines)
    for m in _PROSE_SPAN.finditer(text):
        yield text.count("\n", 0, m.start()) + 1, " ".join(m.group(1).split())


class TestDocsSpeakTheNewSpellings(MayaTestCase):
    """Every page names rig's members as they are spelled since the property ban
    and the per-object rule (user decisions 2026-10-16); the one-sentence rule is
    stated where the operators are introduced."""

    def test_prose_names_no_renamed_member(self):
        for relpath in _PAGES:
            found = []
            for line, span in _prose_spans(relpath):
                if span in _SHOWN_ON_PURPOSE or " is now " in span:   # or an old name's hint
                    continue
                if span in _BARE_CHECKED:
                    found.append((line, span))
                for m in _PROSE_DOT.finditer(span):
                    if not _VALUE_RECEIVER.match(m.group(1)):
                        found.append((line, span))
                        break
            with self.subTest(page=relpath):
                self.assertEqual(found, [], f"{relpath} names a renamed member")

    def test_docstrings_name_no_renamed_member(self):
        """No ``:attr:`` reference of rig's source names a renamed member."""
        ref = re.compile(r":attr:`[~.\w]*?\b(" + "|".join(_RENAMED) + r")`")
        root = os.path.dirname(rig.__file__)
        found = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in ("_tests", "__pycache__")]
            for filename in filenames:
                if filename.endswith(".py"):
                    path = os.path.join(dirpath, filename)
                    with open(path, encoding="utf-8") as fh:
                        for n, line in enumerate(fh, 1):
                            if ref.search(line):
                                found.append((os.path.relpath(path, root), n, line.strip()))
        self.assertEqual(found, [])

    def test_the_rule_is_stated_where_the_operators_are(self):
        rule = ("a bare name is the Maya attribute if one exists, otherwise the method; "
                "with brackets it is the method")
        readme = " ".join(_text("README.md").split("## The construction rule")[0]
                          .split("## The language in one table")[1].split())
        doc = " ".join(rig.__doc__.split("Operator conventions:")[0].split())
        for where, text in (("README.md, the language in one table", readme),
                            ("help(rig), before the operators", doc),
                            ("nodetypes/README.md", " ".join(_text(os.path.join("nodetypes", "README.md")).split()))):
            with self.subTest(where=where):
                self.assertTrue(rule.lower() in text.replace("**", "").lower(),
                                f"{where} does not state the rule")
        for where, text in (("README.md", readme), ("help(rig)", doc)):
            with self.subTest(where=where, rule="per object"):
                for words in ("points into the scene", "keeps its data as attributes"):
                    self.assertTrue(words in text, f"{where} does not say {words!r}")

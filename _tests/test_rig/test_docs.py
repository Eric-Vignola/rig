"""The ``.md`` pages run as documented.

* The node-class pages (``rig/nodetypes/CHEATSHEET.md`` and ``README.md``): every
  ``python`` block of a page runs top to bottom in one namespace, as the
  cheatsheet says (a block after a ``<!-- notest -->`` marker is skipped). A
  block must not raise, must not build a DSL network (``==``, ``sorted()`` or
  ``//`` on a DSL ``Plug`` build nodes, or raise, where the typed ``Attribute``
  spelling does not), and each ``print(...)  # expected`` comment must match the
  line it printed, in blocks where every printed line is annotated.
* The clip block of the root page (``rig/CHEATSHEET.md``), which does not run whole:
  that block alone runs, and every line it prints must be its comment.
* Every page: each name a block imports from ``rig`` exists, and no page names
  ``PlugList`` or ``PyNode``, which round 4a removed (``List`` and ``Node`` are
  the names).
* Every page and every python file of rig, its tests included: no method the
  names review retired is named, called or not, outside the hint tables, the Was
  cell of a ``| Was | Now |`` row and the tables of the tests that pin them.
* The review's docstrings, its keyword rule and its undo steps
  (`TestDocstringsSayWhatTheMethodsDo`, `TestTheKeywordRule`,
  `TestTheUndoTablesNameTheNewSteps`).
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
    ``-- ...`` explanation; from Maya 2027, what the explanation gives after
    ``Maya 2027:``, where it does (that Maya's default shader is openPBR_shader1)."""
    later = int(cmds.about(majorVersion=True)) >= 2027
    expected = []
    for line in src.splitlines():
        if line.strip().startswith("print(") and "  # " in line:
            want, _, note = line.split("  # ", 1)[1].strip().partition(" -- ")
            if later and "Maya 2027: " in note:
                want = note.split("Maya 2027: ", 1)[1]
            expected.append(want.strip())
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

    def test_the_clip_example_of_the_root_cheatsheet(self):
        """The root page does not run whole; this example does, and prints what it says."""
        blocks = [src for _, src in _blocks("CHEATSHEET.md") if "ClipData" in src]
        self.assertEqual(len(blocks), 1)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(compile(blocks[0], "CHEATSHEET.md#clip", "exec"), {"__name__": "__doc__"})
        self.assertEqual([line.rstrip() for line in out.getvalue().splitlines()], _expected(blocks[0]))


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


# The methods the names review retired (user decision 2026-10-30), and the keys value's
# count: `TestRetiredMethods` sees them called, where the lint above reads only a name
# with no brackets, only in prose, and blanks every Was | Now row. Each name has its
# hint (`_OLD_NODE_NAMES`, `_OLD_PLUG_NAMES`, the `_OLD_NAMES` of a handle or of the keys
# value), which keeps the spelling on purpose.
_OLD_METHODS = (
    # Attribute
    "logical_index element_by_logical_index element_by_physical_index delete_logical_index "
    "get_num_children get_num_elements "
    # DGNode, DisplayLayer, AnimCurve and the keys value
    "ensure_valid get_clean_name for_node get_num_keys num_keys "
    # Geometry
    "add_component_tag remove_component_tag rename_component_tag set_component_tag_contents "
    "select_component_tag_contents get_component_tags has_component_tag get_component_tag_index "
    "get_component_tag_name component_tag_expression_subset_state get_component_tag_history "
    "get_component_tag_category set_component_tag_category get_component_tag_contents "
    "get_component_tag_indices is_component_tag_empty get_component_tag_data "
    "serialize_component_tags get_local_shape_attr get_world_shape_attr "
    # Mesh, its ColorSet, NurbsCurve, NurbsSurface, Follicle
    "get_per_face_vertex_normals get_num_vertices get_num_polygons get_num_weight_points "
    "merge_maps merge_maps_list sanitize_map_values paint_map get_paintable_maps "
    "transfer_component_tags as_current_color_set get_num_cvs set_uv_values "
    # Joint, BlendShape, SkinCluster, Transform
    "convert_orients_to_rotation convert_rotation_to_orients hierarchy_to_rotations "
    "hierarchy_to_orients orient_joint orient_chain target_data_attrs get_num_targets "
    "set_targets_data copy_skincluster connect_bind_pre_matrices set_xfrom_attrs_locked").split()
# the class-route mirror of a retired name is caught with it, and so is a family spelled
# with a wildcard; `_ensure_valid`, the hot readers' inline check, is another name
_OLD_METHOD = re.compile(r"\b(?:_raw_)?(?:" + "|".join(_OLD_METHODS) + r"|element_by_\w*)\b")
# the statements that keep an old spelling on purpose: the hint tables, the tables of
# the tests that pin them, and the pattern above
_PIN_TABLES = {
    "_OLD_NODE_NAMES", "_OLD_PLUG_NAMES", "_OLD_PLUG_METHODS", "_OLD_NAMES", "_OLD_METHODS", "_OLD_METHOD",
    "_OLD_KEYWORDS", "_RENAMED", "_RENAMED_NODE",
}


def _rig_files(*suffixes):
    """The path, relative to rig, of every file of rig that ends in one of `suffixes`."""
    root = os.path.dirname(rig.__file__)
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(files):
            if name.endswith(suffixes):
                yield os.path.relpath(os.path.join(base, name), root)


def _pin_lines(source):
    """The line numbers of the `_PIN_TABLES` statements of the python `source`."""
    lines = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) in _PIN_TABLES for t in node.targets):
            lines.update(range(node.lineno, node.end_lineno + 1))
    return lines


def _was_cells(relpath):
    """The Was cell of every ``| Was | Now |`` row of the page `relpath`."""
    cells, in_table = [], False
    for line in _text(relpath).split("\n"):
        if line.startswith("| Was | Now"):
            in_table = True
        elif not line.startswith("|"):
            in_table = False
        elif in_table:
            cells.append(line.split(" | ", 1)[0])
    return cells


def _old_method_lines(relpath, text):
    """The (line number, line) of each line of `text`, the file `relpath`, that names an
    old method: a page's prose and code, or a python file's code, comments and docstrings.
    A Was | Now row is read from its Now cell (only the Was cell keeps the old spelling)."""
    python   = relpath.endswith(".py")
    pinned   = _pin_lines(text) if python else ()
    in_table = False
    found    = []
    for n, line in enumerate(text.split("\n"), 1):
        if not python:
            if line.startswith("| Was | Now"):
                in_table = True
            elif not line.startswith("|"):
                in_table = False
        read = line.split(" | ", 1)[-1] if in_table else line
        if n not in pinned and _OLD_METHOD.search(read):
            found.append((n, line.strip()))
    return found


class TestRetiredMethods(MayaTestCase):
    """No page or python file of rig names a method the names review retired."""

    def test_no_file_names_an_old_method(self):
        root = os.path.dirname(rig.__file__)
        for relpath in _rig_files(".py", ".md"):
            with open(os.path.join(root, relpath), encoding="utf-8") as fh:
                found = _old_method_lines(relpath, fh.read())
            with self.subTest(file=relpath):
                self.assertEqual(found, [], f"{relpath} names a retired method")

    def test_the_check_sees_what_the_prose_lint_cannot(self):
        # the negative control: a call, a Now cell, a comment, a docstring, a mirror and a
        # family's wildcard are found; a Was cell, a hint table and a longer name are not
        old  = _OLD_METHODS[0]
        page = (
            f"| Was | Now |\n|---|---|\n| `node.{old}()` | `node.new()` |\n| `node.x` | `node.{old}()` |\n"
            f"\n`node.{old}()`\n"
        )
        self.assertEqual([n for n, _ in _old_method_lines("page.md", page)], [4, 6])
        wild   = _OLD_METHODS[1].replace(old, "*")
        source = (
            f"_OLD_NODE_NAMES = {{\n    '{old}': 'new',\n}}\nnode.{old}()\n# {old}\n"
            f"type(x)._raw_{old}(x)\nnode.my_{old}()\n"
            f'def f():\n    """Reads the {old}."""\n# {wild}\n'
        )
        self.assertEqual([n for n, _ in _old_method_lines("code.py", source)], [4, 5, 6, 9, 10])

    def test_every_old_method_has_its_row(self):
        was = " ".join(_was_cells("CHEATSHEET.md"))
        self.assertEqual([name for name in _OLD_METHODS if not re.search(rf"\b{name}\b", was)], [])

    def test_every_old_method_has_its_hint(self):
        from rig._internal.anim_keys import _Keys
        from rig.nodetypes._base import _OLD_PLUG_NAMES
        from rig.nodetypes.dg_node import _OLD_NODE_NAMES
        from rig.nodetypes.mesh import ColorSet

        hinted = set(_OLD_NODE_NAMES) | set(_OLD_PLUG_NAMES) | set(ColorSet._OLD_NAMES) | set(_Keys._OLD_NAMES)
        self.assertEqual(set(_OLD_METHODS) - hinted, set())


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


class TestDocstringsSayWhatTheMethodsDo(MayaTestCase):
    """The docstrings the names review found wrong, each with the words that were wrong and
    the words it holds now, and its typos (user decision 2026-10-30)."""

    def test_no_docstring_keeps_its_wrong_words(self):
        from rig.nodetypes import Mesh
        from rig.nodetypes._base import Attribute
        from rig.nodetypes.dag_node import DAGNode
        from rig.nodetypes.dg_node import DGNode
        from rig.nodetypes.transform import Transform

        for cls, name, wrong, right in (
            (Attribute, "list_connections", "on this node", "connected to this attribute"),
            (DGNode, "list_connections", "connected attrs on this node", "connected to this node"),
            (DAGNode, "get_children", "shape objects", "shapes and transforms"),
            (Transform, "serialize", "from a given string", "TransformData"),
            (Mesh, "get_map_names", "paintable maps", "paintable or not"),
            (Mesh, "enter_paint_tool", "Enable painting", "Paint Attributes Tool"),
        ):
            with self.subTest(method=f"{cls.__name__}.{name}"):
                doc = getattr(cls, name).__doc__
                self.assertNotIn(wrong, doc)
                self.assertIn(right, " ".join(doc.split()))

    def test_no_typo_of_the_review_is_left(self):
        typo = re.compile(r"\b(?:Deletse|erros|Rebuils|intermidiate|identifyer|an [Uu][Vv])\b")
        root = os.path.dirname(rig.__file__)
        for relpath in _rig_files(".py"):
            if relpath.startswith("_tests"):
                continue
            with open(os.path.join(root, relpath), encoding="utf-8") as fh:
                found = typo.findall(fh.read())
            with self.subTest(file=relpath):
                self.assertEqual(found, [])


class TestTheKeywordRule(MayaTestCase):
    """``failfast=`` where the default answers, ``quiet=`` where the default raises: both stay,
    and the conventions state the rule (user decision 2026-10-30). A test named ``guard`` holds
    on the tree before it too: what stays."""

    TEST_START_NEW_SCENE = True

    def test_the_conventions_state_it(self):
        page    = _text(os.path.join("nodetypes", "README.md"))
        section = page.split("\n## Conventions\n")[1].split("\n---\n")[0]
        for words in (
            "`failfast=` where the default answers, `quiet=` where the default raises",
            "`is_type`", "`is_valid`", "`get_target_index`", "`find_map_attr`", "`find_attr`", "`find_alias`",
        ):
            with self.subTest(words=words):
                self.assertIn(words, " ".join(section.split()))

    def test_failfast_raises_where_the_default_answers(self):
        from rig.nodetypes import BlendShape, Mesh, Node
        from rig.nodetypes.transform import Transform

        cube = Node(cmds.polyCube(name="cube", ch=False)[0])
        mesh = Mesh("cubeShape")
        bs   = BlendShape.create(Node(cmds.polyCube(name="target", ch=False)[0]), cube)
        gone = Transform.create(name="gone")
        gone.delete()
        for label, answer, default, error, failfast in (
            ("is_type", lambda: Transform.is_type("cubeShape"), False, ValueError,
             lambda: Transform.is_type("cubeShape", failfast=True)),
            ("get_target_index", lambda: bs.get_target_index("nope"), -1, RuntimeError,
             lambda: bs.get_target_index("nope", failfast=True)),
            ("find_map_attr", lambda: mesh.find_map_attr("visibility"), None, ValueError,
             lambda: mesh.find_map_attr("visibility", failfast=True)),
            ("is_valid", lambda: gone.is_valid(), False, RuntimeError,
             lambda: gone.is_valid(failfast=True)),
        ):
            with self.subTest(label):
                self.assertEqual(answer(), default)
                with self.assertRaises(error):
                    failfast()

    def test_guard_quiet_answers_where_the_default_raises(self):
        from rig.nodetypes.transform import Transform

        box = Transform.create(name="box")
        for label, error, raises, quiet in (
            ("find_attr", AttributeError, lambda: box.find_attr("nope"), lambda: box.find_attr("nope", quiet=True)),
            ("find_alias", RuntimeError, lambda: box.find_alias("nope"), lambda: box.find_alias("nope", quiet=True)),
        ):
            with self.subTest(label):
                self.assertIsNone(quiet())
                with self.assertRaises(error):
                    raises()


class TestTheUndoTablesNameTheNewSteps(MayaTestCase):
    """The one-step calls the names review added are in both pages' undo paragraphs (user
    decision 2026-10-30): the root CHEATSHEET's table with their step names."""

    def test_both_pages_name_them(self):
        rows    = [line for line in _text("CHEATSHEET.md").splitlines() if line.startswith("| `Mesh.create`")]
        methods = "BlendShape.create", "BlendShape.set_target_data", "SkinCluster.connect_bind_pre_matrix"
        self.assertEqual(len(rows), 1)
        for method in methods:
            with self.subTest(method):
                self.assertIn(f"`rig.{method}`", rows[0])
        self.assertIn(
            "`{}`, `{}` and `{}` are one step each".format(*methods),
            " ".join(_text(os.path.join("nodetypes", "README.md")).split()),
        )

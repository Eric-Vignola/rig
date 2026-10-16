"""Round 5, step A5: the lookup bundle.

* H6: ``DGNode.find_attr``'s lookup (``DGNode._find_attr_fresh``) tells a plug
  path (a name with a "[" or a ".") with two ``in`` tests when the name has no
  newline, where they decide as the regex did (the regex still decides a name
  with one); it asks ``_canonicalize_component_alias_plug`` about a component
  alias only, and ``_is_unresolved_multi_child`` about an element or a child
  plug only (each helper's own first test); it filters only when given a
  filter; and it says whether it built the attr from its MPlug just now.
  ``DGNode.__getattr__``, when ``node.find_attr`` is DGNode's own, gives the
  plug of such a fresh attr the attr's str instead of converting the MPlug a
  second time.
* H8: ``memoize._attribute_key`` of a plug whose owner names it as a DGNode
  does (``_dg_named_owner``) runs the two checks naming it makes
  (``_ensure_dg_named``) and reads its owner's identity, without the name.

Tests:

* ``TestTheOutcomesOfA74d089``: every lookup of ``_NAMES`` (plain names, short
  names, names with "[" and ".", newlines, carriage returns and spaces, aliases,
  dynamic attrs, generic attrs, component aliases on meshes, curves and
  surfaces and through their transforms, a shape's attrs through its
  transform, an instanced shape through each path, children of unindexed
  multis, missing names, ints and a float), through ``getattr`` (twice),
  ``find_attr``, ``find_attr(quiet=True)``, a ``data_type`` filter and
  ``find_attr`` on the node the ``getattr`` filled, gives what a74d089 gave
  (``r5_lookup_oracle_a74d089.json`` beside this file, written by
  ``runs\\r5\\A5\\gen_oracle.py`` running ``_lookup_outcomes`` on the snapshot
  of a74d089): the result (type, str buffer, name, owner, handles, MPlug,
  state keys, the cache it left, the cached instance) or the error with its
  whole chain (``__cause__`` / ``__context__``), and what it printed.
* ``TestThePlugPathTest``: the two ``in`` tests decide as the regex for every
  name with no newline (all names of up to four characters over an alphabet of
  the characters that matter), and the regex is asked only about a name with a
  newline.
* ``TestTheSkippedHelpers``: each skipped helper returns its input whenever its
  first test fails (over every attribute of a DG node, a mesh, a curve, a
  skinCluster and a blendShape), and is called exactly when that test passes;
  ``_filtered`` is called only with a filter.
* ``TestAFreshLookup``: ``_find_attr_fresh`` says True exactly for an attr it
  built; a ``getattr`` miss converts its MPlug to a str once (two times before),
  and the plug's str is the MPlug's name, also after a rename between two
  accesses and through each path of an instanced node.
* ``TestALookupOfItsOwn``: a class with its own ``find_attr``, a class-level or
  an instance-level patch of it, and a class whose ``__getattribute__`` hands
  another one are asked as before, and their plugs are converted from the MPlug.
* ``TestMemoKeys``: ``_attribute_key`` equals the key through the name on DG,
  DAG, instanced, owner-less, component and user-class plugs (a class with its
  own ``name``, a DG class holding a DAG node) and builds no name for a
  DG-named plug; a deleted, a freed and a freed-attribute plug raise as their
  name does, with no API 2.0 call; patched classes take the name; a memo entry
  is found through either key.

``runs\\r5\\A5\\make_mutants.py`` / ``run_mutants.py`` check that these tests
catch eight wrong versions of the two items.
"""

import contextlib
import io
import itertools
import json
import logging
import os
import re
import sys
import warnings
from collections import Counter
from unittest import mock

from maya import cmds
from maya.api import OpenMaya

from rig import Node, Plug
from rig._internal import memoize
from rig.nodetypes import _base
from rig.nodetypes import dg_node
from rig.nodetypes._base import Attribute, _plug_identity_name, set_custom_type
from rig.nodetypes.dg_node import DGNode
from rig._tests._base import MayaTestCase


# --------------------------------------------------------------------------- #
#  The outcomes table (`_lookup_outcomes` runs on a74d089 too: public API, and
#  the node and plug state both sides keep)
# --------------------------------------------------------------------------- #


def _chain(error):
    """[type name, message, has a cause, suppresses its context] of `error` and of
    each exception it was raised from (`__cause__`) or while handling
    (`__context__`), the whole chain Python prints."""
    out  = []
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        out.append(
            [type(error).__name__, str(error), error.__cause__ is not None, error.__suppress_context__]
        )
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return out


class _Output(logging.Handler):
    """What a call prints: Maya's command output, the ``rig`` loggers' records,
    Python warnings, ``sys.stdout`` / ``sys.stderr``."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.maya    = []
        self.records = []

    def emit(self, record):
        self.records.append([record.name, record.levelname, record.getMessage()])

    def __enter__(self):
        self._callback = OpenMaya.MCommandMessage.addCommandOutputCallback(
            lambda message, kind, *args: self.maya.append([int(kind), message])
        )
        self._logger = logging.getLogger("rig")
        self._level  = self._logger.level
        self._logger.addHandler(self)
        self._logger.setLevel(logging.DEBUG)
        self._warnings = warnings.catch_warnings(record=True)
        self.warned    = self._warnings.__enter__()
        warnings.simplefilter("always")
        self._out, self._err = io.StringIO(), io.StringIO()
        self._streams = (contextlib.redirect_stdout(self._out), contextlib.redirect_stderr(self._err))
        for stream in self._streams:
            stream.__enter__()
        return self

    def __exit__(self, *exc):
        for stream in reversed(self._streams):
            stream.__exit__(None, None, None)
        self._warnings.__exit__(None, None, None)
        self._logger.setLevel(self._level)
        self._logger.removeHandler(self)
        OpenMaya.MMessage.removeCallback(self._callback)
        return False

    def state(self):
        printed = [
            self.maya,
            self.records,
            [[w.category.__name__, str(w.message)] for w in self.warned],
            self._out.getvalue(),
            self._err.getvalue(),
        ]
        return printed if any(printed) else None


def _owner_label(value, node):
    owner = vars(value).get("_node")
    if owner is node:
        return "the node"
    return None if owner is None else type(owner).__name__


def _handle_label(value, node):
    handle = vars(value).get("_handle")
    if handle is None:
        return None
    return "the node's" if handle is vars(node).get("_objhandle") else "its own"


def _result(value, node):
    """What a lookup returned: its type, and for a plug its str buffer, name,
    owner, node handle, attribute handle, MPlug name and state keys ("state":
    the keys of a fresh Attribute's state)."""
    if not isinstance(value, str):
        return [type(value).__name__, None if value is None else str(value)]
    state = vars(value)
    try:
        name = str(value)
    except Exception as error:  # noqa: BLE001 - the error is the thing compared
        name = ["raises", type(error).__name__, str(error)]
    mplug = state.get("_mplug")
    keys  = sorted(state)
    return [
        type(value).__name__,
        str.__str__(value),
        name,
        _owner_label(value, node),
        _handle_label(value, node),
        state.get("_attrh") is not None,
        None if mplug is None else mplug.name(),
        "state" if keys == sorted(_base._attr_state(None, None)) else keys,
    ]


def _cached(node):
    return sorted(vars(node)["_attr_dict"])


def _call(node, lookup):
    """The outcome of ``lookup(node)``: its result or its error chain, what it
    printed, and the cache keys of `node` once it ran."""
    with _Output() as output:
        try:
            outcome = ["returns", _result(lookup(node), node)]
        except Exception as error:  # noqa: BLE001 - the error is the thing compared
            outcome = ["raises", _chain(error)]
    return [outcome, output.state(), _cached(node)]


def _lookups(node_name, name):
    """{call: outcome} of the lookups of `name` on the node `node_name`: each
    through a node object of its own (an empty cache), except the second
    ``getattr`` and the last ``find_attr``, on the node object the first
    ``getattr`` filled."""
    first = Node(node_name)
    out   = {}
    if isinstance(name, str):
        out["getattr"]       = _call(first, lambda n: getattr(n, name))
        out["getattr again"] = _call(first, lambda n: getattr(n, name))
    out["find_attr"]       = _call(Node(node_name), lambda n: n.find_attr(name))
    out["find_attr quiet"] = _call(Node(node_name), lambda n: n.find_attr(name, quiet=True))
    out["find_attr double"] = _call(
        Node(node_name), lambda n: n.find_attr(name, data_type="double")
    )
    held = []

    def again(node):
        found = node.find_attr(name)
        held.append(any(found is cached for cached in vars(node)["_attr_dict"].values()))
        return found

    out["find_attr on the first"] = _call(first, again)
    out["find_attr on the first"].append(held)
    return out


def _lookup_scene():
    """The scene every node's lookups start from (fixed names)."""
    cmds.file(new=True, force=True)
    cmds.createNode("multiplyDivide", name="md")
    cmds.addAttr("md", longName="dyn", attributeType="double")
    cmds.addAttr("md", longName="dynv", attributeType="double3")
    for axis in "XYZ":
        cmds.addAttr("md", longName="dynv" + axis, attributeType="double", parent="dynv")
    cmds.aliasAttr("nick", "md.input1X")
    cmds.namespace(add="ns")
    cmds.createNode("multiplyDivide", name="ns:md")
    cmds.createNode("plusMinusAverage", name="pma")
    cmds.setAttr("pma.input3D[2].input3Dx", 1.0)  # an element that exists
    cmds.createNode("choice", name="ch")
    cmds.createNode("transform", name="xf")
    for group in ("grp1", "grp2"):
        cmds.createNode("transform", name=group)
        cmds.createNode("transform", name="dup", parent=group)
    cmds.polyCube(name="cube", constructionHistory=False)
    cmds.createNode("transform", name="T2")
    cmds.parent("|cube|cubeShape", "T2", add=True, shape=True)  # an instanced shape
    cmds.curve(name="crv", degree=1, point=[(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    cmds.rename(cmds.listRelatives("crv", shapes=True)[0], "crvShape")
    cmds.sphere(name="sph", constructionHistory=False)
    cmds.select(clear=True)
    cmds.joint(name="j1")
    cmds.select(clear=True)
    cmds.polyCube(name="skinned", constructionHistory=False)
    cmds.skinCluster("j1", "skinned", name="skin")
    cmds.polyCube(name="base", constructionHistory=False)
    cmds.polyCube(name="tgt", constructionHistory=False)
    cmds.blendShape("tgt", "base", name="bls")
    cmds.select(clear=True)


# names with a "[" or a "." and a newline, a carriage return or a space (the
# regex tells a newline apart, the `in` tests do not)
_ODD_NAMES = [
    pattern
    for base in ("input1X", "input1.input1X", "input3D[0]", "t.tx", ".")
    for sep in ("\n", "\r\n", " ")
    for pattern in (base + sep, sep + base, base + sep + "x")
] + ["a\n\nb.c", "input1X\n\n", "\ninput1X\n", "input1.input1X\n\n", "\n"]

# {node: the names looked up on it}
_NAMES = {
    "md": [
        "input1X", "i1x", "input1", "input1.input1X", "input1.i1x", "nick", "dyn", "dynv",
        "dynvX", "dynv.dynvX", "output", "message", "nothere", "input1.nothere", "input1[0]",
        "input1X[0]", "[0]", ".", "", "a.b", 0, 5, 1.5,
    ] + _ODD_NAMES,
    "ns:md": ["input1X", "input1.input1X", "nothere"],
    "pma": [
        "input3D", "input3D[0]", "input3D[2]", "input3D[2].input3Dx", "input3D[0].input3Dx",
        "input3Dx", "i3", "input3D[x]", "input3D[-1]", "input1D[3]", "output3D.output3Dx",
        "output3Dx",
    ],
    "ch": ["input", "input[0]", "input[3]", "output", "selector"],
    "xf": [
        "tx", "t", "t.tx", "translate.translateX", "worldMatrix", "worldMatrix[0]", "wm[0]",
        "wm", "nothere", "vtx", "f",
    ] + _ODD_NAMES,
    "grp1|dup": ["tx", "translate.tx", "worldMatrix[0]"],
    "cube": [
        "outMesh", "vtx", "pt", "pnts", "map", "uv", "f", "e", "cv", "tx", "worldMesh",
        "worldMesh[0]", "visibility",
    ],
    "|cube|cubeShape": [
        "visibility", "worldMatrix", "worldMatrix[0]", "worldMatrix[1]", "instObjGroups",
        "vtx", "pnts", "pt", "outMesh", "f", "map",
    ],
    "|T2|cubeShape": [
        "visibility", "worldMatrix", "worldMatrix[0]", "worldMatrix[1]", "instObjGroups",
        "vtx", "pnts", "pt", "outMesh", "f", "map",
    ],
    "crv": ["cv", "tx", "pt"],
    "crvShape": ["cv", "cv[1]", "f", "form", "pt", "controlPoints", "vtx"],
    "sph": ["cv"],
    "sphShape": ["cv", "cv[1][2]", "f"],
    "skin": ["weights", "weightList", "weightList[0]", "weightList[0].weights", "wl[0].w",
             "matrix", "matrix[0]"],
    "bls": ["tgt", "weight", "weight[0]", "w[0]", "inputTargetGroup",
            "inputTarget[0].inputTargetGroup", "inputTarget[0].inputTargetGroup[0]"],
}


def _lookup_outcomes():
    """{"<node> <name>": {call: outcome}} (`_lookups`), each node's names in the
    scene `_lookup_scene` makes for it."""
    table = {}
    try:
        for node_name, names in _NAMES.items():
            _lookup_scene()
            for name in names:
                table[f"{node_name} {name!r}"] = _lookups(node_name, name)
    finally:
        cmds.file(new=True, force=True)
    return table


# a74d089's outcomes of `_NAMES` (runs\r5\A5\gen_oracle.py on its snapshot)
_ORACLE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "r5_lookup_oracle_a74d089.json"
)

# re-pinned (round 5 A6): an attr's caches of its child attributes are made on
# first use, so a74d089's state keys (written out for a component plug, whose
# keys are not a fresh Attribute's) lose them; no lookup here makes them
_LAZY_CACHES = ("_Attribute__child_name_dict", "_Attribute__child_id_dict")


def _without_lazy_caches(value):
    """`value`, an outcome of the oracle, with `_LAZY_CACHES` left out of every
    list of state keys it holds."""
    if isinstance(value, list):
        if any(isinstance(each, str) and each in _LAZY_CACHES for each in value):
            return [each for each in value if each not in _LAZY_CACHES]
        return [_without_lazy_caches(each) for each in value]
    if isinstance(value, dict):
        return {key: _without_lazy_caches(each) for key, each in value.items()}
    return value


class TestTheOutcomesOfA74d089(MayaTestCase):
    """Every lookup of `_NAMES` gives what a74d089 gave (`_ORACLE_PATH`)."""

    def test_every_lookup_as_a74d089(self):
        with open(_ORACLE_PATH) as fh:
            oracle = _without_lazy_caches(json.load(fh))
        table = json.loads(json.dumps(_lookup_outcomes()))
        self.assertEqual(sorted(table), sorted(oracle))
        for case in sorted(oracle):
            with self.subTest(case=case):
                self.assertEqual(table[case], oracle[case])

    def test_the_table_reaches_every_branch(self):
        # the oracle holds plugs and errors of each kind the lookup gives
        with open(_ORACLE_PATH) as fh:
            oracle = json.load(fh)
        kinds    = Counter()
        messages = []
        for calls in oracle.values():
            for outcome in calls.values():
                if outcome[0][0] == "returns":
                    kinds[outcome[0][1][0]] += 1
                else:
                    kinds[outcome[0][1][0][0]] += 1
                    messages.append(outcome[0][1][0][1])
        for kind in ("Plug", "Attribute", "NoneType", "AttributeError", "Components"):
            with self.subTest(kind=kind):
                self.assertGreater(kinds[kind], 0)
        for text in ("Attribute not found", "a child of an unindexed multi"):
            with self.subTest(message=text):
                self.assertTrue(any(text in message for message in messages))


# --------------------------------------------------------------------------- #
#  H6: the lookup's tests
# --------------------------------------------------------------------------- #

_PATH_REGEX = r"^.*[\[\.].*$"


def _the_lookups_path_test(text):
    """`_find_attr_fresh`'s test of a plug path, as written there."""
    return ("[" in text or "." in text) and ("\n" not in text or re.match(_PATH_REGEX, text))


class TestThePlugPathTest(MayaTestCase):
    """The `in` tests decide as the regex for a name with no newline."""

    def test_the_in_tests_decide_as_the_regex_without_a_newline(self):
        alphabet = ["a", ".", "[", "]", "\n", "\r", " ", "\t", "_"]
        checked  = 0
        for size in range(5):
            for letters in itertools.product(alphabet, repeat=size):
                text  = "".join(letters)
                regex = bool(re.match(_PATH_REGEX, text))
                self.assertEqual(bool(_the_lookups_path_test(text)), regex, repr(text))
                if "\n" not in text:
                    self.assertEqual("[" in text or "." in text, regex, repr(text))
                checked += 1
        self.assertEqual(checked, sum(len(alphabet) ** n for n in range(5)))

    def test_the_regex_is_asked_about_a_name_with_a_newline_only(self):
        node  = Node(cmds.createNode("multiplyDivide", name="md"))
        other = Node("md")
        with mock.patch.object(dg_node.re, "match", wraps=re.match) as regex:
            node.find_attr("input1.input1X")
            node.find_attr("input3D[0]", quiet=True)
            node.find_attr("input1X")
            other.input1X
            getattr(other, "input2.input2Y")
            self.assertEqual(regex.call_count, 0)
            with self.assertRaises(AttributeError):
                node.find_attr("input1.input1X\nx")
        self.assertEqual([c.args for c in regex.call_args_list], [(_PATH_REGEX, "input1.input1X\nx")])


class TestTheSkippedHelpers(MayaTestCase):
    """`_canonicalize_component_alias_plug` and `_is_unresolved_multi_child`
    return their input when their first test fails, and are asked only when it
    passes; `_filtered` is asked only with a filter."""

    TEST_START_NEW_SCENE = True

    def _plugs(self):
        """(node, attr name, MPlug) for every attribute of a few node types,
        their first elements and those elements' children."""
        cmds.createNode("multiplyDivide", name="md")
        cmds.createNode("plusMinusAverage", name="pma")
        cmds.polyCube(name="cube", constructionHistory=False)
        cmds.curve(name="crv", degree=1, point=[(0, 0, 0), (1, 0, 0), (2, 0, 0)])
        cmds.rename(cmds.listRelatives("crv", shapes=True)[0], "crvShape")
        cmds.select(clear=True)
        cmds.joint(name="j1")
        cmds.select(clear=True)
        cmds.polyCube(name="skinned", constructionHistory=False)
        cmds.skinCluster("j1", "skinned", name="skin")
        cmds.polyCube(name="tgt", constructionHistory=False)
        cmds.blendShape("tgt", "skinned", name="bls")
        found = []
        for name in ("md", "pma", "cubeShape", "crvShape", "skin", "bls"):
            node = Node(name)
            fn   = node.fn_set
            for attr in cmds.listAttr(name) or []:
                if "." in attr:
                    continue
                try:
                    plug = fn.findPlug(attr, False)
                except RuntimeError:
                    continue
                found.append((node, attr, plug))
                if plug.isArray:
                    element = plug.elementByLogicalIndex(0)
                    found.append((node, attr, element))
                    if element.isCompound:
                        found.extend(
                            (node, attr, element.child(i)) for i in range(element.numChildren())
                        )
        return found

    def test_each_helper_returns_its_input_when_its_first_test_fails(self):
        plugs = self._plugs()
        self.assertGreater(len(plugs), 500)
        aliases = dg_node._COMPONENT_ALIASES
        for node, attr, plug in plugs:
            if not (plug.isElement or plug.isChild):
                self.assertIs(dg_node._is_unresolved_multi_child(plug), False, plug.name())
            if attr not in aliases:
                self.assertIs(node._canonicalize_component_alias_plug(attr, plug), plug)
            self.assertIs(node._canonicalize_component_alias_plug(5, plug), plug)

    def test_the_helpers_are_asked_when_their_first_test_passes(self):
        md    = Node(cmds.createNode("multiplyDivide", name="md"))
        pma   = Node(cmds.createNode("plusMinusAverage", name="pma"))
        cube  = Node(cmds.polyCube(name="cube", constructionHistory=False)[0])
        shape = cube.get_children(type="mesh")[0]
        canonicalize = DGNode.__dict__["_canonicalize_component_alias_plug"]
        with mock.patch.object(
            DGNode, "_canonicalize_component_alias_plug", autospec=True, side_effect=canonicalize
        ) as aliased, mock.patch.object(
            # the walk `_is_unresolved_multi_child` answers through, which `find_attr`
            # asks itself (it also says whether to hold the attr: follow-up FU 3)
            dg_node, "_multi_child_walk", wraps=dg_node._multi_child_walk
        ) as unresolved:
            for node, name, alias, element in (
                (md, "operation", False, False),
                (md, "input1", False, False),
                (md, "input1X", False, True),  # a child of input1
                (md, "input1.input1X", False, True),
                (pma, "input3D[0]", False, True),
                (pma, "input3D[0].input3Dx", False, True),
                (pma, "input3D", False, False),
                (shape, "pnts", True, False),
                (shape, "pt", True, False),
                (shape, "vtx", True, False),
                (shape, "outMesh", False, False),
            ):
                with self.subTest(name=name):
                    aliased.reset_mock()
                    unresolved.reset_mock()
                    Node(str(node)).find_attr(name)
                    self.assertEqual(aliased.call_count, int(alias))
                    self.assertEqual(unresolved.call_count, int(element))

    def test_filtered_is_asked_only_with_a_filter(self):
        md = Node(cmds.createNode("multiplyDivide", name="md"))
        with mock.patch.object(dg_node, "_filtered", wraps=dg_node._filtered) as filtered:
            md.find_attr("input1X")
            Node("md").find_attr("input2X")
            Node("md").input2Y
            self.assertEqual(filtered.call_count, 0)
            self.assertIsNotNone(Node("md").find_attr("input2Z", data_type="float"))
            self.assertIsNone(Node("md").find_attr("input2Z", data_type="double"))
            self.assertEqual(filtered.call_count, 2)


# --------------------------------------------------------------------------- #
#  H6: the str of a fresh lookup's plug
# --------------------------------------------------------------------------- #


class _Builds:
    """While active, the `_new_attr` and `_attr_state` calls `dg_node` makes (the
    module globals `_find_attr_fresh` and `__getattr__` call)."""

    def __enter__(self):
        self._patches = (
            mock.patch.object(dg_node, "_new_attr", wraps=dg_node._new_attr),
            mock.patch.object(dg_node, "_attr_state", wraps=dg_node._attr_state),
        )
        self.new_attr, self.attr_state = (patch.__enter__() for patch in self._patches)
        return self

    def __exit__(self, *exc):
        for patch in reversed(self._patches):
            patch.__exit__(None, None, None)
        return False

    @property
    def counts(self):
        return (self.new_attr.call_count, self.attr_state.call_count)


class TestAFreshLookup(MayaTestCase):
    """The lookup says whether it built the attr; `__getattr__` then gives the
    plug the attr's str."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.createNode("multiplyDivide", name="md")
        cmds.addAttr("md", longName="dyn", attributeType="double")

    def test_the_lookup_says_true_for_an_attr_it_built(self):
        node  = Node("md")
        fresh = DGNode._find_attr_fresh
        attr, built = fresh(node, "input1X")
        self.assertTrue(built)
        self.assertIs(node.find_attr("input1X"), attr)
        self.assertEqual(fresh(node, "input1X"), (attr, False))  # cached
        self.assertEqual(fresh(node, "i1x"), (attr, False))
        self.assertEqual(fresh(node, attr), (attr, False))  # passed through
        typed, built = fresh(node, node.input1X)  # a Plug: its Attribute
        self.assertIs(type(typed), Attribute)
        self.assertFalse(built)
        dyn, built = fresh(node, "dyn")
        self.assertTrue(built)
        self.assertIsNot(fresh(node, "dyn")[0], dyn)  # a dynamic attr is not cached
        self.assertEqual(fresh(node, "nothere", quiet=True), (None, False))
        self.assertEqual(fresh(Node("md"), "input2X", data_type="float")[1], True)
        self.assertEqual(fresh(Node("md"), "input2X", data_type="double"), (None, False))
        self.assertEqual(fresh(node, "input1X", data_type="float"), (attr, False))
        self.assertEqual(fresh(node, "input1X", data_type="double"), (None, False))

    def test_a_miss_converts_its_mplug_once(self):
        node = Node("md")
        with _Builds() as builds:
            plug = node.input1X
        # the attr of the lookup, then the plug from its str (a74d089: two
        # `_new_attr`, each converting the MPlug)
        self.assertEqual(builds.counts, (1, 1))
        self.assertIs(type(plug), Plug)
        self.assertEqual(str.__str__(plug), str(vars(plug)["_mplug"]))
        self.assertEqual(str.__str__(plug), "md.input1X")
        self.assertIs(vars(plug)["_node"], node)
        self.assertIs(vars(plug)["_handle"], vars(node)["_objhandle"])
        self.assertEqual(sorted(vars(plug)), sorted(vars(node.input1Y)))
        with _Builds() as builds:
            again = node.input1X  # a hit: the plug converts its MPlug
        self.assertEqual(builds.counts, (1, 0))
        self.assertEqual(vars(again).keys(), vars(plug).keys())
        with _Builds() as builds:
            node.dyn  # a dynamic attr is looked up every time
        self.assertEqual(builds.counts, (1, 1))

    def test_the_str_is_the_mplugs_name_after_a_rename(self):
        node  = Node("md")
        first = node.dyn
        plug  = node.input1X
        cmds.rename("md", "renamed")
        second = node.dyn  # a new lookup
        hit    = node.input1X  # a hit of the cache
        self.assertEqual(str.__str__(first), "md.dyn")
        self.assertEqual(str.__str__(second), "renamed.dyn")
        self.assertEqual(str.__str__(plug), "md.input1X")
        self.assertEqual(str.__str__(hit), "renamed.input1X")
        self.assertEqual(str.__str__(Node("renamed").input1X), "renamed.input1X")
        for each in (second, hit):
            self.assertEqual(str.__str__(each), str(vars(each)["_mplug"]))

    def test_each_path_of_an_instanced_node(self):
        cmds.polyCube(name="cube", constructionHistory=False)
        cmds.createNode("transform", name="T2")
        cmds.parent("|cube|cubeShape", "T2", add=True, shape=True)
        for path, expected in (
            ("|cube|cubeShape", "cube|cubeShape.visibility"),
            ("|T2|cubeShape", "T2|cubeShape.visibility"),
        ):
            with self.subTest(path=path):
                node = Node(path)
                with _Builds() as builds:
                    plug = node.visibility
                self.assertEqual(builds.counts, (1, 1))
                self.assertEqual(str.__str__(plug), expected)
                self.assertEqual(plug.full_name, expected)
                self.assertEqual(cmds.getAttr(plug), True)

    def test_a_shapes_attr_through_its_transform(self):
        cmds.polyCube(name="cube", constructionHistory=False)
        node = Node("cube")
        with _Builds() as builds:
            plug = node.outMesh
        self.assertEqual(builds.counts, (0, 1))  # the attr is built by `Attribute`
        self.assertEqual(str.__str__(plug), str(vars(plug)["_mplug"]))
        self.assertEqual(str.__str__(plug), "cubeShape.outMesh")
        self.assertIsNone(vars(plug)["_node"])
        self.assertEqual(type(plug.node).__name__, "Mesh")


class TestALookupOfItsOwn(MayaTestCase):
    """A `find_attr` other than DGNode's own is asked, as before, and its plugs
    are converted from the MPlug."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.createNode("multiplyDivide", name="md")

    def test_a_class_with_its_own_find_attr(self):
        asked = []

        class _OwnLookup(DGNode):
            CUSTOM_NODE_TYPE = "r5lookupOwnFind"

            def find_attr(self, attr, *args, **kwargs):
                asked.append(attr)
                return super().find_attr(attr, *args, **kwargs)

        set_custom_type("md", "r5lookupOwnFind")
        node = Node("md")
        self.assertIsInstance(node, _OwnLookup)
        with _Builds() as builds:
            plug = node.input1X
        self.assertEqual(asked, ["input1X"])
        self.assertEqual(builds.counts, (2, 0))  # the attr, then the plug from the MPlug
        self.assertEqual(str.__str__(plug), "md.input1X")

    def test_a_patched_find_attr(self):
        original = DGNode.__dict__["find_attr"]
        asked    = []

        def find_attr(node, attr, *args, **kwargs):
            asked.append(attr)
            return original(node, attr, *args, **kwargs)

        node = Node("md")
        with mock.patch.object(DGNode, "find_attr", find_attr):
            with _Builds() as builds:
                plug = node.input1X
            self.assertEqual(asked, ["input1X"])
            self.assertEqual(builds.counts, (2, 0))
            self.assertEqual(str.__str__(plug), "md.input1X")
        node = Node("md")
        with mock.patch.object(node, "find_attr", wraps=node.find_attr) as patched:
            with _Builds() as builds:
                node.input1Y
            patched.assert_called_once_with("input1Y")
            self.assertEqual(builds.counts, (2, 0))
        self.assertNotIn("find_attr", vars(node))
        with _Builds() as builds:
            node.input1Z
        self.assertEqual(builds.counts, (1, 1))

    def test_a_class_whose_attribute_lookup_hands_another_find_attr(self):
        asked = []

        class _OwnLookupRead(DGNode):
            CUSTOM_NODE_TYPE = "r5lookupOwnRead"

            def __getattribute__(self, name):
                if name == "find_attr":
                    bound = object.__getattribute__(self, name)

                    def find_attr(attr, *args, **kwargs):
                        asked.append(attr)
                        return bound(attr, *args, **kwargs)

                    return find_attr
                return object.__getattribute__(self, name)

        set_custom_type("md", "r5lookupOwnRead")
        node = Node("md")
        self.assertIsInstance(node, _OwnLookupRead)
        with _Builds() as builds:
            plug = node.input1X
        self.assertEqual(asked, ["input1X"])
        self.assertEqual(builds.counts, (2, 0))
        self.assertEqual(str.__str__(plug), "md.input1X")


# --------------------------------------------------------------------------- #
#  H8: memo keys
# --------------------------------------------------------------------------- #

_FULL_NAME_CODE = Attribute.__dict__["full_name"].fget.__code__


class _Names:
    """While active, the calls of `Attribute.full_name` (sys.setprofile on its
    code object: nothing is patched, so the owner test still passes)."""

    def _profile(self, frame, event, arg):
        if event == "call" and frame.f_code is _FULL_NAME_CODE:
            self.count += 1

    def __enter__(self):
        self.count  = 0
        self._saved = sys.getprofile()
        sys.setprofile(self._profile)
        return self

    def __exit__(self, *exc):
        sys.setprofile(self._saved)
        return False


def _named_key(plug):
    """The key `_attribute_key` gave at a74d089: through the plug's name."""
    node_str = plug.full_name.split(".", 1)[0]
    identity = memoize._named_dg_identity(plug, node_str)
    if identity is None:
        identity = memoize._node_identity(node_str)
    return (identity, _plug_identity_name(plug))


def _outcome(call):
    try:
        return ("returns", call())
    except Exception as error:  # noqa: BLE001 - the error is the thing compared
        return ("raises", type(error).__name__, str(error), _chain(error.__context__))


def _infinite_undo(test):
    """Turn the undo queue on and infinite for `test`, restored after it."""
    saved = (
        cmds.undoInfo(query=True, state=True),
        cmds.undoInfo(query=True, infinity=True),
        cmds.undoInfo(query=True, length=True),
    )
    cmds.undoInfo(state=True, infinity=True)

    def restore():
        if not saved[1]:
            cmds.undoInfo(infinity=False, length=saved[2])
        cmds.undoInfo(state=saved[0])

    test.addCleanup(restore)


class TestMemoKeys(MayaTestCase):
    """`_attribute_key` of a DG-named plug reads its owner's identity; the key
    is the one its name gives, and it raises as its name does."""

    TEST_START_NEW_SCENE = True

    def _scene(self):
        cmds.createNode("multiplyDivide", name="md")
        cmds.addAttr("md", longName="dyn", attributeType="double")
        cmds.addAttr("md", longName="dynv", attributeType="double3")
        for axis in "XYZ":
            cmds.addAttr("md", longName="dynv" + axis, attributeType="double", parent="dynv")
        cmds.aliasAttr("nick", "md.input1X")
        cmds.namespace(add="ns")
        cmds.createNode("multiplyDivide", name="ns:md")
        cmds.createNode("plusMinusAverage", name="pma")
        cmds.createNode("choice", name="ch")
        cmds.createNode("transform", name="xf")
        for group in ("grp1", "grp2"):
            cmds.createNode("transform", name=group)
            cmds.createNode("transform", name="dup", parent=group)
        cmds.polyCube(name="cube", constructionHistory=False)
        cmds.createNode("transform", name="T2")
        cmds.parent("|cube|cubeShape", "T2", add=True, shape=True)
        cmds.sphere(name="sph", constructionHistory=False)

    def _dg_plugs(self):
        md, pma, ch = Node("md"), Node("pma"), Node("ch")
        return [
            ("scalar", md.input2X),
            ("compound", md.input1),
            ("child", md.input1.input1Y),
            ("element", pma.input3D[1]),
            ("element child", pma.input3D[1].input3Dx),
            ("multi root", pma.input3D),
            ("alias", md.nick),
            ("dynamic", md.dyn),
            ("dynamic child", md.dynv.dynvX),
            ("message", md.message),
            ("namespaced", Node("ns:md").input1X),
            ("typed attribute", md.find_attr("input2X")),
            ("generic output", ch.output),
            ("generic slot", ch.input[0]),
            ("a plug that cast its node", self._cast(Plug("md.outputX"))),
        ]

    @staticmethod
    def _cast(plug):
        plug.node
        return plug

    def _other_plugs(self):
        """(label, plug) of the plugs keyed through their name (the component
        plugs through their own `full_name`)."""
        sph = Node("sph")
        return [
            ("DAG", Node("xf").tx),
            ("DAG, a short name two nodes share", Node("grp1|dup").tx),
            ("DAG, the other", Node("grp2|dup").tx),
            ("instanced, first path", Node("|cube|cubeShape").visibility),
            ("instanced, second path", Node("|T2|cubeShape").visibility),
            ("instanced element", Node("|T2|cubeShape").worldMatrix[1]),
            ("instanced array, second path", Node("|T2|cubeShape").worldMatrix),
            ("no owner (a Plug)", Plug("md.input1Y")),
            ("no owner (an Attribute)", Attribute("md.input1Z")),
            ("component", sph.cv),
            ("component element", sph.cv[1, 2]),
        ]

    def test_the_keys_are_the_named_keys(self):
        self._scene()
        for label, plug in self._dg_plugs() + self._other_plugs():
            with self.subTest(case=label):
                self.assertEqual(memoize._attribute_key(plug), _named_key(plug))
                self.assertEqual(memoize._stable_key(plug), _named_key(plug))
        self.assertNotEqual(
            memoize._attribute_key(Node("grp1|dup").tx), memoize._attribute_key(Node("grp2|dup").tx)
        )
        self.assertEqual(
            memoize._attribute_key(Node("|cube|cubeShape").v),
            memoize._attribute_key(Node("|T2|cubeShape").v),
        )
        self.assertEqual(
            memoize._stable_key([Node("md").input1X, Node("pma").output1D]),
            (_named_key(Node("md").input1X), _named_key(Node("pma").output1D)),
        )

    def test_a_dg_named_plug_is_keyed_without_its_name(self):
        self._scene()
        for label, plug in self._dg_plugs():
            with self.subTest(case=label):
                self.assertIsNotNone(_base._dg_named_owner(plug))
                with mock.patch.object(
                    memoize, "_node_identity", wraps=memoize._node_identity
                ) as by_name:
                    with _Names() as names:
                        memoize._attribute_key(plug)
                self.assertEqual(names.count, 0)
                by_name.assert_not_called()
        for label, plug in self._other_plugs():
            with self.subTest(case=label):
                self.assertIsNone(_base._dg_named_owner(plug))
                with _Names() as names:
                    memoize._attribute_key(plug)
                if not label.startswith("component"):  # a component names itself
                    self.assertEqual(names.count, 1)

    def test_a_deleted_plug_raises_as_its_name_does(self):
        self._scene()
        _infinite_undo(self)
        plug   = Node("md").input1X
        before = memoize._attribute_key(plug)
        cmds.delete("md")
        api2 = mock.MagicMock(wraps=memoize.OpenMaya)
        with mock.patch.object(memoize, "OpenMaya", api2):
            raised = _outcome(lambda: memoize._attribute_key(plug))
        self.assertEqual(api2.mock_calls, [])
        self.assertEqual(raised, ("raises", "RuntimeError", "md already deleted!", []))
        self.assertEqual(raised, _outcome(lambda: plug.full_name))
        cmds.undo()
        self.assertEqual(memoize._attribute_key(plug), before)
        self.assertEqual(before, _named_key(plug))

    def test_a_freed_attribute_or_node_raises_as_its_name_does(self):
        self._scene()
        _infinite_undo(self)
        dyn = Node("md").dyn
        str(dyn)
        cmds.deleteAttr("md.dyn")
        # deleted to the undo queue: its attribute lives, and its name answers
        self.assertEqual(memoize._attribute_key(dyn), _named_key(dyn))
        cmds.flushUndo()  # the attribute is freed, its node lives on
        raised = _outcome(lambda: memoize._attribute_key(dyn))
        self.assertEqual(raised, ("raises", "RuntimeError", "md.dyn already deleted!", []))
        self.assertEqual(raised, _outcome(lambda: dyn.full_name))
        held = Node("pma").output1D
        str(held)
        cmds.file(new=True, force=True)
        raised = _outcome(lambda: memoize._attribute_key(held))
        self.assertEqual(raised[:2], ("raises", "RuntimeError"))
        self.assertTrue(raised[2].endswith(
            "(freed by a new scene, a file open or a reference unload) already deleted!"
        ))
        self.assertEqual(raised, _outcome(lambda: held.full_name))

    def test_a_patched_class_is_keyed_through_its_name(self):
        self._scene()
        plug     = Node("md").input1X
        key      = memoize._attribute_key(plug)
        original = DGNode.__dict__["ensure_valid"]
        checked  = []

        def ensure_valid(node):
            checked.append(node)
            return original(node)

        for owner, name, value in (
            (DGNode, "ensure_valid", ensure_valid),
            (DGNode, "name", property(lambda node: node.fn_set.name())),
            (DGNode, "fn_set", property(lambda node: node._fn_set)),
            (Attribute, "full_name", property(Attribute.__dict__["full_name"].fget)),
        ):
            with self.subTest(patched=f"{owner.__name__}.{name}"):
                with mock.patch.object(owner, name, value):
                    self.assertIsNone(_base._dg_named_owner(plug))
                    self.assertEqual(memoize._attribute_key(plug), key)
        self.assertTrue(checked)  # through the name
        node = Node("md")
        node.__dict__["ensure_valid"] = lambda: checked.append("instance")
        own = node.input1X
        self.assertIsNone(_base._dg_named_owner(own))
        del checked[:]
        self.assertEqual(memoize._attribute_key(own), key)
        self.assertEqual(checked, ["instance"])

    def test_a_user_class_with_its_own_name(self):
        self._scene()

        class _OwnName(DGNode):
            CUSTOM_NODE_TYPE = "r5lookupOwnName"

            @property
            def name(self):
                return self.fn_set.name()

        set_custom_type("md", "r5lookupOwnName")
        plug = Node("md").input1X
        self.assertIsNone(_base._dg_named_owner(plug))
        with _Names() as names:
            key = memoize._attribute_key(plug)
        self.assertEqual(names.count, 1)
        self.assertEqual(key, _named_key(plug))

    def test_a_dag_node_held_by_a_dg_class_is_keyed_through_its_name(self):
        # its class names it as a DGNode does (by its fn set: the short name),
        # but a DAG node resolves by name: a short name two nodes share cannot
        # key it (the memo then calls through), a unique one finds it
        self._scene()

        class _DgOnDag(DGNode):
            CUSTOM_NODE_TYPE = "r5lookupDgOnDag"

        cmds.createNode("transform", name="solo", parent="grp1")
        plugs = {}
        for path in ("grp1|solo", "xf"):
            set_custom_type(path, "r5lookupDgOnDag")
            plugs[path] = Node(path).tx  # a unique short name: the class can wrap it
        cmds.createNode("transform", name="solo", parent="grp2")  # not unique now
        for path, plug in plugs.items():
            with self.subTest(path=path):
                self.assertIsInstance(plug.node, _DgOnDag)
                self.assertIs(_base._dg_named_owner(plug), plug.node)
                self.assertEqual(
                    _outcome(lambda: memoize._attribute_key(plug)), _outcome(lambda: _named_key(plug))
                )
        self.assertEqual(
            _outcome(lambda: memoize._attribute_key(plugs["grp1|solo"]))[:2], ("raises", "TypeError")
        )
        self.assertEqual(_outcome(lambda: memoize._attribute_key(plugs["xf"]))[0], "returns")

    def test_a_memo_entry_is_found_with_either_key(self):
        self._scene()
        runs = []

        @memoize.memoize
        def probe(plug):
            runs.append(str(plug))
            return len(runs)

        md = Node("md")
        self.assertEqual(probe(md.input2X), 1)
        self.assertEqual(probe(Node("md").input2X), 1)  # another node object
        self.assertEqual(probe(Plug("md.input2X")), 1)  # no owner: named
        self.assertEqual(probe(md.find_attr("input2X")), 1)
        cmds.rename("md", "renamed")
        self.assertEqual(probe(md.input2X), 1)  # the key survives the rename
        self.assertEqual(probe(md.input2Y), 2)
        self.assertEqual(runs, ["md.input2X", "renamed.input2Y"])

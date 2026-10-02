"""Round 5, B1: the GC tag written through the API 2.0.

``container._gc_tag(node_name, node)`` gives a node rig just created (``container.createNode``
of a GC-eligible type, a typed create of one) the same hidden, locked ``__rig__`` bool as
``cmds.addAttr`` + ``cmds.setAttr(lock=True)`` (the route without a node object), through
``_api_gc_tag``: ``MFnNumericAttribute.create`` (bool, default False), ``hidden``,
``MFnDependencyNode.addAttribute``, ``MPlug.setBool(True)``, ``isLocked = True``. The attribute
is the node's own data, so the undo and redo of the node's creation take it along: the write
has no undo step, where the cmds route has two (addAttr, setAttr). That is the one approved
difference; everything else is compared against the cmds route (``_api_gc_tag`` patched to
decline): the scene (every node's attributes, values, locks, aliases), the saved file, what
is raised and printed, the attribute messages the node sends, ``cleanup()``'s reports.

The guards (each sends the tag to the cmds route, with nothing changed): no node object, or
one that is not valid (deleted, undone, freed, half built), a node whose name is not the one
the caller holds (renamed), a node with a dynamic attribute already (the tag must be its
first: a command's attribute made before it would come back after it once the creation is
undone and redone; an alias, a clash), and a failed ``addAttribute`` (a locked node, an
extension attribute of that name: cmds then warns that the name clashes). A failure once the
attribute is added (its value, its lock) is skipped, as cmds skips a failed setAttr.
"""

import contextlib
import io
import logging
import os
import shutil
import tempfile
import warnings
from unittest import mock

from maya import cmds
from maya.api import OpenMaya

from rig import cleanup, container, destroy, Node, prune_memoize_caches
from rig._internal import container as container_module
from rig._internal.container import _GC_ELIGIBLE_TYPES, _RIG_TAG
from rig.nodetypes import _base, DGNode
from rig.nodetypes._base import _handle_valid
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk

TAG = _RIG_TAG

# the attribute flags compared (attributeQuery)
_FLAGS = (
    "hidden", "keyable", "channelBox", "readable", "writable", "storable", "connectable",
    "cachedInternally", "usedAsColor", "usedAsFilename", "internal", "affectsAppearance",
    "affectsWorldspace", "indeterminant", "multi", "indexMatters", "attributeType",
    "minExists", "maxExists", "softMinExists", "softMaxExists", "listDefault", "niceName",
    "longName", "shortName", "categories", "usesMultiBuilder",
)


def _mobject(name):
    sel = OpenMaya.MSelectionList()
    sel.add(name)
    return sel.getDependNode(0)


@contextlib.contextmanager
def _cmds_route():
    """Every tag through ``cmds.addAttr`` + ``cmds.setAttr``: the route before B1."""
    with mock.patch.object(container_module, "_api_gc_tag", return_value=False):
        yield


class _Output(logging.Handler):
    """What a call prints: Maya's command output, the ``rig`` loggers, Python warnings,
    ``sys.stdout`` / ``sys.stderr``."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.maya, self.records = [], []

    def emit(self, record):
        self.records.append((record.name, record.levelname, record.getMessage()))

    def __enter__(self):
        self._callback = OpenMaya.MCommandMessage.addCommandOutputCallback(
            lambda message, kind, *args: self.maya.append((int(kind), message))
        )
        self._logger = logging.getLogger("rig")
        self._level = self._logger.level
        self._logger.addHandler(self)
        self._logger.setLevel(logging.DEBUG)
        self._warnings = warnings.catch_warnings(record=True)
        self.warned = self._warnings.__enter__()
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
        return {
            "maya": self.maya,
            "logs": self.records,
            "warnings": [(w.category.__name__, str(w.message)) for w in self.warned],
            "stdout": self._out.getvalue(),
            "stderr": self._err.getvalue(),
        }


class _Messages:
    """The attribute-changed and attribute-added messages of the node ``mobject``."""

    def __init__(self, mobject):
        self.mobject, self.seen, self.ids = mobject, [], []

    def _changed(self, msg, plug, other, data):
        self.seen.append(("changed", int(msg), plug.partialName(includeNodeName=True, useLongNames=True)))

    def _added(self, msg, plug, data):
        self.seen.append(("added", int(msg), plug.partialName(includeNodeName=True, useLongNames=True)))

    def __enter__(self):
        if self.mobject is not None:
            self.ids = [
                OpenMaya.MNodeMessage.addAttributeChangedCallback(self.mobject, self._changed),
                OpenMaya.MNodeMessage.addAttributeAddedOrRemovedCallback(self.mobject, self._added),
            ]
        return self

    def __exit__(self, *exc):
        for i in self.ids:
            OpenMaya.MMessage.removeCallback(i)
        return False


def _tag_state(name):
    """``name``'s ``__rig__`` (exists, then value, lock and every flag), or None."""
    if not cmds.attributeQuery(TAG, node=name, exists=True):
        return None
    plug = f"{name}.{TAG}"
    return (
        cmds.getAttr(plug),
        cmds.getAttr(plug, lock=True),
        tuple(cmds.attributeQuery(TAG, node=name, **{flag: True}) for flag in _FLAGS),
    )


def _scene_state():
    """Every node's user attributes, aliases, locked plugs, its tag and its values; the
    containers' members, the selection, the references' edits."""
    nodes = {}
    for name in sorted(cmds.ls(long=True)):
        user = sorted(cmds.listAttr(name, userDefined=True) or [])
        nodes[name] = (
            cmds.nodeType(name),
            user,
            cmds.aliasAttr(name, query=True),
            sorted(cmds.listAttr(name, locked=True) or []),
            cmds.lockNode(name, query=True, lock=True),
            _tag_state(name),
            tuple(
                (a, repr(cmds.getAttr(f"{name}.{a}")))
                for a in user
                if cmds.attributeQuery(a, node=name, exists=True)
                and not cmds.attributeQuery(a, node=name, multi=True)
                and cmds.getAttr(f"{name}.{a}", type=True) not in ("TdataCompound", "message", "attributeAlias")
            ),
        )
    for name in cmds.ls(type="multiplyDivide"):
        nodes[name] += (cmds.getAttr(f"{name}.input1X"), cmds.getAttr(f"{name}.input1X", lock=True))
    return {
        "nodes": nodes,
        "containers": {c: sorted(cmds.container(c, query=True, nodeList=True) or []) for c in cmds.ls(type="container")},
        "current": cmds.container(query=True, current=True),
        "selection": cmds.ls(selection=True),
        "edits": {r: cmds.referenceQuery(r, editStrings=True) for r in cmds.ls(type="reference") if r != "sharedReferenceNode"},
    }


def _undo_all():
    """The steps left on the undo queue (undone to count them)."""
    steps = 0
    quiet = cmds.scriptEditorInfo(query=True, suppressInfo=True)
    cmds.scriptEditorInfo(suppressInfo=True)
    try:
        while not cmds.undoInfo(query=True, undoQueueEmpty=True) and steps < 500:
            cmds.undo()
            steps += 1
    finally:
        cmds.scriptEditorInfo(suppressInfo=quiet)
    return steps


def _fresh():
    cmds.file(new=True, force=True)
    cmds.undoInfo(state=True, infinity=True)
    cmds.flushUndo()


def _outcome(setup, api, undo=True):
    """What ``_gc_tag`` does after ``setup()`` (which returns ``(name, node)``): with
    ``api`` the call the creators make (``_gc_tag(name, node)``), else the cmds route.
    The result or the error (type, message, context, cause), the output, the node's
    attribute messages, the scene, the cmds calls, and the undo steps the tag added."""
    _fresh()
    setup()
    setup_steps = _undo_all()
    _fresh()
    name, node = setup()
    # the node's own handle says whether its MObject may be read (a freed node's may not)
    held = node.__dict__["_mobject"] if isinstance(node, DGNode) and _handle_valid(vars(node)) else None
    if not undo:
        cmds.undoInfo(state=False)
    calls = {"addAttr": 0, "setAttr": 0}

    def counted(fn, key):
        def wrapper(*args, **kwargs):
            calls[key] += 1
            return fn(*args, **kwargs)
        return wrapper

    with mock.patch.object(cmds, "addAttr", counted(cmds.addAttr, "addAttr")), mock.patch.object(
        cmds, "setAttr", counted(cmds.setAttr, "setAttr")
    ):
        with _Messages(held) as messages, _Output() as output:
            try:
                if api:
                    container_module._gc_tag(name, node)
                else:
                    with _cmds_route():
                        container_module._gc_tag(name, node)
                result = None
            except Exception as error:  # noqa: BLE001 - the error is the thing compared
                result = (type(error).__name__, str(error), repr(error.__context__), repr(error.__cause__),
                          error.__suppress_context__)
    if not undo:
        cmds.undoInfo(state=True)
    state = _scene_state()
    return {
        "result": result,
        "output": output.state(),
        "messages": messages.seen,
        "scene": state,
        "cmds_calls": calls,
        "undo_steps": _undo_all() - setup_steps if undo else 0,
    }


def _md(name="md"):
    created = cmds.createNode("multiplyDivide", name=name, skipSelect=True)
    return created, Node(created)


# -- the setups: each returns (the name the caller holds, the node object it holds)


def _plain():
    return _md()


def _quat():
    cmds.loadPlugin("quatNodes", quiet=True)
    created = cmds.createNode("quatAdd", name="md", skipSelect=True)
    return created, Node(created)


def _a_transform():
    created = cmds.createNode("transform", name="md", skipSelect=True)
    return created, Node(created)


def _current_container():
    cmds.container(name="ctn")
    cmds.container("ctn", edit=True, current=True)
    out = _md()
    cmds.container("ctn", edit=True, current=False)
    return out


def _black_boxed_member():
    out = _md()
    cmds.container(name="ctn", addNode=["md"], force=True)
    cmds.setAttr("ctn.blackBox", True)
    return out


def _relative_names():
    cmds.namespace(add="ns")
    cmds.namespace(set="ns")
    cmds.namespace(relativeNames=True)
    return _md()


def _namespaced():
    cmds.namespace(add="ns")
    return _md("ns:md")


# the folder of the referenced file `_referenced` reads (made by setUpClass)
_REF_DIR = []


def _ref_source():
    return os.path.join(_REF_DIR[0], "gc_tag_ref_src.ma").replace("\\", "/")


def _referenced():
    cmds.file(_ref_source(), reference=True, namespace="ref")
    return "ref:md", Node("ref:md")


def _locked():
    out = _md()
    cmds.lockNode("md", lock=True)
    return out


def _locked_container_member():
    out = _md()
    cmds.container(name="ctn", addNode=["md"], force=True)
    cmds.lockNode("ctn", lock=True)
    return out


def _tagged_already():
    out = _md()
    with _cmds_route():
        container_module._gc_tag("md")
    return out


def _user_bool():
    out = _md()
    cmds.addAttr("md", longName=TAG, attributeType="bool")
    return out


def _short_name():
    out = _md()
    cmds.addAttr("md", longName="other", shortName=TAG, attributeType="bool")
    return out


def _compound_child():
    out = _md()
    # with undo off: Maya cannot undo a compound's child ("Undo of the last child not
    # supported"), and `_outcome` undoes the setup to count its steps
    cmds.undoInfo(stateWithoutFlush=False)
    try:
        cmds.addAttr("md", longName="cmp", attributeType="compound", numberOfChildren=1)
        cmds.addAttr("md", longName=TAG, attributeType="double", parent="cmp")
    finally:
        cmds.undoInfo(stateWithoutFlush=True)
    return out


def _alias():
    out = _md()
    cmds.aliasAttr(TAG, "md.input1X")
    return out


def _a_dynamic_attribute():
    out = _md()
    cmds.addAttr("md", longName="note", attributeType="double")
    return out


def _deleted():
    out = _md()
    cmds.delete("md")
    return out


def _freed():
    name, node = _md()
    cmds.file(new=True, force=True)
    return name, node


def _renamed():
    name, node = _md()
    cmds.rename("md", "md_new")
    return name, node


def _renamed_name_reused():
    name, node = _md()
    cmds.rename("md", "md_old")
    cmds.createNode("multiplyDivide", name="md", skipSelect=True)
    return name, node


def _half_built():
    name, _ = _md()
    return name, object.__new__(type(Node(name)))


def _name_only():
    return _md()[0], None


def _not_a_node_object():
    name, _ = _md()
    return name, name


# (label, setup, whether the API writes it, undo on)
_CASES = (
    ("a multiplyDivide", _plain, True, True),
    ("undo off", _plain, True, False),
    ("a plug-in node type (quatAdd)", _quat, True, True),
    ("a DAG node (a transform)", _a_transform, True, True),
    ("made while a user's container is current", _current_container, True, True),
    ("a black-boxed container's member", _black_boxed_member, True, True),
    ("a relative name (namespace -relativeNames)", _relative_names, True, True),
    ("a namespaced node", _namespaced, True, True),
    ("a referenced node", _referenced, True, True),
    ("a locked node: addAttribute refused", _locked, False, True),
    ("a lockNode'd container's member: addAttribute refused", _locked_container_member, False, True),
    ("tagged already (a dynamic attribute: cmds, which warns)", _tagged_already, False, True),
    ("a user's __rig__ bool (a dynamic attribute: cmds, which warns)", _user_bool, False, True),
    ("another attribute's short name (a dynamic attribute: cmds, which warns)", _short_name, False, True),
    ("a compound's child named __rig__ (a dynamic attribute: cmds, which warns)", _compound_child, False, True),
    ("an alias named __rig__ (a dynamic attribute: cmds sets the aliased plug)", _alias, False, True),
    ("a dynamic attribute a command made first (cmds)", _a_dynamic_attribute, False, True),
    ("deleted to the undo queue (cmds: no object matches)", _deleted, False, True),
    ("freed by a new scene (cmds: no object matches)", _freed, False, True),
    ("renamed since (cmds: no object matches)", _renamed, False, True),
    ("renamed, a new node took the name (cmds tags it)", _renamed_name_reused, False, True),
    ("a half-built node object (cmds tags by name)", _half_built, False, True),
    ("a name only", _name_only, False, True),
    ("a str for the node object", _not_a_node_object, False, True),
)


def _reset():
    """A new scene, the root namespace, absolute names."""
    cmds.namespace(relativeNames=False)
    cmds.namespace(set=":")
    cmds.file(new=True, force=True)
    cmds.flushUndo()


class TestTheTagAsTheCmdsRoute(UndoWalk, MayaTestCase):
    """Every state a node can be in when it is tagged: the API write, or the cmds route a
    guard sends it to, does what the cmds route does, but for its undo steps."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _REF_DIR.append(tempfile.mkdtemp(prefix="rig_gc_tag_"))
        cmds.file(new=True, force=True)
        cmds.createNode("multiplyDivide", name="md", skipSelect=True)
        cmds.file(rename=_ref_source())
        cmds.file(save=True, force=True, type="mayaAscii")
        cmds.file(new=True, force=True)

    @classmethod
    def tearDownClass(cls):
        cmds.file(new=True, force=True)
        shutil.rmtree(_REF_DIR.pop(), ignore_errors=True)
        super().tearDownClass()

    def tearDown(self):
        _reset()
        super().tearDown()

    def test_every_state(self):
        for label, setup, writes, undo in _CASES:
            with self.subTest(case=label):
                api = _outcome(setup, True, undo)
                _reset()
                ref = _outcome(setup, False, undo)
                _reset()
                # which route ran: the API write calls no cmds; a guard's fallback
                # calls them as the cmds route does
                self.assertEqual(ref["cmds_calls"]["addAttr"], 1)
                if writes:
                    self.assertEqual(api["cmds_calls"], {"addAttr": 0, "setAttr": 0})
                    # the approved difference: no undo step of its own (cmds: two)
                    self.assertEqual(api["undo_steps"], 0)
                    self.assertEqual(ref["undo_steps"], 2 if undo else 0)
                else:
                    self.assertEqual(api["cmds_calls"], ref["cmds_calls"])
                    self.assertEqual(api["undo_steps"], ref["undo_steps"])
                for key in ("result", "output", "messages", "scene"):
                    self.assertEqual(api[key], ref[key], key)

    def test_the_tag_and_its_messages(self):
        api = _outcome(_plain, True)
        self.assertIsNone(api["result"])
        self.assertEqual(
            [kind for kind, _, plug in api["messages"]], ["changed", "added", "changed", "changed"]
        )
        value, locked, flags = api["scene"]["nodes"]["md"][5]
        self.assertIs(value, True)
        self.assertIs(locked, True)
        self.assertIs(flags[0], True)  # hidden

    def test_the_warning_of_a_clash_is_cmds(self):
        api = _outcome(_tagged_already, True)
        self.assertEqual(api["cmds_calls"]["addAttr"], 1)
        self.assertIn("clashes with an existing attribute", str(api["output"]["maya"]))

    def test_an_extension_attribute_of_that_name(self):
        attr_fn = OpenMaya.MFnNumericAttribute()
        ext = attr_fn.create(TAG, TAG, OpenMaya.MFnNumericData.kBoolean, False)
        node_class = OpenMaya.MNodeClass("multiplyDivide")
        cmds.file(new=True, force=True)
        node_class.addExtensionAttribute(ext)
        try:
            api = _outcome(_plain, True)
            ref = _outcome(_plain, False)
        finally:
            cmds.file(new=True, force=True)
            cmds.flushUndo()
            node_class.removeExtensionAttribute(ext)
        self.assertEqual(api["cmds_calls"]["addAttr"], 1)
        self.assertIn("clashes with an existing attribute", str(api["output"]["maya"]))
        for key in ("result", "output", "messages", "scene", "undo_steps"):
            self.assertEqual(api[key], ref[key], key)

    def test_a_failure_after_the_add_is_skipped(self):
        # the value or the lock refused once the attribute is added: skipped, as the cmds
        # route skips a setAttr that fails (the attribute stays, False and unlocked)
        class _RefusingPlug:
            def __init__(self, *args):
                pass

            def setBool(self, value):
                raise RuntimeError("refused")

        _fresh()
        name, node = _md()
        with mock.patch.object(container_module.OpenMaya, "MPlug", _RefusingPlug):
            container_module._gc_tag(name, node)
        api = _scene_state()
        _fresh()
        name, node = _md()
        with _cmds_route(), mock.patch.object(cmds, "setAttr", side_effect=RuntimeError("refused")):
            container_module._gc_tag(name, node)
        self.assertEqual(api, _scene_state())
        self.assertEqual(api["nodes"]["md"][5][:2], (False, False))


class TestTheTagInUse(UndoWalk, MayaTestCase):
    """The tag the creators write, as rig and Maya use it."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self._registered = dict(_base._NODE_CLASS_DICT)

    def tearDown(self):
        # drop any class a test registered, then the dispatch it cached
        _base._NODE_CLASS_DICT.clear()
        _base._NODE_CLASS_DICT.update(self._registered)
        _base._CLASS_BY_TYPE.clear()
        _base._CASTABLE_TYPES.clear()
        super().tearDown()

    def _both(self, script):
        """``script()``'s return and the scene, through the API and through cmds."""
        out = []
        for route in (contextlib.nullcontext, _cmds_route):
            cmds.file(new=True, force=True)
            prune_memoize_caches()  # the same caches on both sides
            self.undo_queue_on()
            with route():
                result = script()
            out.append((result, _scene_state()))
        return out

    def test_file_identity_on_40_nodes(self):
        folder = tempfile.mkdtemp(prefix="rig_gc_tag_")
        try:
            texts = []
            for route in (contextlib.nullcontext, _cmds_route):
                cmds.file(new=True, force=True)
                with route():
                    with container("box"):
                        for i in range(20):
                            container.createNode(("multiplyDivide", "plusMinusAverage", "network", "condition")[i % 4])
                    for i in range(20):
                        container.createNode(("decomposeMatrix", "composeMatrix", "reverse", "clamp")[i % 4])
                self.assertEqual(len(cmds.ls("*." + TAG, objectsOnly=True, recursive=True)), 40)
                path = os.path.join(folder, "tags_%d.ma" % len(texts)).replace("\\", "/")
                cmds.file(rename=path)
                cmds.file(save=True, force=True, type="mayaAscii")
                with open(path) as fh:
                    texts.append([
                        line for line in fh.read().splitlines()
                        if not line.startswith("//") and "rename -uid" not in line and 'fileInfo "UUID"' not in line
                    ])
            self.assertEqual(texts[0], texts[1])
            self.assertEqual(sum('-ln "__rig__"' in line for line in texts[0]), 40)
            self.assertEqual(sum('setAttr -l on ".__rig__" yes;' in line for line in texts[0]), 40)
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)

    def test_undo_and_redo_keep_the_tag(self):
        src = Node(cmds.createNode("transform", name="src"))
        cmds.flushUndo()
        node = container.createNode("multiplyDivide", name="u")
        node.input1X << src.tx
        tagged = _scene_state()
        self.assertEqual(_tag_state("u")[:2], (True, True))
        # createNode and the connect: the tag has no step
        self.undo_steps(2)
        self.assertFalse(cmds.objExists("u"))
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
        self.redo_steps(2)
        self.assertEqual(_scene_state(), tagged)
        self.undo_steps(1)
        self.redo_steps(1)
        self.assertEqual(_scene_state(), tagged)
        self.undo_all()
        self.redo_all()
        self.assertEqual(_scene_state(), tagged)
        self.assertEqual(_tag_state("u")[:2], (True, True))

    def test_undo_and_redo_keep_the_attribute_order(self):
        # a dynamic attribute a command made before the tag (a post_create's) sends the tag
        # to cmds: through the API it would come back first once the creation is undone and
        # redone; the tag made first keeps its place on both routes
        class _Noted(DGNode):
            NATIVE_NODE_TYPE = "multiplyDivide"

            @classmethod
            def post_create(cls, new_node_name, *args, **kwargs):
                cmds.addAttr(new_node_name, longName="note", attributeType="double")
                return super().post_create(new_node_name, *args, **kwargs)

        def attrs():
            return {name: cmds.listAttr(name, userDefined=True) for name in ("noted", "plain")}

        def script():
            with container("o"):
                _Noted.create(name="noted")
                container.createNode("multiplyDivide", name="plain")
            cmds.addAttr("plain", longName="later", attributeType="double")
            built = attrs()
            self.undo_all()
            self.redo_all()
            return built, attrs()

        api, ref = self._both(script)
        self.assertEqual(api, ref)
        built, redone = api[0]
        self.assertEqual(built, {"noted": ["note", TAG], "plain": [TAG, "later"]})
        self.assertEqual(redone, built)

    def test_undo_steps_of_a_create(self):
        # outside a scope: createNode (cmds route: + addAttr, setAttr); in a scope: + the
        # member; a typed create: one chunk either way
        class _MD(DGNode):
            NATIVE_NODE_TYPE = "multiplyDivide"

        def steps(make):
            out = []
            for route in (contextlib.nullcontext, _cmds_route):
                cmds.file(new=True, force=True)
                self.undo_queue_on()
                with route():
                    make()
                out.append(_undo_all())
            return out

        self.assertEqual(steps(lambda: container.createNode("multiplyDivide")), [1, 3])

        def in_scope():
            with container("s"):
                container.createNode("multiplyDivide", name="first")
                container.createNode("multiplyDivide", name="second")
        # the container, its first member's hyperLayout, 2 creates, 2 members (+ 2 x 2 on cmds)
        api, ref = steps(in_scope)
        self.assertEqual(ref - api, 4)

        def typed():
            with container("t"):
                _MD.create(name="typed")
        api, ref = steps(typed)
        self.assertEqual(api, ref)

    def test_delete_attr_refused(self):
        def script():
            container.createNode("multiplyDivide", name="d")
            try:
                cmds.deleteAttr("d." + TAG)
            except RuntimeError as error:
                return str(error)
            return None

        (api, api_scene), (ref, ref_scene) = self._both(script)
        self.assertIn("is locked and may not be removed", api)
        self.assertEqual((api, api_scene), (ref, ref_scene))

    def test_duplicate_carries_it(self):
        def script():
            container.createNode("multiplyDivide", name="d")
            return cmds.duplicate("d")

        api, ref = self._both(script)
        self.assertEqual(api, ref)
        copy = api[0][0]
        self.assertEqual(api[1]["nodes"][copy][5][:2], (True, True))

    def test_destroy_refused(self):
        def script():
            node = container.createNode("multiplyDivide", name="d")
            try:
                node << destroy(TAG)
            except Exception as error:  # noqa: BLE001 - the error is the thing compared
                return type(error).__name__, str(error)
            return None

        api, ref = self._both(script)
        self.assertIsNotNone(api[0])
        self.assertEqual(api, ref)

    def test_math_nodes_read_true(self):
        def script():
            a = Node.create("transform", name="a")
            out = a.tx * 2
            name = str(out).split(".")[0]
            return name, cmds.getAttr(name + "." + TAG), cmds.getAttr(name + "." + TAG, lock=True)

        api, ref = self._both(script)
        self.assertEqual(api[0][1:], (True, True))
        self.assertEqual(api, ref)

    def test_an_untaggable_type_is_skipped(self):
        # a type off the whitelist is never tagged; a node that refuses the attribute (a
        # typed create whose post_create locks it) is skipped silently, as by cmds
        class _Locking(DGNode):
            NATIVE_NODE_TYPE = "multiplyDivide"

            @classmethod
            def post_create(cls, new_node_name, *args, **kwargs):
                node = super().post_create(new_node_name, *args, **kwargs)
                cmds.lockNode(new_node_name, lock=True)
                return node

        def script():
            out = []
            with _Output() as output:
                with container("u"):
                    out.append(str(container.createNode("transform", name="xf")))
                    out.append(str(Node.create("locator", name="loc")))
                    try:
                        out.append(str(_Locking.create(name="locked")))
                    except Exception as error:  # noqa: BLE001 - the error is the thing compared
                        out.append((type(error).__name__, str(error)))
            return out, output.state()

        api, ref = self._both(script)
        self.assertEqual(api, ref)
        made = api[0][0]
        self.assertEqual(made[2], "locked")
        for name in made + cmds.ls(made[:2], dag=True, long=True):
            self.assertIsNone(_tag_state(name), name)
        self.assertEqual(cmds.lockNode("locked", query=True, lock=True), [True])

    def test_every_eligible_type(self):
        def script():
            made = {}
            for node_type in sorted(_GC_ELIGIBLE_TYPES):
                try:
                    made[node_type] = str(container.createNode(node_type))
                except Exception as error:  # noqa: BLE001 - a type this Maya does not have
                    made[node_type] = (type(error).__name__, str(error))
            return made

        api, ref = self._both(script)
        self.assertEqual(api, ref)
        tagged = [n for n in api[0].values() if isinstance(n, str)]
        self.assertGreater(len(tagged), 60)
        for name in tagged:
            self.assertEqual(api[1]["nodes"][name][5][:2], (True, True), name)

    def _orphans(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        with container("rig1"):
            b.ty << a.tx * 2 + 1
            a.tx * 3  # an orphan
            (a.tx + 1) * 4  # an orphan chain
        a.ty * 5  # an orphan at the scene root
        b.tz << a.ty - 1

    def test_cleanup_finds_every_tagged_node(self):
        def script():
            self._orphans()
            dry = cleanup(dry_run=True)
            scoped = cleanup(container="rig1", dry_run=True)
            tagged = sorted(cmds.ls("*." + TAG, objectsOnly=True, recursive=True))
            done = cleanup()
            return dry, scoped, tagged, done, sorted(cmds.ls())

        api, ref = self._both(script)
        self.assertTrue(api[0][0]["by_owner"])
        self.assertEqual(api, ref)

    def test_save_and_reopen(self):
        folder = tempfile.mkdtemp(prefix="rig_gc_tag_")
        try:
            def script():
                self._orphans()
                before = cleanup(dry_run=True)
                path = os.path.join(folder, "reopen.ma").replace("\\", "/")
                cmds.file(rename=path)
                cmds.file(save=True, force=True, type="mayaAscii")
                cmds.file(new=True, force=True)
                cmds.file(path, open=True, force=True)
                return before, cleanup(dry_run=True), sorted(cmds.ls("*." + TAG, objectsOnly=True, recursive=True))

            api, ref = self._both(script)
            before, after, tagged = api[0]
            self.assertTrue(before["by_owner"])
            for key in ("by_owner", "deleted_containers"):
                self.assertEqual(before[key], after[key], key)
            self.assertTrue(tagged)
            self.assertEqual(api, ref)
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)

    def test_a_referenced_files_tagged_nodes(self):
        folder = tempfile.mkdtemp(prefix="rig_gc_tag_")
        try:
            def script():
                self._orphans()
                path = os.path.join(folder, "src_%d.ma" % len(os.listdir(folder))).replace("\\", "/")
                cmds.file(rename=path)
                cmds.file(save=True, force=True, type="mayaAscii")
                cmds.file(new=True, force=True)
                cmds.file(path, reference=True, namespace="ref")
                return (
                    sorted(cmds.ls("*." + TAG, objectsOnly=True, recursive=True)),
                    cleanup(dry_run=True),
                )

            api, ref = self._both(script)
            self.assertTrue(any(name.startswith("ref:") for name in api[0][0]))
            self.assertEqual(api[0], ref[0])
            # the scenes differ only by the referenced file's name
            self.assertEqual(sorted(api[1]["nodes"]), sorted(ref[1]["nodes"]))
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)

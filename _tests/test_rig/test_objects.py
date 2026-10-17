"""The objects outside nodes and plugs (user decisions 2026-10-16, "Objects outside nodes
and plugs": the rule is per object).

An object that points into the scene exposes every public member as a method; a value
keeps its data as attributes, frozen where a write would be lost or leak; plumbing is
private.

* `TestComponentsReadThroughMethods`: ``get_shape()``, ``get_kind()``, ``is_all()``,
  ``get_indices()``, ``get_count()``, ``get_names()``; ``geometry`` is gone
  (``c.get_shape().get_node_type()``); ``c >> None`` stays.
* `TestColorSetReadsThroughMethods`: ``get_name()``, ``get_mesh()``, ``is_valid()``,
  ``get_representation()``, ``get_data()`` / ``set_data(colours)``; a read writes nothing
  (the current set stays, the scene is not marked modified).
* `TestContainerStack`: ``rig.container.get_stack()`` is read-only ``(name, Container or
  None)`` tuples; ``stack`` and ``containers`` are gone.
* `TestTheDoorIsStored`: ``NodeNotFoundError.door`` is read once, with the message.
* `TestMembershipTokens`: the layer / material tokens have no ``name`` (``str(token)``,
  the repr); ``Tag('x').name`` stays (a value); removes / purges are private; a token
  whose node is deleted raises AttributeError.
* `TestContainerOptionsAreReadOnly`: a write raises naming ``rig.set_options(...)``;
  reads, ``set_options`` and ``force_nodes`` work.
* `TestPlumbingIsPrivate`: the membership kinds' constants, ``compoundType``,
  ``SetSkinWeightsCommand``, the ``normalise`` and journal imports of the public
  modules; the undo command's name is spelled once.
* `TestValuesFrozenWhereItBites`: keys arrays are read-only; the ``kargs`` of ``lock`` /
  ``unlock`` / ``hide`` / ``unhide`` are a read-only mapping; ``Enum`` keeps the default
  it was given.

Everything new is read by name inside the tests, so that on a tree without the step
each test fails on its own (the negative control).
"""

import copy
import importlib
import inspect
import os
import pickle
import pkgutil
from unittest import mock

import numpy as np
from maya import cmds

import rig
from rig import AnimCurve, Components, ContainerOptions, Enum, List, Node, Tag, container
from rig import hide, lock, unhide, unlock
from rig._internal import container as container_module
from rig._internal import journal
from rig._internal import members
from rig.nodetypes import DisplayLayer, Joint, NodeNotFoundError
from rig.nodetypes.mesh import ColorSet, Mesh
from rig.shade import Blinn, Material
from rig._tests._base import MayaTestCase


def _cube(name="cube"):
    return Node(cmds.polyCube(name=name, constructionHistory=False)[0])


def _properties(cls):
    """The public properties of ``cls`` (its MRO)."""
    return sorted(
        name for name, value in inspect.getmembers_static(cls)
        if not name.startswith("_") and isinstance(value, property)
    )


# --------------------------------------------------------------------- #
#  Scene handles
# --------------------------------------------------------------------- #


class TestComponentsReadThroughMethods(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_every_read_is_a_method(self):
        cube  = _cube()
        faces = cube.f[:3]
        self.assertEqual(faces.get_indices().tolist(), [0, 1, 2])
        self.assertEqual(faces.get_count(), 3)
        self.assertEqual(faces.get_names(), ["|cube|cubeShape.f[0:2]"])
        self.assertEqual(faces.get_kind(), "f")
        self.assertEqual(str(faces.get_shape()), "cubeShape")
        self.assertEqual(faces.get_shape().get_node_type(), "mesh")
        self.assertIs(faces.is_all(), False)
        self.assertIs(cube.f.is_all(), True)
        # >> None stays the operator read of the ids
        np.testing.assert_array_equal(faces >> None, faces.get_indices())

    def test_no_property_is_left(self):
        self.assertEqual(_properties(Components), [])
        faces = _cube().f
        for name in ("shape", "kind", "geometry", "indices", "count", "names"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(faces, name))

    def test_the_whole_kind_reads_the_shape_at_each_call(self):
        cube  = _cube()
        faces = cube.f
        self.assertEqual(faces.get_count(), 6)
        cmds.polySmooth("cube", divisions=1, constructionHistory=False)
        self.assertEqual(faces.get_count(), 24)
        self.assertEqual(len(faces.get_indices()), 24)


class TestColorSetReadsThroughMethods(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        _cube()
        self.mesh = Mesh("cubeShape")
        self.red  = self.mesh.add_color_set("red", ColorSet.Representation.RGBA)
        self.red.set_data(np.tile([1.0, 0.0, 0.0, 1.0], (8, 1)))
        self.blue = self.mesh.add_color_set("blue", ColorSet.Representation.RGB)

    def test_every_member_is_a_method(self):
        self.assertEqual(_properties(ColorSet), [])
        self.assertEqual(self.red.get_name(), "red")
        self.assertIs(self.red.get_mesh(), self.mesh)
        self.assertIs(self.red.is_valid(), True)
        self.assertIs(self.red.get_representation(), ColorSet.Representation.RGBA)
        np.testing.assert_allclose(self.red.get_data()[0], [1.0, 0.0, 0.0, 1.0])
        self.assertEqual(self.red.get_data().shape, (8, 4))
        for name in ("name", "mesh", "representation", "data"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(self.red, name))

    def test_a_read_writes_nothing(self):
        fn = self.mesh.get_fn_set()
        fn.setCurrentColorSetName("blue")
        path = os.path.join(self._folder(), "sets.ma").replace("\\", "/")
        cmds.file(rename=path)
        cmds.file(save=True, type="mayaAscii", force=True)
        self.assertFalse(cmds.file(query=True, modified=True))
        undo_name = cmds.undoInfo(query=True, undoName=True)
        for cset in (self.red, self.blue):
            with self.subTest(cset=cset.get_name()):
                cset.get_data()
                cset.get_representation()
                self.assertEqual(fn.currentColorSetName(), "blue")
                self.assertFalse(cmds.file(query=True, modified=True))
                self.assertEqual(cmds.undoInfo(query=True, undoName=True), undo_name)

    def test_set_data_is_one_undo_step(self):
        before = self.red.get_data()
        cmds.undoInfo(state=True, infinity=True)
        self.red.set_data(np.tile([0.0, 0.0, 1.0, 1.0], (8, 1)))
        np.testing.assert_allclose(self.red.get_data()[0], [0.0, 0.0, 1.0, 1.0])
        cmds.undo()
        np.testing.assert_allclose(self.red.get_data(), before)

    def test_the_repr_and_a_typo(self):
        self.assertEqual(repr(self.red), "ColorSet<cubeShape.red>")
        with self.assertRaises(AttributeError):
            self.red.nmae = "x"

    def _folder(self):
        import shutil
        import tempfile

        folder = tempfile.mkdtemp(prefix="rig_objects_")
        self.addCleanup(shutil.rmtree, folder, True)
        return folder


class TestContainerStack(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_read_only_tuples_outermost_first(self):
        self.assertEqual(container.get_stack(), ())
        with container("outer") as outer:
            with container("inner"):
                stack = container.get_stack()
        self.assertEqual(stack, (("outer", outer), ("inner", None)))
        self.assertIsInstance(stack, tuple)
        self.assertTrue(all(type(frame) is tuple for frame in stack))
        self.assertEqual(container.get_stack(), ())

    def test_stack_and_containers_are_gone(self):
        for name in ("stack", "containers"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(container, name))
        self.assertEqual(_properties(type(container)), [])


class TestTheDoorIsStored(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def miss(self, call):
        with self.assertRaises(NodeNotFoundError) as caught:
            call()
        return caught.exception

    def test_the_door_is_read_once_with_the_message(self):
        with mock.patch.object(Joint, "_define_door", wraps=Joint._define_door) as door:
            error = self.miss(lambda: Joint("spnie_01"))
            door.assert_not_called()
            self.assertEqual(error.door, "Joint.define('spnie_01')")   # builds the message
            door.assert_called_once_with("spnie_01")
            self.assertEqual(str(error), "no joint named 'spnie_01'; Joint.define('spnie_01') finds or makes it")
            error.door
            door.assert_called_once_with("spnie_01")
        self.assertIn("_door", vars(error))

    def test_a_scene_change_leaves_the_door_and_the_message_agreeing(self):
        # no namespace ns9: define makes no namespace, so there is no door
        error = self.miss(lambda: Joint("ns9:jx"))
        text  = str(error)
        self.assertIsNone(error.door)
        cmds.namespace(add="ns9")   # now Joint.define('ns9:jx') would make it
        self.assertIsNone(error.door)
        self.assertEqual(str(error), text)
        # read before the message: the same answer, kept
        error = self.miss(lambda: Joint("ns9:jy"))
        self.assertEqual(error.door, "Joint.define('ns9:jy')")
        cmds.namespace(removeNamespace="ns9")
        self.assertEqual(error.door, "Joint.define('ns9:jy')")
        self.assertTrue(str(error).endswith("; Joint.define('ns9:jy') finds or makes it"))


class TestMembershipTokens(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.cube  = _cube()
        self.layer = DisplayLayer.define("lay")
        self.red   = Blinn.create(name="red")

    def tokens(self):
        return (-self.layer, DisplayLayer(), -self.red, Blinn(), Material(), -self.red.get_engine())

    def test_a_token_has_no_name_attribute(self):
        for token in self.tokens():
            with self.subTest(token=repr(token)):
                with self.assertRaisesRegex(AttributeError, r"a token has no name attribute: str\(token\) is its name"):
                    token.name
        self.assertEqual(str(-self.layer), "lay")
        self.assertEqual(repr(-self.layer), '-DisplayLayer("lay")')
        self.assertEqual(str(-self.red), "red")
        self.assertEqual(repr(-self.red), '-Blinn("red")')

    def test_a_tag_keeps_its_name(self):
        self.assertEqual(Tag(" ab ").name, "ab")
        self.assertEqual((-Tag("ab")).name, "ab")
        self.assertIsNone(Tag().name)
        self.assertEqual(str(Tag("ab")), "ab")

    def test_the_protocol_is_private(self):
        public = ("name", "removes", "purges", "KIND", "ACCEPTS", "WANT_SHAPES", "EXCLUSIVE", "HAS_IDS")
        for token in self.tokens() + (Tag("ab"), -Tag("ab"), Tag()):
            for name in public:
                if name == "name" and isinstance(token, Tag):
                    continue
                with self.subTest(token=repr(token), name=name):
                    self.assertFalse(hasattr(token, name))
        self.assertTrue((-Tag("ab"))._remove and Tag()._purges)

    def test_a_token_of_a_deleted_node_raises_attribute_error(self):
        layer_token, red_token = -self.layer, -self.red
        cmds.delete("lay", "red")
        # re-pinned (user decision 6C 2026-10-17): the message shows the token's
        # repr, its node marked (a repr never raises), as for a live token
        for token, noun, shown in (
            (layer_token, "layer", r'-DisplayLayer\("lay", deleted\)'),
            (red_token, "material", r'-Blinn\("red", deleted\)'),
        ):
            for name in ("get_name", "name", "anything"):
                with self.subTest(noun=noun, name=name):
                    with self.assertRaisesRegex(AttributeError, f"^{shown} is a removal token, not an? {noun}; "):
                        getattr(token, name)
                    self.assertFalse(hasattr(token, name))


# --------------------------------------------------------------------- #
#  Process state
# --------------------------------------------------------------------- #


class TestContainerOptionsAreReadOnly(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.addCleanup(rig.set_options, **rig.get_options())

    def test_a_write_raises_naming_set_options(self):
        for name, value in (("use_shorthand", False), ("maya_version", 2022), ("constant_folding", False)):
            with self.subTest(name=name):
                before = getattr(ContainerOptions, name)
                with self.assertRaisesRegex(
                    AttributeError, rf"^ContainerOptions is read-only: write rig.set_options\({name}=\.\.\.\)"
                ):
                    setattr(ContainerOptions, name, value)
                self.assertEqual(getattr(ContainerOptions, name), before)
                with self.assertRaisesRegex(AttributeError, r"rig.set_options"):
                    setattr(ContainerOptions(), name, value)
                with self.assertRaisesRegex(AttributeError, r"rig.set_options"):
                    delattr(ContainerOptions, name)
        self.assertEqual(rig.get_options()["maya_version"], None)

    def test_a_misspelt_option_is_refused_not_made(self):
        with self.assertRaisesRegex(AttributeError, r"has no option 'use_shortand'"):
            ContainerOptions.use_shortand = False
        self.assertFalse(hasattr(ContainerOptions, "use_shortand"))

    def test_reads_set_options_and_force_nodes_work(self):
        rig.set_options(use_shorthand=False)
        self.assertIs(ContainerOptions.use_shorthand, False)
        self.assertIs(rig.get_options()["use_shorthand"], False)
        rig.set_options(use_shorthand=True)
        self.assertIs(ContainerOptions.use_shorthand, True)
        with rig.force_nodes():
            self.assertIs(ContainerOptions.constant_folding, False)
        self.assertIs(ContainerOptions.constant_folding, True)


# --------------------------------------------------------------------- #
#  Plumbing
# --------------------------------------------------------------------- #

_JOURNAL_NAMES = ("barrier", "cmds_edit", "one_step", "outside", "rig_code", "statement",
                  "user_call", "mel_eval", "run_statement", "cmds_route")


def _public_modules():
    """rig's public modules: no ``_`` part in the dotted name, no tests, no examples."""
    found = [rig]
    for info in pkgutil.walk_packages(rig.__path__, "rig."):
        parts = info.name.split(".")[1:]
        if any(p.startswith("_") for p in parts) or parts[0] in ("examples",):
            continue
        if parts[-1] == "undoable_api_command":   # the plug-in Maya loads: no rig import
            continue
        found.append(importlib.import_module(info.name))
    return found


class TestPlumbingIsPrivate(MayaTestCase):
    def test_the_membership_kinds_constants(self):
        from rig.membership import _LayerMember
        from rig.shade import _MaterialMember

        for cls in (members._MemberSpec, Tag, _LayerMember, _MaterialMember):
            for name in ("KIND", "ACCEPTS", "WANT_SHAPES", "EXCLUSIVE", "HAS_IDS"):
                with self.subTest(cls=cls.__name__, name=name):
                    self.assertFalse(hasattr(cls, name))
        self.assertEqual((Tag._KIND, _LayerMember._KIND, _MaterialMember._KIND), ("tag", "layer", "material"))

    def test_compound_type_and_the_skin_weights_command(self):
        from rig import Euler
        from rig.nodetypes import skincluster

        spec = Euler("r")
        self.assertNotIn("compoundType", vars(spec))
        self.assertEqual(spec._compound_type, "doubleAngle")
        self.assertFalse(hasattr(skincluster, "SetSkinWeightsCommand"))
        self.assertTrue(hasattr(skincluster, "_SetSkinWeightsCommand"))

    def test_no_public_module_leaks_normalise_or_a_journal_name(self):
        leaks = []
        for module in _public_modules():
            if getattr(module, "normalise", None) is members.normalise:
                leaks.append(f"{module.__name__}.normalise")
            for name in _JOURNAL_NAMES:
                if getattr(module, name, None) is getattr(journal, name):
                    leaks.append(f"{module.__name__}.{name}")
        self.assertEqual(leaks, [])
        self.assertIn("rig.nodetypes.dg_node", [m.__name__ for m in _public_modules()])

    def test_the_undo_command_is_named_once(self):
        from rig.nodetypes import plugins
        from rig.nodetypes.plugins import undoable_api_command

        self.assertEqual(plugins.UNDOABLE_API_COMMAND, undoable_api_command.COMMAND_NAME)
        with open(plugins.__file__, encoding="utf-8") as handle:
            self.assertNotIn('"rigUndoableAPICommand"', handle.read())


# --------------------------------------------------------------------- #
#  Values
# --------------------------------------------------------------------- #


class TestValuesFrozenWhereItBites(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_keys_arrays_are_read_only(self):
        keys = AnimCurve([0.0, 1.0, 2.0], times=[0, 5, 10], in_slopes=[0.1, np.nan, 0.2],
                         out_slopes=[0.1, np.nan, 0.2], weighted=True,
                         in_weights=[1.0, 2.0, 1.0], out_weights=[1.0, 2.0, 1.0])
        cube = _cube()
        cube.tx << keys
        read = AnimCurve.of(cube.tx)[0].get_keys()
        for value in (keys, read):
            for name in ("values", "times", "in_slopes", "out_slopes", "in_weights", "out_weights"):
                with self.subTest(keys=value is read, name=name):
                    arr = getattr(value, name)
                    with self.assertRaisesRegex(ValueError, "read-only"):
                        arr[0] = 99.0
                    self.assertNotEqual(getattr(value, name)[0], 99.0)
                    mine = arr.copy()
                    mine[0] = 99.0
                    self.assertEqual(mine[0], 99.0)

    def test_the_modifiers_kargs_are_read_only(self):
        for spec, flags in ((lock, {"lock": True}), (unlock, {"lock": False}),
                            (hide, {"keyable": False, "channelBox": False}),
                            (unhide, {"keyable": True, "channelBox": False})):
            with self.subTest(flags=flags):
                with self.assertRaises(TypeError):
                    spec.kargs["lock"] = False
                self.assertEqual(dict(spec.kargs), flags)
                for other in (copy.copy(spec), copy.deepcopy(spec), pickle.loads(pickle.dumps(spec))):
                    self.assertEqual(dict(other.kargs), flags)
        cube = _cube()
        cube.tx << copy.copy(lock)
        self.assertTrue(cmds.getAttr("cube.tx", lock=True))
        List([cube.tx, cube.ty]) << unlock << lock
        self.assertTrue(cmds.getAttr("cube.ty", lock=True))

    def test_enum_keeps_the_default_it_was_given(self):
        spec = Enum("mode", dv="True")
        self.assertEqual(spec.kargs["defaultValue" if "defaultValue" in spec.kargs else "dv"], "True")
        for name in ("a", "b"):
            node = Node(cmds.createNode("transform", name=name))
            node << spec
            self.assertEqual(cmds.addAttr(f"{name}.mode", query=True, defaultValue=True), 1.0)
            self.assertEqual(spec.kargs["defaultValue" if "defaultValue" in spec.kargs else "dv"], "True")

"""Container-aware typed creators (round 4a M5: D13; R2: Node.create).

Inside ``with container()`` a typed create (``Transform.create()``,
``Mesh.create(...)``, ``Follicle.create_on_mesh(...)``, ...) joins the scope with
``Node.create``'s rules: membership (the uuid on every frame), the flattened
scope's name prefix on an explicit name, ``skipSelect``, the GC tag under the
eligibility rule, and every node the call made is registered. Only the
outermost typed create acts. ``container=False`` opts out; the scene
registries (display layers, sets and shading engines, references) stay out
unless ``container=True`` (then registered, never prefixed). Outside a scope
nothing changes. ``Node.create`` (R2: PyNode is gone) runs the typed create of a
registered type and ``container.createNode`` for any other type.
"""

import inspect
import os
import shutil
import tempfile
from unittest import mock

from maya import cmds
from rig import container, Layer, Node, set_options
from rig._internal import container as container_module
from rig._internal.container import ContainerOptions
from rig.bridges import commands as rc
from rig.nodetypes import (
    _base,
    DGNode,
    DisplayLayer,
    Follicle,
    Joint,
    Mesh,
    ObjectSet,
    ShadingEngine,
    SkinCluster,
    Transform,
)
from rig.nodetypes import dg_node as dg_node_module
from rig.nodetypes.dg_node import _typed_creator
from rig.shade import Blinn
from rig._tests._base import MayaTestCase


RIG_TAG = "__rig__"

_OPTIONS = ("skip_selection", "create_containers", "flatten_containers", "cleanup_on_exit")


def _members(ctn):
    """The nodes of a Maya container, as it lists them."""
    return sorted(cmds.container(str(ctn), query=True, nodeList=True) or [])


def _owner(node):
    """The container holding ``node``, or None."""
    return cmds.container(query=True, findContainer=[str(node)])


def _uuid(node):
    return cmds.ls(str(node), uuid=True)[0]


def _frame_uuids():
    """The uuids each open frame recorded, outermost first."""
    return [list(frame.members) for frame in container.stack]


def _tagged(node):
    return cmds.attributeQuery(RIG_TAG, node=str(node), exists=True)


def _tag_state(node):
    """The ``__rig__`` tag as a tuple (exists, hidden, locked, value)."""
    if not _tagged(node):
        return (False,)
    plug = f"{node}.{RIG_TAG}"
    return (
        True,
        cmds.attributeQuery(RIG_TAG, node=str(node), hidden=True),
        cmds.getAttr(plug, lock=True),
        cmds.getAttr(plug),
    )


class TestTypedCreateInContainer(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self._options    = {k: getattr(ContainerOptions, k) for k in _OPTIONS}
        self._registered = dict(_base._NODE_CLASS_DICT)
        cmds.select(clear=True)

    def tearDown(self):
        for key, value in self._options.items():
            setattr(ContainerOptions, key, value)
        # drop any class a test registered, then the dispatch it cached
        _base._NODE_CLASS_DICT.clear()
        _base._NODE_CLASS_DICT.update(self._registered)
        _base._CLASS_BY_TYPE.clear()
        _base._CASTABLE_TYPES.clear()
        cmds.namespace(set=":")
        self.assertEqual(container_module._TYPED_DEPTH, 0)
        self.assertFalse(container.is_active)
        super().tearDown()

    # -- helpers -- #

    def _cube(self, name="cube"):
        return cmds.polyCube(name=name, ch=False)[0]

    def _mesh_data(self):
        cube = self._cube("src_cube")
        data = Mesh(cmds.listRelatives(cube, shapes=True)[0]).serialize(include_uvs=False)
        cmds.delete(cube)
        return data

    def _md_class(self):
        class _MD(DGNode):
            NATIVE_NODE_TYPE = "multiplyDivide"

        return _MD

    # -- the hooks -- #

    def test_hooks_are_installed(self):
        self.assertIs(dg_node_module._TYPED_CREATE_HOOK, container_module._typed_create)
        self.assertIs(_base._NODE_CREATE_HOOK, container_module._node_create)
        self.assertFalse(hasattr(container_module, "_pynode_create"))
        self.assertFalse(hasattr(_base, "_PYNODE_CREATE_HOOK"))
        self.assertTrue(DGNode._CONTAINER_AWARE)
        self.assertTrue(Transform._CONTAINER_AWARE)
        self.assertTrue(SkinCluster._CONTAINER_AWARE)
        for cls in (DisplayLayer, ObjectSet, ShadingEngine):
            self.assertFalse(cls._CONTAINER_AWARE, cls.__name__)
        from rig.nodetypes import Reference

        self.assertFalse(Reference._CONTAINER_AWARE)

    def test_decorated_creators_keep_their_signatures(self):
        params = list(inspect.signature(Follicle.create_on_mesh).parameters)
        self.assertEqual(params[:3], ["mesh", "ref_object", "name"])
        self.assertIn("Creates a follicle", Follicle.create_on_mesh.__doc__)
        params = list(inspect.signature(Transform.create_hierarchy).parameters)
        self.assertEqual(params, ["hierarchy", "parent", "world_space"])

    # -- 1 membership and prefix -- #

    def test_membership_and_prefix(self):
        with container("box") as box:
            top = Transform.create(name="x")
            self.assertEqual(_frame_uuids(), [[_uuid(top)]])
        self.assertEqual(str(top), "x")
        self.assertEqual(_members(box), ["x"])

        with container("outer") as outer:
            with container("inner"):
                typed  = Transform.create(name="x")
                short  = Transform.create(n="y")
                dsl    = Node.create("transform", name="z")
                joint  = Joint.create(name="j")
                plain  = Transform.create()  # no explicit name: no prefix
                frames = _frame_uuids()
        self.assertEqual(str(typed), "inner_x")
        self.assertEqual(str(short), "inner_y")
        self.assertEqual(str(dsl), "inner_z")
        self.assertEqual(str(joint), "inner_j")
        self.assertEqual(str(plain), "transform1")
        for node in (typed, short, dsl, joint, plain):
            uid = _uuid(node)
            self.assertEqual(len(frames), 2)
            for frame in frames:
                self.assertIn(uid, frame)
            self.assertEqual(_owner(node), str(outer))
        # the typed classes are kept
        self.assertIs(type(typed), Transform)
        self.assertIs(type(joint), Joint)

    def test_a_deleted_name_reused_in_a_scope_joins_again(self):
        # E1: same-name reuse after delete
        with container("box") as box:
            first = Transform.create(name="x")
            uid   = _uuid(first)
            cmds.delete(str(first))
            again = Transform.create(name="x")
            self.assertNotEqual(_uuid(again), uid)
        self.assertEqual(str(again), "x")
        self.assertEqual(_members(box), ["x"])

    def test_undo_and_redo_of_a_typed_create_in_a_scope(self):
        # E1: undo removes the node and its membership, redo restores both
        state    = cmds.undoInfo(query=True, state=True)
        infinity = cmds.undoInfo(query=True, infinity=True)
        cmds.undoInfo(state=True, infinity=True)
        try:
            with container("box") as box:
                cmds.undoInfo(openChunk=True)
                try:
                    Transform.create(name="x")
                finally:
                    cmds.undoInfo(closeChunk=True)
                cmds.undo()
                self.assertFalse(cmds.objExists("x"))
                self.assertEqual(_members(box), [])
                cmds.redo()
            self.assertTrue(cmds.objExists("x"))
            self.assertEqual(_members(box), ["x"])
        finally:
            cmds.undoInfo(state=state, infinity=infinity)

    def test_a_new_scene_inside_an_open_scope(self):
        # E2: the scope's container is freed; a typed create behaves as Node.create
        for how in ("new", "open"):
            with self.subTest(how=how):
                self.new_scene()
                folder = tempfile.mkdtemp()
                path   = os.path.join(folder, "m5_scene.ma")
                try:
                    cmds.createNode("transform", name="held")
                    cmds.file(rename=path)
                    cmds.file(save=True, type="mayaAscii", force=True)
                    self.new_scene()
                    with container("box"):
                        with container("inner"):
                            if how == "new":
                                self.new_scene()
                            else:
                                cmds.file(path, open=True, force=True)
                            typed = Transform.create(name="x")
                            dsl   = Node.create("transform", name="y")
                            plain = Node.create("multiplyDivide", name="m")
                    self.assertEqual(
                        [str(typed), str(dsl), str(plain)], ["inner_x", "inner_y", "inner_m"]
                    )
                finally:
                    self.new_scene()
                    shutil.rmtree(folder, ignore_errors=True)

    def test_a_typed_create_before_the_hook_is_set_runs_plain(self):
        # the mid-import state: no hook, container= still consumed
        class _Rig(Transform):
            @classmethod
            @_typed_creator
            def build(cls, name=None):
                return cls.create(name=name)

        hook = dg_node_module._TYPED_CREATE_HOOK
        dg_node_module._TYPED_CREATE_HOOK = None
        try:
            with container("box") as box:
                with container("inner"):
                    node  = Transform.create(name="x", container=True)
                    built = _Rig.build(name="b", container=True)
        finally:
            dg_node_module._TYPED_CREATE_HOOK = hook
        self.assertEqual(str(node), "x")
        self.assertEqual(str(built), "b")
        self.assertEqual(_members(box), [])

    # -- 2 selection -- #

    def test_selection_inside_a_scope_and_outside(self):
        md_cls = self._md_class()
        cmds.createNode("transform", name="picked")
        cmds.select("picked")
        with container("box"):
            Transform.create(name="t")
            Joint.create(name="j")
            md_cls.create(name="md")
            Node.create("transform", name="p")
            Node.create("multiplyDivide", name="m")
            self.assertEqual(cmds.ls(selection=True), ["picked"])
            # the caller's flag wins
            Transform.create(name="chosen", skipSelect=False)
            self.assertEqual(cmds.ls(selection=True), ["chosen"])
            set_options(skip_selection=False)
            Transform.create(name="selected")
            self.assertEqual(cmds.ls(selection=True), ["selected"])
            set_options(skip_selection=True)
        # outside a scope a typed create selects, as before
        Transform.create(name="free")
        self.assertEqual(cmds.ls(selection=True), ["free"])

    def test_dg_create_forwards_skip_select(self):
        md_cls = self._md_class()
        cmds.select(clear=True)
        md_cls.create(name="selected")
        self.assertEqual(cmds.ls(selection=True), ["selected"])
        md_cls.create(name="quiet", skipSelect=True)
        self.assertEqual(cmds.ls(selection=True), ["selected"])
        md_cls.create(name="quiet_ss", ss=True)
        self.assertEqual(cmds.ls(selection=True), ["selected"])

    # -- 3 implicit nodes -- #

    def test_mesh_create_registers_transform_and_shape(self):
        data = self._mesh_data()
        with container("box") as box:
            mesh = Mesh.create(data, name="m")
        self.assertEqual(str(mesh), "mShape")
        self.assertEqual(_members(box), ["m", "mShape"])

    def test_follicle_registers_shape_and_transform(self):
        plane = cmds.polyPlane(name="plane", ch=False)[0]
        ref   = cmds.spaceLocator(name="ref")[0]
        cmds.xform(ref, t=(0.1, 0, 0.2), ws=True)
        with container("outer") as outer:
            with container("inner"):
                positional = Follicle.create_on_mesh(plane, ref, "fol")
                keyword    = Follicle.create_on_mesh(plane, ref, name="folk")
                default    = Follicle.create_on_mesh(plane, ref)
        self.assertEqual(str(positional), "inner_folShape")
        self.assertEqual(str(positional.get_parent()), "inner_fol")
        self.assertEqual(str(keyword), "inner_folkShape")
        self.assertEqual(str(keyword.get_parent()), "inner_folk")
        self.assertEqual(str(default), "follicleShape")  # no explicit name: no prefix
        self.assertEqual(
            _members(outer),
            sorted([
                "inner_fol", "inner_folShape", "inner_folk", "inner_folkShape",
                "follicle", "follicleShape",
            ]),
        )
        self.assertIsNone(_owner(plane))
        self.assertIsNone(_owner(ref))

    # -- 4 nesting -- #

    def test_skincluster_registers_its_own_nodes_and_no_temporary_joint(self):
        # the Orig shape under the user's mesh and the shared bindPose belong
        # to nodes that existed before the call, so they are not registered
        # (see test_r4a_review.TestScopeRegistersItsOwnNodes)
        j1   = cmds.createNode("joint", name="j1")
        j2   = cmds.createNode("joint", name="j2", parent=j1)
        cube = self._cube()
        with container("box") as box:
            skin = SkinCluster.create(cube, [j1, j2])
            before = _frame_uuids()
            # set_influence_objects makes a temporary joint and deletes it
            skin.set_influence_objects([j1])
            after = _frame_uuids()
        members = _members(box)
        self.assertEqual(members, [str(skin)])
        self.assertTrue(cmds.objExists("cubeShapeOrig"))
        self.assertTrue(cmds.ls(type="dagPose"))
        self.assertEqual(after, before)
        for uid in after[0]:
            name = cmds.ls(uid)
            self.assertTrue(name, uid)  # no dead uuid
            self.assertNotEqual(cmds.nodeType(name[0]), "joint")
        self.assertIsNone(_owner(j1))
        self.assertIsNone(_owner(cube))

    def test_mesh_create_prefixes_once(self):
        data = self._mesh_data()
        with container("outer") as outer:
            with container("inner"):
                mesh = Mesh.create(data, name="m")
        self.assertEqual(str(mesh), "inner_mShape")
        self.assertEqual(_members(outer), ["inner_m", "inner_mShape"])

    def test_the_one_node_a_create_made_is_registered_as_its_object(self):
        # R2: a typed create that makes only the node it returns (DGNode.create's
        # template with DGNode's _create on a DG class or DAGNode's on a transform
        # class, and DGNode's post_create: Transform, Joint, a DG class; so
        # Node.create of those types) runs untracked and registers the node
        # object, whose uuid is read directly; any other create is tracked and
        # registers every node it made, by name (the one it returns as its object)
        class _WithHelper(Transform):
            @classmethod
            def post_create(cls, new_node_name, *args, **kwargs):
                cmds.createNode("multiplyDivide", name="helper", skipSelect=True)
                return super().post_create(new_node_name, *args, **kwargs)

        data    = self._mesh_data()
        md_cls  = self._md_class()
        by_name = mock.Mock(wraps=container_module._node_uuid)
        tracked = mock.Mock(wraps=container_module._call_tracking_creation)
        with mock.patch.object(container_module, "_node_uuid", by_name), mock.patch.object(
            container_module, "_call_tracking_creation", tracked
        ):
            with container("outer") as outer:
                with container("inner"):
                    made = [
                        Transform.create(name="t"),
                        Node.create("joint", name="j"),
                        md_cls.create(name="md"),
                        Node.create("transform", name="n"),
                    ]
                    counts = (by_name.call_count, tracked.call_count)
                    mesh   = Mesh.create(data, name="m")
                    counts_mesh = (by_name.call_count, tracked.call_count)
                    helped = _WithHelper.create(name="w")
                    frames = _frame_uuids()
        self.assertEqual(counts, (0, 0))
        self.assertEqual(counts_mesh, (2, 1))
        self.assertEqual(tracked.call_count, 2)
        self.assertEqual([str(x) for x in made], ["inner_t", "inner_j", "inner_md", "inner_n"])
        self.assertEqual(str(helped), "inner_w")
        self.assertEqual(
            _members(outer),
            ["helper", "inner_j", "inner_m", "inner_mShape", "inner_md", "inner_n", "inner_t",
             "inner_w"],
        )
        for node in made + [mesh, mesh.get_parent(), helped, "helper"]:
            for frame in frames:
                self.assertIn(_uuid(node), frame)
        self.assertTrue(_tagged(made[2]))
        self.assertFalse(_tagged(made[0]))
        self.assertEqual(cmds.ls(selection=True), [])

    def test_create_hierarchy_keeps_its_names_and_registers_every_node(self):
        root  = Node.create("joint", name="root_joint")
        child = Node.create("joint", name="child_joint", parent=root)
        Node.create("joint", name="child_joint", parent=child)
        hierarchy = root.serialize_hierarchy()
        expected  = sorted(cmds.ls(type="joint", long=True))

        self.new_scene()
        with container("outer") as outer:
            with container("inner"):
                created = Transform.create_hierarchy(hierarchy)
                frames  = _frame_uuids()
        self.assertEqual(sorted(cmds.ls(type="joint", long=True)), expected)
        self.assertEqual(len(created), 3)
        for name in created:
            self.assertEqual(_owner(name), str(outer))
            for frame in frames:
                self.assertIn(_uuid(name), frame)

    # -- 5 registries -- #

    def test_display_layer_get_or_create_stays_out_of_a_flattened_scope(self):
        """Historical id: pinned DisplayLayer / ObjectSet.get_or_create in a flattened
        scope; it now pins their define (round 4b NC4): no prefix, not registered."""
        with container("outer") as outer:
            with container("inner"):
                first  = DisplayLayer.define("L")
                second = DisplayLayer.define("L")
                sets   = ObjectSet.define("S")
                frames = _frame_uuids()
        self.assertEqual(str(first), "L")
        self.assertEqual(first, second)
        self.assertEqual(cmds.ls("L*", type="displayLayer"), ["L"])
        self.assertEqual(str(sets), "S")
        for node in (first, sets):
            self.assertIsNone(_owner(node))
            for frame in frames:
                self.assertNotIn(_uuid(node), frame)
        self.assertIsNone(cmds.container(str(outer), query=True, nodeList=True))

    def test_registry_with_container_true_is_a_member_never_prefixed(self):
        with container("outer") as outer:
            with container("inner"):
                layer = DisplayLayer.create(name="L", container=True)
                found = ObjectSet.create(name="S", container=True)
        self.assertEqual(str(layer), "L")
        self.assertEqual(str(found), "S")
        self.assertEqual(_members(outer), ["L", "S"])

    def test_materials_and_layers_leave_the_same_scene_inside_a_scope(self):
        def build():
            cube = Node(self._cube())
            cube << Blinn("red")
            cube << Layer.define("lay")   # re-pinned (round 4b NC6): Layer is DisplayLayer

        def scene():
            return sorted(
                (name, cmds.nodeType(name))
                for name in cmds.ls()
                if cmds.nodeType(name) != "container"
            )

        build()
        outside = scene()
        self.new_scene()
        with container("outer") as outer:
            with container("inner"):
                build()
        self.assertEqual(scene(), outside)
        for name in ("red", "redSG", "lay"):
            self.assertIsNone(_owner(name))
        self.assertEqual(_members(outer), [])

    # -- 6 GC parity -- #

    def test_gc_tag_parity_with_node_create(self):
        md_cls = self._md_class()

        class _Meta(DGNode):
            NATIVE_NODE_TYPE = "network"
            CUSTOM_NODE_TYPE = "m5Meta"

        with container("box") as box:
            typed = md_cls.create(name="typed_md")
            dsl   = Node.create("multiplyDivide", name="dsl_md")
            meta  = _Meta.create(name="meta")
            xform = Transform.create(name="xform")
        self.assertEqual(_tag_state(typed), (True, True, True, True))
        self.assertEqual(_tag_state(typed), _tag_state(dsl))
        self.assertEqual(_tag_state(meta), (False,))
        self.assertEqual(_tag_state(xform), (False,))
        self.assertEqual(_members(box), ["dsl_md", "meta", "typed_md", "xform"])
        # outside a scope nothing changes: a typed create tags nothing
        self.assertEqual(_tag_state(md_cls.create(name="free_md")), (False,))

    # -- 7 container=False -- #

    def test_container_false_opts_out(self):
        plane = cmds.polyPlane(name="plane", ch=False)[0]
        ref   = cmds.spaceLocator(name="ref")[0]
        cmds.xform(ref, t=(0.1, 0, 0.2), ws=True)
        with container("outer") as outer:
            with container("inner"):
                loose = Transform.create(name="loose", container=False)
                fol   = Follicle.create_on_mesh(plane, ref, name="fol", container=False)
                md    = Node.create("multiplyDivide", name="md", container=False)
                kept  = Transform.create(name="kept", container=True)
                frames = _frame_uuids()
        # container=False means what it means for container.createNode, on
        # every creator: the nodes are not registered; the flattened prefix
        # (and an eligible type's GC tag) still apply
        self.assertEqual(str(loose), "inner_loose")
        self.assertEqual(str(fol), "inner_folShape")
        self.assertEqual(str(md), "inner_md")
        self.assertTrue(_tagged(md))
        for node in (loose, fol, fol.get_parent(), md):
            self.assertIsNone(_owner(node))
            for frame in frames:
                self.assertNotIn(_uuid(node), frame)
        self.assertEqual(str(kept), "inner_kept")
        self.assertEqual(_members(outer), ["inner_kept"])
        # consumed outside a scope too
        self.assertEqual(str(Transform.create(name="free", container=False)), "free")
        self.assertEqual(str(Node.create("multiplyDivide", name="fmd", container=False)), "fmd")

    # -- 8 cleanup_on_exit -- #

    def test_cleanup_on_exit_keeps_a_typed_transform(self):
        md_cls = self._md_class()
        with container("box", cleanup_on_exit=True):
            keep = Transform.create(name="keep")
            md_cls.create(name="orphan_typed")
            Node.create("multiplyDivide", name="orphan_dsl")
        self.assertTrue(cmds.objExists(str(keep)))
        # the tagged orphans go the same way
        self.assertFalse(cmds.objExists("orphan_typed"))
        self.assertFalse(cmds.objExists("orphan_dsl"))

        with container("outer"):
            with container("inner", cleanup_on_exit=True):
                inner = Transform.create(name="keep")
        self.assertEqual(str(inner), "inner_keep")
        self.assertTrue(cmds.objExists("inner_keep"))

    # -- 9 Node.create (R2: the former PyNode.create) -- #

    def test_node_create_joins_through_both_branches(self):
        with container("outer") as outer:
            with container("inner"):
                typed = Node.create("transform", name="t")
                plain = Node.create("multiplyDivide", name="md")
                self.assertEqual(cmds.ls(selection=True), [])
        self.assertIs(type(typed), Transform)
        self.assertEqual(str(typed), "inner_t")
        self.assertEqual(str(plain), "inner_md")
        self.assertIs(type(plain), type(Node(cmds.createNode("multiplyDivide"))))
        self.assertTrue(_tagged(plain))
        self.assertEqual(_members(outer), ["inner_md", "inner_t"])
        # outside a scope: container.createNode, as Node.create always did (the
        # GC tag, no selection; PyNode.create's plain cmds.createNode is gone)
        cmds.select(clear=True)
        free = Node.create("multiplyDivide", name="free")
        self.assertEqual(str(free), "free")
        self.assertTrue(_tagged(free))
        self.assertEqual(cmds.ls(selection=True), [])

    def test_node_create_inside_a_typed_create(self):
        class _Rig(Transform):
            @classmethod
            @_typed_creator
            def build(cls, name=None):
                Node.create("multiplyDivide", name="deep")
                Node.create("transform", name="deep_t")
                return cls.create(name=name)

        with container("outer") as outer:
            with container("inner"):
                # the container's hyperLayout exists before the typed create (a
                # first add inside a tracked create registers the hyperLayout
                # it makes as a member, as at R1)
                Node.create("transform", name="first")
                top = _Rig.build(name="top")
        self.assertEqual(str(top), "inner_top")
        # a registered type is a typed create, so it is nested: no prefix,
        # tracked by the outer one
        self.assertTrue(cmds.objExists("deep_t"))
        # a type with no class is container.createNode, which joins the scope
        # itself (prefix, GC tag), as Node.create always did (R2: the nested
        # plain path of PyNode.create, D13b, is gone)
        self.assertFalse(cmds.objExists("deep"))
        self.assertTrue(_tagged("inner_deep"))
        self.assertEqual(_members(outer), ["deep_t", "inner_deep", "inner_first", "inner_top"])

    # -- 10 neutral scopes -- #

    def test_neutral_scopes_record_uuids_and_add_nothing(self):
        for label in ("create_containers", "enabled"):
            with self.subTest(scope=label):
                self.new_scene()
                if label == "create_containers":
                    set_options(create_containers=False)
                    scope = container("box")
                else:
                    scope = container("box", enabled=False)
                with scope as box:
                    typed  = Transform.create(name="x")
                    dsl    = Node.create("transform", name="y")
                    frames = _frame_uuids()
                set_options(create_containers=True)
                self.assertIsNone(box)
                self.assertEqual(str(typed), "x")
                self.assertEqual(str(dsl), "y")
                self.assertEqual(frames, [[_uuid(typed), _uuid(dsl)]])
                self.assertEqual(cmds.ls(type="container"), [])

    # -- 11 exceptions -- #

    def test_an_exception_resets_the_depth(self):
        class _Boom(Transform):
            @classmethod
            def _create(cls, *args, **kwargs):
                raise RuntimeError("boom")

        class _BoomBuild(Transform):
            @classmethod
            @_typed_creator
            def build(cls):
                Transform.create(name="half")
                raise RuntimeError("boom")

        with container("box") as box:
            with self.assertRaises(RuntimeError):
                _Boom.create(name="never")
            self.assertEqual(container_module._TYPED_DEPTH, 0)
            with self.assertRaises(RuntimeError):
                _BoomBuild.build()
            self.assertEqual(container_module._TYPED_DEPTH, 0)
            after = Transform.create(name="after")
        with self.assertRaises(RuntimeError):
            _Boom.create(name="never")
        self.assertEqual(container_module._TYPED_DEPTH, 0)
        self.assertEqual(str(after), "after")
        # a failed create registers nothing (as an rc.* call that raises)
        self.assertEqual(_members(box), ["after"])
        self.assertTrue(cmds.objExists("half"))

    # -- 12 keywords -- #

    def test_other_keywords_reach_create_unchanged(self):
        """A keyword the class declares in ``_CREATE_FLAGS`` reaches its
        ``_create`` unchanged (the same object); since NC3 any other keyword
        is an attribute of the new node, set once it exists (``caching``)."""
        seen = []

        class _Rec(DGNode):
            NATIVE_NODE_TYPE = "network"
            CUSTOM_NODE_TYPE = "m5Rec"
            # the keywords its _create takes (NC3: any other is an attribute)
            _CREATE_FLAGS = DGNode._CREATE_FLAGS | {"foo", "flags"}

            @classmethod
            def _create(cls, *args, **kwargs):
                seen.append((args, dict(kwargs)))
                return cmds.createNode("network", name=kwargs["name"], skipSelect=True)

        class _RecT(Transform):
            @classmethod
            def post_create(cls, new_node_name, *args, **kwargs):
                seen.append(dict(kwargs))
                return super().post_create(new_node_name, *args, **kwargs)

        flags = [1, 2]
        with container("outer"):
            with container("inner"):
                _Rec.create("pos", name="r", foo=1, flags=flags, container=None, caching=True)
                _RecT.create(name="t")
        self.assertEqual(seen[0], (("pos",), {"name": "inner_r", "foo": 1, "flags": [1, 2]}))
        self.assertIs(seen[0][1]["flags"], flags)
        self.assertEqual(seen[1], {"name": "inner_t", "skipSelect": True})
        self.assertIs(cmds.getAttr("inner_r.caching"), True)

    def test_display_layer_and_skincluster_flags_pass_through(self):
        cube = self._cube()
        j1   = cmds.createNode("joint", name="j1")
        j2   = cmds.createNode("joint", name="j2", parent=j1)
        cmds.select(cube)
        with container("outer"):
            with container("inner"):
                empty  = DisplayLayer.create(name="empty_layer", empty=True)
                filled = DisplayLayer.create(name="filled_layer", empty=False)
                skin   = SkinCluster.create(cube, [j1, j2], name="sk", maximumInfluences=1)
        self.assertIsNone(cmds.editDisplayLayerMembers(str(empty), query=True))
        # createDisplayLayer took the selection (empty=False): no flag was
        # added to its call
        self.assertIn(cube, cmds.editDisplayLayerMembers(str(filled), query=True))
        self.assertEqual(str(skin), "inner_sk")
        self.assertEqual(cmds.skinCluster(str(skin), query=True, maximumInfluences=True), 1)

    # -- 13 namespaces -- #

    def test_a_current_namespace(self):
        cmds.namespace(add="ns")
        cmds.namespace(set="ns")
        with container("outer") as outer:
            with container("inner"):
                typed = Transform.create(name="x")
                dsl   = Node.create("transform", name="y")
        cmds.namespace(set=":")
        self.assertEqual(str(typed), "ns:inner_x")
        self.assertEqual(str(dsl), "ns:inner_y")
        self.assertEqual(_owner(typed), str(outer))
        self.assertEqual(_owner(dsl), str(outer))

    # -- 14 the bridges -- #

    def test_bridge_tracking_is_the_shared_one(self):
        self.assertIs(rc._call_tracking_creation, container_module._call_tracking_creation)
        with container("build"):
            rc.polyCube(name="box")
            rc.createNode("transform", name="loose", container=False)
        self.assertEqual(_members("build"), ["box", "boxShape", "polyCube1"])

    # -- outside a scope -- #

    def test_outside_a_scope_nothing_changes(self):
        md_cls = self._md_class()
        free = Transform.create(name="free")
        self.assertEqual(cmds.ls(selection=True), ["free"])
        self.assertIsNone(_owner(free))
        self.assertEqual(_tag_state(md_cls.create(name="md")), (False,))
        self.assertEqual(container_module._TYPED_DEPTH, 0)

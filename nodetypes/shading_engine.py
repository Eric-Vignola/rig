"""
Shading engine class

A ``shadingEngine`` is an objectSet with render wiring: a ``surfaceShader``
input, a slot in ``renderPartition`` (which is what makes membership
exclusive), a ``materialInfo`` node and lightLinker entries.
``cmds.createNode("shadingEngine")`` makes none of that and the resulting set
refuses every member, so creation goes through ``cmds.sets(renderable=True)``.
Membership is read through ``MFnSet`` so per-face entries survive; the
inherited ``get_members()`` drops them.

Usage::

    from rig.nodetypes import ShadingEngine

    sg = ShadingEngine.for_material("myBlinn")     # find, or build 'myBlinnSG'
    sg.assign(["pCube1.f[0:1]"], touched=["|pCube1|pCubeShape1"])
    sg.get_face_members()                          # [(Mesh("pCubeShape1"), array([0, 1]))]
    sg.assign(["|pCube1|pCubeShape1"])             # whole object
    sg.get_face_members()                          # [(Mesh("pCubeShape1"), None)]
    sg.get_material()                              # Blinn("myBlinn")

    cube << sg ; faces << -sg ; cube in sg ; cube >> sg   # the membership grammar (rig.shade)

An engine node on the right of ``<<`` / ``>>`` / ``in`` names exactly that
engine (``faces << ShadingEngine("altSG")``, while a shader node goes to the
engine it feeds); ``-sg`` is its removal token. It is no kind:
``ShadingEngine()`` is a TypeError, and ``Material()`` takes the members out
of every engine.
"""

from __future__ import annotations

from typing import Any, Iterator, Sequence

import numpy as np
from maya import cmds
from maya.api import OpenMaya
from rig.nodetypes._base import _cast
from rig.nodetypes.dag_node import DAGNode
from rig.nodetypes.dg_node import DGNode
from rig.nodetypes.object_set import ObjectSet


def _absolute(name: str) -> str:
    """The absolute name (``:x``, ``:lib:x``) of the node a name Maya returned
    names."""
    sel = OpenMaya.MSelectionList()
    sel.add(name)
    return OpenMaya.MFnDependencyNode(sel.getDependNode(0)).absoluteName()


def _shown(absolute: str) -> str:
    """An absolute name as a user reads it (``x``, ``lib:x``)."""
    return absolute[1:] if absolute[:1] == ":" else absolute


class ShadingEngine(ObjectSet):
    """
    Shading engine class
    """

    # the maya native node type string
    NATIVE_NODE_TYPE = "shadingEngine"

    # the shading engine every new shadeable shape is a member of
    DEFAULT = "initialShadingGroup"

    # ``cmds.sets``' name (``renderable``, ``noSurfaceShader`` and ``empty`` are
    # always set); ``create``'s other keywords are the engine's attributes
    _CREATE_FLAGS = frozenset({"name", "n"})

    # an engine is a membership collection but no kind: ``ShadingEngine()`` /
    # ``ShadingEngine(None)`` are refused, naming the kind token of materials
    _KIND_HINT = "Material() takes the members out of every shading engine"

    # --- the membership grammar (rig.shade)

    def _member(self) -> Any:
        """The membership this engine runs on the right of ``<<`` / ``>>``:
        exactly this engine."""
        from rig.nodetypes.material_node import _material_member

        return _material_member()(self)

    def __neg__(self) -> Any:
        """``-sg``: the removal token (``faces << -sg`` takes the faces out of
        this engine)."""
        from rig.nodetypes.material_node import _material_member

        return _material_member()(self, remove=True)

    def __invert__(self) -> Any:
        raise TypeError(
            f"~{self!r} is unassigned; -sg removes members and Material() removes "
            f"them from every shading engine"
        )

    def __contains__(self, lhs: Any) -> bool:
        """``x in sg``: whether every face of every shadeable shape of ``x``
        (or the whole object) is in exactly this engine."""
        return self._member().contains(lhs)

    def _refuse_default(self, verb: str) -> None:
        """A Maya default engine (``initialShadingGroup``, ``initialParticleSE``)
        takes no delete or rename: TypeError, nothing written (Maya itself
        prints an error and deletes nothing, or raises a RuntimeError)."""
        name = self.name
        if cmds.ls(name, defaultNodes=True):
            raise TypeError(f"'{name}' cannot be {verb}: it is a Maya default shading engine")

    def delete(self, nodes=None, **kwargs) -> None:
        """Deletes this engine (its members are left in no engine: green;
        ``rig.shade.repair()`` re-homes them), or ``nodes`` when given, as
        :meth:`DGNode.delete` does. Refuses a Maya default engine
        (``Default()``): TypeError."""
        if nodes is None:
            self._refuse_default("deleted")
        super().delete(nodes, **kwargs)

    def rename(self, new_name: Any) -> None:
        """Renames this engine. Refuses a Maya default engine (``Default()``):
        TypeError."""
        self._refuse_default("renamed")
        super().rename(new_name)

    # --- creation

    @classmethod
    def _create(cls, *args, **kwargs) -> str:
        """[Internal] Creates an empty, fully wired shading engine and returns its name.

        ``cmds.createNode`` yields a shading engine with no renderPartition slot,
        materialInfo or lightLinker entries, and such a set refuses every member
        ("Source node will not allow the connection"); ``cmds.sets`` wires all of them.

        Args:
            kwargs: the ``_CREATE_FLAGS`` given to ``create``: ``name`` / ``n``
                names the set; ``renderable``, ``noSurfaceShader`` and
                ``empty`` are always set.
        """
        name = kwargs.pop("name", kwargs.pop("n", None)) or cls.NATIVE_NODE_TYPE
        return cmds.sets(
            renderable=True, noSurfaceShader=True, empty=True, name=name, **kwargs
        )

    @classmethod
    def for_material(
        cls, material: str | DGNode, create: bool = True
    ) -> ShadingEngine | None:
        """Returns the shading engine that renders a material.

        The candidates are the shading engines fed through ``.surfaceShader``,
        ``initialParticleSE`` excluded (standardSurface1 feeds both defaults):

        - none: with ``create``, a ``<material>SG`` is built and connected and the
          material is listed in ``defaultShaderList1`` the way
          ``cmds.shadingNode(asShader=True)`` does. Otherwise None. The engine
          of a referenced shader is built at the root (``:<leaf>SG``): a node
          made in the reference's namespace would be unreferenced there.
        - one: that engine.
        - many: ``<material>SG`` when it is one of them, else a ValueError naming them.

        Names are compared absolutely (``namespace -relativeNames`` changes how
        Maya prints them, not which node they name).

        Args:
            material: A surface shader node.
            create: If True, build the engine when the material has none.
        """
        sel = OpenMaya.MSelectionList()
        try:
            sel.add(material.name if isinstance(material, DGNode) else str(material))
        except RuntimeError:
            found = cmds.ls(str(material))
            raise ValueError(f"Material must match one node: {material} -> {found}") from None
        shader   = OpenMaya.MFnDependencyNode(sel.getDependNode(0))
        material = shader.absoluteName()

        engines = []
        plugs   = cmds.listConnections(
            material, type="shadingEngine", source=False, destination=True, plugs=True
        )
        for plug in plugs or []:
            node, _, attr = plug.rpartition(".")
            if attr != "surfaceShader":
                continue
            node = _absolute(node)
            if node != ":initialParticleSE" and node not in engines:
                engines.append(node)

        leaf         = material.rsplit(":", 1)[-1]
        default_name = f":{leaf}SG" if shader.isFromReferencedFile else f"{material}SG"
        if len(engines) == 1:
            return cls._wrap(engines[0])
        if engines:
            if default_name in engines:
                return cls._wrap(default_name)
            shown = [_shown(name) for name in engines]
            raise ValueError(
                f"{_shown(material)} feeds {len(engines)} shading engines {shown} and none "
                f"is named {_shown(default_name)}; pick one explicitly."
            )
        if not create:
            return None

        engine = cls.create(name=default_name)
        engine.set_material(material)
        shaders = cmds.listConnections(
            "defaultShaderList1.shaders", source=True, destination=False
        )
        if material not in {_absolute(name) for name in shaders or []}:
            cmds.connectAttr(
                f"{material}.message", "defaultShaderList1.shaders", nextAvailable=True
            )
        return engine

    # --- material

    def get_material(self) -> DGNode | None:
        """Returns the surface shader feeding this engine, or None."""
        material = cmds.listConnections(
            f"{self.name}.surfaceShader", source=True, destination=False
        )
        return _cast(material[0]) if material else None

    def set_material(self, material: str | DGNode) -> None:
        """Feeds this engine from a material, replacing the previous one.

        Args:
            material: A surface shader node (its ``outColor`` is used) or an
                output plug string such as ``"myRamp.outColor"``.
        """
        source = str(material)
        if "." not in source:
            source = f"{source}.outColor"
        cmds.connectAttr(source, f"{self.name}.surfaceShader", force=True)

    def get_material_info(self) -> list[str]:
        """Returns the materialInfo nodes attached to this engine."""
        return cmds.listConnections(f"{self.name}.message", type="materialInfo") or []

    # --- membership

    def _iter_dag_members(self) -> Iterator[tuple[OpenMaya.MDagPath, np.ndarray | None]]:
        """[Internal] Yields (dag path, face ids) per DAG member as Maya stores it:
        None for whole-object membership, an array for a face component (every
        face of the mesh included). Other component kinds are skipped."""
        members = self.fn_set.getMembers(False)
        for i in range(members.length()):
            try:
                dagpath, component = members.getComponent(i)
            except TypeError:
                # a DG member; shading engines hold none
                continue
            if component.isNull():
                yield dagpath, None
            elif component.hasFn(OpenMaya.MFn.kMeshPolygonComponent):
                ids = OpenMaya.MFnSingleIndexedComponent(component).getElements()
                yield dagpath, np.array(sorted(ids), dtype=int)

    def get_face_members(self) -> list[tuple[DAGNode, np.ndarray | None]]:
        """Returns (shape, face ids) pairs for every member of this engine.

        Read through ``MFnSet`` so per-face entries survive (``get_members()`` drops
        them). ``None`` means the whole object; a component naming every face of its
        mesh is reported as ``None`` too, since Maya never collapses that state by
        itself. Instances are reported per DAG path.
        """
        result = []
        for dagpath, ids in self._iter_dag_members():
            if ids is not None and dagpath.hasFn(OpenMaya.MFn.kMesh):
                if len(ids) == OpenMaya.MFnMesh(dagpath).numPolygons:
                    ids = None
            result.append((_cast(dagpath), ids))
        return result

    def assign(
        self,
        members:   Sequence[str],
        touched:   Sequence[str] = (),
        normalise: bool          = False,
    ) -> None:
        """Moves objects and faces into this engine; a member leaves whatever
        engine held it (``forceElement``).

        Maya carves a whole-object membership down to the complementary faces
        when some of its faces move elsewhere, but never collapses per-face
        membership back, and leaves the groupId nodes it made behind. So for the
        shapes named in ``touched`` (the shapes ``members`` live on), this method
        re-assigns at object level every non-instanced shape whose faces all
        ended in this engine (instances are members per DAG path) when
        ``normalise`` is True, and deletes those shapes' groupId nodes that ended
        with no connections. Nothing is touched when ``touched`` is empty. The
        membership operator never asks for this (a tidy-up is its own step,
        off the assignment's undo chunk), so the default is the raw Maya
        behaviour.

        Faces into the engine that already owns their whole object is a silent
        no-op in Maya and is left alone: that state already holds.

        Args:
            members: Object and component strings, as cmds.sets() takes them.
            touched: The shapes the members live on, by any name.
            normalise: If True, collapse all-faces memberships to object level.
        """
        members = [str(x) for x in members]
        touched = [_long_name(x) for x in touched]
        before  = self._group_ids(touched)

        cmds.sets(members, edit=True, forceElement=self.name)

        if normalise:
            self._renormalise(touched)
        for group_id in before | self._group_ids(touched):
            if cmds.objExists(group_id) and not cmds.listConnections(group_id):
                cmds.delete(group_id)

    def _renormalise(self, touched: Sequence[str]) -> None:
        """[Internal] Re-assigns at object level the touched, non-instanced meshes
        whose faces all sit in this engine as a face component."""
        if not touched:
            return
        paths = []
        for dagpath, ids in self._iter_dag_members():
            if ids is None or dagpath.isInstanced() or not dagpath.hasFn(OpenMaya.MFn.kMesh):
                continue
            path = dagpath.fullPathName()
            if path in touched and len(ids) == OpenMaya.MFnMesh(dagpath).numPolygons:
                paths.append(path)
        if paths:
            cmds.sets(paths, edit=True, forceElement=self.name)

    @staticmethod
    def _group_ids(shapes: Sequence[str]) -> set[str]:
        """[Internal] Returns the groupId nodes wired to the given shapes."""
        result = set()
        for shape in shapes:
            result.update(cmds.listConnections(shape, type="groupId") or [])
        return result


def _long_name(node: str | DAGNode) -> str:
    """Returns the full DAG path of a node that matches exactly one path."""
    found = cmds.ls(str(node), long=True)
    if len(found) != 1:
        raise ValueError(f"Shape must match one node: {node} -> {found}")
    return found[0]

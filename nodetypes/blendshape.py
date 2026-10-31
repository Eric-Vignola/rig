"""
Blendshape class
"""

from __future__ import annotations

import logging
import os
import re

import numpy as np
from maya import cmds
from maya.api import OpenMaya
from numpy.typing import ArrayLike
from rig._internal.journal import cmds_edit as _cmds_edit, statement as _statement
from rig._internal.undo import _undo_chunk
from rig.nodetypes import _files
from rig.nodetypes._base import _connected_attrs, Attribute, _name_of
from rig.nodetypes.deformer import Deformer
from rig.nodetypes.geometry import iter_component_ranges
from rig.nodetypes.transform import Transform

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# what Maya refuses in the alias of a target, whatever the code page (test_class_blendshape
# asks Maya for each ASCII symbol and digit, first and further in): a digit or a "-" first, a
# "<" further in, a space and any other symbol but "_". A list of what it takes would refuse
# working names, a letter of the code page for one; a letter outside it stays Maya's own error.
_NOT_AN_ALIAS = re.compile(r"^[-0-9]|.<|[ !-,./:;=-@\[-^`{-~]")


_BLENDSHAPE_LOAD_DOC = """Loads blendShape targets from a cgmath file (a MorphList): one name, two forms.

    ``BlendShape.load(targets, geometry, **flags_and_attrs)``, from the class,
    CREATES a new blendShape on `geometry`, always (as `create` does: a second
    load stacks a second blendShape)::

        face = BlendShape.load("body_face.npz", body, name="face_bs")

    ``BlendShape('face_bs').load(targets)``, from a blendShape, UPDATES that
    blendShape's targets in place by name, the inverse of
    ``BlendShape('face_bs').save(...)``: each target of the file replaces the
    offsets of its target of that name, or is added when there is none; the
    blendShape's other targets stay. A target the file replaces gives up its
    live target mesh (its ``inputGeomTarget`` input, logged at INFO), as a
    value replaces a plug's driver: a live mesh would write its own offsets
    over the file's at the next evaluation::

        BlendShape("face_bs").load("body_face.npz")

    Args:
        targets: The file (a str or os.PathLike, ``~`` expanded; a cgmath
            ``.npz`` / ``.json`` / ``.pkl``), or the MorphList / MorphData
            itself (never changed: the load works on a copy). The file comes
            first, the geometry second: the other order is a TypeError.
        geometry: (creating) The base, as `create` takes it.
        flags_and_attrs: (creating) as `create` takes them (``name=`` ...).

    Returns:
        The new BlendShape (creating), or this BlendShape (updating).

    The file is read, and what it holds checked, before anything is made or
    changed: a missing file is a FileNotFoundError, a file of another class a
    TypeError naming what it holds, a file from an older cgmath cgmath's
    LegacyFileError, a morph with no name, or with a new target's name Maya
    refuses (`set_target_data`), a ValueError. A MorphData holds a target's
    sparse offsets and its name only: the weights come back 0, and in-betweens
    are not carried.

    Each call is ONE undo step named ``rig.BlendShape.load`` and leaves the
    selection as it was.
    """


class BlendShape(Deformer):
    """
    Blenshape class
    """

    # the maya native node type string
    NATIVE_NODE_TYPE = "blendShape"

    # built from its shapes (``Node.create("blendShape")`` with none raises,
    # naming them: ``cmds.blendShape`` alone would deform the selection)
    _CREATE_TAKES_INPUTS = "*targets, base"

    # ``cmds.blendShape``'s create-mode flags (Maya 2025 docs, each accepted in
    # a create by runs/r4b/NC3/probe_flags_bs.py; the symmetry flags work only
    # in edit mode); ``create``'s other keywords are the blendShape's
    # attributes. ``envelope``, ``origin`` and ``topologyCheck`` are attributes
    # too: the flag.
    _CREATE_FLAGS = frozenset(
        {
            "after", "af", "afterReference", "ar", "automatic", "at", "before", "bf",
            "envelope", "en", "exclusive", "ex", "frontOfChain", "foc", "ignoreSelected", "is",
            "inBetween", "ib", "inBetweenType", "ibt", "includeHiddenSelections", "ihs",
            "name", "n", "origin", "o", "parallel", "par", "split", "sp",
            "suppressDialog", "sd", "tangentSpace", "ts", "target", "t",
            "topologyCheck", "tc", "useComponentTags", "uct", "weight", "w",
            "weightCount", "wc",
        }
    )

    _DEFINE_REFUSED = (
        "a blendShape is keyed by its geometry: BlendShape.create(*targets, base) or "
        "BlendShape.load(path, base) makes one; BlendShape('x') refers to one"
    )

    @classmethod
    def _create(cls, *args, **kwargs) -> str:
        """[Internal] Creates a blendshape node and returns the mesh name.

        Args:
            args: args supported by cmds.blendShape(). A MorphData or MorphList
                can be given in place of a target geometry, in which case a
                target is built for each morph and its offsets are applied
                immediately after the node is created.
            kwargs: the ``_CREATE_FLAGS`` given to ``create``, for
                cmds.blendShape()

        One undo step named ``rig.BlendShape.create``: the blendShape and the
        morphs' targets. A morph with no name, or with a name Maya refuses
        (`set_target_data`), is a ValueError before anything is made.
        """
        from cgmath.geometry import MorphData, MorphList

        # split the morph data out of the geometries, it is applied once the
        # node exists
        morphs = MorphList()
        geoms  = []
        for arg in args:
            if isinstance(arg, MorphData):
                morphs.append(arg)
            elif isinstance(arg, MorphList):
                morphs.extend(arg)
            else:
                geoms.append(arg)
        _check_morph_names("BlendShape.create", morphs)
        _check_target_names("BlendShape.create", [m.name for m in morphs], OpenMaya.MNodeClass(cls.NATIVE_NODE_TYPE))

        # default frontOfChain to True
        kwargs["frontOfChain"] = kwargs.get("frontOfChain", kwargs.get("foc", True))
        if "foc" in kwargs:
            kwargs.pop("foc")

        with _undo_chunk("rig.BlendShape.create"):
            name = _cmds_edit.blendShape(*geoms, **kwargs)[0]

            if morphs:
                cls._wrap(name).set_target_data(morphs)

        return name

    # --- attr helpers

    def get_target_data_attrs(self, index: int) -> tuple[Attribute, Attribute, Attribute]:
        """Returns 2 attribute objects pointing to the 2 data attrs of a given target:

        .inputTarget[0].inputTargetGroup[x].inputTargetItem[6000].inputGeomTarget
        .inputTarget[0].inputTargetGroup[x].inputTargetItem[6000].inputPointsTarget
        .inputTarget[0].inputTargetGroup[x].inputTargetItem[6000].inputComponentsTarget
        """
        group_attr = self.find_attr("inputTarget")[0].inputTargetGroup
        group_attr = group_attr[index]
        item_attr  = group_attr.inputTargetItem[6000]
        return (
            item_attr.inputGeomTarget,
            item_attr.inputPointsTarget,
            item_attr.inputComponentsTarget,
        )

    # --- target operations

    def get_target_count(self) -> int:
        """Returns the number of targets"""
        weight = self.find_attr("weight")
        return type(weight).get_element_count(weight)

    def get_targets(self) -> list[str]:
        """Returns a list of targets in this node."""
        return cmds.listAttr(_name_of(self), string="weight", multi=True)

    def get_target_indices(self) -> list[int]:
        """Returns a list of target index in this node."""
        return self.find_attr("weight").get_logical_indices()

    def get_target_index(self, target_name: str, failfast: bool = False) -> int:
        """Returns the logical index of a target, or -1 if not found.

        Args:
            target_name: A target name to work with.
            failfast: If True, raise error if target not found.
        """
        target_attr = self.find_alias(target_name, quiet=True)
        if target_attr is not None:
            return target_attr.get_logical_index()
        if failfast:
            raise RuntimeError(f"Target {_name_of(self)}.{target_name} not found.")
        return -1

    def get_target_name(self, target_id: int) -> str | None:
        """Returns the name a target, or None if not found.

        Args:
            target_id: A target index to work with.
        """
        weight = self.get_target_weight_attr(target_id)
        return type(weight).get_alias(weight)

    def set_target_name(self, target: int | str, name: str) -> None:
        """Sets the name of a target.

        Args:
            target: A target name or index.
            name: New name.
        """
        if not isinstance(target, int):
            target_id = self.get_target_index(target, failfast=True)
        else:
            target_id = target
        weight = self.get_target_weight_attr(target_id)
        type(weight).set_alias(weight, name)

    def get_target_weight_attr(self, target: str | int) -> Attribute:
        """Returns the weight attribute a target.
        Using index is faster than name.

        Args:
            target: A target name or index.
        """
        if not isinstance(target, int):
            target_id = self.get_target_index(target, failfast=True)
        else:
            target_id = target
        return self.find_attr("weight")[target_id]

    def get_target_weight(self, target: str | int) -> float:
        """Returns the weight of a target.
        Using index is faster than name.

        Args:
            target: A target name or index.
        """
        return self.get_target_weight_attr(target).get()

    def set_target_weight(self, target: str | int, val: float) -> None:
        """Sets the weight of a target.
        Using index is faster than name.

        Args:
            target: A target name or index.
            val: Weight value.
        """
        return self.get_target_weight_attr(target).set(val)

    def add_empty_target(self, target_name: str) -> int:
        """Adds a new empty target.

        Args:
            target_name: The new target to add.

        Raises:
            RuntimeError: If target already exists.

        Returns:
            The target index.
        """
        if not type(self)._raw_get_fn_set(self).findAlias(target_name).isNull():
            raise RuntimeError(f"Target {_name_of(self)}.{target_name} already exists.")

        weight_attr            = self.find_attr("weight")
        index                  = weight_attr.get_next_available_index()
        weight_attr            = weight_attr[index]
        type(weight_attr).set_alias(weight_attr, target_name)
        type(weight_attr).set_keyable(weight_attr, True)
        weight_attr.set(0)
        return index

    def rebuild_target(
        self,
        target: int | str | MorphData,
        name:   str | None            = None,
    ) -> Transform:
        """Rebuilds a geometry from data stored in a given target."""
        from cgmath.geometry import MorphData

        target_id = -1
        if isinstance(target, str):
            target_id = self.get_target_index(target, failfast=True)
        elif isinstance(target, int):
            target_id = target

        # duplicates the base mesh
        geom        = self.get_geometries()[0].get_parent()
        target_mesh = geom.duplicate_geometry(name=name)

        # compute offsetted points
        org_mesh_data = self.get_original_geometries()[0].serialize(
            world_space=False, include_uvs=False
        )
        points      = org_mesh_data.points
        target_data = self.get_target_data(target_id, original_mesh_data=org_mesh_data)
        points[target_data.indices] += target_data.offsets

        # apply points
        shape = target_mesh.get_children(shapes=True)[0]
        shape.set_points(points, world_space=False)

        return target_mesh

    # --- serialization

    def get_target_data(
        self,
        target:             str        | int,
        tolerance:          float      | None = None,
        neighbors:          np.ndarray | None = None,
        original_mesh_data: MeshData   | None = None,
    ) -> MorphData:
        """Serialize a target.
        Using index is faster than name.

        Args:
            target: A target name or index.
            tolerance: Tolerance for removing tiny offsets.
                If None use MorphData's default tolerance.
            neighbors: a numpy array of vertex neighboring data.
                used for pruning offsets.
            original_mesh_data: the original(pre-deformation) mesh data of the
                source mesh this blendshape is applied to, used for computing target
                data from a live mesh. If None, serialize a fresh mesh data (slower), and
                if no original mesh data is provided, serialize directly from the blendshape node.

        Returns:
            A `MorphData` object.
        """
        from cgmath.geometry import MeshData, MorphData

        if not isinstance(target, int):
            target_id = self.get_target_index(target, failfast=True)
        else:
            target_id = target

        # get weight alias/target name and weight value
        weight_attr = self.get_target_weight_attr(target)
        target_name = type(weight_attr).get_alias(weight_attr)
        # get target attributes
        geom_attr, offset_attr, comp_attr = self.get_target_data_attrs(target_id)

        # if a live target mesh is connected, compute target data from mesh data
        # (the blendshape offsets attr is empty when a live mesh is connected)
        target_mesh = geom_attr.list_connections(
            source=True, destination=False, plugs=False
        )
        # read at once, never held: no handle of their nodes
        target_mesh = _connected_attrs(geom_attr, src=True, dst=False)
        if target_mesh:
            kwargs           = {"include_uvs": False, "world_space": False}
            target_mesh      = type(target_mesh[0])._raw_get_node(target_mesh[0])
            target_mesh_data = type(target_mesh).serialize(target_mesh, **kwargs)
            if not original_mesh_data:
                original_mesh_data = self.get_original_geometries()
                if original_mesh_data:
                    original = original_mesh_data[0]
                    original_mesh_data = type(original).serialize(original, **kwargs)

            if original_mesh_data:
                target_data = MorphData.from_mesh_data(
                    original_mesh_data, target_mesh_data, target_name=target_name
                )
                target_data.prune_offsets(tolerance=tolerance, neighbors=neighbors)
                return target_data

        data = MorphData(
            name    = target_name,
            indices = np.empty((0,), dtype=int),
            offsets = np.empty((0, 3), dtype=float),
        )

        comp_data = comp_attr.get_data_fn_set()
        if comp_data:
            offsets = offset_attr.get()
            if offsets:
                data.offsets = np.array(offsets)[:, :-1]
                indices      = []
                for i in range(comp_data.length()):
                    comp = OpenMaya.MFnSingleIndexedComponent(comp_data.get(i))
                    indices.extend(comp.getElements())
                data.indices = np.array(indices)

                # remove offsets below the given tolerance
                data.prune_offsets(tolerance=tolerance, neighbors=neighbors)

                # sort numerically
                # maya allows unsorted indices, but sorted indices is more optimal
                data.sort()

        return data

    def set_target_data(
        self,
        target:    str       | int | MorphData | MorphList | list[MorphData],
        data:      MorphData | tuple[ArrayLike, ArrayLike] | None            = None,
        *,
        tolerance: float     | None                                          = None,
        force:     bool                                                      = True,
    ) -> None:
        """Sets the offset data of a target, or of the targets some morphs
        name: the inverse of `get_target_data` and `serialize`::

            morph.set_target_data("smile", data)            # one target, by name
            morph.set_target_data(1, (indices, offsets))    # one target, by index
            morph.set_target_data(data)                     # a MorphData: the target of its name
            morph.set_target_data(other.serialize())        # a MorphList: each onto the target of its name

        Using index is faster than name.

        Args:
            target: A target name or index to set, with `data`. Or, with no
                `data`, what `get_target_data` or `serialize` returns: a
                MorphData (set onto the target of its name), a MorphList or a
                list of MorphData (each onto the target of its name).
            data: For a target name or index, a MorphData object (its own name
                is not read) or (indices, offsets).
            tolerance: Tolerance for removing tiny offsets.
                If None use MorphData's default tolerance.
            force: If True, a new target will be added if no target exists. If
                False, every name is checked before the first edit.

        The data is copied: the caller's object is never changed. One undo step,
        named ``rig.BlendShape.set_target_data``, however many targets it sets.

        Raises:
            TypeError: before any edit: `data` given with a MorphData or a
                MorphList, a target name or index without `data`, a MorphList or
                a list of MorphData as its `data`, a list of anything but
                MorphData.
            ValueError: before any edit: a morph with no name to set it by, or
                a new target's name Maya refuses: a space, a symbol, a digit
                first, an attribute of the blendShape. ``weight[1]`` is one:
                `serialize`'s name for a weight element that has no name.
            RuntimeError: before any edit: with ``force=False``, a name no
                target has.
        """
        morphs = _target_morphs(target, data)
        new    = []
        for name, morph in morphs:
            # remove tiny offsets first
            morph.prune_offsets(tolerance=tolerance)
            if not isinstance(name, int) and self.get_target_index(name, failfast=not force) < 0:
                new.append(name)
        _check_target_names("BlendShape.set_target_data", new, type(self)._raw_get_fn_set(self))

        with _undo_chunk("rig.BlendShape.set_target_data"):
            for name, morph in morphs:
                self._set_target_morph(name, morph)

    def _set_target_morph(self, target: str | int, data: MorphData) -> None:
        """Writes the offsets of `data` to a target by name or index, added when
        the name has none (`set_target_data`, which has checked ``force=``)."""
        if not isinstance(target, int):
            target_id = self.get_target_index(target)

            # if the shape doesn't exist, add a new entry
            if target_id < 0:
                target_id = self.add_empty_target(target)

        else:
            target_id = target

        _, offset_attr, comp_attr = self.get_target_data_attrs(target_id)
        offset_attr.set(len(data.offsets), *data.offsets.tolist(), type="pointArray")

        # convert indices to range strings
        #
        # this reduces the number of indices in the component list attr. it makes
        # all related operations much more efficient, also smaller file size and memory
        # footprint.
        comps = list(iter_component_ranges("vtx", data.indices))
        comp_attr.set(len(comps), *comps, type="componentList")

    def serialize(self, match_name: list[str] | str | None = None) -> MorphList:
        """Serialize this blendshape.

        Args:
            match_name: One or more of target names or regex to match.
                If None, use all targets.

        Returns:
            A list of `MorphData` objects.
        """
        from cgmath.geometry import MorphList

        names       = [match_name] if isinstance(match_name, str) else match_name
        regex       = re.compile(r"|".join([rf"^{_name_of(x)}$" for x in names])) if names else None
        weight_attr = self.find_attr("weight")
        org_mesh    = self.get_original_geometries()

        # if there is an original mesh, use it to compute target data, in its
        # object space, as get_target_data reads a live target (a moved base
        # would otherwise offset every target by its move)
        if org_mesh:
            org_mesh = org_mesh[0].serialize(include_uvs=False, world_space=False)

        target_data_list = []
        for i in self.get_target_indices():
            if regex:
                target      = weight_attr[i]
                target_name = type(target).get_alias(target)
                if not regex.match(target_name):
                    continue

            target_data_list.append(
                self.get_target_data(i, original_mesh_data=org_mesh)
            )

        return MorphList(target_data_list)

    # --- files

    def save(self, path: str | os.PathLike, *, match_name: list[str] | str | None = None) -> str:
        """Saves this blendShape's targets to a cgmath file: its MorphList
        (`serialize`), the inverse of `load`.

        Args:
            path: A str or os.PathLike (``~`` expanded). The extension sets the
                format: ``.npz`` (recommended), ``.json`` or ``.pkl`` (never
                load a ``.pkl`` you did not write).
            match_name: The targets to save, as `serialize` takes them. None:
                every target.

        Returns:
            `path`, as written (``~`` expanded).

        Each target's sparse offsets, its name and the order are saved; its
        weight, its in-betweens and a live target mesh's connection are not (a
        MorphData holds none of them). A read of the scene: no undo step, the
        selection unchanged. The path is checked before anything is serialized
        (an extension cgmath does not save: ValueError; a missing folder:
        FileNotFoundError).
        """
        path = _files.check_save_path(path, "BlendShape.save")
        self.serialize(match_name=match_name).save(path)
        return path

    @_statement
    def _load_new(cls, targets, geometry, **kwargs):
        _files.check_order("BlendShape.load(targets, geometry)", targets, geometry)
        data = _read_targets("BlendShape.load", targets)
        with _undo_chunk("rig.BlendShape.load"):
            return cls.create(data, geometry, **kwargs)

    @_statement
    def _load_into(self, targets, *args, **kwargs):
        method = f"BlendShape({_name_of(self)!r}).load"
        if args or kwargs:
            got = [type(a).__name__ for a in args] + [f"{k}=" for k in kwargs]
            raise TypeError(
                f"{method} updates its targets in place from the file alone (got {', '.join(got)} "
                f"too); BlendShape.load(path, geometry, ...) makes a new blendShape"
            )
        data = _read_targets(method, targets)
        new  = [t.name for t in data if self.get_target_index(t.name) < 0]
        _check_target_names(method, new, type(self)._raw_get_fn_set(self))
        with _undo_chunk("rig.BlendShape.load"):
            self._free_live_targets(method, [t.name for t in data])
            self.set_target_data(data)
        return self

    def _free_live_targets(self, method: str, names: list) -> None:
        """Disconnects the live target meshes of the targets ``names`` (each
        ``inputGeomTarget`` of their items, in-betweens included, on every base) that
        a load replaces: a live mesh would write its own offsets over the file's at the
        next evaluation. Logged at INFO; the load's chunk holds the disconnects."""
        from rig._internal.plug import Plug, _disconnect_incoming

        fn      = OpenMaya.MFnDependencyNode(type(self)._raw_get_mobject(self))
        inputs  = fn.findPlug("inputTarget", False)
        indices = [i for i in (self.get_target_index(n) for n in names) if i >= 0]
        if not indices:
            return
        for g in inputs.getExistingArrayAttributeIndices():
            groups = inputs.elementByLogicalIndex(g).child(fn.attribute("inputTargetGroup"))
            for t in indices:
                items = groups.elementByLogicalIndex(t).child(fn.attribute("inputTargetItem"))
                for i in items.getExistingArrayAttributeIndices():
                    live = items.elementByLogicalIndex(i).child(fn.attribute("inputGeomTarget"))
                    if not live.isDestination:
                        continue
                    plug = Plug(live)
                    LOGGER.info(
                        "%s: target %r: disconnecting its live mesh %s (the file's offsets "
                        "replace it)", method, self.get_target_name(t), live.source().name(),
                    )
                    _disconnect_incoming(plug)

    load = _files.creates_or_updates(_load_new, _load_into, _BLENDSHAPE_LOAD_DOC)
    del _load_new, _load_into


def _read_targets(method: str, targets):
    """`BlendShape.load`'s targets, read and checked (`_files.read`) before any
    edit, each with a name: a MorphList (`set_target_data` copies it, so the
    caller's object is never changed)."""
    from cgmath.geometry import MorphData, MorphList

    data = _files.read(targets, (MorphList, MorphData), method, "the targets")
    data = MorphList([data]) if isinstance(data, MorphData) else data
    _check_morph_names(method, data)
    return data


def _check_morph_names(method: str, morphs) -> None:
    """A morph with no name has no target to go to: a ValueError from `method`,
    read before any edit."""
    for i, morph in enumerate(morphs):
        if not morph.name:
            raise ValueError(
                f"{method}: the morph at position {i} has no name, so no target to go to: "
                f"name it, or give its target: set_target_data(target, morph)"
            )


def _check_target_names(method: str, names: list[str], attrs) -> None:
    """Each of `names` is a new target's: a name Maya would refuse for its alias is
    a ValueError from `method`, read before any edit (the refusal comes after the
    edits of the targets before it otherwise, and `BlendShape.create` leaves its
    node behind). `attrs` answers ``hasAttribute(name)``: the blendShape's fn set,
    or its MNodeClass before it exists."""
    for name in names:
        if _NOT_AN_ALIAS.search(name) or attrs.hasAttribute(name):
            raise ValueError(
                f"{method}: a new target cannot be named {name!r}: Maya takes no space, no symbol "
                f"but _ and -, no digit or - first and no attribute of a blendShape"
            )


def _target_morphs(target, data) -> list[tuple[str | int, MorphData]]:
    """`BlendShape.set_target_data`'s arguments, checked before any edit: a
    (target name or index, MorphData) pair for each target to set, the MorphData
    a copy of the caller's data."""
    from cgmath.geometry import MorphData, MorphList

    method = "BlendShape.set_target_data"
    if isinstance(target, (MorphData, MorphList, list)):
        if data is not None:
            raise TypeError(
                f"{method}: a {type(target).__name__} goes alone, onto the targets of its names; "
                f"data= goes with a target name or index: set_target_data(target, data)"
            )
        morphs = [target] if isinstance(target, MorphData) else list(target)
        for morph in morphs:
            if not isinstance(morph, MorphData):
                raise TypeError(f"{method}: a list of MorphData holds no {type(morph).__name__}")
        _check_morph_names(method, morphs)
        return [(morph.name, morph.copy()) for morph in morphs]

    if data is None:
        raise TypeError(
            f"{method}({target!r}): a target name or index takes its data: "
            f"set_target_data(target, data); a MorphData, a MorphList or a list of MorphData "
            f"goes alone: set_target_data(morphs)"
        )
    if isinstance(data, MorphData):
        return [(target, data.copy())]
    if isinstance(data, MorphList) or (isinstance(data, (list, tuple)) and any(isinstance(d, MorphData) for d in data)):
        raise TypeError(
            f"{method}({target!r}, ...): a target name or index takes one MorphData or "
            f"(indices, offsets); a MorphList or a list of MorphData goes alone: set_target_data(morphs)"
        )
    return [(target, MorphData(name="name", indices=np.array(data[0]), offsets=np.array(data[1])))]


# the public names (user decision 2026-10-17: import leaks hidden)
__all__ = [
    "BlendShape",
]

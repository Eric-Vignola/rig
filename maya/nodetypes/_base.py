"""
Node base classes and utils
"""

from __future__ import annotations

import uuid
from typing import Any

from maya import cmds
from maya.api import OpenMaya
from rig.maya.attribute import Attribute


CUSTOM_TYPE_ATTR = "__custom_node_type__"


def is_valid_maya_uid(uid_string: str) -> bool:
    """
    Validates if a string matches Maya's unique ID format using the uuid module.
    """
    try:
        uuid.UUID(uid_string)
        return True
    except ValueError:
        return False


class NodeMeta(type):
    """
    A metaclass that register node classes to the cached dict in PyNode.
    """

    def __new__(mcs, class_name, bases, attrs):
        cls_obj   = type.__new__(mcs, class_name, bases, attrs)

        node_type = attrs.get("CUSTOM_NODE_TYPE")
        if not node_type:
            node_type = attrs.get("NATIVE_NODE_TYPE")
        if node_type:
            PyNode._NODE_CLASS_DICT[node_type] = cls_obj

        return cls_obj


class PyNode:
    """
    A factory class that for casting arbitrary objects into the most suitable
    object classes.

    **If you know the type of the object you are working with, it is recommended to
    explicitly cast it into the associated class instead of going through the `PyNode`
    factory process, which yields better performance.**
    """

    _NODE_CLASS_DICT = {}

    def __new__(
        cls,
        obj: str | OpenMaya.MObject | OpenMaya.MDagPath | OpenMaya.MPlug,
        *args,
        **kwargs,
    ) -> Any:
        """Cast a object (node or attr) into the most suitable class.

        Args:
            obj: Node: name string, MObject, or MDagPath.
                Attribute: name string or MPlug.

        Returns:
            An object instance.
        """
        super(PyNode, cls).__new__(cls, *args, **kwargs)

        if isinstance(obj.__class__, NodeMeta) or isinstance(obj, Attribute):
            return obj
        elif isinstance(obj, OpenMaya.MPlug):
            return Attribute(obj)
        elif isinstance(obj, OpenMaya.MObject):
            obj = _mobject_to_str(obj)
        elif isinstance(obj, OpenMaya.MDagPath):
            obj = obj.partialPathName()
        elif not isinstance(obj, str):
            t = type(obj)
            raise ValueError(f"{obj} ({t}) is not a str, MObject, MDagPath, or MPlug")

        # if obj is given as a unique id, convert to string
        if is_valid_maya_uid(obj):
            str_from_uid = cmds.ls(obj, uid=True)
            if not str_from_uid:
                raise TypeError(f"No object matches uuid: {obj}.")

            # node name is properly converted to a name string
            obj = str_from_uid[0]

        # TODO need a better way to identify attribute strings
        if obj.rfind(".") != -1:
            return Attribute(obj)

        cls_obj     = None
        custom_type = get_custom_type(obj)
        if custom_type:
            cls_obj = cls._NODE_CLASS_DICT.get(custom_type)
        if not cls_obj:
            # set default node type to DAG or DG
            default_type = "dagNode" if cmds.ls(obj, dag=True) else "entity"
            cls_obj      = cls._NODE_CLASS_DICT.get(default_type)

            # override cls_obj with a defined node class if any
            for t in reversed(cmds.nodeType(obj, inherited=True)):
                if t in cls._NODE_CLASS_DICT:
                    cls_obj = cls._NODE_CLASS_DICT.get(t)
                    break

        if cls_obj:
            return cls_obj(obj)

        raise ValueError(f"Failed casting {obj}")

    @classmethod
    def create(cls, node_type, *args, **kwargs) -> Any:
        """Thin wrapper around cmds.createNode()."""
        node_cls = cls._NODE_CLASS_DICT.get(node_type)
        if node_cls:
            return node_cls.create(*args, **kwargs)
        else:
            result = cmds.createNode(node_type, *args, **kwargs)
            if isinstance(result, (list, tuple)):
                return [cls(x) for x in result]
            elif result:
                return cls(result)
            return result

    @classmethod
    def find_all(cls, node_type: str, exact_type: bool = True) -> list[Any]:
        """Thin wrapper around cmds.ls()."""
        node_cls = cls._NODE_CLASS_DICT.get(node_type)
        if node_cls:
            return node_cls.find_all(exact_type=exact_type)
        else:
            raise NotImplementedError(f"Node type {node_type} not implemented")


def _mobject_to_str(mobject: OpenMaya.MObject) -> str:
    """Converts an node MObject to a name string."""
    if mobject.isNull():
        raise ValueError("MObject is null.")
    elif mobject.apiType() in (
        OpenMaya.MFn.kWorld,
        OpenMaya.MFn.kInvalid,
        OpenMaya.MFn.kUnknown,
    ):
        raise ValueError("Invalid MObject API Type: " + mobject.apiTypeStr())
    elif mobject.hasFn(OpenMaya.MFn.kDagNode):
        dag_path = OpenMaya.MDagPath.getAPathTo(mobject)
        return dag_path.partialPathName()
    elif mobject.hasFn(OpenMaya.MFn.kDependencyNode):
        return OpenMaya.MFnDependencyNode(mobject).name()
    else:
        raise ValueError("MObject is not a node.")


def get_custom_type(node_name: str) -> str | None:
    """Returns the value of the custom type attr, if exists."""
    if cmds.attributeQuery(CUSTOM_TYPE_ATTR, node=node_name, exists=True):
        return cmds.getAttr(f"{node_name}.{CUSTOM_TYPE_ATTR}")


def set_custom_type(node_name: str, custom_type: str) -> None:
    """Sets the custom type of a given node."""
    attr = f"{node_name}.{CUSTOM_TYPE_ATTR}"
    if cmds.attributeQuery(CUSTOM_TYPE_ATTR, node=node_name, exists=True):
        cmds.deleteAttr(attr)
    cmds.addAttr(node_name, ln=CUSTOM_TYPE_ATTR, dt="string")
    cmds.setAttr(attr, custom_type, type="string")
    cmds.setAttr(attr, lock=True)
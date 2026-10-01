"""
The node lookup errors: one family for a name that does not give one node.

::

    NodeLookupError(LookupError, TypeError, ValueError)   no single node: raised as is for a pattern ('red*')
        NodeNotFoundError                                    no node has the name (or the uuid)
        AmbiguousNodeError                                   several do: DAG paths, or ':x' and ':char:x'
    NodeTypeError(TypeError, ValueError)                   the node is of another type than the class names

Every class is a TypeError and a ValueError (and the lookup ones a LookupError),
so an ``except TypeError`` / ``except ValueError`` written for what ``Node(x)``
and the node classes raised before catches them too.

An error keeps its facts as attributes (``name``, the class ``label`` such as
``"joint"``, the ``candidates``) and builds its message when it is first
printed. The hints read the scene then, each in its own ``try``: "did you mean
'spine_01'?" (the closest leaf names among the first ones ``cmds.ls`` lists,
about 500: of the class's node type for a class's reference, so
``Joint('spnie_01')`` compares joints only, else of every node),
"'char:root' exists" (the same leaf in another namespace) and "'inner_k' exists
(the scope prefix)" (the name a create inside the active flattened
``with container()`` scope gives). A node class's miss then names the
``define`` that makes the node ("; Joint.define('spnie_01') finds or makes
it"), when that makes it from the name. A hint that cannot be read (the scene
was closed or replaced before the print) is left out, so printing an error
never raises, and a caller that discards the error (``Node.wrap``,
``SkinCluster`` influences) never pays for them.

Usage::

    from rig import Node, NodeNotFoundError
    from rig.nodetypes import Joint

    try:
        Node("spnie_01")
    except NodeNotFoundError as error:
        print(error)          # no node named 'spnie_01' (did you mean 'spine_01'?)
        print(error.name)     # spnie_01
    try:
        Joint("spnie_01")
    except NodeNotFoundError as error:
        print(error)          # no joint named 'spnie_01' (did you mean 'spine_01'?); Joint.define('spnie_01') finds or makes it
"""

from __future__ import annotations

import difflib
from typing import Callable, Iterable

from maya import cmds


# ``rig._internal.container`` sets it when it loads (the D31 pattern keeps
# nodetypes free of ``rig._internal`` imports): ``name -> str | None``, the name
# a create inside the active flattened ``with container()`` scope gives
# ``name`` (the scope prefix on its leaf), None outside one. Read when a
# NodeNotFoundError is raised, since the scope may be gone when it is printed.
_SCOPE_HINT_HOOK = None

# the "did you mean" hint compares the missing leaf with the first names
# ``cmds.ls`` lists, about 500 of them: ``cmds.ls(type=..., head=500)`` for a
# class's miss (its node type), ``cmds.ls(head=1000)`` for any other, whose
# ``head`` also counts the ~470 default nodes ``ls`` does not list. A print
# then costs a few ms at 30,000 nodes (``cmds.ls()`` and difflib over it took
# 125 ms there).
_HINT_HEAD_TYPED = 500
_HINT_HEAD       = 1000

# how many candidates a message lists before "..."
_SHOWN = 10


def _leaf(name: str) -> str:
    """The node name without its path and namespace (``|g|ns:a`` is ``a``)."""
    return name.rsplit("|", 1)[-1].rsplit(":", 1)[-1]


def _namespace(name: str) -> str:
    """The namespace a node name is spelled in, ``""`` for the root
    (``|g|ns:a`` is ``ns``; ``:a`` and ``a`` are ``""``)."""
    return name.rsplit("|", 1)[-1].rpartition(":")[0].strip(":")


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _quoted(names: Iterable[str]) -> str:
    """``'a'``, ``'a' or 'b'``, ``'a', 'b' or 'c'``."""
    names = [repr(n) for n in names]
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} or {names[-1]}"


def _is_pattern(name: str) -> bool:
    return any(c in name for c in "*?[]")


class NodeLookupError(LookupError, TypeError, ValueError):
    """A name that does not give one node. The base of :class:`NodeNotFoundError`
    and :class:`AmbiguousNodeError`, raised as is for a pattern (``'red*'``): a
    pattern is a search (``cmds.ls('red*')``, ``Transform.find_all()``), never a
    node's name.

    Attributes:
        name: the name as it was given.
        label: what the name was to be, ``"node"`` or a class's node type
            (``"joint"``).
    """

    def __init__(self, name: str = "", label: str = "node") -> None:
        super().__init__(name)
        self.name     = name
        self.label    = label
        self._message = None

    def __str__(self) -> str:
        if self._message is None:
            hints = []
            for hint in self._hints():
                try:
                    text = hint()
                except Exception:  # noqa: BLE001 -- the scene changed or closed: no hint
                    text = None
                if text:
                    hints.append(text)
            text = self._text()
            self._message = f"{text} ({'; '.join(hints)})" if hints else text
        return self._message

    def _text(self) -> str:
        if _is_pattern(self.name):
            return (
                f"{self.name!r} is a pattern, not {_article(self.label)} {self.label} name; "
                f"a pattern is a search: cmds.ls({self.name!r}), or Transform.find_all() "
                f"for the nodes of a class"
            )
        return f"{self.name!r} names no single {self.label}"

    def _hints(self) -> tuple[Callable[[], str | None], ...]:
        return ()


class NodeNotFoundError(NodeLookupError):
    """No node has the name (``uuid=True``: the uuid). The message is
    ``no joint named 'spnie_01'`` with its hints (did you mean, another
    namespace, the scope prefix), or ``no node has the uuid '...'``. A node
    class's reference ends it with the call that finds or makes the node
    (``door``), read when it is printed, as the hints are:
    ``...; Joint.define('spnie_01') finds or makes it``."""

    # the node class whose reference raised the error (``Joint``), set by the
    # reference; None for ``Node(x)`` and an error made directly
    node_class = None

    def __init__(self, name: str = "", label: str = "node", uuid: bool = False) -> None:
        super().__init__(name, label)
        self.uuid = uuid
        # the flattened scope's spelling of the name, read now (see _SCOPE_HINT_HOOK)
        self.scope_name = None
        hook = _SCOPE_HINT_HOOK
        if hook is not None and not uuid and name and "|" not in name:
            try:
                self.scope_name = hook(name)
            except Exception:  # noqa: BLE001 -- a hint must not fail the raise
                pass

    def __str__(self) -> str:
        if self._message is None:
            text = super().__str__()
            door = self.door
            self._message = f"{text}; {door} finds or makes it" if door else text
        return self._message

    @property
    def door(self) -> str | None:
        """The call that finds or makes the missing node
        (``"Joint.define('spnie_01')"``), read from the scene now: the
        reference's class's ``define`` when it makes the node from the name
        (``DGNode._define_door``); None for a uuid, ``Node(x)``, an error made
        directly, or a scene that cannot be read."""
        if self.node_class is None or self.uuid:
            return None
        try:
            return self.node_class._define_door(self.name)
        except Exception:  # noqa: BLE001 -- the scene changed or closed: no door
            return None

    def _text(self) -> str:
        if self.uuid:
            return f"no node has the uuid {self.name!r}"
        return f"no {self.label} named {self.name!r}"

    def _hints(self) -> tuple[Callable[[], str | None], ...]:
        if self.uuid or not self.name:
            return ()
        return (self._scope_hint, self._namespace_hint, self._close_name_hint)

    def _scope_hint(self) -> str | None:
        scoped = self.scope_name
        if scoped and scoped != self.name and cmds.objExists(scoped):
            return f"{scoped!r} exists (the scope prefix)"
        return None

    def _namespace_hint(self) -> str | None:
        """The nodes of the same leaf name in any namespace."""
        found = [n for n in cmds.ls(_leaf(self.name), recursive=True) or [] if n != self.name]
        if not found:
            return None
        shown = ", ".join(repr(n) for n in found[:3]) + (", ..." if len(found) > 3 else "")
        return f"{shown} {'exists' if len(found) == 1 else 'exist'}"

    def _close_name_hint(self) -> str | None:
        """The scene names whose leaf is closest to the missing leaf, among
        the first ones ``cmds.ls`` lists (``_HINT_HEAD_TYPED``,
        ``_HINT_HEAD``): of the node type the class's reference takes
        (``node_class._hint_type()``), else of every node. Of the names that
        share a leaf, the one in the missing name's own namespace is shown
        (``'spine_01'`` rather than a reference's ``'char:spine_01'``)."""
        leaf      = _leaf(self.name)
        home      = _namespace(self.name)
        node_type = None if self.node_class is None else self.node_class._hint_type()
        if node_type:
            names = cmds.ls(type=node_type, head=_HINT_HEAD_TYPED)
        else:
            names = cmds.ls(head=_HINT_HEAD)
        by_leaf = {}
        for name in names or []:
            key = _leaf(name)
            if key not in by_leaf or _namespace(name) == home != _namespace(by_leaf[key]):
                by_leaf[key] = name
        by_leaf.pop(leaf, None)  # the same leaf elsewhere is the namespace hint
        close = difflib.get_close_matches(leaf, list(by_leaf), n=3, cutoff=0.75)
        return f"did you mean {_quoted(by_leaf[c] for c in close)}?" if close else None


class AmbiguousNodeError(NodeLookupError):
    """Several nodes have the name: DAG nodes under different parents
    (``candidates`` are their full paths; "use a path"), or a bare name at the
    root namespace and in the current one (``namespaces=True``, ``candidates``
    ``[':x', ':char:x']``; "spell the namespace"). An instanced node is one
    node. ``define`` raises it too, with its own ``message``, when the node it
    would make shares its name with nodes elsewhere (``'root' exists at
    |char_grp|root``): the name would then be ambiguous."""

    def __init__(
        self,
        name: str = "",
        candidates: Iterable[str] = (),
        label: str = "node",
        namespaces: bool = False,
        message: str | None = None,
    ) -> None:
        super().__init__(name, label)
        self.candidates = list(candidates)
        self.namespaces = namespaces
        self._text_given = message

    def _text(self) -> str:
        if self._text_given:
            return self._text_given
        if self.namespaces:
            return (
                f"{self.name!r} is ambiguous: it names {' and '.join(self.candidates)}; "
                f"spell the namespace"
            )
        shown = ", ".join(self.candidates[:_SHOWN])
        if len(self.candidates) > _SHOWN:
            shown += ", ..."
        return (
            f"{self.name!r} is ambiguous: it names {len(self.candidates)} nodes: {shown}; "
            f"use a path"
        )


class NodeTypeError(TypeError, ValueError):
    """The node a name gives is of another type than the class names:
    ``'grp' is a transform, not a joint`` and an optional ``hint`` (the
    spelling that works).

    ``define`` raises it with its own ``message`` when Maya would give the
    node it makes another name than its key: a DG node holds the name
    (``'knob' is taken by a multiplyDivide``), or the made node came out
    renamed.

    Attributes:
        name: the name as it was given.
        label: the type the class names (``"joint"``).
        node_type: the node's type (``"transform"``).
        hint: the text after the ``;``, or ``""``.
    """

    def __init__(
        self,
        name: str = "",
        label: str = "node",
        node_type: str = "",
        hint: str = "",
        message: str | None = None,
    ) -> None:
        super().__init__(name)
        self.name      = name
        self.label     = label
        self.node_type = node_type
        self.hint      = hint
        self._message  = message

    def __str__(self) -> str:
        if self._message:
            return self._message
        text = (
            f"{self.name!r} is {_article(self.node_type)} {self.node_type}, "
            f"not {_article(self.label)} {self.label}"
        )
        return f"{text}; {self.hint}" if self.hint else text


__all__ = ["AmbiguousNodeError", "NodeLookupError", "NodeNotFoundError", "NodeTypeError"]

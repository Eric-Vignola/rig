"""
Animation curve node classes: ``AnimCurve`` (any curve) and ``AnimCurveTL`` / ``TA`` / ``TU``
/ ``TT`` (one per timed curve type).

``AnimCurve`` is a node class that also makes keys, as ``Material`` is a node class that a
node on the right of ``<<`` assigns:

* ``AnimCurve("pCube1_translateX")`` refers to any curve (the class of every curve type that
  has no class of its own: the driven-key ``animCurveU*`` types for now), ``AnimCurveTL("x")``
  to exactly that type;
* ``AnimCurve([0, 1, 0], times=[0, 12, 24], ...)``, a call whose first argument is a sequence,
  returns KEYS, a value that writes nothing until ``plug << keys`` keys a plug (see
  `rig._internal.anim_keys`). So ``isinstance(AnimCurve([0]), AnimCurve)`` is False: the call
  made keys, not a node. A number, or no argument at all, is a TypeError naming both forms;
  on a typed class, ``AnimCurveTL([...])`` is a TypeError naming ``AnimCurve([...])`` and
  ``AnimCurveTL.create([...])``: the type of a curve follows the plug it keys.

The typed classes make curves: ``AnimCurveTU.create([0, 1, 0], name="blink")`` always makes a
new unconnected one, ``AnimCurveTU.define("blink", [0, 1, 0])`` finds or makes it (a found one
gets its keys replaced, ``update=True``). ``curve.get_keys()`` reads its keys back as a value
(``other.tx << curve.get_keys()`` copies the animation), ``curve.set_keys(...)`` replaces
them, ``AnimCurve.of(x)`` lists the curves driving ``x``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from maya.api import OpenMaya, OpenMayaAnim

from rig._internal.journal import cmds_edit, statement
from rig.nodetypes._base import _check_attrs, _refuse_locked_node, _type_label
from rig.nodetypes.dg_node import (
    _ABSTRACT_TYPES,
    _attribute_keywords,
    _define,
    _DEFINE_OWN,
    DGNode,
)
from rig.nodetypes.errors import _article

_FN = OpenMayaAnim.MFnAnimCurve

# the curve types rig keys by time (a driven key's animCurveU* is keyed by its driver)
_TIMED = {
    _FN.kAnimCurveTA: "animCurveTA",
    _FN.kAnimCurveTL: "animCurveTL",
    _FN.kAnimCurveTT: "animCurveTT",
    _FN.kAnimCurveTU: "animCurveTU",
}


def _anim_keys():
    from rig._internal import anim_keys  # the plug module imports this package

    return anim_keys


def _split_options(kwargs: dict) -> dict:
    """The key options of a create / define / set_keys call, taken out of ``kwargs`` (what
    is left: the curve's Maya attributes and the command's flags)."""
    options = _anim_keys().OPTIONS
    return {key: kwargs.pop(key) for key in list(kwargs) if key in options}


class AnimCurve(DGNode):
    """Any animation curve, and the keys maker.

    ``AnimCurve("x")`` refers to the curve ``x`` (any type). ``AnimCurve(values, times=None,
    *, tangents=None, in_tangents=None, out_tangents=None, slopes=None, in_slopes=None,
    out_slopes=None, weighted=False, in_weights=None, out_weights=None, infinity="constant",
    pre_infinity=None, post_infinity=None)`` returns keys for ``plug << ...``:

    * ``values``: (N,) or (N, C) numbers in UI units (degrees, the scene's linear unit,
      frames for a time curve), or (N,) enum field names; ``times``: (N,) frames in the
      scene's time unit, strictly increasing (None: 0, 1, 2 ...);
    * tangents: keyTangent's names (``linear`` ``step`` ``stepnext`` ``flat`` ``spline``
      ``clamped`` ``plateau`` ``auto`` ``fixed`` ...); None is linear, with a step out
      tangent on a bool or an enum (as Maya keys them);
    * slopes: value units per frame, making that side ``fixed`` (NaN: none for that key);
    * weights need ``weighted=True``; infinity: setInfinity's names;
    * every per-key option is one value for all keys, (N,) or (N, C).

    Values, times and options are checked when the keys are made (ValueError /
    TypeError); the rest (the channel count, the curve type, enum names, locks) at ``<<``.
    """

    NATIVE_NODE_TYPE = "animCurve"
    FN_SET = _FN

    # how a lookup error names AnimCurve's nodes
    _TYPE_LABEL = "animation curve"

    @classmethod
    def _KEYS_CALL(cls, args: tuple, kwargs: dict) -> Any:  # noqa: N802 (the _refer hook)
        """The class call's keys branch (`rig.nodetypes._base._refer`): keys for a
        sequence, a TypeError for a number or no argument, None for a name (the
        reference)."""
        first = args[0] if args else None
        anim_keys = _anim_keys()
        if args and anim_keys._is_keys_sequence(first):
            if cls is not AnimCurve:
                name = cls.__name__
                raise TypeError(
                    f"{name}([...]) refers to a curve by name: AnimCurve([...]) makes keys "
                    f"(their curve type follows the plug they key: plug << AnimCurve([...])), "
                    f"and {name}.create([...]) makes {_article(cls.NATIVE_NODE_TYPE)} "
                    f"{cls.NATIVE_NODE_TYPE} with them"
                )
            return anim_keys.keys(*args, **kwargs)
        if args and type(first) is anim_keys._Keys:
            if len(args) > 1 or kwargs:
                raise TypeError(
                    "AnimCurve(keys, ...): an AnimCurve value holds its own times and options; "
                    "give the values themselves to set them"
                )
            return first
        if cls is AnimCurve and (not args and not kwargs or isinstance(first, (bool, int, float, np.number))):
            got = "AnimCurve()" if not args else f"AnimCurve({first!r})"
            one = f"AnimCurve([{first!r}])" if args else "AnimCurve([0, 1, 0])"
            raise TypeError(
                f"{got} is no keys and names no curve: {one} makes keys (a sequence of "
                f"values), AnimCurve('x') refers to the curve x"
            )
        return None

    # --- the classes

    @classmethod
    def _mismatch_hint(cls, node: Any) -> str:
        """The reference's hint for a curve of another type: the generic reference."""
        if cls is AnimCurve or not isinstance(node, AnimCurve):
            return ""
        return f"; AnimCurve({node.name!r}) takes any curve"

    @classmethod
    def _abstract(cls, verb: str) -> TypeError:
        return TypeError(
            f"AnimCurve.{verb}() makes no curve: 'animCurve' is an abstract type; name "
            f"AnimCurveTL, AnimCurveTA, AnimCurveTU or AnimCurveTT (or key a plug: "
            f"plug << AnimCurve([...]) makes the curve its type takes)"
        )

    @classmethod
    @statement
    def create(
        cls,
        keys:      Any         = None,
        times:     Any         = None,
        *,
        name:      str | None  = None,
        parent:    Any         = None,
        container: bool | None = None,
        **kwargs:  Any,
    ) -> "AnimCurve":
        """Makes a new, unconnected curve of this class's type, always (Maya picks the final
        name: a second ``AnimCurveTU.create(name="blink")`` is ``blink1``).

        ``keys`` is the values (with ``times`` and the key options of ``AnimCurve(...)``) or
        an ``AnimCurve`` value (then no times or options: TypeError); None makes a curve with
        no key. Every other keyword is an attribute of the curve (``preInfinity=...``,
        ``curveColor=...``), checked before anything is made and set after the keys, so an
        attribute wins over the keys' ``infinity=``. The keys and the attributes are one undo
        step (``rig.create``); a value an attribute refuses deletes the new curve before its
        error propagates. Inside ``with container()`` it joins the scope as every typed
        create does. The selection is kept (``skipSelect=True`` unless given). Without rig's
        undo plug-in, keys raise RuntimeError before anything is made."""
        if cls is AnimCurve:
            raise cls._abstract("create")
        if parent is not None:
            raise TypeError(f"{cls.__name__}.create() takes no parent=: an animation curve is a DG node")
        call    = f"{cls.__name__}.create()"
        options = _split_options(kwargs)
        anim_keys = _anim_keys()
        value   = anim_keys.coerce(keys, times, options, call)
        entry   = None
        if value is not None:
            if value._names is not None:
                raise TypeError(
                    f"{call}: field names key an enum plug (plug << AnimCurve([...])); a curve "
                    f"made on its own takes numbers"
                )
            anim_keys._columns(value, 1, call)
            entry = anim_keys._plan_values(value, cls.NATIVE_NODE_TYPE, 0, False, call)
            anim_keys.require_plugin(call)
        flags = cls._CREATE_FLAGS
        attrs = {}
        if kwargs and not flags.issuperset(kwargs):
            attrs = _attribute_keywords(cls.NATIVE_NODE_TYPE, kwargs, flags, call)
        if "ss" not in kwargs and "skipSelect" not in kwargs:
            kwargs["skipSelect"] = True  # a curve never changes the selection
        if entry is None and not attrs:
            return DGNode.create.__func__(cls, name=name, container=container, **kwargs)
        from rig._internal.undo import _undo_chunk

        with _undo_chunk("rig.create"):
            node = DGNode.create.__func__(cls, name=name, container=container, **kwargs)
            try:
                if entry is not None:
                    anim_keys._write(node.fn_set, entry)
                for attr, v in attrs.items():
                    getattr(node, attr) << v
            except BaseException:
                try:
                    cmds_edit.delete(node.name)
                except (RuntimeError, ValueError):
                    pass
                raise
        return node

    @classmethod
    @statement
    def define(
        cls,
        name:      str,
        keys:      Any         = None,
        times:     Any         = None,
        *,
        parent:    Any         = None,
        update:    bool        = True,
        container: bool | None = None,
        **kwargs:  Any,
    ) -> "AnimCurve":
        """Finds the curve of this class at the key ``name`` names, or makes it there with
        the keys (``create``): ``AnimCurveTU.define("blink", [0, 1, 0], times=[0, 2, 4])``.
        A found curve gets its keys replaced (and its attributes set) when they differ,
        ``update=True`` (the script is the source of truth), in one undo step
        (``rig.define``): the same node and UUID. Keys it holds already write nothing.
        ``update=False`` leaves a found curve as it is. The key rules and refusals are
        ``DGNode.define``'s (a curve of another type at the key is a NodeTypeError naming
        ``AnimCurve('x')``); a found curve that is locked (lockNode) or from a referenced file
        refuses new keys (InjectionError)."""
        if cls is AnimCurve:
            raise cls._abstract("define")
        own = _DEFINE_OWN.intersection(kwargs)
        if own:
            raise TypeError(
                f"{cls.__name__}.define() takes its name first and the keys after it "
                f"(got {', '.join(f'{key}=' for key in sorted(own))})"
            )
        if parent is not None:
            raise TypeError(f"{cls.__name__}.define() takes no parent=: an animation curve is a DG node")
        call    = f"{cls.__name__}.define({name!r})"
        options = _split_options(kwargs)
        anim_keys = _anim_keys()
        value   = anim_keys.coerce(keys, times, options, call)
        if value is not None:
            if value._names is not None:
                raise TypeError(f"{call}: field names key an enum plug; a curve takes numbers")
            anim_keys._columns(value, 1, call)
            anim_keys._plan_values(value, cls.NATIVE_NODE_TYPE, 0, False, call)  # checked first
        flags = cls._CREATE_FLAGS
        attrs = {}
        if kwargs and not flags.issuperset(kwargs):
            attrs = _attribute_keywords(cls.NATIVE_NODE_TYPE, kwargs, flags, f"{cls.__name__}.define()")
        made = []

        def accept(found):
            return found if isinstance(found, cls) else cls._coerce(found)

        def make(made_name, _parent):
            node = cls.create(value, name=made_name, container=container, **kwargs, **attrs)
            made.append(node)
            return node

        node = _define(
            lambda args: f"{cls.__name__}.define({args})", cls.__name__, _type_label(cls), name, None,
            dag=False, update=update, attrs={}, accept=accept, make=make,
            aware=cls._CONTAINER_AWARE, joins=container, typed=True, hint=cls._mismatch_hint,
            plain=True,
        )
        if made or not update or (value is None and not attrs):
            return node
        if attrs:
            _check_attrs(attrs, node=node.name)
        entry = node._plan_set(value, call) if value is not None else None
        if (entry is None or entry.same) and not attrs:
            return node
        from rig._internal.undo import _undo_chunk

        with _undo_chunk("rig.define"):
            if entry is not None and not entry.same:
                anim_keys._write(node.fn_set, entry)
            for attr, v in attrs.items():
                getattr(node, attr) << v
        return node

    @classmethod
    def of(cls, x: Any) -> Any:
        """The curves directly driving ``x``, a ``List`` of curve nodes typed by class
        (``AnimCurveTL("pCube1_translateX")``), each once, empty when there are none: a plug
        (a compound reads every child, a multi every element), a node (every plug it has a
        curve on) or a ``List`` of them. A pairBlend or a unitConversion is not looked
        through: these are the curves ``<<`` would update. ``AnimCurveTL.of(x)`` keeps the
        curves of its type."""
        found = _anim_keys().curves_of(x)
        if cls is AnimCurve:
            return found
        from rig._internal.list import List

        return List(c for c in found if isinstance(c, cls))

    # --- the keys

    def _timed_type(self, verb: str) -> str:
        curve_type = _TIMED.get(self.fn_set.animCurveType)
        if curve_type is None:
            raise TypeError(
                f"{self.name!r} is {_article(self.node_type)} {self.node_type}, keyed by its "
                f"driver: {verb} reads and writes curves keyed by time (TL, TA, TU, TT)"
            )
        return curve_type

    def _plan_set(self, value: Any, call: str) -> Any:
        """The checked write of ``value`` into this curve (``entry.same``: it holds them)."""
        anim_keys  = _anim_keys()
        curve_type = self._timed_type(call)
        fn = self.fn_set
        _refuse_locked_node(fn, "set the keys of", self.name)
        if fn.isFromReferencedFile:
            from rig.nodetypes._base import _injection_error

            raise _injection_error(
                f"cannot set the keys of {self.name!r}: it is from a referenced file, whose keys "
                f"the file holds (key the plug: it gets a new curve). Nothing was changed"
            )
        if value._names is not None:
            raise TypeError(f"{call}: field names key an enum plug; a curve takes numbers")
        anim_keys._columns(value, 1, call)
        entry = anim_keys._plan_values(value, curve_type, 0, False, call)
        entry.same = anim_keys._same(anim_keys._read(fn, curve_type), entry)
        if not entry.same:
            anim_keys.require_plugin(call)
        return entry

    @statement
    def set_keys(self, keys: Any, times: Any = None, **options: Any) -> "AnimCurve":
        """Replaces every key of this curve (one undo step, ``rig.AnimCurve.set_keys``) and
        returns it: ``keys`` is the values (with ``times`` and the key options of
        ``AnimCurve(...)``) or an ``AnimCurve`` value. Keys it holds already write nothing.
        The values are in UI units. Refused (InjectionError, nothing changed): a locked
        (lockNode) curve, a curve from a referenced file."""
        call  = f"{type(self).__name__}({self.name!r}).set_keys()"
        value = _anim_keys().coerce(keys, times, options, call)
        if value is None:
            raise TypeError(f"{call} takes the keys: values, or an AnimCurve value")
        entry = self._plan_set(value, call)
        if entry.same:
            return self
        from rig._internal.undo import _undo_chunk

        with _undo_chunk("rig.AnimCurve.set_keys"):
            _anim_keys()._write(self.fn_set, entry)
        return self

    def get_keys(self) -> Any:
        """The keys of this curve as an ``AnimCurve`` value, in UI units: values and times,
        every key's in and out tangent type, the slopes where a side is fixed (NaN
        elsewhere), the weights of a weighted curve, the infinities. ``other.tx <<
        curve.get_keys()`` reproduces the curve (tangents that are not fixed recompute
        identically from the same keys). ValueError for a curve with no key."""
        return _anim_keys().read_keys(self.fn_set, self._timed_type("get_keys()"))

    @property
    def num_keys(self) -> int:
        """The number of keys."""
        return self.fn_set.numKeys

    def evaluate(self, times: Any) -> Any:
        """The curve's value at ``times`` (frames in the scene's time unit), in UI units
        (degrees, the linear unit, frames for a time curve): a float for one time, an
        ndarray of the shape of ``times`` for several."""
        curve_type = self._timed_type("evaluate()")
        fn   = self.fn_set
        unit = OpenMaya.MTime.uiUnit()
        arr  = np.asarray(times, dtype=float)
        if curve_type == "animCurveTT":
            out = [fn.evaluate(OpenMaya.MTime(t, unit)).asUnits(unit) for t in arr.ravel().tolist()]
        else:
            factor = _anim_keys()._factor(curve_type)
            out = [fn.evaluate(OpenMaya.MTime(t, unit)) / factor for t in arr.ravel().tolist()]
        result = np.array(out, dtype=float).reshape(arr.shape)
        return float(result) if result.ndim == 0 else result


class AnimCurveTL(AnimCurve):
    """An ``animCurveTL``: keys a distance (``tx``, ``Float(at="doubleLinear")``) by time."""

    NATIVE_NODE_TYPE = "animCurveTL"
    _TYPE_LABEL = None


class AnimCurveTA(AnimCurve):
    """An ``animCurveTA``: keys an angle (``rx``, ``Angle``, an ``Euler`` child) by time."""

    NATIVE_NODE_TYPE = "animCurveTA"
    _TYPE_LABEL = None


class AnimCurveTU(AnimCurve):
    """An ``animCurveTU``: keys a unitless number (``sx``, ``v``, an enum, ``Float``,
    ``Int``, ``Bool``, a ``Typed`` slot) by time."""

    NATIVE_NODE_TYPE = "animCurveTU"
    _TYPE_LABEL = None


class AnimCurveTT(AnimCurve):
    """An ``animCurveTT``: keys a time (``Time``) by time; its values are frames."""

    NATIVE_NODE_TYPE = "animCurveTT"
    _TYPE_LABEL = None


assert "animCurve" in _ABSTRACT_TYPES

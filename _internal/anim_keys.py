"""
Keys: the value ``rig.AnimCurve([...])`` makes, and the edits that key a plug with it.

``AnimCurve(values, times=None, **options)`` (a call of the node class with a sequence,
`rig.nodetypes.anim_curve`) returns a `_Keys`: values, times and key options that write
nothing until they meet ``<<``. ``plug << keys`` keys the plug with one animation curve per
channel and returns the plug (`inject`); a ``List`` broadcasts and pairs (`inject_rows`); a
per-channel list (``t << [AnimCurve(xs), skip, 0]``) keys the channels that hold keys
(`fanout`). Every check runs before the first edit (`_plan_plug`), then every curve is made
or updated in ONE undo step, ``rig.animcurve`` (`_apply`).

* **The curve type follows the plug**, by Maya's own rule
  (``MFnAnimCurve.timedAnimCurveTypeForPlug``): an animCurveTL for a distance, TA for an
  angle, TT for a time, TU for every other number (a bool, an enum, an int, a double, a
  ``Typed`` slot). A bool or an enum gets a step out tangent by default; every other key is
  linear.
* **Units are the UI's**, as for ``plug << value``: degrees, the scene's linear unit, frames
  for a TT curve's values. Times are frames in the scene's time unit.
* **A plug that is driven** has its input replaced (the ``plug << plug`` rule, D1): the
  input on that channel is disconnected and the curve connected; no pairBlend, no
  ``blendParent1``, and the old node is never deleted. A re-run finds the plug's own curve
  (its input is the output of a curve of the type the plug takes, which drives nothing else,
  whose own input is free and which is not from a referenced file) and updates it in place:
  the same node and UUID, every key replaced. A curve shared with other plugs is left alone
  and this plug gets a new one. Keys equal to the curve's write nothing and add no undo step.
* **A locked plug refuses**, as a value does (D3: keep raising): InjectionError, even when
  the keys are the ones its curve holds, so a build that locks keyed plugs unlocks them
  before it runs again.
* The keys of a curve are edited through the API (``MFnAnimCurve`` with one
  ``MAnimCurveChange`` per curve, journaled by `_CurveEdit`); a new curve is made with a raw
  ``cmds.createNode`` (it commutes with the journal), never with ``setKeyframe``,
  ``keyTangent`` or ``cutKey`` (``cutKey -clear`` deletes the emptied curve node). Without
  rig's undo plug-in the statement raises RuntimeError before any node is made.

A leaf module for the node classes: it imports `rig._internal.plug` and the container
lazily.
"""

from __future__ import annotations

import logging
import math
import numbers
from collections.abc import Iterator
from typing import Any

import numpy as np
from maya import cmds
from maya.api import OpenMaya, OpenMayaAnim

from rig.nodetypes._base import _name_of
from rig._internal import journal
from rig._internal.undo import _undo_chunk

LOGGER = logging.getLogger(__name__)

_FN = OpenMayaAnim.MFnAnimCurve

# the undo step of every keying statement
STEP = "rig.animcurve"

# the timed curve types, by MFnAnimCurve's enum
_CURVE_TYPES = {
    _FN.kAnimCurveTA: "animCurveTA",
    _FN.kAnimCurveTL: "animCurveTL",
    _FN.kAnimCurveTT: "animCurveTT",
    _FN.kAnimCurveTU: "animCurveTU",
}
_TIMED = frozenset(_CURVE_TYPES.values())

# keyTangent's names (``smooth`` is the API's name of ``spline``); matched with the enum
# tolerance (case, spaces, ``_`` and ``-`` ignored). The first line is what errors list.
_TANGENTS = {
    "linear":     _FN.kTangentLinear,
    "step":       _FN.kTangentStep,
    "stepnext":   _FN.kTangentStepNext,
    "flat":       _FN.kTangentFlat,
    "spline":     _FN.kTangentSmooth,
    "clamped":    _FN.kTangentClamped,
    "plateau":    _FN.kTangentPlateau,
    "auto":       _FN.kTangentAuto,
    "fixed":      _FN.kTangentFixed,
    "fast":       _FN.kTangentFast,
    "slow":       _FN.kTangentSlow,
    "autoease":   _FN.kTangentAutoEase,
    "automix":    _FN.kTangentAutoMix,
    "autocustom": _FN.kTangentAutoCustom,
    "global":     _FN.kTangentGlobal,
}
_TANGENT_LISTED = (
    "linear", "step", "stepnext", "flat", "spline", "clamped", "plateau", "auto", "fixed",
    "fast", "slow", "autoease", "automix",
)
_TANGENT_ALIASES = {"smooth": "spline"}
_TANGENT_NAMES = {code: name for name, code in _TANGENTS.items()}

# setInfinity's names
_INFINITY = {
    "constant":      _FN.kConstant,
    "linear":        _FN.kLinear,
    "cycle":         _FN.kCycle,
    "cycleRelative": _FN.kCycleRelative,
    "oscillate":     _FN.kOscillate,
}
_INFINITY_NAMES = {code: name for name, code in _INFINITY.items()}

# the per-key options and the others, by keyword
_PER_KEY = (
    "tangents", "in_tangents", "out_tangents", "slopes", "in_slopes", "out_slopes",
    "in_weights", "out_weights",
)
OPTIONS = frozenset(_PER_KEY + ("weighted", "infinity", "pre_infinity", "post_infinity"))

# the numeric types a curve keys (an MFnNumericAttribute of one value)
_NUMERIC = frozenset({
    OpenMaya.MFnNumericData.kBoolean,
    OpenMaya.MFnNumericData.kByte,
    OpenMaya.MFnNumericData.kChar,
    OpenMaya.MFnNumericData.kShort,
    OpenMaya.MFnNumericData.kInt,
    OpenMaya.MFnNumericData.kInt64,
    OpenMaya.MFnNumericData.kFloat,
    OpenMaya.MFnNumericData.kDouble,
})

# an unattached function set: Maya's curve-type rule needs no curve
_TYPE_RULE = _FN()


def _norm(name: str) -> str:
    """A name with case, spaces, ``_`` and ``-`` ignored (the enum tolerance)."""
    return "".join(ch for ch in name.lower() if ch not in " _-")


_TANGENT_BY_NORM = {_norm(k): v for k, v in _TANGENTS.items()}
_TANGENT_BY_NORM.update({_norm(k): _TANGENTS[v] for k, v in _TANGENT_ALIASES.items()})
_INFINITY_BY_NORM = {_norm(k): v for k, v in _INFINITY.items()}


def _tangent_code(name: Any, option: str) -> int:
    if isinstance(name, str):
        code = _TANGENT_BY_NORM.get(_norm(name))
        if code is not None:
            return code
    raise TypeError(
        f"AnimCurve {option}={name!r}: a tangent is one of {', '.join(_TANGENT_LISTED)} "
        f"(keyTangent's names; case, spaces and '_' are ignored)"
    )


def _infinity_code(name: Any, option: str) -> int:
    if isinstance(name, str):
        code = _INFINITY_BY_NORM.get(_norm(name))
        if code is not None:
            return code
    raise TypeError(
        f"AnimCurve {option}={name!r}: an infinity is one of {', '.join(_INFINITY)} "
        f"(setInfinity's names; case, spaces and '_' are ignored)"
    )


# --------------------------------------------------------------------------------------- #
#  The value
# --------------------------------------------------------------------------------------- #


def _is_keys_sequence(values: Any) -> bool:
    """True for what ``AnimCurve(x)`` reads as values: a list, tuple, range, ndarray or
    iterator, never a str (a name: the curve node ``AnimCurve('x')`` refers to)."""
    if isinstance(values, (str, bytes)):
        return False
    return isinstance(values, (list, tuple, range, np.ndarray, Iterator))


def _per_key(value: Any, n: int, option: str, kind: str) -> Any:
    """A per-key option as an array of ndim 0 (one for all keys), 1 ``(N,)`` or 2 ``(N, C)``;
    ``kind`` is ``"tangent"`` (codes) or ``"number"`` (floats, NaN allowed)."""
    if kind == "tangent":
        if isinstance(value, str):
            return np.array(_tangent_code(value, option))
        try:
            raw = np.asarray(value, dtype=object)
        except Exception:  # noqa: BLE001 (re-raised, clear)
            raw = None
        if raw is None or raw.ndim not in (1, 2):
            raise TypeError(
                f"AnimCurve {option}= takes a tangent name, or one per key ((N,) or (N, C)); "
                f"got {value!r}"
            )
        codes = np.vectorize(lambda x: _tangent_code(x, option), otypes=[np.int64])(raw)
        arr = codes
    else:
        try:
            arr = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            raise TypeError(
                f"AnimCurve {option}= takes numbers: one for every key, (N,) or (N, C); got {value!r}"
            ) from None
        if arr.ndim > 2:
            raise ValueError(f"AnimCurve {option}= is a number, (N,) or (N, C); got the shape {arr.shape}")
        if np.isinf(arr).any():
            raise ValueError(f"AnimCurve {option}= holds an infinite number")
    if arr.ndim and arr.shape[0] != n:
        raise ValueError(
            f"AnimCurve {option}= has {arr.shape[0]} rows for {n} keys: one per key ((N,) or "
            f"(N, C)), or one value for all keys"
        )
    return arr


def _per_key_2d(arr: Any) -> Any:
    """A per-key option for broadcasting against another: (N,) is one per key, (N, 1)."""
    return arr[:, None] if np.ndim(arr) == 1 else arr


def _read_only(value: Any) -> Any:
    """A read-only copy of the array ``value`` (a keys value's read); anything else (None,
    a name) as it is."""
    if not isinstance(value, np.ndarray):
        return value
    arr = np.array(value)
    arr.flags.writeable = False
    return arr


class _Keys:
    """Keys waiting for a destination: what ``AnimCurve(values, ...)`` returns (not a node:
    ``isinstance(AnimCurve([0]), AnimCurve)`` is False). ``plug << keys`` keys the plug.

    Read-only attributes (UI units): ``values`` ((N,) or (N, C) floats, or (N,) enum field
    names), ``times`` ((N,) frames), ``in_tangents`` / ``out_tangents`` (the names given,
    or None: the default), ``in_slopes`` / ``out_slopes`` (value units per frame, NaN where
    a key has none; None when none was given), ``weighted``, ``in_weights`` /
    ``out_weights``, ``pre_infinity`` / ``post_infinity`` (names). Plain numpy and str, so a
    value pickles. The arrays are read-only (user decision 2026-10-16: a write into one
    raises instead of being lost): ``keys.values.copy()`` is a writable copy."""

    __slots__ = (
        "_values", "_times", "_names", "_itt", "_ott", "_islope", "_oslope", "_weighted",
        "_iw", "_ow", "_pre", "_post", "_mtimes",
    )

    def __init__(self, values: Any, times: Any = None, **options: Any) -> None:
        unknown = sorted(set(options) - OPTIONS)
        if unknown:
            raise TypeError(
                f"AnimCurve() got an unknown option {unknown[0]!r}; the options are "
                f"{', '.join(sorted(OPTIONS))}"
            )
        if not _is_keys_sequence(values):
            raise TypeError(
                f"AnimCurve({values!r}): keys are a sequence of values, AnimCurve([{values!r}]) "
                f"for one key; AnimCurve('x') refers to the curve node x"
            )
        if isinstance(values, Iterator):
            values = list(values)
        names = None
        if not isinstance(values, np.ndarray):
            from rig.nodetypes._base import Attribute

            if any(isinstance(v, Attribute) for v in values):
                raise TypeError(
                    "AnimCurve values are numbers, not plugs: a plug drives a channel with "
                    "plug << other_plug"
                )
        if not isinstance(values, np.ndarray) and any(isinstance(v, str) for v in values):
            # enum field names (ints may be mixed in): read at '<<', on the enum
            for v in values:
                if not (isinstance(v, str) or (isinstance(v, numbers.Real) and math.isfinite(v))):
                    raise TypeError(
                        f"AnimCurve values with field names take names and ints, one per key; got {v!r}"
                    )
            names = np.array(list(values), dtype=object)
            vals = np.zeros(len(names))
        else:
            try:
                if not isinstance(values, np.ndarray) and np.asarray(values).dtype == object:
                    raise TypeError  # None, a plug, a ragged list: no numbers
                vals = np.asarray(values, dtype=float)
            except (TypeError, ValueError):
                raise TypeError(
                    f"AnimCurve values are numbers ((N,) or (N, C)) or enum field names; got {values!r}"
                ) from None
            if vals.ndim not in (1, 2):
                raise ValueError(
                    f"AnimCurve values are (N,) or (N, C): one row per key; got the shape {vals.shape}"
                )
            if not np.isfinite(vals).all():
                raise ValueError("AnimCurve values must be finite (a NaN or an inf was given)")
        n = len(vals)
        if n < 1:
            raise ValueError("AnimCurve holds at least one key; got no values")
        if times is None:
            tms = np.arange(n, dtype=float)
        else:
            if isinstance(times, Iterator):
                times = list(times)
            try:
                tms = np.asarray(times, dtype=float)
            except (TypeError, ValueError):
                raise TypeError(f"AnimCurve times are frames (numbers); got {times!r}") from None
            if tms.shape != (n,):
                raise ValueError(f"AnimCurve: {n} values need {n} times; got the shape {tms.shape}")
            if not np.isfinite(tms).all():
                raise ValueError("AnimCurve times must be finite (a NaN or an inf was given)")
            if n > 1 and not (np.diff(tms) > 0).all():
                raise ValueError(
                    "AnimCurve times must be strictly increasing (Maya merges keys of one time "
                    "and sorts the rest)"
                )

        def per_key(option: str, kind: str) -> Any:
            value = options.get(option)
            return None if value is None else _per_key(value, n, option, kind)

        tangents = per_key("tangents", "tangent")
        itt      = per_key("in_tangents", "tangent")
        ott      = per_key("out_tangents", "tangent")
        slopes   = per_key("slopes", "number")
        islope   = per_key("in_slopes", "number")
        oslope   = per_key("out_slopes", "number")
        weighted = options.get("weighted", False)
        if not isinstance(weighted, (bool, np.bool_)):
            raise TypeError(f"AnimCurve weighted= is True or False; got {weighted!r}")
        weighted = bool(weighted)
        iw = per_key("in_weights", "number")
        ow = per_key("out_weights", "number")
        if (iw is not None or ow is not None) and not weighted:
            raise ValueError("AnimCurve in_weights= / out_weights= need weighted=True (a weighted curve)")
        for w, option in ((iw, "in_weights"), (ow, "out_weights")):
            if w is not None and (w[~np.isnan(w)] <= 0).any():
                raise ValueError(f"AnimCurve {option}= are positive numbers (NaN: none for that key)")
        infinity = options.get("infinity")
        inf  = _infinity_code("constant" if infinity is None else infinity, "infinity")
        pre  = options.get("pre_infinity")
        post = options.get("post_infinity")
        pre  = inf if pre is None else _infinity_code(pre, "pre_infinity")
        post = inf if post is None else _infinity_code(post, "post_infinity")

        itt = tangents if itt is None else itt
        ott = tangents if ott is None else ott
        islope = slopes if islope is None else islope
        oslope = slopes if oslope is None else oslope
        # a slope or a weight makes its side "fixed" (Maya fixes a tangent whose weight is
        # set): a tangent of another type given for that side is a conflict
        for types, slope, weight, side in ((itt, islope, iw, "in"), (ott, oslope, ow, "out")):
            if types is None or (slope is None and weight is None):
                continue
            fixing = [x for x in (slope, weight) if x is not None]
            try:
                t_b, *f_b = np.broadcast_arrays(*(_per_key_2d(x) for x in (types, *fixing)))
            except ValueError:
                raise ValueError(
                    f"AnimCurve {side} tangents and {side} slopes / weights have shapes that do "
                    f"not match: {[np.shape(x) for x in (types, *fixing)]}"
                ) from None
            sets = np.zeros(t_b.shape, dtype=bool)
            for x in f_b:
                sets |= ~np.isnan(x)
            if (sets & (t_b != _FN.kTangentFixed)).any():
                raise ValueError(
                    f"AnimCurve: a key has an {side} slope or weight and an {side} tangent that "
                    f"is not 'fixed' (a slope or a weight makes its side fixed: drop the "
                    f"tangent, or give 'fixed')"
                )
        # the (N, C) options agree on C
        cols = {a.shape[1] for a in (vals, itt, ott, islope, oslope, iw, ow)
                if a is not None and np.ndim(a) == 2 and a.shape[1] != 1}
        if len(cols) > 1:
            raise ValueError(
                f"AnimCurve values and options give different channel counts {sorted(cols)}"
            )
        self._values   = vals
        self._names    = names
        self._times    = tms
        self._itt      = itt
        self._ott      = ott
        self._islope   = islope
        self._oslope   = oslope
        self._weighted = weighted
        self._iw       = iw
        self._ow       = ow
        self._pre      = pre
        self._post     = post
        self._mtimes   = None

    # -- reads -- #

    @property
    def values(self) -> np.ndarray:
        return _read_only(self._names if self._names is not None else self._values)

    @property
    def times(self) -> np.ndarray:
        return _read_only(self._times)

    @property
    def num_keys(self) -> int:
        return len(self._times)

    def _names_of(self, codes: Any) -> Any:
        if codes is None:
            return None
        if np.ndim(codes) == 0:
            return _TANGENT_NAMES.get(int(codes), int(codes))
        return np.vectorize(lambda c: _TANGENT_NAMES.get(int(c), str(c)), otypes=[object])(codes)

    @property
    def in_tangents(self) -> Any:
        return _read_only(self._names_of(self._itt))

    @property
    def out_tangents(self) -> Any:
        return _read_only(self._names_of(self._ott))

    @property
    def in_slopes(self) -> Any:
        return _read_only(self._islope)

    @property
    def out_slopes(self) -> Any:
        return _read_only(self._oslope)

    @property
    def weighted(self) -> bool:
        return self._weighted

    @property
    def in_weights(self) -> Any:
        return _read_only(self._iw)

    @property
    def out_weights(self) -> Any:
        return _read_only(self._ow)

    @property
    def pre_infinity(self) -> str:
        return _INFINITY_NAMES[self._pre]

    @property
    def post_infinity(self) -> str:
        return _INFINITY_NAMES[self._post]

    def __repr__(self) -> str:
        def short(arr):
            items = arr.tolist()
            if len(items) > 8:
                return "[" + ", ".join(repr(x) for x in items[:6]) + ", ...]"
            return repr(items)

        text = f"AnimCurve({short(self.values)}, times={short(self._times)}"
        for option in ("in_tangents", "out_tangents"):
            codes = self._itt if option == "in_tangents" else self._ott
            if codes is not None:
                named = self._names_of(codes)
                text += f", {option}={named!r}" if np.ndim(codes) == 0 else f", {option}=[...]"
        for option, arr in (("in_slopes", self._islope), ("out_slopes", self._oslope)):
            if arr is not None:
                text += f", {option}=[...]" if np.ndim(arr) else f", {option}={float(arr)!r}"
        if self._weighted:
            text += ", weighted=True"
        if self._pre == self._post:
            if self._pre != _FN.kConstant:
                text += f", infinity={self.pre_infinity!r}"
        else:
            text += f", pre_infinity={self.pre_infinity!r}, post_infinity={self.post_infinity!r}"
        return text + ")"

    def __getstate__(self) -> dict:
        state = {slot: getattr(self, slot) for slot in self.__slots__ if slot != "_mtimes"}
        return state

    def __setstate__(self, state: dict) -> None:
        for slot, value in state.items():
            object.__setattr__(self, slot, value)
        object.__setattr__(self, "_mtimes", None)

    def __bool__(self) -> bool:
        return True

    def _mtime_array(self) -> Any:
        """The times as an MTimeArray in the scene's time unit (kept while the unit is)."""
        unit = OpenMaya.MTime.uiUnit()
        cached = self._mtimes
        if cached is None or cached[0] != unit:
            cached = (unit, OpenMaya.MTimeArray([OpenMaya.MTime(t, unit) for t in self._times.tolist()]))
            self._mtimes = cached
        return cached[1]


def keys(values: Any, times: Any = None, **options: Any) -> _Keys:
    """``AnimCurve(values, times=None, **options)`` (the class call's keys branch)."""
    return _Keys(values, times, **options)


def coerce(value: Any, times: Any, options: dict, call: str) -> "_Keys | None":
    """The keys a node class method was given: values (with ``times`` and options), an
    ``AnimCurve`` value (then neither), or None (none: neither either)."""
    if value is None:
        if times is not None or options:
            raise TypeError(f"{call}: times and key options need values (the keys)")
        return None
    if type(value) is _Keys:
        if times is not None or options:
            raise TypeError(
                f"{call}: an AnimCurve value holds its own times and options; give the values "
                f"themselves to set them here"
            )
        return value
    return _Keys(value, times, **options)


# --------------------------------------------------------------------------------------- #
#  Units
# --------------------------------------------------------------------------------------- #


def _factor(curve_type: str) -> float:
    """Internal value units per UI unit of a curve type (TA radians, TL cm, TU 1)."""
    if curve_type == "animCurveTA":
        return OpenMaya.MAngle(1.0, OpenMaya.MAngle.uiUnit()).asRadians()
    if curve_type == "animCurveTL":
        return OpenMaya.MDistance(1.0, OpenMaya.MDistance.uiUnit()).asCentimeters()
    return 1.0


def _fps() -> float:
    """Frames of the scene's time unit per second."""
    return OpenMaya.MTime(1.0, OpenMaya.MTime.kSeconds).asUnits(OpenMaya.MTime.uiUnit())


# --------------------------------------------------------------------------------------- #
#  Reading a curve
# --------------------------------------------------------------------------------------- #


class _State:
    """What a curve holds, in UI units, as the plan compares and writes it."""

    __slots__ = ("times", "values", "itt", "ott", "islope", "oslope", "weighted", "iw", "ow", "pre", "post")


def _read(fn: Any, curve_type: str) -> _State:
    """The keys of the curve of ``fn`` (UI units): times, values, tangent codes, slopes
    (value units per frame; NaN where a side is not fixed), weights (weighted curves), the
    infinities."""
    s = _State()
    n = fn.numKeys
    unit = OpenMaya.MTime.uiUnit()
    inputs = [fn.input(i) for i in range(n)]
    s.times = np.array([t.asUnits(unit) for t in inputs], dtype=float)
    if curve_type == "animCurveTT":
        s.values = np.array([fn.evaluate(t).asUnits(unit) for t in inputs], dtype=float)
        factor = None
    else:
        factor = _factor(curve_type)
        s.values = np.array([fn.value(i) for i in range(n)], dtype=float) / factor
    s.itt = np.array([fn.inTangentType(i) for i in range(n)], dtype=np.int64)
    s.ott = np.array([fn.outTangentType(i) for i in range(n)], dtype=np.int64)
    s.islope = np.full(n, np.nan)
    s.oslope = np.full(n, np.nan)
    if factor is not None:
        fps = _fps()
        fixed = _FN.kTangentFixed
        for side, types, out in ((True, s.itt, s.islope), (False, s.ott, s.oslope)):
            for i in np.flatnonzero(types == fixed).tolist():
                x, y = fn.getTangentXY(i, side)
                out[i] = (y / factor) / (x * fps) if x else (math.copysign(math.inf, y) if y else 0.0)
    s.weighted = bool(fn.isWeighted)
    if s.weighted:
        s.iw = np.array([fn.getTangentAngleWeight(i, True)[1] for i in range(n)], dtype=float)
        s.ow = np.array([fn.getTangentAngleWeight(i, False)[1] for i in range(n)], dtype=float)
    else:
        s.iw = s.ow = None
    s.pre = fn.preInfinityType
    s.post = fn.postInfinityType
    return s


def read_keys(fn: Any, curve_type: str) -> _Keys:
    """``curve.get_keys()``: the keys value that reproduces the curve of ``fn``."""
    s = _read(fn, curve_type)
    if not len(s.times):
        raise ValueError(f"{fn.name()!r} holds no keys")
    options: dict = {
        "in_tangents": [_TANGENT_NAMES.get(int(c), "auto") for c in s.itt],
        "out_tangents": [_TANGENT_NAMES.get(int(c), "auto") for c in s.ott],
    }
    if not np.isnan(s.islope).all():
        options["in_slopes"] = s.islope
    if not np.isnan(s.oslope).all():
        options["out_slopes"] = s.oslope
    if s.weighted:
        # the weights of the fixed sides (a weight given makes its side fixed; the other
        # sides' weights recompute from their tangent type)
        options["weighted"] = True
        fixed = _FN.kTangentFixed
        iw = np.where(s.itt == fixed, s.iw, np.nan)
        ow = np.where(s.ott == fixed, s.ow, np.nan)
        if not np.isnan(iw).all():
            options["in_weights"] = iw
        if not np.isnan(ow).all():
            options["out_weights"] = ow
    options["pre_infinity"] = _INFINITY_NAMES.get(s.pre, "constant")
    options["post_infinity"] = _INFINITY_NAMES.get(s.post, "constant")
    return _Keys(s.values, s.times, **options)


def _same(s: _State, want: "_Leaf") -> bool:
    """True when the curve already holds what ``want`` would write (checked before any
    edit: unchanged keys make no edit and no undo step)."""
    n = len(want.times)
    if len(s.times) != n or s.weighted != want.weighted or s.pre != want.pre or s.post != want.post:
        return False
    if not (np.array_equal(s.itt, want.itt) and np.array_equal(s.ott, want.ott)):
        return False
    if not np.allclose(s.times, want.times, rtol=0.0, atol=1e-9):
        return False
    if not np.allclose(s.values, want.values, rtol=1e-9, atol=1e-9):
        return False
    for have, slope in ((s.islope, want.islope), (s.oslope, want.oslope)):
        if slope is None:
            continue
        given = ~np.isnan(slope)
        if given.any() and not np.allclose(have[given], slope[given], rtol=1e-6, atol=1e-9):
            return False
    if want.weighted:
        for have, w in ((s.iw, want.iw), (s.ow, want.ow)):
            if w is None:
                continue
            given = ~np.isnan(w)
            if given.any() and not np.allclose(have[given], w[given], rtol=1e-6, atol=1e-9):
                return False
    return True


# --------------------------------------------------------------------------------------- #
#  The plan: every check, no write
# --------------------------------------------------------------------------------------- #


class _Leaf:
    """One channel's edit: what to write (UI units), into which curve (None: a new one)."""

    __slots__ = (
        "leaf", "curve_type", "curve", "same", "name", "replaces", "times", "source", "values",
        "itt", "ott", "islope", "oslope", "weighted", "iw", "ow", "pre", "post",
    )


class _Plan:
    """A plug's edits: its leaves, the compound input to disconnect first, and whether a
    compound above the plug is connected whole (its input split into one per child
    first, `rig._internal.plug._free_of_whole_wire`)."""

    __slots__ = ("dst", "leaves", "compound_input", "whole_above")


def _kind_text(leaf: Any) -> str:
    """What a non-numeric attribute holds, for a refusal (read on its path only)."""
    try:
        kind = cmds.getAttr(_name_of(leaf), type=True)
    except (RuntimeError, ValueError):
        kind = None
    return kind or "non-numeric"


def _leaf_refusal(mplug: Any, name: str) -> "str | None":
    """Why the curve cannot key the plug ``mplug`` (a scalar), or None."""
    attr = mplug.attribute()
    if attr.hasFn(OpenMaya.MFn.kNumericAttribute):
        if OpenMaya.MFnNumericAttribute(attr).numericType() in _NUMERIC:
            return None
    elif attr.hasFn(OpenMaya.MFn.kUnitAttribute) or attr.hasFn(OpenMaya.MFn.kEnumAttribute):
        return None
    elif attr.hasFn(OpenMaya.MFn.kTypedAttribute):
        if OpenMaya.MFnTypedAttribute(attr).attrType() == OpenMaya.MFnData.kAny:
            return None  # a generic slot holds numbers too: an animCurveTU
    return name


def _channels(arr: Any, column: int, n: int) -> Any:
    """Column ``column`` of a per-key option (ndim 0, 1 or 2), as an (N,) array."""
    if arr is None:
        return None
    if np.ndim(arr) == 0:
        return np.full(n, arr.item(), dtype=arr.dtype)
    if np.ndim(arr) == 1:
        return np.asarray(arr)
    return np.asarray(arr[:, 0 if arr.shape[1] == 1 else column])


def _curve_type_of(leaf: Any) -> str:
    """Maya's curve type for the plug ``leaf`` (``timedAnimCurveTypeForPlug``)."""
    return _CURVE_TYPES[_TYPE_RULE.timedAnimCurveTypeForPlug(leaf._mplug)]


def _stepped(mplug: Any) -> bool:
    """A bool or an enum: its default out tangent is a step (as Maya's)."""
    attr = mplug.attribute()
    if attr.hasFn(OpenMaya.MFn.kEnumAttribute):
        return True
    if attr.hasFn(OpenMaya.MFn.kNumericAttribute):
        return OpenMaya.MFnNumericAttribute(attr).numericType() == OpenMaya.MFnNumericData.kBoolean
    return False


def _own_curve(mplug: Any, curve_type: str) -> Any:
    """The curve the plug ``mplug`` updates in place on a re-run, or None: its input is the
    output of a curve of type ``curve_type`` that drives nothing else, whose input is free
    and which is not from a referenced file. Returns ``(mobject, fn)``."""
    src = mplug.source()
    if src.isNull:
        return None
    node = src.node()
    if not node.hasFn(OpenMaya.MFn.kAnimCurve):
        return None
    fn = _FN(node)
    if _CURVE_TYPES.get(fn.animCurveType) != curve_type:
        return None
    if len(src.destinations()) != 1:
        return None
    if fn.isFromReferencedFile:
        return None
    if not fn.findPlug("input", False).source().isNull:
        return None  # a driven key
    return node, fn


def _curve_name(mplug: Any) -> str:
    """Maya's name for a new curve of ``mplug``: ``<node>_<attribute>``, the node's short
    name without its namespace (the curve goes in the current one, as setKeyframe puts it),
    an element's ``[i]`` written ``_i_``, a child's ``.`` written ``_``."""
    node = OpenMaya.MFnDependencyNode(mplug.node()).name().rsplit("|", 1)[-1].rsplit(":", 1)[-1]
    attr = mplug.partialName(False, False, False, False, False, True)
    return f"{_name_of(node)}_{attr.replace('[', '_').replace(']', '_').replace('.', '_')}"


def _resolve_side(given_types: Any, slope: Any, weight: Any, default: int, n: int) -> Any:
    """A side's (N,) tangent codes: the types given, else the default; a key with a slope or
    a weight is fixed."""
    codes = np.full(n, default, dtype=np.int64) if given_types is None else given_types.astype(np.int64)
    for fixing in (slope, weight):
        if fixing is not None:
            codes = codes.copy()
            codes[~np.isnan(fixing)] = _FN.kTangentFixed
    return codes


def _plan_values(keys: _Keys, curve_type: str, column: int, stepped: bool, where: str,
                 enum_plug: Any = None) -> _Leaf:
    """The write of one curve of ``curve_type`` from column ``column`` of ``keys``, checked
    (no plug, no curve yet: those are the caller's)."""
    from rig.nodetypes._base import _enum_value

    e = _Leaf()
    n = keys.num_keys
    e.curve_type = curve_type
    e.times = keys._times
    if keys._names is not None:
        if enum_plug is None:
            raise TypeError(
                f"{where}: field names key an enum attribute; this curve keys numbers "
                f"(give the values, or key the enum plug)"
            )
        e.values = np.array(
            [float(_enum_value(enum_plug, v, where)) if isinstance(v, str) else float(v) for v in keys._names]
        )
    else:
        vals = keys._values
        e.values = vals if vals.ndim == 1 else vals[:, 0 if vals.shape[1] == 1 else column]
    itt = _channels(keys._itt, column, n)
    ott = _channels(keys._ott, column, n)
    e.islope = _channels(keys._islope, column, n)
    e.oslope = _channels(keys._oslope, column, n)
    e.weighted = keys._weighted
    e.iw = _channels(keys._iw, column, n)
    e.ow = _channels(keys._ow, column, n)
    if curve_type == "animCurveTT" and (
        any(x is not None and not np.isnan(x).all() for x in (e.islope, e.oslope)) or e.weighted
    ):
        raise ValueError(f"{where}: an animCurveTT (a time curve) takes no slopes or weights")
    linear = _FN.kTangentLinear
    e.itt = _resolve_side(itt, e.islope, e.iw, linear, n)
    e.ott = _resolve_side(ott, e.oslope, e.ow, _FN.kTangentStep if stepped else linear, n)
    e.pre = keys._pre
    e.post = keys._post
    e.source = keys
    return e


def _columns(keys: _Keys, count: int, dst: Any) -> None:
    """``ValueError`` unless every (N, C) array of ``keys`` has C 1 or ``count``."""
    for arr in (keys._values, keys._itt, keys._ott, keys._islope, keys._oslope, keys._iw, keys._ow):
        if arr is not None and np.ndim(arr) == 2 and arr.shape[1] not in (1, count):
            what = f"{count} channel" + ("s" if count != 1 else "")
            raise ValueError(
                f"{_name_of(dst)} << AnimCurve(...): (N, {arr.shape[1]}) keys for {what}; give (N,) "
                f"for the same keys on every channel or (N, {count}) for one column per channel"
            )


def require_plugin(what: str = "keys") -> None:
    """RuntimeError, before any node is made, when rig's undo plug-in cannot be had: the
    curves' key edits have no other undoable route (as ``Mesh.set_points``)."""
    from rig.nodetypes import plugins

    if not plugins._fast_path_ready():
        if OpenMaya.MGlobal.isUndoing() or OpenMaya.MGlobal.isRedoing():
            return  # a callback's statement while Maya undoes: the plug-in is not the reason
        raise RuntimeError(
            f"rig needs its undo plug-in 'undoable_api_command' for {what} (AnimCurve), and it "
            f"could not be loaded ({plugins._plugin_whereabouts()}); nothing was made"
        )


def _indexed_name(attr: Any, mplug: Any) -> str:
    """``attr``'s name for a message, with its element index: Maya names an element of a
    per-instance array without it (``a.worldMatrix`` for ``a.wm[0]``)."""
    name = _name_of(attr)
    if mplug.isElement and not name.endswith("]"):
        node = name.split(".", 1)[0]
        return f"{_name_of(node)}.{mplug.partialName(False, False, True, False, False, True)}"
    return name


def _plan_plug(keys: _Keys, dst: Any, where: "str | None" = None) -> _Plan:
    """Every check of ``dst << keys``, with no write: the plug's channels, their kind and
    curve type, its locks (D3: a locked plug refuses, even when the keys equal its curve's),
    the shapes, the enum names, each channel's curve (its own, updated in place; else a new
    one) and whether it holds these keys already."""
    from rig.nodetypes._base import Attribute, _ensure_owner_alive, _is_generic_slot
    from rig._internal.plug import InjectionError, _assert_settable, _check_whole_wire

    if not isinstance(dst, Attribute):
        raise TypeError(f"an AnimCurve keys a plug: node.<attr> << AnimCurve(...); got {dst!r}")
    _ensure_owner_alive(dst)
    mplug = dst._mplug
    where = where or f"{_indexed_name(dst, mplug)} << AnimCurve(...)"
    if mplug.isArray:
        raise TypeError(
            f"{where}: {_name_of(dst)!r} is a multi attribute: key an element ({_name_of(dst)}[i]) or every "
            f"element ({_name_of(dst)}[:]); a curve appended to the multi would add a new curve on every "
            f"re-run and never find its own"
        )
    plan = _Plan()
    plan.dst = dst
    plan.compound_input = False
    if mplug.isCompound:
        leaves = [dst.child(i) for i in range(mplug.numChildren())]
        for leaf in leaves:
            child = leaf._mplug
            if child.isCompound or child.isArray:
                raise TypeError(
                    f"{where}: an animation curve keys a numeric attribute or a compound of "
                    f"them; {str(leaf)!r} is a compound or a multi (key its children)"
                )
        plan.compound_input = mplug.isDestination
    else:
        leaves = [dst]
    for leaf in leaves:
        if _leaf_refusal(leaf._mplug, str(leaf)) is not None:
            raise TypeError(
                f"{where}: an animation curve keys a numeric attribute; "
                f"{_indexed_name(leaf, leaf._mplug)!r} is {_kind_text(leaf)} (Maya keys numbers, "
                f"angles, distances and times)"
            )
        if not OpenMaya.MFnAttribute(leaf._mplug.attribute()).writable:
            raise InjectionError(
                f"Cannot key {_indexed_name(leaf, leaf._mplug)!r}: it is an output (not "
                f"writable), which no curve can drive. Nothing was changed"
            )
    _columns(keys, len(leaves), dst)
    # D3: a locked plug refuses, as a value does, whatever its curve holds
    _assert_settable(dst, leaves=leaves)
    # a compound above the plug connected whole (``c.t`` into ``cube.t`` for ``cube.tx``):
    # its input is split into one per child before the curve replaces this one's (a curve
    # next to the whole input would leave the channel two drivers); checked here
    plan.whole_above = _check_whole_wire(dst, "key")
    entries = []
    for column, leaf in enumerate(leaves):
        lp = leaf._mplug
        curve_type = "animCurveTU" if _is_generic_slot(leaf) else _curve_type_of(leaf)
        enum_plug = lp if lp.attribute().hasFn(OpenMaya.MFn.kEnumAttribute) else None
        e = _plan_values(keys, curve_type, column, _stepped(lp), where, enum_plug)
        e.leaf = leaf
        own = _own_curve(lp, curve_type)
        e.replaces = None
        if own is not None:
            node, fn = own
            from rig.nodetypes._base import _refuse_locked_node

            _refuse_locked_node(fn, "set the keys of")
            e.curve = OpenMaya.MObjectHandle(node)
            e.same = _same(_read(fn, curve_type), e)
            e.name = None
        else:
            e.curve = None
            e.same = False
            e.name = _curve_name(lp)
            if lp.isDestination:
                e.replaces = lp.source().name()
        entries.append(e)
    plan.leaves = entries
    if any(not e.same for e in entries):
        require_plugin()
        if any(e.curve is None for e in entries):
            from rig._internal.container import container

            container._refuse_locked_leaf()
    return plan


# --------------------------------------------------------------------------------------- #
#  The edits
# --------------------------------------------------------------------------------------- #


class _CurveEdit:
    """A journal op: the ``MAnimCurveChange`` of one curve's key edit, applied already
    (undo walks it back, redo forwards). Its calls make no ``cmds`` call."""

    __slots__ = ("change",)

    RECORDS_NO_CMDS = True
    UNDO_EDIT = "the keys of an animation curve"

    def __init__(self, change: Any) -> None:
        self.change = change

    def doIt(self) -> None:  # noqa: N802 (the journal's op protocol)
        self.change.redoIt()

    def undoIt(self) -> None:  # noqa: N802
        self.change.undoIt()


def _same_or_none(a: float, b: float) -> bool:
    """Both NaN (none given), or the same number."""
    return (math.isnan(a) and math.isnan(b)) or a == b


def _write(fn: Any, e: _Leaf) -> None:
    """Writes ``e`` into the curve of ``fn``, replacing every key, in one ``MAnimCurveChange``
    the journal holds: the old keys removed last first (``remove``, never ``cutKey``), the
    weighted flag, the new keys (``addKeys``; a TT curve one ``addKey`` per key), the tangent
    types, the slopes (a key whose sides differ unlocked first: Maya locks them, and a
    locked key's other side follows), the weights, the infinities."""
    change = OpenMayaAnim.MAnimCurveChange()
    try:
        for i in range(fn.numKeys - 1, -1, -1):
            fn.remove(i, change)
        if bool(fn.isWeighted) != e.weighted:
            fn.setIsWeighted(e.weighted, change)
        n = len(e.times)
        itt, ott = e.itt, e.ott
        unit = OpenMaya.MTime.uiUnit()
        if e.curve_type == "animCurveTT":
            for i, (t, v) in enumerate(zip(e.times.tolist(), e.values.tolist())):
                fn.addKey(OpenMaya.MTime(t, unit), OpenMaya.MTime(v, unit), int(itt[i]), int(ott[i]), change)
        else:
            uin = int(itt[0]) if (itt == itt[0]).all() else _FN.kTangentLinear
            uout = int(ott[0]) if (ott == ott[0]).all() else _FN.kTangentLinear
            factor = _factor(e.curve_type)
            values = e.values * factor if factor != 1.0 else e.values
            fn.addKeys(e.source._mtime_array(), OpenMaya.MDoubleArray(values.tolist()), uin, uout, False, change)
            for i in np.flatnonzero(itt != uin).tolist():
                fn.setInTangentType(i, int(itt[i]), change)
            for i in np.flatnonzero(ott != uout).tolist():
                fn.setOutTangentType(i, int(ott[i]), change)
        nan = np.full(n, np.nan)
        si = nan if e.islope is None else e.islope
        so = nan if e.oslope is None else e.oslope
        wi = e.iw if e.weighted and e.iw is not None else nan
        wo = e.ow if e.weighted and e.ow is not None else nan
        given = ~(np.isnan(si) & np.isnan(so) & np.isnan(wi) & np.isnan(wo))
        for i in np.flatnonzero(given).tolist():
            a, b, x, y = float(si[i]), float(so[i]), float(wi[i]), float(wo[i])
            # a key keeps Maya's tangent lock only when both sides get the same tangent
            twin = _same_or_none(a, b) and _same_or_none(x, y) and not (math.isnan(a) and math.isnan(x))
            if not twin:
                fn.setTangentsLocked(i, False, change)
                if e.weighted:
                    fn.setWeightsLocked(i, False, change)
            for side, slope, weight in ((True, a, x), (False, b, y)):
                if math.isnan(slope) and math.isnan(weight):
                    continue
                if e.weighted:
                    # the angle-and-weight form: a slope keeps the weight, a weight the angle
                    angle, held = fn.getTangentAngleWeight(i, side)
                    if not math.isnan(slope):
                        angle = OpenMaya.MAngle(math.atan(slope))
                    fn.setTangent(i, angle, held if math.isnan(weight) else weight, side, change, True)
                else:
                    fn.setTangent(i, 1.0, slope, side, change, True)
        if fn.preInfinityType != e.pre:
            fn.setPreInfinityType(e.pre, change)
        if fn.postInfinityType != e.post:
            fn.setPostInfinityType(e.post, change)
    except BaseException:
        try:
            change.undoIt()  # never left half written and unjournaled
        except Exception:  # noqa: BLE001 (the first error is the one raised)
            pass
        raise
    journal.add(_CurveEdit(change))


def _new_curve(curve_type: str, name: str) -> Any:
    """A new unconnected curve (``cmds.createNode``, raw: a node made after the pending
    journal commutes with it), joining the open ``with container()`` scope; returns its
    MObject. The name is not given the scope's prefix: it comes from the keyed node."""
    from rig._internal.container import container

    made = cmds.createNode(curve_type, name=name, skipSelect=True)
    sel = OpenMaya.MSelectionList()
    sel.add(_name_of(made))
    mobj = sel.getDependNode(0)
    if container._stack:
        from rig.nodetypes._base import _cast

        container._add(_cast(mobj), made=True)
    return mobj


def _apply(plans: list) -> None:
    """Every plan's edits in one undo step (``rig.animcurve``); nothing at all when every
    curve holds its keys already and no compound above or at a keyed plug is connected
    whole. A compound connected whole gives up that input first: one driver per channel,
    so a curve that holds its keys under such a wire (two drivers, as Maya's setKeyframe
    leaves them) is kept, and the whole wire is split (above the plug) or removed (on it)."""
    if not any(
        plan.whole_above or plan.compound_input or any(not e.same for e in plan.leaves)
        for plan in plans
    ):
        return
    from rig._internal.plug import Plug, _disconnect_incoming, _free_of_whole_wire, _set_or_connect

    with _undo_chunk(STEP):
        for plan in plans:
            if plan.whole_above:
                _free_of_whole_wire(plan.dst, "key")
                plan.compound_input = plan.dst._mplug.isCompound and plan.dst._mplug.isDestination
            if plan.compound_input:
                LOGGER.info("key %s: replacing its input %s", plan.dst, plan.dst._mplug.source().name())
                _disconnect_incoming(plan.dst)
            for e in plan.leaves:
                if e.same:
                    continue
                if e.curve is None:
                    mobj = _new_curve(e.curve_type, e.name)
                    _write(_FN(mobj), e)
                    out = Plug(OpenMaya.MFnDependencyNode(mobj).findPlug("output", False))
                    if e.replaces:
                        LOGGER.info("key %s: replacing its input %s with %s", e.leaf, e.replaces, out)
                    _set_or_connect(out, e.leaf, _dst_compound=False)
                else:
                    _write(_FN(e.curve.object()), e)


# --------------------------------------------------------------------------------------- #
#  The operator forms
# --------------------------------------------------------------------------------------- #


def inject(dst: Any, keys: _Keys) -> Any:
    """``dst << keys``: keys ``dst`` (a scalar plug, or each channel of a compound) and
    returns ``dst``."""
    _apply([_plan_plug(keys, dst)])
    return dst


def inject_rows(rows: list) -> None:
    """``List(plugs) << keys`` / ``<< [keys, keys]``: every row planned first (a refused row
    writes nothing, not even the rows before it), then every curve in one undo step. Every
    left-hand side is a plug, every right-hand side keys."""
    from rig.nodetypes._base import Attribute

    plans = []
    for i, (lhs, rhs) in enumerate(rows):
        if type(rhs) is not _Keys:
            raise TypeError(
                f"a pair broadcast with AnimCurve keys takes keys on every right-hand side; [{i}] "
                f"is {rhs!r}: write one '<<' per kind"
            )
        if not isinstance(lhs, Attribute):
            raise TypeError(
                f"element [{i}] ({lhs!r}) is not a plug: an AnimCurve keys a plug, "
                f"List(nodes).<attr> << AnimCurve(...)"
            )
        plans.append(_plan_plug(rhs, lhs, f"List row {i}, {lhs} << AnimCurve(...)"))
    # a plug named by two rows (``List([a, a]).tx << ...``) is keyed once, by the last row,
    # as the last of two values stays: each row was planned against the scene before any
    # edit, so both would otherwise make a new curve and leave the first one unconnected
    last: dict = {}
    for plan in plans:
        for e in plan.leaves:
            earlier = last.get(e.leaf)
            if earlier is not None:
                earlier.same = True  # written by the later row instead
            last[e.leaf] = e
    _apply(plans)


def fanout(pairs: list, fanout_channel: Any) -> None:
    """A per-channel list holding keys (``t << [AnimCurve(xs), skip, 0]``): the keys slots
    are planned first, then every slot runs, in one undo step (``rig.animcurve``)."""
    planned = {}
    for i, (src, dst) in enumerate(pairs):
        if type(src) is _Keys:
            planned[i] = _plan_plug(src, dst)
    with _undo_chunk(STEP):
        for i, (src, dst) in enumerate(pairs):
            if i in planned:
                _apply([planned[i]])
            else:
                fanout_channel(src, dst)


def node_error(node: Any) -> TypeError:
    """``node << keys``: a curve keys a plug."""
    return TypeError(
        f"Cannot inject AnimCurve keys into the node {_name_of(node)!r}: a curve keys a plug, "
        f"{_name_of(node)}.<attr> << AnimCurve(...)"
    )


def rshift_error(lhs: Any) -> TypeError:
    """``x >> keys``: keys go on the right of ``<<``."""
    return TypeError(
        f"'{_name_of(lhs)} >> AnimCurve(...)' is not a form: keys go on the right of '<<' "
        f"({_name_of(lhs)} << AnimCurve(...) keys it); AnimCurve.of(x) lists the curves driving x"
    )


def curves_of(x: Any) -> list:
    """``AnimCurve.of(x)``: the curve nodes directly driving the plugs of ``x`` (a plug, a
    compound's children, a multi's elements; a node: every plug it has a curve on; a List
    of them), each once, in order. A pairBlend or a unitConversion is not looked through.
    A plug of a node freed (a new scene, a file open) raises "... already deleted!" before
    its MPlug is read; a List holding a deleted or freed element raises (the List rule)."""
    from rig._internal import journal
    from rig.nodetypes._base import Attribute, Node, _cast, _ensure_owner_alive

    if journal.LIST_TYPE:
        journal.check_lists((x,))
    found: list = []
    seen: dict = {}

    def add_source(mp):
        if not mp.isDestination:
            return
        src = mp.source()
        if src.isNull:
            return
        node = src.node()
        if node.hasFn(OpenMaya.MFn.kAnimCurve):
            handle = OpenMaya.MObjectHandle(node)
            known = seen.setdefault(handle.hashCode(), [])
            if not any(h == handle for h in known):
                known.append(handle)
                found.append(_cast(node))

    def walk(mp):
        add_source(mp)
        if mp.isArray:
            for i in mp.getExistingArrayAttributeIndices():
                walk(mp.elementByLogicalIndex(i))
        elif mp.isCompound:
            for c in range(mp.numChildren()):
                walk(mp.child(c))

    def visit(item):
        if isinstance(item, Attribute):
            # a freed node's MPlug points at freed memory: checked before it is read
            _ensure_owner_alive(item)
            walk(item._mplug)
        elif isinstance(item, Node):
            fn = OpenMaya.MFnDependencyNode(type(item)._raw_get_mobject(item))
            for mp in fn.getConnections():
                add_source(mp)
        elif isinstance(item, (list, tuple)):
            for sub in item:
                visit(sub)
        else:
            raise TypeError(
                f"AnimCurve.of() takes a plug, a node or a List of them; got {item!r}"
            )
    from rig._internal.list import List

    visit(x)
    return List(found)

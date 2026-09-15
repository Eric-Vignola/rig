"""
Asymmetric iterator helpers used for vectorised broadcasts.

Pure-Python -- no Maya dependency. Ported from Eric Vignola's BSD-3-licensed
"rig" library (https://github.com/Eric-Vignola/rig), 2023, with Python 2
fallbacks removed.
"""

from __future__ import annotations

from typing import Any, Generator


def _is_sequence(obj: Any) -> bool:
    """Return ``True`` if ``obj`` is a non-string, non-dict sequence."""
    if isinstance(obj, str):
        return False

    try:
        len(obj)
        if isinstance(obj, dict):
            return False
    except Exception:
        return False

    return True


def _yield(obj: Any, index: int) -> Any:
    """Return the element of ``obj`` at the given ``index``.

    For sets, iterate to the requested index (sets are unordered).
    For dicts, return the key at that index.
    For sequences (list/tuple/etc.), index normally.
    For scalars, return ``obj`` itself.

    Empty collections raise :class:`IndexError` with a clear message
    rather than silently returning the container.
    """
    # set
    if isinstance(obj, set):
        if not obj:
            raise IndexError("_yield: empty set has no element at any index")
        for i, elem in enumerate(obj):
            if i == index:
                return elem
        raise IndexError(
            f"_yield: index {index} out of range for set of length {len(obj)}"
        )

    # list, tuple, sequence-like
    if _is_sequence(obj):
        seq = list(obj)
        if not seq:
            raise IndexError("_yield: empty sequence has no element at any index")
        return seq[index]

    # dict, operate on the keys
    if isinstance(obj, dict):
        if not obj:
            raise IndexError("_yield: empty dict has no key at any index")
        for i, key in enumerate(obj):
            if i == index:
                return key
        raise IndexError(
            f"_yield: index {index} out of range for dict of length {len(obj)}"
        )

    # scalar -- return as-is
    return obj


def sequences(*args: Any) -> Generator[list, None, None]:
    """Asymmetric generator that broadcasts unequal-length sequences.

    Yields one row at a time, capping shorter inputs to their last entry.

    Example::

        >>> for x in sequences(5, "hey", ["wow", "cool"], [["a"], ["b"], ["c"]]):
        ...     print(x)
        [5, 'hey', 'wow', ['a']]
        [5, 'hey', 'cool', ['b']]
        [5, 'hey', 'cool', ['c']]
    """
    counts = [
        len(x) if (isinstance(x, (set, tuple, list, dict)) or _is_sequence(x)) else 1
        for x in args
    ]
    max_size = max(counts) if counts else 0

    for i in range(max_size):
        result = []
        for j, c in enumerate(counts):
            index = min(i, c - 1)
            result.append(_yield(args[j], index))
        yield result


def dictionaries(**kargs: Any) -> Generator[dict, None, None]:
    """Asymmetric ``dict`` generator. Same broadcast rules as :func:`sequences`.

    Example::

        >>> for x in dictionaries(neat=[5, 6, 7], great=3, woot=[[1, 2], [3, 4, 5]]):
        ...     print(x)
        {'neat': 5, 'great': 3, 'woot': [1, 2]}
        {'neat': 6, 'great': 3, 'woot': [3, 4, 5]}
        {'neat': 7, 'great': 3, 'woot': [3, 4, 5]}
    """
    keys = list(kargs.keys())
    vals = list(kargs.values())

    counts   = [len(x) if isinstance(x, (set, tuple, list, dict)) else 1 for x in vals]
    max_size = max(counts) if counts else 0

    for i in range(max_size):
        result = {}
        for j, c in enumerate(counts):
            index           = min(i, c - 1)
            result[keys[j]] = _yield(vals[j], index)
        yield result


def arguments(*args: Any, **kargs: Any) -> Generator[tuple, None, None]:
    """Combined asymmetric generator for positional + keyword arguments.

    Yields ``(args_list, kwargs_dict)`` tuples. Used by ``@vectorize`` to
    broadcast a function's call-site arguments.
    """
    keys = None
    vals = None
    counts:    list[int] = []
    kw_counts: list[int] = []
    max_count = 0

    if args:
        counts = [
            len(x)
            if (_is_sequence(x) or isinstance(x, (set, tuple, list, dict)))
            else 1
            for x in args
        ]
        max_count = max(counts)

    if kargs:
        keys = list(kargs.keys())
        vals = list(kargs.values())
        kw_counts = [
            len(x)
            if (_is_sequence(x) or isinstance(x, (set, tuple, list, dict)))
            else 1
            for x in vals
        ]
        max_count = max(kw_counts + [max_count])

    for i in range(max_count):
        result = []
        for j, c in enumerate(counts):
            index = min(i, c - 1)
            result.append(_yield(args[j], index))

        kw_result = {}
        for j, c in enumerate(kw_counts):
            index              = min(i, c - 1)
            kw_result[keys[j]] = _yield(vals[j], index)

        yield result, kw_result
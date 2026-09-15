"""
Utility module for parsing maya node names.
"""

from __future__ import annotations

import itertools
import re
from typing import Any, Iterator


def get_short_name(name: Any) -> str:
    """Returns the short name of a given node."""
    return str(name).rsplit("|", 1)[-1]


def get_clean_name(name: Any) -> str:
    """Returns clean name of a given node (no namespace)"""
    return get_short_name(name).rsplit(":", 1)[-1]


def get_suffix(shape_name: Any) -> str:
    """Given a string name, return last string split by underscores"""
    name = str(shape_name).split("_")
    if len(name) > 1:
        return name[-1]
    return shape_name


def replace_suffix(name: Any, suffix: str) -> str:
    """Replaces a node name's suffix.

    Args:
        suffix: The suffix to use. Can NOT contain `_`.

    Returns:
        New name with the suffix replaced.
    """
    if suffix.find("_") != -1:
        raise ValueError("Suffix can NOT contain '_'")
    name = get_short_name(name)
    i    = name.rfind("_")
    # if no suffix, append it
    if i == -1:
        return f"{name}_{suffix}"
    return name[: i + 1] + suffix


def strip_prefix(shape_name: Any) -> str:
    """Given a string name, remove first string separated by an underscore"""
    name = str(shape_name).split("_")
    return "_".join(name[1:])


def has_pattern(name: str, pattern: str = r"[a-z]{3}\d{5}") -> bool:
    """Returns True if shape contains regex pattern.

    Args:
        name (str): String name to search within for regex pattern.
        pattern (str): Regex pattern to search within string.
    """
    tag = re.compile(pattern)
    if tag.search(name):
        return True
    return False


def iter_component_ranges(prefix: str, ids: list[int]) -> Iterator[str]:
    """A generator that converts a list of indices to a list of range strings.

    e.g.
    ```python
    iter_component_ranges("vtx", [1, 2, 3, 5, 7, 8])
    # ["vtx[1:3]", "vtx[5]", "vtx[7:8]"]
    ```
    """
    gen = itertools.groupby(enumerate(ids), lambda pair: pair[1] - pair[0])
    for _, bit in gen:
        b     = list(bit)
        start = b[0][1]
        end   = b[-1][1]
        if start == end:
            yield f"{prefix}[{start}]"
        else:
            yield f"{prefix}[{start}:{end}]"
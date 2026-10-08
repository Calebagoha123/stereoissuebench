#!/usr/bin/env python3
"""Equal-ink marker sizing, shared by every figure that encodes model identity as a
shape.

Matplotlib's ``markersize`` is a *bounding-box* dimension, not an amount of ink, and
the five shapes in the model vocabulary do not fill their box equally: a triangle
covers half of it, a circle 79%, the filled X 63%. Drawing them all at one ``ms``
therefore renders Llama (``^``) and Sonnet (``v``) at half the visual weight of
Gemma (``s``), which reads as a size difference the data does not have.

``marker_ms(marker, base)`` returns the ``ms`` at which a shape carries the same ink
as a square of side ``base``; ``scatter_s`` is the same correction for
``ax.scatter``, whose ``s`` is an area in points^2. The multipliers are measured off
the marker paths at import, so adding a shape to the vocabulary needs no new
constants.
"""
from __future__ import annotations

import numpy as np
from matplotlib.markers import MarkerStyle


def _marker_area(marker: str) -> float:
    """Ink area of a marker path at ms = 1, in the unit box matplotlib draws it in."""
    ms = MarkerStyle(marker)
    path = ms.get_path().transformed(ms.get_transform())
    total = 0.0
    for poly in path.to_polygons():
        x, y = poly[:, 0], poly[:, 1]
        total += abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return total


_SCALE: dict[str, float] = {}


def _scale(marker: str) -> float:
    if marker not in _SCALE:
        area = _marker_area(marker)
        # Square = 1.0 reference. Fall back to no correction for exotic markers
        # (text-path markers, "*", the thumbs-up Path) whose area comes out 0.
        _SCALE[marker] = (_marker_area("s") / area) ** 0.5 if area > 0 else 1.0
    return _SCALE[marker]


def marker_ms(marker: str, base: float) -> float:
    """``ms`` for `marker` carrying the same ink as a square of side `base` points."""
    return base * _scale(marker)


def scatter_s(marker: str, base: float) -> float:
    """``s`` (points^2) for ``ax.scatter`` matching a square of side `base` points."""
    return (base * _scale(marker)) ** 2

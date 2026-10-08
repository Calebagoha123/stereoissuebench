#!/usr/bin/env python3
"""Shared typography for the thesis figures: match the document's own type.

``~/Desktop/SDS/thesis/main.tex`` is ``\\documentclass[12pt]{article}`` with T1
fontenc and no font package, so the thesis is set in **Computer Modern** (not Times
-- an earlier guess). The figures therefore use a CM serif stack for text and
``mathtext.fontset="cm"`` for math, so ``$\\hat{\\Delta}_k$`` in an axis label and in
a caption are the same shapes.

No ``text.usetex``: real LaTeX typesetting would additionally mean escaping every
``%`` in the tick and segment labels, for a difference invisible at figure scale
once the face matches.

Font order: CMU Serif and Latin Modern Roman are the packaged CM clones present on
TeX installs (so a rebuild on Brains picks the better hinted face), then matplotlib's
own bundled ``cmr10``, then DejaVu Serif as a coverage fallback.

Two consequences of cmr10 that callers must respect:

* it has no bold face, so ``fontweight="bold"`` silently renders as regular -- carry
  emphasis with colour or size instead;
* its glyph coverage is essentially ASCII, so arrows and multiplication signs must be
  written as mathtext (``r"$\\leftarrow$"``, ``r"$\\times$"``) rather than as literal
  unicode, and ``axes.unicode_minus`` must stay False.
"""
from __future__ import annotations

SERIF_RC = {
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Latin Modern Roman", "cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,      # cmr10 has no U+2212
    "axes.formatter.use_mathtext": True,  # numeric ticks through mathtext (cmr10 asks)
}

# Mathtext stand-ins for the glyphs cmr10 lacks.
ARROW_L = r"$\leftarrow$"
ARROW_R = r"$\rightarrow$"
TIMES = r"$\times$"


def apply(plt) -> None:
    """Merge the serif stack into rcParams. Call after a script's own rcParams block
    so figure-specific settings still win."""
    plt.rcParams.update(SERIF_RC)

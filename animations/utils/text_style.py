"""Text that Pango lays out properly at small sizes.

``Text._text2svg`` hands ``font_size`` straight to Pango, so a 13 pt label is
laid out on a 13 px grid: glyph advances round to whole pixels, and manim then
scales that SVG up to fill the frame. The rounding scales up with it, which is
why small labels come out with ragged, uneven gaps between letters -- "is
ometric", "co ntours" -- as if the words had been drawn rather than typeset.

Laying the same string out at a large size and scaling the mobject down puts it
on a grid fine enough that the rounding is invisible. Metrically the result is
identical to asking for the small size; only the quantisation changes.

Use :func:`label` instead of :class:`~manim.mobject.text.text_mobject.Text`
anywhere the on-screen size is below roughly 30 pt. ``DecimalNumber`` needs no
such help: it builds its glyphs at the default 48 pt and scales them itself.
"""

from __future__ import annotations

from manim import Text

#: Layout size. Large enough that per-glyph rounding is well under a tenth of a
#: pixel at the sizes this project uses, small enough to stay cheap to render.
BASE_FONT_SIZE = 96.0


def label(text: str, font_size: float = 24.0, **kwargs) -> Text:
    """A :class:`Text` that *looks* like ``font_size`` but is laid out at 96 pt."""
    mob = Text(text, font_size=BASE_FONT_SIZE, **kwargs)
    mob.scale(font_size / BASE_FONT_SIZE)
    return mob

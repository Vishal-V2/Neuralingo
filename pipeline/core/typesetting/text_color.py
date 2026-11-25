"""The colour of a bubble's original lettering, so the English can be drawn in it.

Black text (most bubbles) returns None and keeps the default look (black, white
outline). Coloured text (red shouting, blue whispers) and light text (white on a dark
bubble) return their colour.
"""
import cv2
import numpy as np

from pipeline.core.cleaning.text_masking import text_pixels


# Text whose colour is at most this saturated (CIELAB chroma) is grey...
MIN_CHROMA = 20
# ...and grey text at most this light (CIELAB L, 0-100) is black: the default.
MAX_DARK_LIGHTNESS = 45
# Text at least this light gets a black outline, darker text a white one.
LIGHT_TEXT = 60
# Fewer text pixels than this (after trimming the soft edges): too few to judge.
MIN_PIXELS = 30


def _lab(bgr: np.ndarray) -> tuple[float, float, float]:
    """CIELAB (L 0-100, a, b) of one BGR colour."""
    L, a, b = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    return L * 100 / 255, a - 128, b - 128


def text_color(image: np.ndarray | None) -> tuple[tuple[int, int, int], bool] | None:
    """(RGB, light) of the lettering in a text box's original crop, or None for black text.

    light: the text is light, so it needs a dark outline.
    """
    if image is None or image.size == 0:
        return None
    mask = text_pixels(image)
    # Letter cores only: anti-aliased edges are blended with the background.
    core = cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0
    if core.sum() < MIN_PIXELS:
        core = mask > 0
    if core.sum() < MIN_PIXELS:
        return None
    b, g, r = np.median(image[core], axis=0)
    L, a, b_ = _lab((b, g, r))
    if np.hypot(a, b_) < MIN_CHROMA and L <= MAX_DARK_LIGHTNESS:
        return None
    return (int(r), int(g), int(b)), L >= LIGHT_TEXT

"""Speech bubbles filled with one flat colour, found without the bubble model.

The bubble model is trained on ordinary manga bubbles (white, outlined) and misses
bubbles drawn as solid coloured shapes, e.g. blue or crimson ellipses with white
lettering on a black backdrop. Such a bubble is easy to find directly:

  * a region of one flat colour (no colour edges inside it apart from the letters),
  * that is solid once its holes are filled (an ellipse or box, not a ragged shape),
  * whose holes look like lettering (several letter-sized pieces, densely packed).

A semi-transparent bubble over busy art isn't one flat region: lines and shapes of the
art show through as faint edges that cut its fill into pieces of nearly the same
colour. Neighbouring pieces close in colour are merged back into one region first.

Same-coloured bubbles that overlap form one region with two deep notches where their
outlines cross; it is cut between the notches so each bubble is its own region.

Regions whose text a detected bubble already covers are left alone, so this only adds
bubbles the models missed. The exception is text the text model found on its own (a
frameless region) inside the flat bubble: that region becomes the bubble (its box
grows to the bubble's, its text pieces are kept), so it's typeset as a bubble.
"""
import cv2
import numpy as np

from pipeline.core.box_data import SpeechBubble, TextCluster
import constants
from pipeline.core.detection.text_cluster import COVERED_FRACTION, _fraction_inside

BUBBLE_CLASS_NAME = "standard"
CONFIDENCE = 0.5
# Neighbouring pixels differing by less than this (CIELAB) belong to the same flat fill.
FLAT_STEP = 6
# Merging a semi-transparent bubble's pieces: starting from a flat region of at least
# CORE_AREA (share of the page), neighbouring flat regions (within MERGE_REACH px: a
# sharp edge leaves a 3 px band between flat regions)
# smaller than it and within MERGE_DISTANCE (CIELAB) of its colour are absorbed, for up
# to MERGE_ROUNDS rounds (pieces can be several bands of art away from the region
# merging started from). Art showing through a bubble at ~80% opacity shifts its
# colour by up to ~28; lettering differs from its bubble by far more (white on blue:
# ~160). The edge pixels between merged pieces join the fill if within MERGE_DISTANCE.
CORE_AREA = 0.001
MERGE_REACH = 4
MERGE_DISTANCE = 35
MERGE_ROUNDS = 12
# Bubble size as a share of the page (smaller: specks; larger: the backdrop).
MIN_AREA = 0.003
MAX_AREA = 0.25
# Filled-region area / its convex hull area; bubbles are convex (ellipses, boxes).
MIN_SOLIDITY = 0.95
# Overlapping bubbles: a notch in the outline at least this deep (share of the
# region's shorter side, and in px) marks where two outlines cross.
NOTCH_SHARE = 0.05
MIN_NOTCH = 8
MAX_CUTS = 3
# Lettering: holes of at least MIN_PIECE px (area) adding up to at least MIN_LETTERS
# letters, letter-sized (median longer side at least MIN_LETTER px, not specks),
# packed densely within their own bounding box: real lettering fills 0.25-0.49 of it
# (the low end: short columns of different lengths in small, thin kana),
# while a few art lines or sound effects scattered over a flat patch of art fill under
# 0.17. Letters are counted by shape, not by hole: heavy, tightly packed lettering
# runs together, so one hole can be a whole column (one letter wide, N tall = N
# letters); holes thinner than MIN_LETTER (art lines, dots, small kana parts) don't
# count. How much of the bubble the lettering covers varies too much to tell (one
# short column in a big bubble covers ~11%), so it's only capped. Holes within RIM px
# of the outline are ignored (anti-aliased edge).
MIN_LETTERS = 4
MIN_PIECE = 15
MIN_LETTER = 15
MIN_TEXT_DENSITY = 0.2
# Thin, regular-weight lettering fills less of its box (~0.16); a bubble holding at
# least MANY_LETTERS letters is accepted down to MANY_LETTERS_DENSITY. The art patches
# measured so far hold at most 10 letter-sized holes.
MANY_LETTERS = 15
MANY_LETTERS_DENSITY = 0.12
MAX_TEXT_SHARE = 0.5
RIM = 4
# A frameless region at least this much inside a flat bubble's box is that bubble's text.
FRAMELESS_INSIDE = 0.8
# Margin (px) around the lettering for the text box; stays inside the fill so the
# cleaning step sees the bubble's colour around the letters.
TEXT_PADDING = 6


def _flat_regions(image: np.ndarray):
    """Connected areas of near-constant colour: (lab, labels, stats), the latter two like
    connectedComponentsWithStats."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    step = np.zeros(image.shape[:2], np.float32)
    step[:, :-1] = np.linalg.norm(lab[:, 1:] - lab[:, :-1], axis=2)
    step[:-1] = np.maximum(step[:-1], np.linalg.norm(lab[1:] - lab[:-1], axis=2))
    flat = cv2.erode((step < FLAT_STEP).astype(np.uint8), np.ones((3, 3), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(flat, connectivity=4)
    return lab, labels, stats


def _candidate_fills(lab: np.ndarray, labels: np.ndarray, stats: np.ndarray, page_area: int):
    """Yield (x, y, fill) per candidate bubble fill: a flat region, merged with its
    neighbouring near-same-colour pieces (see CORE_AREA); fill is a 0/1 mask whose
    top-left corner is at (x, y) on the page."""
    n = len(stats)
    areas = stats[:, cv2.CC_STAT_AREA]
    flat_labels = labels.ravel()
    counts = np.maximum(1, np.bincount(flat_labels, minlength=n))
    means = np.stack([np.bincount(flat_labels, weights=lab[..., c].ravel(), minlength=n) for c in range(3)],
                     axis=1) / counts[:, None]
    page_h, page_w = labels.shape
    kernel = np.ones((2 * MERGE_REACH + 1,) * 2, np.uint8)
    taken = np.zeros(n, bool)
    for core in np.argsort(-areas):
        if core == 0 or taken[core] or areas[core] < CORE_AREA * page_area or areas[core] > MAX_AREA * page_area:
            continue
        taken[core] = True
        members = [core]
        for _ in range(MERGE_ROUNDS):
            x0 = max(0, stats[members, 0].min() - MERGE_REACH)
            y0 = max(0, stats[members, 1].min() - MERGE_REACH)
            x1 = min(page_w, (stats[members, 0] + stats[members, 2]).max() + MERGE_REACH)
            y1 = min(page_h, (stats[members, 1] + stats[members, 3]).max() + MERGE_REACH)
            window = labels[y0:y1, x0:x1]
            mask = np.isin(window, members).astype(np.uint8)
            near = np.unique(window[(cv2.dilate(mask, kernel) > 0) & (mask == 0)])
            near = near[(near > 0) & ~taken[near] & (areas[near] < areas[core])]
            near = near[np.linalg.norm(means[near] - means[core], axis=1) < MERGE_DISTANCE]
            if not len(near):
                break
            taken[near] = True
            members.extend(int(i) for i in near)

        x = stats[members, 0].min()
        y = stats[members, 1].min()
        x1 = (stats[members, 0] + stats[members, 2]).max()
        y1 = (stats[members, 1] + stats[members, 3]).max()
        window = labels[y:y1, x:x1]
        if len(members) == 1:
            fill = (window == core).astype(np.uint8)
        else:
            fill = np.isin(window, members).astype(np.uint8)
            # The edge pixels between the pieces (not flat, so in no region) in the colour.
            between = (window == 0) & (cv2.dilate(fill, kernel) > 0)
            close = np.linalg.norm(lab[y:y1, x:x1] - means[core], axis=2) < MERGE_DISTANCE
            fill[between & close] = 1
        if MIN_AREA * page_area <= fill.sum() <= MAX_AREA * page_area:
            yield x, y, fill


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    return filled


def _split_overlapping(filled: np.ndarray) -> list[np.ndarray]:
    """Cut a region made of overlapping bubbles between its two deepest outline notches."""
    pieces, done = [filled], []
    for _ in range(MAX_CUTS):
        if not pieces:
            break
        piece = pieces.pop()
        contours, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Slightly simplified: a jagged outline can cross itself, which convexityDefects
        # refuses; if it still does, the region isn't split.
        contour = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1.0, True)
        x, y, w, h = cv2.boundingRect(contour)
        min_depth = max(MIN_NOTCH, NOTCH_SHARE * min(w, h))
        try:
            defects = cv2.convexityDefects(contour, cv2.convexHull(contour, returnPoints=False))
        except cv2.error:
            done.append(piece)
            continue
        notches = sorted((d for d in (defects[:, 0] if defects is not None else []) if d[3] / 256 >= min_depth),
                         key=lambda d: -d[3])[:2]
        if len(notches) < 2:
            done.append(piece)
            continue
        cut = piece.copy()
        a, b = (tuple(int(v) for v in contour[d[2]][0]) for d in notches)
        cv2.line(cut, a, b, 0, thickness=3)
        n, labels = cv2.connectedComponents(cut, connectivity=4)
        parts = [(labels == i).astype(np.uint8) for i in range(1, n)]
        parts = [p for p in parts if p.sum() >= 0.1 * piece.sum()]
        if len(parts) < 2:
            done.append(piece)
        else:
            pieces.extend(parts)
    return done + pieces


def detect_flat_bubbles(image: np.ndarray, existing: list[SpeechBubble],
                        log: list[str] | None = None) -> tuple[list[SpeechBubble], list[SpeechBubble]]:
    """Returns (new bubbles, existing frameless regions turned into flat bubbles in place).

    log: if given, each candidate region's box and what was decided (and why) is
    appended (see tools/detect_diagnose.py); numbers only.
    """
    def note(box, message: str) -> None:
        if log is not None:
            log.append(f"flat region {box}: {message}")

    page_h, page_w = image.shape[:2]
    page_area = page_h * page_w
    lab, labels, stats = _flat_regions(image)
    found: list[SpeechBubble] = []
    upgraded: list[SpeechBubble] = []
    for x, y, fill in _candidate_fills(lab, labels, stats, page_area):
        for bubble in _split_overlapping(_fill_holes(fill)):
            contours, _ = cv2.findContours(bubble, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contour = max(contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(contour)
            box = (int(x + bx), int(y + by), int(x + bx + bw), int(y + by + bh))
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = bubble.sum() / max(1.0, hull_area)
            if solidity < MIN_SOLIDITY:
                note(box, f"rejected: not bubble-shaped (solidity {solidity:.2f} < {MIN_SOLIDITY})")
                continue
            inner = cv2.erode(bubble, np.ones((2 * RIM + 1,) * 2, np.uint8)) > 0
            text = (inner & (fill == 0)).astype(np.uint8)
            if text.sum() > MAX_TEXT_SHARE * bubble.sum():
                note(box, f"rejected: holes cover {text.sum() / bubble.sum():.2f} of it (> {MAX_TEXT_SHARE})")
                continue
            _, piece_labels, piece_stats, _ = cv2.connectedComponentsWithStats(text, connectivity=8)
            kept = np.flatnonzero(piece_stats[:, cv2.CC_STAT_AREA] >= MIN_PIECE)
            kept = kept[kept > 0]
            pieces = piece_stats[kept]
            sides = np.sort(pieces[:, 2:4], axis=1)  # (shorter, longer) side of each hole
            letter_like = sides[sides[:, 0] >= MIN_LETTER]
            letter_count = np.maximum(1, np.round(letter_like[:, 1] / letter_like[:, 0])).sum()
            if letter_count < MIN_LETTERS:
                note(box, f"rejected: {letter_count:.0f} letter-like holes (< {MIN_LETTERS})")
                continue
            median = np.median(np.maximum(pieces[:, 2], pieces[:, 3]))
            if median < MIN_LETTER:
                note(box, f"rejected: holes too small to be letters (median {median:.0f}px < {MIN_LETTER})")
                continue
            px0, py0 = pieces[:, 0].min(), pieces[:, 1].min()
            px1, py1 = (pieces[:, 0] + pieces[:, 2]).max(), (pieces[:, 1] + pieces[:, 3]).max()
            density = np.isin(piece_labels[py0:py1, px0:px1], kept).sum() / max(1, (px1 - px0) * (py1 - py0))
            min_density = MANY_LETTERS_DENSITY if letter_count >= MANY_LETTERS else MIN_TEXT_DENSITY
            if density < min_density:
                note(box, f"rejected: holes too scattered for lettering (density {density:.2f} < {min_density}"
                          f" with {letter_count:.0f} letters)")
                continue

            text_box = (int(x + max(bx, px0 - TEXT_PADDING)), int(y + max(by, py0 - TEXT_PADDING)),
                        int(x + min(bx + bw, px1 + TEXT_PADDING)), int(y + min(by + bh, py1 + TEXT_PADDING)))
            covering = next((sb for sb in existing + found
                             if _fraction_inside(text_box, sb.position) >= COVERED_FRACTION), None)
            if covering is not None:
                if (covering.class_name == constants.FRAMELESS_CLASS_NAME
                        and _fraction_inside(covering.position, box) >= FRAMELESS_INSIDE):
                    note(box, f"bubble; frameless text region {tuple(covering.position)} inside it becomes it")
                    covering.position = box
                    covering.class_name = BUBBLE_CLASS_NAME
                    covering.bounding_image = image[box[1]:box[3], box[0]:box[2]]
                    upgraded.append(covering)
                else:
                    note(box, f"bubble, but its text is already covered by {covering.class_name} {tuple(covering.position)}")
                continue
            note(box, f"bubble added, text box {text_box}")
            found.append(SpeechBubble(
                position=box,
                confidence=CONFIDENCE,
                class_name=BUBBLE_CLASS_NAME,
                bounding_image=image[box[1]:box[3], box[0]:box[2]],
                text_clusters=[TextCluster(
                    position=text_box,
                    confidence=CONFIDENCE,
                    bounding_image=image[text_box[1]:text_box[3], text_box[0]:text_box[2]],
                )],
            ))
    return found, upgraded

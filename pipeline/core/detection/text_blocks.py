"""Paragraph handling for text-heavy pages (page_mode: text).

Comic pipelines expect short vertical text inside bubbles. Pages like character
profiles or afterwords instead have blocks of horizontal paragraphs, which the
text model reports as overlapping boxes of different sizes that cut lines off,
and which are too long for manga_ocr to read in one go.

build_text_blocks() takes the frameless regions found on such a page and:
  1. merges overlapping/nested regions into one region per text area,
  2. leaves vertical regions as single frameless regions,
  3. splits horizontal regions into text lines (blank-row gaps), grows them to pick
     up lines the detector cut off or missed, and splits them into paragraphs.
Each paragraph becomes a PARAGRAPH_CLASS_NAME region whose cluster carries its
line boxes, so OCR can read it line by line and typesetting can lay it out as a
left-aligned paragraph.
"""
import cv2
import numpy as np

import constants
from pipeline.core.box_data import SpeechBubble, TextCluster


PARAGRAPH_CLASS_NAME = "paragraph"

# Regions overlapping by at least this fraction of the smaller one are merged.
MERGE_OVERLAP = 0.5
# A row counts as containing text if at least this fraction of its pixels is ink.
ROW_INK_FRACTION = 0.004
# Ink runs separated by less than this fraction of the median line height are one line
# (e.g. dakuten or a long dash sitting slightly apart).
LINE_JOIN_GAP = 0.1
# Runs shorter than this fraction of the median line height are noise, not lines.
MIN_LINE_HEIGHT = 0.4
# A gap larger than this fraction of the median line height starts a new paragraph...
PARAGRAPH_GAP = 0.8
# ...as does a jump in line height (e.g. a bigger title line) by more than this ratio.
PARAGRAPH_HEIGHT_RATIO = 1.3
# When growing a block, look this many median line heights beyond its edge.
GROW_SEARCH = 2.5
# A grown line must be within this ratio of the median line height to count as text.
GROW_HEIGHT_RATIO = (0.6, 1.6)
# While growing, ink shapes larger than this many line heights are artwork, not text.
GLYPH_MAX_SIZE = 1.5
# Only blocks with at least this many lines are grown.
GROW_MIN_LINES = 2
# A grown line must be at least this many line heights wide.
GROW_MIN_WIDTH = 1.5
# A grown line needs at least this share of its ink in character-sized shapes.
GROW_MIN_GLYPH_SHARE = 0.7
# A line more than this fraction inside an existing bubble belongs to that bubble.
IN_BUBBLE_FRACTION = 0.5
# Padding (px) added around each paragraph box.
PADDING = 6


def _area(box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection(a, b) -> int:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def _union(a, b) -> tuple[int, int, int, int]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _merge_overlapping(boxes: list[tuple]) -> list[tuple]:
    """Repeatedly merge boxes that overlap by MERGE_OVERLAP of the smaller one."""
    boxes = list(boxes)
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if _intersection(a, b) >= MERGE_OVERLAP * max(min(_area(a), _area(b)), 1):
                    boxes[i] = _union(a, b)
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return boxes


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return mask > 0


def _glyph_ink(ink: np.ndarray, line_height: float) -> np.ndarray:
    """Keep only ink shapes small enough to be characters.

    Artwork next to the text (outlines, hair, smoke lines) forms connected shapes far
    larger than a character; leaving them in fills the blank rows between text lines.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    too_big = (stats[:, cv2.CC_STAT_HEIGHT] > GLYPH_MAX_SIZE * line_height) | \
              (stats[:, cv2.CC_STAT_WIDTH] > GLYPH_MAX_SIZE * line_height)
    too_big[0] = True  # background label
    return ~too_big[labels] & ink


def _row_runs(ink: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) row ranges that contain ink."""
    rows = ink.mean(axis=1) >= ROW_INK_FRACTION
    runs, start = [], None
    for y, has_ink in enumerate(rows):
        if has_ink and start is None:
            start = y
        elif not has_ink and start is not None:
            runs.append((start, y))
            start = None
    if start is not None:
        runs.append((start, len(rows)))
    return runs


def _find_lines(ink: np.ndarray, line_height: float | None = None) -> list[tuple[int, int]]:
    """Text line row ranges in an ink mask: join near runs, drop noise.

    line_height: the expected line height. Without it, it's estimated from the mask,
    which is unreliable when the mask also contains artwork (e.g. when growing a block).
    """
    runs = _row_runs(ink)
    if not runs:
        return []
    median = line_height or float(np.median([e - s for s, e in runs]))
    joined = [runs[0]]
    for s, e in runs[1:]:
        if s - joined[-1][1] < LINE_JOIN_GAP * median:
            joined[-1] = (joined[-1][0], e)
        else:
            joined.append((s, e))
    median = line_height or float(np.median([e - s for s, e in joined]))
    return [(s, e) for s, e in joined if e - s >= MIN_LINE_HEIGHT * median]


def _elongation(ink: np.ndarray) -> float:
    """Median length/thickness of the row bands of ink (the lines, if text runs along rows)."""
    ratios = []
    for s, e in _find_lines(ink):
        cols = np.where(ink[s:e].any(axis=0))[0]
        ratios.append((cols[-1] - cols[0] + 1) / max(e - s, 1))
    return float(np.median(ratios)) if ratios else 0.0


def _is_horizontal(gray_page: np.ndarray, box) -> bool:
    """Text runs in the direction whose bands of ink are the most elongated.

    Horizontal lines are bands far wider than tall; vertical columns are bands far
    taller than wide. Counting breaks instead misleads (gaps between characters in a
    single vertical column look like line breaks), and so does the region's shape (a
    block of many horizontal lines can be taller than it is wide).
    """
    x0, y0, x1, y1 = box
    ink = _ink_mask(gray_page[y0:y1, x0:x1])
    return _elongation(ink) >= _elongation(ink.T)


def _without_bubbles(ink: np.ndarray, x0: int, y0: int, bubbles: list[SpeechBubble]) -> np.ndarray:
    """Blank out ink inside detected bubbles (ink is the crop starting at x0, y0)."""
    ink = ink.copy()
    h, w = ink.shape
    for b in bubbles:
        bx0, by0, bx1, by1 = b.position
        ink[max(0, by0 - y0):max(0, min(h, by1 - y0)), max(0, bx0 - x0):max(0, min(w, bx1 - x0))] = False
    return ink


def _inside_bubble(box, bubbles: list[SpeechBubble]) -> bool:
    return any(_intersection(box, b.position) >= IN_BUBBLE_FRACTION * max(_area(box), 1) for b in bubbles)


def _grow(gray_page: np.ndarray, box, bubbles: list[SpeechBubble]) -> tuple:
    """Extend a horizontal block up/down over neighbouring lines of similar height."""
    page_h = gray_page.shape[0]
    x0, y0, x1, y1 = box

    lines = _find_lines(_ink_mask(gray_page[y0:y1, x0:x1]))
    # With fewer lines there's no reliable line height to judge new lines against
    # (e.g. two tightly spaced handwritten lines read as one tall line).
    if len(lines) < GROW_MIN_LINES:
        return box
    line_h = float(np.median([e - s for s, e in lines]))
    lo, hi = GROW_HEIGHT_RATIO[0] * line_h, GROW_HEIGHT_RATIO[1] * line_h
    reach = int(GROW_SEARCH * line_h)

    def text_lines(strip_start: int, strip_end: int) -> list[tuple[int, int]]:
        """Absolute row ranges in the strip that look like a line of text."""
        raw = _without_bubbles(_ink_mask(gray_page[strip_start:strip_end, x0:x1]), x0, strip_start, bubbles)
        glyphs = _glyph_ink(raw, line_h)
        found = []
        for s, e in _find_lines(glyphs, line_h):
            if not lo <= e - s <= hi:
                continue
            # Within the line's own width, text is nearly all character-sized shapes;
            # a band of artwork is mostly large shapes with a few small specks.
            cols = np.where(glyphs[s:e].any(axis=0))[0]
            # A lone stroke of artwork can be line-height tall; a line is a few characters wide.
            if cols[-1] - cols[0] + 1 < GROW_MIN_WIDTH * line_h:
                continue
            raw_ink = raw[s:e, cols[0]:cols[-1] + 1].sum()
            if glyphs[s:e, cols[0]:cols[-1] + 1].sum() < GROW_MIN_GLYPH_SHARE * raw_ink:
                continue
            found.append((strip_start + s, strip_start + e))
        return found

    # Downwards: look at a strip below the block; take its first line if it looks like text.
    while y1 < page_h:
        # Re-read from the last line's top so a line cut by the box edge is seen whole.
        start = max(y0, y1 - int(hi))
        nxt = next(((s, e) for s, e in text_lines(start, min(page_h, y1 + reach)) if e > y1), None)
        if nxt is None or _inside_bubble((x0, nxt[0], x1, nxt[1]), bubbles):
            break
        y1 = nxt[1]

    # Upwards, symmetrically.
    while y0 > 0:
        candidates = text_lines(max(0, y0 - reach), min(y1, y0 + int(hi)))
        prev = next(((s, e) for s, e in reversed(candidates) if s < y0), None)
        if prev is None or _inside_bubble((x0, prev[0], x1, prev[1]), bubbles):
            break
        y0 = prev[0]

    return (x0, y0, x1, y1)


def _paragraphs(gray_page: np.ndarray, box, bubbles: list[SpeechBubble]) -> list[list[tuple]]:
    """Split a horizontal block into paragraphs; each is a list of absolute line boxes."""
    x0, y0, x1, y1 = box
    ink = _without_bubbles(_ink_mask(gray_page[y0:y1, x0:x1]), x0, y0, bubbles)
    lines = _find_lines(ink)
    if not lines:
        return []
    # Second pass without artwork shapes, using the first pass's typical line height.
    median = float(np.median([e - s for s, e in lines]))
    ink = _glyph_ink(ink, median)
    lines = _find_lines(ink, median)
    if not lines:
        return []

    line_boxes = []
    for s, e in lines:
        cols = np.where(ink[s:e].any(axis=0))[0]
        line_boxes.append((x0 + int(cols[0]), y0 + s, x0 + int(cols[-1]) + 1, y0 + e))

    paragraphs = [[line_boxes[0]]]
    for prev, cur in zip(line_boxes, line_boxes[1:]):
        gap = cur[1] - prev[3]
        h_prev, h_cur = prev[3] - prev[1], cur[3] - cur[1]
        ratio = max(h_prev, h_cur) / max(min(h_prev, h_cur), 1)
        if gap > PARAGRAPH_GAP * median or ratio > PARAGRAPH_HEIGHT_RATIO:
            paragraphs.append([cur])
        else:
            paragraphs[-1].append(cur)
    return paragraphs


def _region(image: np.ndarray, box, class_name: str, confidence: float, lines=None) -> SpeechBubble:
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1]
    return SpeechBubble(
        position=box,
        confidence=confidence,
        class_name=class_name,
        bounding_image=crop,
        text_clusters=[TextCluster(position=box, confidence=confidence, bounding_image=crop, lines=lines)],
    )


def build_text_blocks(image: np.ndarray, frameless: list[SpeechBubble], bubbles: list[SpeechBubble]) -> list[SpeechBubble]:
    """Turn a text page's frameless regions into deduplicated vertical regions and paragraphs.

    frameless: the page's frameless regions (replaced by the result).
    bubbles:   the page's other regions; text inside them is left alone.
    """
    if not frameless:
        return []
    page_h, page_w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def confidence_of(box) -> float:
        return max((f.confidence for f in frameless if _intersection(box, f.position) > 0), default=0.0)

    regions = _merge_overlapping([tuple(f.position) for f in frameless])
    # Grow horizontal blocks, then merge again: growing can make blocks touch.
    regions = [_grow(gray, r, bubbles) if _is_horizontal(gray, r) else r for r in regions]
    regions = _merge_overlapping(regions)

    result: list[SpeechBubble] = []
    for box in regions:
        conf = confidence_of(box)
        if not _is_horizontal(gray, box):
            result.append(_region(image, box, constants.FRAMELESS_CLASS_NAME, conf))
            continue
        for lines in _paragraphs(gray, box, bubbles):
            para = (
                max(0, min(l[0] for l in lines) - PADDING), max(0, lines[0][1] - PADDING),
                min(page_w, max(l[2] for l in lines) + PADDING), min(page_h, lines[-1][3] + PADDING),
            )
            result.append(_region(image, para, PARAGRAPH_CLASS_NAME, conf, lines=lines))
    return result

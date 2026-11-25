"""Grow text boxes inside bubbles to cover all of their lettering.

The text model sometimes boxes only part of a bubble's text: some of the columns, or
columns cut short (more often when art shows through a semi-transparent bubble). A
box is grown, shape by shape, over the dark glyph-sized shapes lying within a small
gap of it, as long as each sits on a light surround the way lettering in a bubble
does; dark art and the bubble's outline don't, and are left out.

Sometimes the text model misses a bubble's lettering altogether and boxes something
else inside the bubble's box instead (e.g. dark hatched art in a corner outside the
bubble's outline). Lettering in a bubble that no text box covers is found the same way
(glyph-sized dark shapes on a light surround, grouped into blocks) and boxed; a text
box with no lettering in it is dropped when the bubble has one that has.

Afterwards, two detected bubbles whose text boxes now cover the same lettering (the
bubble model sometimes boxes one bubble twice) keep it only once.
"""
import cv2
import numpy as np

import constants
from pipeline.core.box_data import SpeechBubble, TextCluster

# Ink: pixels darker than this. Art showing faintly through a white bubble is lighter.
INK_MAX_GRAY = 110
# Light surround: pixels lighter than this, at least SURROUND_LIGHT of a glyph's ring
# (GLYPH_RING px wide around its box, minus other ink).
LIGHT_MIN_GRAY = 170
SURROUND_LIGHT = 0.6
GLYPH_RING = 4
# Glyph size, from the shapes inside the original box: the CHAR_PERCENTILE of their
# longer sides (characters split into several shapes; the larger ones are whole).
CHAR_PERCENTILE = 80
# Shapes no larger than this many characters (longer side) are glyphs; larger dark
# shapes are art or the bubble's outline.
GLYPH_MAX = 1.5
# A glyph joins the box when within this many characters of it (column gaps in
# vertical text are ~0.3 characters; line gaps similar).
JOIN_GAP = 0.7
# Search area around the bubble box, in characters: lettering can poke slightly past a
# bubble box that's cut short, but further out is other bubbles' text or art.
SEARCH_MARGIN = 0.25
# Two bubbles' text boxes overlapping this much (intersection over the smaller) hold
# the same text.
SAME_TEXT_OVERLAP = 0.8
# Missing lettering: glyph shapes at least MIN_GLYPH px and at most GLYPH_SHARE of the
# bubble's shorter side are candidates; character size is the CHAR_PERCENTILE of those
# of at least MIN_CHAR px. A block of glyphs (joined within JOIN_GAP characters) is
# lettering if it adds up to at least MIN_LETTERS letters (shapes at least
# LETTER_SIDE characters across; a column of touching characters counts as several),
# its glyph boxes cover at least BLOCK_COVER of its box, and at least BLOCK_LIGHT of
# its box is light (lettering sits on the bubble's light interior).
MIN_GLYPH = 6
MIN_CHAR = 8
GLYPH_SHARE = 0.4
MIN_LETTERS = 4
LETTER_SIDE = 0.4
BLOCK_COVER = 0.3
BLOCK_LIGHT = 0.6
TEXT_PADDING = 4
# Classes not grown: their text regions are built from the lettering already.
SKIP_CLASSES = {constants.FRAMELESS_CLASS_NAME, "caption", "paragraph"}


def _gap(a, b) -> float:
    """Distance between two boxes (0 if they touch or overlap)."""
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return max(dx, dy)


def _glyphs(gray: np.ndarray, char: float | None, skip_edge: bool = False) -> list[tuple[int, int, int, int]]:
    """Boxes (x0, y0, x1, y1) of dark shapes on a light surround, at most GLYPH_MAX
    characters (if char is known). skip_edge: leave out shapes touching the area's
    edge (pieces of something larger cut by it, e.g. the bubble's outline)."""
    ink = (gray < INK_MAX_GRAY).astype(np.uint8)
    light = gray > LIGHT_MIN_GRAY
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    h, w = gray.shape
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 4 or (char is not None and max(bw, bh) > GLYPH_MAX * char):
            continue
        if skip_edge and (x == 0 or y == 0 or x + bw == w or y + bh == h):
            continue
        r0, r1 = max(0, y - GLYPH_RING), min(h, y + bh + GLYPH_RING)
        c0, c1 = max(0, x - GLYPH_RING), min(w, x + bw + GLYPH_RING)
        ring = np.ones((r1 - r0, c1 - c0), bool)
        ring[y - r0:y - r0 + bh, x - c0:x - c0 + bw] = False
        ring &= ink[r0:r1, c0:c1] == 0
        if ring.any() and light[r0:r1, c0:c1][ring].mean() < SURROUND_LIGHT:
            continue
        boxes.append((x, y, x + bw, y + bh))
    return boxes


def _area(box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _overlap(a, b) -> float:
    """Intersection of two boxes over the smaller one's area."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / max(1, smaller)


def grow_cluster(image: np.ndarray, bubble: SpeechBubble, cluster: TextCluster,
                 blocked: list[tuple], foreign: list[tuple]) -> bool:
    """Grow cluster's box over the rest of its lettering. Returns True if it changed.

    blocked: other text boxes, which the grown box must not touch; foreign: other
    bubbles' boxes, whose glyphs are theirs (so overlapping bubbles don't swap text).
    """
    page_h, page_w = image.shape[:2]
    cx0, cy0, cx1, cy1 = cluster.position
    if cx1 <= cx0 or cy1 <= cy0:
        return False
    gray_page = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    inside = _glyphs(gray_page[cy0:cy1, cx0:cx1], None)
    sizes = [max(b[2] - b[0], b[3] - b[1]) for b in inside]
    if len(sizes) < 2:
        return False
    char = float(np.percentile(sizes, CHAR_PERCENTILE))
    if char < 6:
        return False

    margin = int(SEARCH_MARGIN * char)
    bx0, by0, bx1, by1 = bubble.position
    sx0, sy0 = max(0, bx0 - margin), max(0, by0 - margin)
    sx1, sy1 = min(page_w, bx1 + margin), min(page_h, by1 + margin)
    glyphs = [(x0 + sx0, y0 + sy0, x1 + sx0, y1 + sy0)
              for x0, y0, x1, y1 in _glyphs(gray_page[sy0:sy1, sx0:sx1], char, skip_edge=True)]
    def foreign_glyph(g) -> bool:
        gx, gy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
        return any(f[0] <= gx < f[2] and f[1] <= gy < f[3] for f in foreign)

    glyphs = [g for g in glyphs if not foreign_glyph(g) and not any(_gap(g, b) == 0 for b in blocked)]

    box = list(cluster.position)
    joined = True
    while joined:
        joined = False
        rest = []
        for g in glyphs:
            if _gap(g, box) <= JOIN_GAP * char:
                grown = [min(box[0], g[0]), min(box[1], g[1]), max(box[2], g[2]), max(box[3], g[3])]
                if any(_gap(grown, b) == 0 for b in blocked):
                    rest.append(g)
                    continue
                box = grown
                joined = True
            else:
                rest.append(g)
        glyphs = rest

    box = tuple(int(v) for v in box)
    if box == tuple(cluster.position):
        return False
    cluster.position = box
    cluster.bounding_image = image[box[1]:box[3], box[0]:box[2]]
    return True


def grow_text_clusters(image: np.ndarray, bubbles: list[SpeechBubble]) -> None:
    # A bubble boxed twice by the bubble model (boxes nearly the same) isn't "another
    # bubble", and text boxes covering the same text don't block each other; the
    # duplicates are dropped afterwards (drop_duplicate_text).
    clusters = [c for b in bubbles for c in (b.text_clusters or [])]
    for bubble in bubbles:
        if bubble.class_name in SKIP_CLASSES or not bubble.text_clusters:
            continue
        foreign = [b.position for b in bubbles
                   if b is not bubble and _overlap(b.position, bubble.position) < SAME_TEXT_OVERLAP]
        for cluster in bubble.text_clusters:
            blocked = [c.position for c in clusters
                       if c is not cluster and _overlap(c.position, cluster.position) < SAME_TEXT_OVERLAP]
            grow_cluster(image, bubble, cluster, blocked, foreign)


def _letter_blocks(gray: np.ndarray, x0: int, y0: int, glyphs: list[tuple], char: float,
                   notes: list[str] | None = None) -> list[tuple]:
    """Group glyph boxes (page coordinates; gray is the area at (x0, y0)) into blocks of
    lettering; returns the boxes of the blocks that look like text. notes: if given,
    each block and the verdict on it is appended."""
    h, w = gray.shape
    reach = max(1, int(JOIN_GAP * char / 2))
    canvas = np.zeros((h, w), np.uint8)
    for gx0, gy0, gx1, gy1 in glyphs:
        cv2.rectangle(canvas, (gx0 - x0 - reach, gy0 - y0 - reach), (gx1 - x0 + reach - 1, gy1 - y0 + reach - 1), 1, -1)
    n, labels = cv2.connectedComponents(canvas, connectivity=8)
    blocks = []
    for i in range(1, n):
        members = [g for g in glyphs if labels[(g[1] + g[3]) // 2 - y0, (g[0] + g[2]) // 2 - x0] == i]
        if not members:
            continue
        sides = np.sort(np.array([(g[2] - g[0], g[3] - g[1]) for g in members]), axis=1)
        letter_like = sides[sides[:, 0] >= LETTER_SIDE * char]
        letters = np.maximum(1, np.round(letter_like[:, 1] / letter_like[:, 0])).sum() if len(letter_like) else 0
        bx0, by0 = min(g[0] for g in members), min(g[1] for g in members)
        bx1, by1 = max(g[2] for g in members), max(g[3] for g in members)
        cover = sum((g[2] - g[0]) * (g[3] - g[1]) for g in members) / max(1, (bx1 - bx0) * (by1 - by0))
        light = (gray[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0] > LIGHT_MIN_GRAY).mean()
        ok = letters >= MIN_LETTERS and cover >= BLOCK_COVER and light >= BLOCK_LIGHT
        if notes is not None:
            notes.append(f"block {(bx0, by0, bx1, by1)}: {len(members)} shapes, {letters:.0f} letters (min {MIN_LETTERS}), "
                         f"cover {cover:.2f} (min {BLOCK_COVER}), light {light:.2f} (min {BLOCK_LIGHT})"
                         f" -> {'lettering' if ok else 'not lettering'}")
        if ok:
            blocks.append((bx0, by0, bx1, by1))
    return blocks


def add_missing_text(image: np.ndarray, bubbles: list[SpeechBubble], log: list[str] | None = None) -> None:
    """Box lettering inside detected bubbles that no text box covers, and drop text
    boxes with no lettering in them (when the bubble has a text box that has).
    log: if given, what was found and decided per bubble is appended (numbers only)."""
    def note(message: str) -> None:
        if log is not None:
            log.append(message)

    page_h, page_w = image.shape[:2]
    gray_page = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for bubble in bubbles:
        if bubble.class_name in SKIP_CLASSES:
            continue
        note(f"{bubble.class_name} {tuple(bubble.position)}, text boxes {[tuple(c.position) for c in bubble.text_clusters or []]}:")
        x0, y0, x1, y1 = (int(v) for v in bubble.position)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(page_w, x1), min(page_h, y1)
        if x1 - x0 < 2 * MIN_GLYPH or y1 - y0 < 2 * MIN_GLYPH:
            continue
        gray = gray_page[y0:y1, x0:x1]
        limit = GLYPH_SHARE * min(x1 - x0, y1 - y0)
        glyphs = [(gx0 + x0, gy0 + y0, gx1 + x0, gy1 + y0)
                  for gx0, gy0, gx1, gy1 in _glyphs(gray, None, skip_edge=True)
                  if MIN_GLYPH <= max(gx1 - gx0, gy1 - gy0) <= limit]
        sizes = [max(g[2] - g[0], g[3] - g[1]) for g in glyphs if max(g[2] - g[0], g[3] - g[1]) >= MIN_CHAR]
        if len(sizes) < MIN_LETTERS:
            note(f"    only {len(sizes)} glyph-like shapes of {MIN_CHAR}px+ (dark on a light surround); nothing to add")
            continue
        char = float(np.percentile(sizes, CHAR_PERCENTILE))
        glyphs = [g for g in glyphs if max(g[2] - g[0], g[3] - g[1]) <= GLYPH_MAX * char]
        clusters = bubble.text_clusters or []
        note(f"    {len(glyphs)} glyph-like shapes, character size ~{char:.0f}px")

        def covered(g) -> bool:
            gx, gy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
            return any(c.position[0] <= gx < c.position[2] and c.position[1] <= gy < c.position[3] for c in clusters)

        def has_lettering(c) -> bool:
            return bool(_letter_blocks(gray, x0, y0, [g for g in glyphs if _overlap(g, c.position) >= 0.5], char))

        added = []
        uncovered = [g for g in glyphs if not covered(g)]
        note(f"    {len(uncovered)} of them outside the text boxes")
        notes: list[str] = []
        blocks = _letter_blocks(gray, x0, y0, uncovered, char, notes if log is not None else None)
        for line in notes:
            note(f"    {line}")
        for bx0, by0, bx1, by1 in blocks:
            box = (max(0, bx0 - TEXT_PADDING), max(0, by0 - TEXT_PADDING),
                   min(page_w, bx1 + TEXT_PADDING), min(page_h, by1 + TEXT_PADDING))
            added.append(TextCluster(position=box, confidence=bubble.confidence,
                                     bounding_image=image[box[1]:box[3], box[0]:box[2]]))
        if not added:
            note("    nothing added")
            continue
        # Text boxes without lettering (art the text model mistook for text) give way.
        keep = [c for c in clusters if has_lettering(c)]
        note(f"    added {[c.position for c in added]}; dropped (no lettering) "
             f"{[tuple(c.position) for c in clusters if not any(c is k for k in keep)]}")
        bubble.text_clusters = keep + added


def drop_duplicate_text(bubbles: list[SpeechBubble]) -> list[SpeechBubble]:
    """Keep each piece of text once when text boxes cover the same lettering: two boxes
    in one bubble (one inside the other), or two bubbles (the bubble model sometimes
    boxes one bubble twice). Larger boxes in stronger bubbles win; a bubble left without
    text is dropped."""
    kept: list[TextCluster] = []
    emptied = []
    for bubble in sorted(bubbles, key=lambda b: -b.confidence):
        if bubble.class_name in SKIP_CLASSES or not bubble.text_clusters:
            kept.extend(bubble.text_clusters or [])
            continue
        remaining = []
        for c in sorted(bubble.text_clusters, key=lambda c: -_area(c.position)):
            if not any(_overlap(c.position, k.position) >= SAME_TEXT_OVERLAP for k in kept):
                remaining.append(c)
                kept.append(c)
        # Keep the bubble's own order (reading order matters downstream).
        bubble.text_clusters = [c for c in bubble.text_clusters if any(c is r for r in remaining)]
        if not bubble.text_clusters:
            emptied.append(bubble)
    return [b for b in bubbles if not any(b is e for e in emptied)]

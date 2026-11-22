"""Multi-speaker vertical captions drawn straight on the art.

Some pages put several characters' lines side by side as outlined vertical text on
the art (black outline, coloured fill, one fill colour per speaker). The detectors
return the whole group as one frameless box, which manga_ocr reads as a single image
and jumbles across columns.

split_caption() finds the columns from the letter fills (the light, small areas fully
enclosed by the black outline; hair, clothes and background are large connected areas
and drop out), groups neighbouring columns of the same fill colour into one speaker's
line, and returns a CAPTION_CLASS_NAME region with one text cluster per speaker, in
reading order (right to left). Each cluster carries its column boxes (lines) and the
speaker's fill colour, so OCR reads it column by column, cleaning masks only the
letters, and typesetting can draw each speaker's English in their colour.
"""
import cv2
import numpy as np

from pipeline.core.box_data import SpeechBubble, TextCluster
from pipeline.core.detection.text_blocks import _is_horizontal


CAPTION_CLASS_NAME = "caption"

# Pixels darker than this are outline (or dark art); the letter fill is lighter.
OUTLINE_MAX_GRAY = 60
# Letter fill must also average at least this bright: a dark background just above
# OUTLINE_MAX_GRAY breaks up (with image noise) into specks that would pass otherwise.
FILL_MIN_MEAN_GRAY = 100
# Context (px) added around the region so letters at its edge are fully enclosed.
CONTEXT = 12
# Fill shapes smaller than this many pixels are specks, not strokes.
MIN_FILL_AREA = 4
# Fill shapes wider/taller than this many character widths are art, not letters.
FILL_MAX_SIZE = 1.5
# Columns are found from the horizontal fill profile (smoothed over this many px):
PROFILE_SMOOTHING = 5
# x positions whose fill is below this fraction of the profile's peak are gaps.
PROFILE_GAP = 0.05
# Column spacing (px) is searched for in this range.
PITCH_RANGE = (20, 200)
# The first autocorrelation peak at least this share of the strongest is the spacing.
PITCH_PEAK_SHARE = 0.6
# A spacing peak must rise at least this far (correlation, 1.0 = perfect) above the dip
# before it.
PITCH_MIN_PROMINENCE = 0.3
# With no clear spacing within half the fill's width (two columns), look up to this
# fraction of it.
TWO_COLUMN_REACH = 0.75
# The spacing is measured over rows where most columns are present (fill in at least
# this share of the busiest rows' x positions, counted over BUSY_BAND rows), so one
# long column running past the others doesn't drown out the spacing.
BUSY_BAND = 31
BUSY_SHARE = 0.5
# Detectors often cut long captions short: columns are followed past the detected box
# while their fill continues with gaps of at most this many column spacings.
GROW_GAP = 1.0
# ...and only into more fill in the column's colour than this share of an average
# character (not specks).
GROW_MIN_FILL = 0.05
# Rounds of growing (a grown column can reveal a neighbour that started past the box).
GROW_ROUNDS = 4
# Cuts between touching columns are placed at the profile's lowest point within this
# fraction of the column spacing of where the spacing puts them.
CUT_SEARCH = 0.35
# A column needs fill spanning at least this many column spacings vertically...
MIN_COLUMN_HEIGHT = 1.5
# ...at least this many column spacings wide (narrow strips are art highlights)...
MIN_COLUMN_WIDTH = 0.5
# ...and at least this share of the average column's fill (drops stray highlights).
MIN_COLUMN_FILL_SHARE = 0.15
# Columns separated by a blank gap wider than this many spacings aren't part of the
# same caption (columns of one caption all but touch).
COLUMN_RUN_GAP = 0.8
# Neighbouring columns whose fill colours differ by less than this (CIELAB distance)
# are the same speaker. Lower = more sensitive (similar colours are split into
# separate speakers). Overridable per run: detection_group.caption.speaker_color_distance
# in config.yaml.
SPEAKER_SPLIT_DISTANCE = 12
# When growing a column past the detector box, fill within this distance (CIELAB) of
# the column's colour counts as more of the column. Independent of speaker splitting.
GROW_COLOR_DISTANCE = 25
# Margin (px) around each column for the outline, on top of the fill extent.
OUTLINE_PADDING = 8


def fill_mask(gray: np.ndarray, max_size: float | None = None) -> np.ndarray:
    """Letter fills: light shapes enclosed by dark outline, not touching the crop edge,
    and bright enough on average to be lettering rather than background noise.

    max_size: largest fill shape (px) kept; defaults to the crop's narrower side.
    """
    light = (gray >= OUTLINE_MAX_GRAY).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(light, connectivity=4)
    h, w = gray.shape
    if max_size is None:
        max_size = min(h, w)
    x, y = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    cw, ch = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    mean_gray = np.bincount(labels.ravel(), weights=gray.ravel(), minlength=count) / np.maximum(stats[:, cv2.CC_STAT_AREA], 1)
    keep = (stats[:, cv2.CC_STAT_AREA] >= MIN_FILL_AREA) & (cw <= max_size) & (ch <= max_size) \
        & (x > 0) & (y > 0) & (x + cw < w) & (y + ch < h) & (mean_gray >= FILL_MIN_MEAN_GRAY)
    keep[0] = False
    return keep[labels]


def _column_pitch(profile: np.ndarray) -> float | None:
    """Spacing (px) between neighbouring columns, from the profile's autocorrelation.

    Only the stretch of the profile that has fill is used (empty margins would match
    each other at large shifts), each shift's correlation is averaged over the overlap
    that remains, and only clear peaks count: at least PITCH_MIN_PROMINENCE above the
    dip before them (small wiggles at short shifts come from strokes inside letters).
    The profile also repeats at multiples of the spacing, so the first clear peak that's
    nearly as strong as the strongest is taken, not the strongest itself.
    """
    on = np.flatnonzero(profile > PROFILE_GAP * profile.max()) if profile.max() > 0 else []
    if len(on) == 0:
        return None
    profile = profile[on[0]:on[-1] + 1]
    n = len(profile)
    centred = profile - profile.mean()
    corr = np.correlate(centred, centred, mode="full")[n - 1:]
    if corr[0] <= 0:
        return None
    corr = corr / (n - np.arange(n)) / (corr[0] / n)  # overlap-averaged, 1.0 at shift 0
    def clear_peaks(low: int, high: int) -> list[int]:
        peaks = []
        for lag in range(low, min(high, n - 2) + 1):
            if corr[lag] <= 0 or corr[lag] < corr[lag - 1] or corr[lag] < corr[lag + 1]:
                continue
            dip = corr[1:lag].min()  # the trough between shift 0 and this peak
            if corr[lag] - dip >= PITCH_MIN_PROMINENCE:
                peaks.append(lag)
        return peaks

    # Shifts up to half the fill's width have enough overlap to judge reliably. With
    # just two columns the spacing is a bit over half, so look further only if needed.
    peaks = clear_peaks(PITCH_RANGE[0], min(PITCH_RANGE[1], n // 2))
    if not peaks:
        peaks = clear_peaks(n // 2 + 1, min(PITCH_RANGE[1], int(TWO_COLUMN_REACH * n)))
    if not peaks:
        return None
    best = max(corr[lag] for lag in peaks)
    return float(next(lag for lag in peaks if corr[lag] >= PITCH_PEAK_SHARE * best))


def _busy_rows(fill: np.ndarray) -> np.ndarray:
    """Rows where most of the caption's columns are present."""
    occupancy = cv2.dilate(fill.astype(np.uint8), np.ones((BUSY_BAND, 1), np.uint8)).sum(axis=1)
    return occupancy >= BUSY_SHARE * occupancy.max()


def _grow_rows(column: np.ndarray, lab: np.ndarray, color: np.ndarray, y0: int, y1: int, pitch: float) -> tuple[int, int]:
    """Extend [y0, y1) up and down while the column's lettering continues.

    column: the column's fill mask over the full page height; lab: the same area in
    CIELAB; color: the column's fill colour (CIELAB). Only fill in the column's colour
    counts, so streaks and highlights in the art and strokes of a neighbouring column
    poking into the slice are ignored: a step is taken if such fill resumes within
    GROW_GAP spacings and the next spacing's worth of rows holds more than a speck of
    it (GROW_MIN_FILL of an average character). Each end tracks the colour of the
    lettering it last took in, so shaded fill (lighter at one end, darker at the
    other) is followed rather than cut off where it drifts from the average.
    """
    distance = np.linalg.norm(lab - color, axis=2)
    seed_matched = (column[y0:y1] & (distance[y0:y1] < GROW_COLOR_DISTANCE)).sum()
    min_fill = GROW_MIN_FILL * seed_matched / max(1.0, (y1 - y0) / pitch)
    step = max(1, int(round(pitch)))
    max_gap = int(GROW_GAP * pitch)
    page_h = column.shape[0]

    def matching(w0: int, w1: int, ref: np.ndarray) -> np.ndarray:
        """Fill rows in [w0, w1) in colour ref, as a (rows, cols) mask."""
        return column[w0:w1] & (np.linalg.norm(lab[w0:w1] - ref, axis=2) < GROW_COLOR_DISTANCE)

    def step_out(edge: int, ref: np.ndarray, down: bool) -> tuple[int, np.ndarray] | None:
        # Fill in colour within max_gap past the edge, then a spacing's worth of rows from there.
        g0, g1 = (edge, min(page_h, edge + max_gap)) if down else (max(0, edge - max_gap), edge)
        rows = np.flatnonzero(matching(g0, g1, ref).any(axis=1)) + g0
        if not len(rows):
            return None
        start = rows[0] if down else rows[-1] + 1
        w0, w1 = (start, min(page_h, start + step)) if down else (max(0, start - step), start)
        window = matching(w0, w1, ref)
        if window.sum() < min_fill:
            return None
        hit = np.flatnonzero(window.any(axis=1)) + w0
        new_ref = np.median(lab[w0:w1][window], axis=0)
        return (int(hit[-1]) + 1 if down else int(hit[0])), new_ref

    bottom_ref = top_ref = color
    while True:
        grown = False
        below = step_out(y1, bottom_ref, down=True)
        if below:
            (y1, bottom_ref), grown = below, True
        above = step_out(y0, top_ref, down=False)
        if above:
            (y0, top_ref), grown = above, True
        if not grown:
            return y0, y1


def _column_spans(profile: np.ndarray, pitch: float | None) -> list[tuple[int, int]]:
    """(x0, x1) of each column: blocks of fill between gaps in the profile, each cut
    into round(width / pitch) columns where outlines touch and leave no clean gap."""
    on = profile > PROFILE_GAP * profile.max()
    blocks, start = [], None
    for x, v in enumerate(list(on) + [False]):
        if v and start is None:
            start = x
        elif not v and start is not None:
            blocks.append((start, x))
            start = None
    if pitch is None:
        return blocks

    spans = []
    for x0, x1 in blocks:
        n = max(1, round((x1 - x0) / pitch))
        cuts = [x0]
        for k in range(1, n):
            expected = x0 + k * (x1 - x0) / n
            lo, hi = int(expected - CUT_SEARCH * pitch), int(expected + CUT_SEARCH * pitch) + 1
            lo, hi = max(cuts[-1] + 1, lo), min(x1 - 1, hi)
            if hi > lo:
                cuts.append(lo + int(np.argmin(profile[lo:hi])))
        cuts.append(x1)
        spans += list(zip(cuts[:-1], cuts[1:]))
    return spans


def _lab(color_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(color_bgr.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB)[0, 0].astype(float)


# Longest column chunk (in character widths) read in one go; manga_ocr shrinks its
# input to 224x224, so very long columns lose detail and get misread at the end.
OCR_CHUNK_CHARS = 10


def column_ocr_images(column: np.ndarray) -> list[np.ndarray]:
    """A caption column crop as clean black-on-white text, cut into readable chunks.

    column: BGR crop of one column (with its outline padding).
    """
    gray = cv2.cvtColor(column, cv2.COLOR_BGR2GRAY)
    char_size = max(1, column.shape[1] - 2 * OUTLINE_PADDING)
    fill = fill_mask(gray, FILL_MAX_SIZE * char_size)
    text = np.where(fill, 0, 255).astype(np.uint8)

    rows = np.flatnonzero(fill.any(axis=1))
    if len(rows) == 0:
        return []
    top, bottom = rows[0], rows[-1] + 1
    blank = ~fill.any(axis=1)
    chunk = OCR_CHUNK_CHARS * char_size
    images = []
    while bottom - top > chunk * 1.3:
        # Cut at the blank row (gap between characters) nearest the chunk length.
        target = top + chunk
        candidates = np.flatnonzero(blank[top + char_size // 2:bottom - char_size // 2]) + top + char_size // 2
        if len(candidates) == 0:
            break
        cut = int(candidates[np.argmin(np.abs(candidates - target))])
        images.append(text[max(0, top - OUTLINE_PADDING):cut])
        top = cut
        while top < bottom and blank[top]:
            top += 1
    images.append(text[max(0, top - OUTLINE_PADDING):bottom + OUTLINE_PADDING])
    return images


def split_caption(image: np.ndarray, region: SpeechBubble, log: list[str] | None = None,
                  speaker_distance: float = SPEAKER_SPLIT_DISTANCE) -> SpeechBubble | None:
    """Split a region of outlined vertical columns into per-speaker clusters.

    Returns None (leave the region as it is) unless it's taller than wide and has at
    least two outlined columns.
    log: if given, a description of each decision is appended (see
    tools/caption_diagnose.py); numbers only, no text or image content.
    speaker_distance: neighbouring columns closer in colour than this (CIELAB) are
    one speaker (see SPEAKER_SPLIT_DISTANCE).
    """
    def note(message: str) -> None:
        if log is not None:
            log.append(message)

    rx0, ry0, rx1, ry1 = region.position
    if ry1 - ry0 <= rx1 - rx0:
        note(f"rejected: not taller than wide ({rx1 - rx0}x{ry1 - ry0})")
        return None
    # A block of many horizontal lines can be taller than wide too (profile pages).
    if _is_horizontal(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), region.position):
        note("rejected: text runs horizontally (orientation check)")
        return None
    page_h, page_w = image.shape[:2]
    smoothing = np.ones(PROFILE_SMOOTHING) / PROFILE_SMOOTHING

    # Column spacing, from the detected box (first with a generous size limit).
    cx0, cx1 = max(0, rx0 - CONTEXT), min(page_w, rx1 + CONTEXT)
    by0, by1 = max(0, ry0 - CONTEXT), min(page_h, ry1 + CONTEXT)
    box_fill = fill_mask(cv2.cvtColor(image[by0:by1, cx0:cx1], cv2.COLOR_BGR2GRAY))
    if not box_fill.any():
        note("rejected: no outlined letter fill found in the box")
        return None
    note(f"letter fill: {box_fill.mean():.1%} of the box")
    busy_profile = np.convolve(box_fill[_busy_rows(box_fill)].sum(axis=0), smoothing, mode="same")
    pitch = _column_pitch(busy_profile)
    if pitch is None:
        # Tightly packed columns (outlines fully merged) leave only shallow dips, which
        # a wide gap or a stray neighbouring column in the box can drown out: measure
        # on the main block of text alone (the stretch with the most fill and no gap).
        blocks = _column_spans(busy_profile, None)
        if blocks:
            x0, x1 = max(blocks, key=lambda b: busy_profile[b[0]:b[1]].sum())
            pitch = _column_pitch(busy_profile[x0:x1])
            if pitch is not None:
                note("column spacing measured on the main block of text only")
    if pitch is None:
        note("rejected: no regular column spacing found")
        return None
    note(f"column spacing: {pitch:.0f}px")

    # Letter fills over the region's full-height strip of the page, so columns can be
    # followed past the box's top and bottom.
    cy0, cy1 = 0, page_h
    crop = image[cy0:cy1, cx0:cx1]
    fill = fill_mask(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), FILL_MAX_SIZE * pitch)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(float)

    top, bottom = by0, by1
    for _ in range(GROW_ROUNDS):
        active = fill[top:bottom]
        if not active.any():
            note("rejected: no letter fill left at the size limit for this spacing")
            return None
        profile = np.convolve(active.sum(axis=0), smoothing, mode="same")
        found = []
        for x0, x1 in _column_spans(profile, pitch):
            # Start from the column's densest character-sized stretch inside the box and
            # grow outwards, so every extension (even within the box) is checked.
            counts = active[:, x0:x1].sum(axis=1)
            if not counts.any():
                continue
            step = max(1, int(round(pitch)))
            seed = top + int(np.argmax(np.convolve(counts, np.ones(step), mode="valid")))
            # The column's colour: the median over all its fill in the box, where the
            # speaker's colour is the clear majority (any one stretch may be mixed).
            color = np.median(lab[top:bottom, x0:x1][active[:, x0:x1]], axis=0)
            y0, y1 = _grow_rows(fill[:, x0:x1], lab[:, x0:x1], color, seed, seed + step, pitch)
            column = fill[y0:y1, x0:x1]
            cols = np.flatnonzero(column.any(axis=0)) + x0  # tighten to the fill in this column
            found.append(((int(cols[0]), y0, int(cols[-1]) + 1, y1), int(column.sum())))
        if not found:
            note("rejected: no columns found")
            return None
        mean_fill = np.mean([area for _, area in found])
        kept, column_notes = [], []
        for box, area in found:
            reasons = [why for failed, why in (
                (box[3] - box[1] < MIN_COLUMN_HEIGHT * pitch, "too short"),
                (box[2] - box[0] < MIN_COLUMN_WIDTH * pitch, "too narrow"),
                (area < MIN_COLUMN_FILL_SHARE * mean_fill, "too little fill"),
            ) if failed]
            column_notes.append(f"  column x {box[0] + cx0}-{box[2] + cx0}, y {box[1]}-{box[3]}, fill {area}px: "
                                + (", ".join(reasons) if reasons else "kept"))
            if not reasons:
                kept.append(box)
        found = kept
        # A detector box can reach into a neighbouring caption or art: keep only the
        # main run of columns, i.e. split where neighbouring columns are separated by a
        # blank gap wider than COLUMN_RUN_GAP spacings (columns of one caption all but
        # touch) and keep the run with the most columns.
        if len(found) > 1:
            found.sort(key=lambda b: b[0])
            runs = [[found[0]]]
            for box in found[1:]:
                if box[0] - runs[-1][-1][2] > COLUMN_RUN_GAP * pitch:
                    runs.append([box])
                else:
                    runs[-1].append(box)
            main = max(runs, key=len)
            for run in runs:
                if run is not main:
                    for box in run:
                        column_notes.append(f"  column x {box[0] + cx0}-{box[2] + cx0}: dropped, separated from the main run by a gap")
            found = main
        if len(found) < 2:
            for line in column_notes:
                note(line)
            note(f"rejected: {len(found)} column(s) kept, need 2")
            return None
        new_top = min(top, min(b[1] for b in found) - CONTEXT)
        new_bottom = max(bottom, max(b[3] for b in found) + CONTEXT)
        if (new_top, new_bottom) == (top, bottom):
            break
        top, bottom = max(0, new_top), min(page_h, new_bottom)
    for line in column_notes:  # the final round's columns
        note(line)

    # Right to left, as vertical Japanese is read.
    found.sort(key=lambda b: -b[0])
    inner = cv2.erode(fill.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0  # skip anti-aliased edges
    columns = []
    for x0, y0, x1, y1 in found:
        pixels = inner[y0:y1, x0:x1]
        source = pixels if pixels.any() else fill[y0:y1, x0:x1]
        color_lab = np.median(lab[y0:y1, x0:x1][source], axis=0)
        color_bgr = np.median(crop[y0:y1, x0:x1][source], axis=0)
        box = (int(max(0, cx0 + x0 - OUTLINE_PADDING)), int(max(0, cy0 + y0 - OUTLINE_PADDING)),
               int(min(page_w, cx0 + x1 + OUTLINE_PADDING)), int(min(page_h, cy0 + y1 + OUTLINE_PADDING)))
        columns.append((box, color_lab, color_bgr))

    # Neighbouring columns of the same colour are one speaker's line continuing.
    speakers = [[columns[0]]]
    note(f"speaker split distance {speaker_distance:g}")
    for column in columns[1:]:
        distance = float(np.linalg.norm(column[1] - speakers[-1][-1][1]))
        same = distance < speaker_distance
        note(f"  column x {column[0][0]}-{column[0][2]}: colour distance {distance:.1f} to the previous column"
             f" -> {'same speaker' if same else 'new speaker'}")
        if same:
            speakers[-1].append(column)
        else:
            speakers.append([column])

    clusters = []
    for speaker in speakers:
        boxes = [box for box, _, _ in speaker]
        x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        b, g, r = np.median([bgr for _, _, bgr in speaker], axis=0)
        clusters.append(TextCluster(
            position=(x0, y0, x1, y1),
            confidence=region.confidence,
            bounding_image=image[y0:y1, x0:x1],
            lines=boxes,
            color=(int(r), int(g), int(b)),
        ))

    x0, y0 = min(rx0, *(c.position[0] for c in clusters)), min(ry0, *(c.position[1] for c in clusters))
    x1, y1 = max(rx1, *(c.position[2] for c in clusters)), max(ry1, *(c.position[3] for c in clusters))
    return SpeechBubble(
        position=(x0, y0, x1, y1),
        confidence=region.confidence,
        class_name=CAPTION_CLASS_NAME,
        bounding_image=image[y0:y1, x0:x1],
        text_clusters=clusters,
    )

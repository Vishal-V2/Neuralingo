import cv2
import numpy as np
from PIL import Image
from pipeline.core.box_data import TextCluster
from pipeline.core.detection import captions


# Margin (px) around each line when masking paragraphs line by line.
LINE_MASK_PADDING = 4
# Outlined captions: dark pixels within this distance (px) of a letter's fill are its
# outline; the final mask is grown by CAPTION_MASK_GROW to catch the soft edge.
CAPTION_OUTLINE_REACH = 12
CAPTION_MASK_GROW = 3
# The usual mask takes the darker side of an Otsu split as text (dark text on a light
# bubble). If more than this share of the text box's border lands on that side, the
# dark side is really the surroundings (e.g. light lettering in a translucent grey
# bubble over a dark scene), so the text is found by colour instead.
BORDER_INK_LIMIT = 0.5
# Likewise if the dark side covers more than this share of the box's centre: text
# never fills that much of its own box, so the dark side is a dark bubble fill (e.g.
# a dark grey bubble on a lighter background, whose box border is mostly background).
CENTRE_INK_LIMIT = 0.6
# Colour mask: pixels at least this far (CIELAB) from the bubble's interior colour are text.
TEXT_COLOR_DISTANCE = 30
# ...except shapes spanning more than this share of the box both ways (the bubble's
# own outline ring) and shapes touching the box's edge (the surroundings).
OUTLINE_SPAN = 0.8
# If the colour mask finds text covering less than this share of the box, the
# interior colour was probably the letters' own (see _text_by_color).
MIN_TEXT_SHARE = 0.02


def apply_text_mask(mask_canvas: np.ndarray, text_cluster_data: TextCluster) -> np.ndarray:
    if text_cluster_data.lines and text_cluster_data.color is not None:
        return _apply_caption_masks(mask_canvas, text_cluster_data)
    if text_cluster_data.lines:
        return _apply_line_masks(mask_canvas, text_cluster_data)

    TCx_min, TCy_min, _, _ = text_cluster_data.position
    inverted_threshold = text_pixels(text_cluster_data.bounding_image)

    kernel = np.ones((5, 5), np.uint8)
    dilated_threshold = cv2.dilate(inverted_threshold, kernel, iterations=4)
    contours, _ = cv2.findContours(dilated_threshold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    relative_contours = []
    for contour in contours:
        relative = contour + [TCx_min, TCy_min]
        relative_contours.append(relative)

    for contour in relative_contours:
        reshaped = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
        cv2.drawContours(mask_canvas, [reshaped], -1, (0, 0, 0), thickness=cv2.FILLED)

    return mask_canvas


def text_pixels(image: np.ndarray) -> np.ndarray:
    """Text mask (255 = text) of a bubble's text box: the darker side of an Otsu split,
    or the text found by colour when that side is really the surroundings or the fill."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    inverted_threshold = cv2.bitwise_not(threshold)
    border = np.concatenate([inverted_threshold[0], inverted_threshold[-1],
                             inverted_threshold[:, 0], inverted_threshold[:, -1]])
    h, w = inverted_threshold.shape
    centre = inverted_threshold[h // 4:h - h // 4, w // 4:w - w // 4]
    if np.mean(border > 0) > BORDER_INK_LIMIT or np.mean(centre > 0) > CENTRE_INK_LIMIT:
        inverted_threshold = _text_by_color(image)
    return inverted_threshold


def _most_common_color(lab_pixels: np.ndarray) -> np.ndarray:
    """Median of the most common colour (16-level CIELAB bins) among (N, 3) pixels."""
    bins = (lab_pixels // 16).astype(int)
    keys = bins[:, 0] * 256 + bins[:, 1] * 16 + bins[:, 2]
    return np.median(lab_pixels[keys == np.bincount(keys).argmax()], axis=0)


def _text_by_color(image: np.ndarray) -> np.ndarray:
    """Text mask (255 = text) for lettering that isn't simply darker than its bubble.

    The bubble's interior colour is the most common colour in the middle of the box;
    text is what differs clearly from it, minus shapes touching the box's edge (the
    scene around the bubble) and the bubble's outline ring. Heavy lettering can fill
    most of the middle, making the letters' colour the most common there; then (almost
    nothing found) the most common colour along the box's border is used instead.
    """
    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(float)
    centre = lab[h // 4:h - h // 4, w // 4:w - w // 4].reshape(-1, 3)
    mask = _differing_shapes(lab, _most_common_color(centre))
    if np.mean(mask > 0) < MIN_TEXT_SHARE:
        border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        mask = _differing_shapes(lab, _most_common_color(border))
    return mask


def _differing_shapes(lab: np.ndarray, interior: np.ndarray) -> np.ndarray:
    """Shapes differing from the interior colour, not touching the edge, not the outline ring."""
    h, w = lab.shape[:2]
    differs = (np.linalg.norm(lab - interior, axis=2) >= TEXT_COLOR_DISTANCE).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(differs, connectivity=8)
    x, y = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    cw, ch = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    touches_edge = (x == 0) | (y == 0) | (x + cw == w) | (y + ch == h)
    outline = (cw > OUTLINE_SPAN * w) & (ch > OUTLINE_SPAN * h)
    keep = ~(touches_edge | outline)
    keep[0] = False
    return (keep[labels] * 255).astype(np.uint8)


def _apply_line_masks(mask_canvas: np.ndarray, text_cluster_data: TextCluster) -> np.ndarray:
    # Paragraphs on text pages: mask each line's box rather than the whole paragraph
    # rectangle, so artwork or bubble outlines between and beside the lines survive.
    image = text_cluster_data.bounding_image
    h, w = image.shape[:2]
    cx, cy = text_cluster_data.position[:2]
    pad = LINE_MASK_PADDING  # line boxes hug the ink; keep the soft glyph edges
    for x_min, y_min, x_max, y_max in text_cluster_data.lines:
        x0, y0 = max(0, x_min - cx - pad), max(0, y_min - cy - pad)
        x1, y1 = min(w, x_max - cx + pad), min(h, y_max - cy + pad)
        if x1 <= x0 or y1 <= y0:
            continue
        line = TextCluster(position=(cx + x0, cy + y0, cx + x1, cy + y1), confidence=text_cluster_data.confidence,
                           bounding_image=image[y0:y1, x0:x1])
        mask_canvas = apply_text_mask(mask_canvas, line)
    return mask_canvas


def _apply_caption_masks(mask_canvas: np.ndarray, text_cluster_data: TextCluster) -> np.ndarray:
    # Outlined caption columns: mask the letters (fill + its outline) only, not
    # the box, so dark art right behind the text (e.g. hair) isn't repainted.
    image = text_cluster_data.bounding_image
    cx, cy = text_cluster_data.position[:2]
    reach = np.ones((2 * CAPTION_OUTLINE_REACH + 1,) * 2, np.uint8)
    grow = np.ones((2 * CAPTION_MASK_GROW + 1,) * 2, np.uint8)
    for x_min, y_min, x_max, y_max in text_cluster_data.lines:
        x0, y0, x1, y1 = x_min - cx, y_min - cy, x_max - cx, y_max - cy
        column = image[y0:y1, x0:x1]
        if not column.size:
            continue
        gray = captions.letter_gray(column, text_cluster_data.light_outline)
        char_size = max(1, column.shape[1] - 2 * captions.outline_padding(text_cluster_data.light_outline))
        fill = captions.fill_mask(gray, captions.FILL_MAX_SIZE * char_size).astype(np.uint8)
        if text_cluster_data.light_outline:
            # A white outline's soft outer edge is only a little lighter than the art
            # behind it, and inpainting spreads any of it left as white specks: take the
            # whole ring within reach.
            outline = cv2.dilate(fill, reach)
        else:
            outline = (gray < captions.OUTLINE_MAX_GRAY).astype(np.uint8) & cv2.dilate(fill, reach)
        letters = cv2.dilate(fill | outline, grow)
        region = mask_canvas[y_min:y_max, x_min:x_max]
        region[letters[:region.shape[0], :region.shape[1]] > 0] = 0
    return mask_canvas


def batch_apply_text_mask(mask_canvas: np.ndarray, text_clusters_data: list[TextCluster]) -> Image.Image:
    image_np = mask_canvas
    for text_cluster_data in text_clusters_data:
        if text_cluster_data.bounding_image is None:
            continue
        image_np = apply_text_mask(image_np, text_cluster_data)
    
    image_mask_pil = Image.fromarray(image_np)
    image_mask_pil.save("output/image_mask.png")

    return image_mask_pil
from PySide6.QtCore import (
    QObject, Signal, Slot
)
from ultralytics import YOLO
import numpy as np
import cv2
import traceback
from pipeline.core.box_data import SpeechBubble, TextCluster
import constants

from pipeline.core.detection.speech_bubble import SpeechBubbleDetection
from pipeline.core.detection.text_cluster import TextClusterDetection
from pipeline.core.detection.text_blocks import build_text_blocks
from pipeline.core.detection.flat_bubbles import detect_flat_bubbles
from pipeline.core.detection.text_growth import add_missing_text, drop_duplicate_text, grow_text_clusters
from pipeline.core.detection.captions import SPEAKER_SPLIT_DISTANCE, split_caption



# A region becomes a caption only if the caption's columns cover at least this share of
# the region's own text boxes (frameless regions: their box is the text found).
CAPTION_TEXT_COVER = 0.5
# Bubble-model proposals below the threshold that overlap no detected region by more
# than this share of their area are tried as light-outline captions.
WEAK_CAPTION_OVERLAP = 0.3


def _overlap_share(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Share of a's area inside b."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    area = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    return max(0, w) * max(0, h) / area


def _covers_text(caption: SpeechBubble, region: SpeechBubble) -> bool:
    """Whether caption's columns cover most of region's text boxes."""
    texts = [c.position for c in region.text_clusters or []]
    if region.class_name == constants.FRAMELESS_CLASS_NAME or not texts:
        return True
    columns = [line for c in caption.text_clusters for line in (c.lines or [c.position])]
    mask_area = covered = 0
    for tx0, ty0, tx1, ty1 in texts:
        area = max(0, tx1 - tx0) * max(0, ty1 - ty0)
        mask = np.zeros((max(0, ty1 - ty0), max(0, tx1 - tx0)), bool)
        for cx0, cy0, cx1, cy1 in columns:
            mask[max(0, cy0 - ty0):max(0, cy1 - ty0), max(0, cx0 - tx0):max(0, cx1 - tx0)] = True
        mask_area += area
        covered += mask.sum()
    return covered >= CAPTION_TEXT_COVER * max(1, mask_area)


class DetectionWorker(QObject):
    finished = Signal(list)

    def __init__(self, detection_config: dict, speech_bubble_model: YOLO, text_cluster_model: YOLO, image_cv2: np.ndarray):
        super().__init__()
        self.detection_config = detection_config or {}

        sb_thresh = None
        tc_thresh = None
        try:
            sb_thresh = float(self.detection_config.get("speech_bubble", {}).get("confidence_threshold"))
        except Exception:
            pass
        try:
            tc_thresh = float(self.detection_config.get("text_cluster", {}).get("confidence_threshold"))
        except Exception:
            pass
        # Persist thresholds with sane fallbacks
        self.sb_threshold = sb_thresh if sb_thresh is not None else constants.SPEECH_BUBBLE_THRESHOLD
        self.tc_threshold = tc_thresh if tc_thresh is not None else constants.TEXT_CLUSTER_THRESHOLD

        self.speech_bubble_detection = SpeechBubbleDetection(speech_bubble_model, threshold=self.sb_threshold)
        self.text_cluster_detection = TextClusterDetection(text_cluster_model, threshold=self.tc_threshold)
        self.image_cv2 = image_cv2
        # "comic" (default): bubble pages. "text": text-heavy pages such as character
        # profiles, where horizontal paragraphs get split into lines and paragraphs.
        self.page_mode = self.detection_config.get("page_mode", "comic")

        self.speech_bubbles_data = []
        self.flat_bubbles: list[SpeechBubble] = []


    @Slot()
    def run(self):
        # Each step catches its own errors: a step that fails costs only its own work,
        # and the later steps (and the detection image) still run.
        self._step(self._speech_bubble_detection)
        print("[DETECTION] speech_bubbles:", len(self.speech_bubbles_data), [(sb.class_name, sb.confidence) for sb in self.speech_bubbles_data])
        self._step(self._text_cluster_detection)
        self._step(self._uncovered_text_detection)
        self._step(self._flat_bubble_detection)
        if self.page_mode == "text":
            self._step(self._text_block_detection)
        self._step(self._caption_detection)
        print("[DETECTION] clusters_per_bubble:", [len(sb.text_clusters or []) for sb in self.speech_bubbles_data])
        self._step(lambda: self.visualize_detection(self.image_cv2, self.speech_bubbles_data))

        self.finished.emit(self.speech_bubbles_data)


    def _step(self, step) -> None:
        try:
            step()
        except Exception:
            name = getattr(step, "__name__", "step").lstrip("_")
            print(f"[DETECTION] Error in {name}:")
            traceback.print_exc()
    

    def _speech_bubble_detection(self):
        # Threshold retries (and the classical-vision fallback) happen inside detect().
        self.speech_bubbles_data = self.speech_bubble_detection.detect(self.image_cv2)


    def _text_cluster_detection(self):
        if not self.speech_bubbles_data:
            return
        # Per-bubble threshold retries (and the whole-bubble fallback) happen inside detect().
        self.speech_bubbles_data = self.text_cluster_detection.detect(self.speech_bubbles_data)
        # The text model sometimes boxes only part of a bubble's lettering (grow each
        # box over the rest), or misses it and boxes some art instead (box the
        # lettering, drop the art); then keep text the bubble model boxed twice once.
        grow_text_clusters(self.image_cv2, self.speech_bubbles_data)
        add_missing_text(self.image_cv2, self.speech_bubbles_data)
        self.speech_bubbles_data = drop_duplicate_text(self.speech_bubbles_data)


    def _uncovered_text_detection(self):
        # Text drawn straight onto the art (no bubble) is invisible to the bubble
        # model; pick it up with a page-wide text pass and add it as frameless.
        self.speech_bubbles_data += self.text_cluster_detection.detect_uncovered_text(
            self.image_cv2, self.speech_bubbles_data
        )


    def _flat_bubble_detection(self):
        # Bubbles drawn as solid coloured shapes (e.g. blue ellipses on black) that
        # both models miss; adds bubbles whose text nothing already covers, and turns
        # frameless text regions found inside such a bubble into that bubble.
        found, upgraded = detect_flat_bubbles(self.image_cv2, self.speech_bubbles_data)
        self.speech_bubbles_data += found
        self.flat_bubbles = found + upgraded


    def _text_block_detection(self):
        # Text pages only: dedupe frameless regions and split horizontal text into paragraphs.
        frameless = [sb for sb in self.speech_bubbles_data if sb.class_name == constants.FRAMELESS_CLASS_NAME]
        bubbles = [sb for sb in self.speech_bubbles_data if sb.class_name != constants.FRAMELESS_CLASS_NAME]
        self.speech_bubbles_data = bubbles + build_text_blocks(self.image_cv2, frameless, bubbles)


    def _caption_detection(self):
        # Outlined vertical captions on the art with several speakers side by side:
        # one cluster per speaker, so each line is translated as its own sentence.
        # Every region is tried, not just frameless ones: the bubble model sometimes
        # gives such captions a bubble class (e.g. narration), and the text model then
        # finds only fragments inside. split_caption() rejects real speech bubbles
        # (plain text on a white interior has no outlined letter fill to find), but
        # light art on dark inside a bubble's box (e.g. white sparkles in a corner
        # outside its outline) can pass for outlined letters; so a region whose text
        # boxes the caption's columns don't cover keeps its own text.
        caption_config = self.detection_config.get("caption") or {}
        speaker_distance = float(caption_config.get("speaker_color_distance", SPEAKER_SPLIT_DISTANCE))
        for i, sb in enumerate(self.speech_bubbles_data):
            # A flat bubble's fill can pass for a caption's dark outline (white letters
            # on dark blue); it's a bubble, already known.
            if any(sb is flat for flat in self.flat_bubbles):
                continue
            caption = split_caption(self.image_cv2, sb, speaker_distance=speaker_distance)
            if caption is None and sb.class_name == constants.FRAMELESS_CLASS_NAME:
                caption = split_caption(self.image_cv2, sb, speaker_distance=speaker_distance, light_outline=True)
            if caption is not None and _covers_text(caption, sb):
                self.speech_bubbles_data[i] = caption

        # Captions in darker letters outlined in white (on busy art) get only weak
        # bubble-model proposals (0.4-0.55) and weak text-model boxes. Accept a weak
        # proposal nothing else covers if it holds such lettering (split_caption with
        # light_outline checks the outline is a thin ring, not a white bubble).
        for conf, class_name, (x0, y0, x1, y1) in self.speech_bubble_detection.weak_candidates:
            position = (x0, y0, x1, y1)
            if any(_overlap_share(position, sb.position) > WEAK_CAPTION_OVERLAP for sb in self.speech_bubbles_data):
                continue
            weak = SpeechBubble(position=position, confidence=conf, class_name=constants.FRAMELESS_CLASS_NAME,
                                bounding_image=self.image_cv2[y0:y1, x0:x1], text_clusters=[])
            caption = split_caption(self.image_cv2, weak, speaker_distance=speaker_distance, light_outline=True)
            if caption is not None:
                self.speech_bubbles_data.append(caption)


    def visualize_detection(
            self, 
            image_cv2: np.ndarray,
            speech_bubbles_data: list[SpeechBubble]
    ):
        image_copy = image_cv2.copy()

        for speech_bubble_data in speech_bubbles_data:
            SBx_min, SBy_min, SBx_max, SBy_max = speech_bubble_data.position
            cv2.rectangle(image_copy, (SBx_min, SBy_min), (SBx_max, SBy_max), (255, 0, 0), 4)
        
            for text_cluster_data in speech_bubble_data.text_clusters:
                TCx_min, TCy_min, TCx_max, TCy_max = text_cluster_data.position
                cv2.rectangle(image_copy, (TCx_min, TCy_min), (TCx_max, TCy_max), (0, 255, 0), 4)
        
        cv2.imwrite("output/visualized_detection.png", image_copy)
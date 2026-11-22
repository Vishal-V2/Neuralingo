from PySide6.QtCore import (
    QObject, Signal, Slot
)
from ultralytics import YOLO
import numpy as np
import cv2
from pipeline.core.box_data import SpeechBubble, TextCluster
import constants

from pipeline.core.detection.speech_bubble import SpeechBubbleDetection
from pipeline.core.detection.text_cluster import TextClusterDetection
from pipeline.core.detection.text_blocks import build_text_blocks
from pipeline.core.detection.captions import SPEAKER_SPLIT_DISTANCE, split_caption



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


    @Slot()
    def run(self):
        try:
            self._speech_bubble_detection()
            print("[DETECTION] speech_bubbles:", len(self.speech_bubbles_data), [(sb.class_name, sb.confidence) for sb in self.speech_bubbles_data])
            self._text_cluster_detection()
            self._uncovered_text_detection()
            if self.page_mode == "text":
                self._text_block_detection()
            self._caption_detection()
            try:
                counts = [len(sb.text_clusters or []) for sb in self.speech_bubbles_data]
                print("[DETECTION] clusters_per_bubble:", counts)
            except Exception:
                pass
            self.visualize_detection(self.image_cv2, self.speech_bubbles_data)
        except Exception as e:
            print(f"[DETECTION] Error: {e}")

        self.finished.emit(self.speech_bubbles_data)
    

    def _speech_bubble_detection(self):
        # Threshold retries (and the classical-vision fallback) happen inside detect().
        self.speech_bubbles_data = self.speech_bubble_detection.detect(self.image_cv2)


    def _text_cluster_detection(self):
        if not self.speech_bubbles_data:
            return
        # Per-bubble threshold retries (and the whole-bubble fallback) happen inside detect().
        self.speech_bubbles_data = self.text_cluster_detection.detect(self.speech_bubbles_data)


    def _uncovered_text_detection(self):
        # Text drawn straight onto the art (no bubble) is invisible to the bubble
        # model; pick it up with a page-wide text pass and add it as frameless.
        self.speech_bubbles_data += self.text_cluster_detection.detect_uncovered_text(
            self.image_cv2, self.speech_bubbles_data
        )


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
        # (plain text on a white interior has no outlined letter fill to find).
        caption_config = self.detection_config.get("caption") or {}
        speaker_distance = float(caption_config.get("speaker_color_distance", SPEAKER_SPLIT_DISTANCE))
        for i, sb in enumerate(self.speech_bubbles_data):
            caption = split_caption(self.image_cv2, sb, speaker_distance=speaker_distance)
            if caption is not None:
                self.speech_bubbles_data[i] = caption


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
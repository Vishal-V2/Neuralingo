from ultralytics import YOLO
from pipeline.core.box_data import SpeechBubble, TextCluster
import constants
from pipeline.core.detection import overlap
from pipeline.core.detection.speech_bubble import thresholds_to_try
from pipeline.core.detection.text_blocks import MERGE_OVERLAP, _area, _intersection, _merge_overlapping, _union
import cv2
import numpy as np


# Lower thresholds tried, in order, when nothing in a bubble passes the configured threshold.
RETRY_THRESHOLDS = (0.45, 0.35, 0.30, 0.25)

# Page-wide text pass input sizes. At YOLO's default 640px a full page shrinks so much
# that small captions drawn straight onto the art become a few pixels wide, so 1280 is
# also run; but at 1280 long vertical captions get cut short, which 640 sees whole.
# Boxes from both sizes that cover the same text are combined.
FULL_PAGE_IMGSZ = (640, 1280)
# A page-wide text box mostly inside an existing bubble is already handled.
COVERED_FRACTION = 0.5


def _fraction_inside(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Fraction of box a's area that lies inside box b."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max((a[2] - a[0]) * (a[3] - a[1]), 1)


def _grow_frameless(image: np.ndarray, SB: SpeechBubble, box: tuple[int, int, int, int], conf: float) -> None:
    """Grow a frameless region (and its single text cluster) to also cover box."""
    position = _union(tuple(SB.position), box)
    if position == tuple(SB.position):
        return
    x_min, y_min, x_max, y_max = position
    SB.position = position
    SB.bounding_image = image[y_min:y_max, x_min:x_max]
    if SB.text_clusters and len(SB.text_clusters) == 1:
        cluster = SB.text_clusters[0]
        cluster.position = position
        cluster.bounding_image = SB.bounding_image
        cluster.confidence = max(cluster.confidence, conf)


class TextClusterDetection:
    def __init__(self, model: YOLO, threshold: float | None = None):
        self.model = model
        self.threshold = threshold if threshold is not None else constants.TEXT_CLUSTER_THRESHOLD
    

    def _detect_single(self, SBdata: SpeechBubble) -> list[TextCluster]:
        TCdata: list[TextCluster] = []
        SBx_min, SBy_min, _, _ = SBdata.position

        if SBdata.class_name == constants.FRAMELESS_CLASS_NAME:
            TCdata.append(TextCluster(
                position = SBdata.position,
                confidence = SBdata.confidence,
                bounding_image = SBdata.bounding_image,
                jp_text = None,
                tr_text = None
            ))

            return TCdata

        # Run YOLO once at the lowest threshold we might accept, then walk down the
        # threshold ladder by filtering that single result.
        TCresult = self.model(SBdata.bounding_image, conf=min(self.threshold, *RETRY_THRESHOLDS))[0]
        boxes = [(round(box.conf.item(), 2), box) for box in (TCresult.boxes or [])]

        for threshold in thresholds_to_try(self.threshold, RETRY_THRESHOLDS):
            for TCconf, box in boxes:
                if TCconf < threshold:
                    continue

                TCx_min, TCy_min, TCx_max, TCy_max = map(int, box.xyxy.tolist()[0])
                relative_pos = (
                    SBx_min + TCx_min, SBy_min + TCy_min,
                    SBx_min + TCx_max, SBy_min + TCy_max
                )

                bounding_image =  SBdata.bounding_image[TCy_min:TCy_max, TCx_min:TCx_max]

                TCdata.append(TextCluster(
                    position = relative_pos,
                    confidence = TCconf,
                    bounding_image = bounding_image,
                    jp_text = None,
                    tr_text = None,
                ))
            if TCdata:
                return TCdata

        # Fallback: nothing detected even at the lowest retry threshold, so treat
        # the whole bubble as one cluster.
        if not TCdata:
            h, w = SBdata.bounding_image.shape[:2]
            relative_pos = (
                SBx_min, SBy_min,
                SBx_min + w, SBy_min + h
            )
            TCdata.append(TextCluster(
                position=relative_pos,
                confidence=SBdata.confidence,
                bounding_image=SBdata.bounding_image,
                jp_text=None,
                tr_text=None,
            ))
        return TCdata
    

    def _detect_batch(self, SBdata: list[SpeechBubble]) -> list[SpeechBubble]:
        SBupdated_data: list[SpeechBubble] = []

        for SB in SBdata:
            TCdata = self._detect_single(SB)
            SB.text_clusters = TCdata or []
            SBupdated_data.append(SB)
        
        SBupdated_data = overlap.overlap(SBupdated_data)
        
        return SBupdated_data


    def detect_uncovered_text(self, image: np.ndarray, SBdata: list[SpeechBubble]) -> list[SpeechBubble]:
        """Find text outside every detected bubble (e.g. captions drawn on the art).

        Runs the text model over the whole page and returns each confident box that
        isn't already inside a bubble as a frameless bubble holding one cluster.
        Frameless regions in SBdata that a page-wide box overlaps but doesn't fully
        cover (the bubble model often cuts long vertical captions short) are grown
        to cover it.
        """
        boxes: list[tuple[int, int, int, int]] = []
        confidences: list[tuple[tuple[int, int, int, int], float]] = []
        for imgsz in FULL_PAGE_IMGSZ:
            for box in (self.model(image, conf=self.threshold, imgsz=imgsz)[0].boxes or []):
                position = tuple(map(int, box.xyxy.tolist()[0]))
                boxes.append(position)
                confidences.append((position, round(box.conf.item(), 2)))

        def confidence_of(merged) -> float:
            return max(conf for position, conf in confidences if _intersection(merged, position) > 0)

        found: list[SpeechBubble] = []
        for position in _merge_overlapping(boxes):
            TCconf = confidence_of(position)
            frameless = next((SB for SB in SBdata if SB.class_name == constants.FRAMELESS_CLASS_NAME and
                              _intersection(position, SB.position) >= MERGE_OVERLAP * max(min(_area(position), _area(SB.position)), 1)), None)
            if frameless is not None:
                _grow_frameless(image, frameless, position, TCconf)
                continue

            existing = SBdata + found
            if any(_fraction_inside(position, SB.position) >= COVERED_FRACTION for SB in existing):
                continue

            x_min, y_min, x_max, y_max = position
            bounding_image = image[y_min:y_max, x_min:x_max]
            found.append(SpeechBubble(
                position=position,
                confidence=TCconf,
                class_name=constants.FRAMELESS_CLASS_NAME,
                bounding_image=bounding_image,
                text_clusters=[TextCluster(
                    position=position,
                    confidence=TCconf,
                    bounding_image=bounding_image,
                    jp_text=None,
                    tr_text=None,
                )],
            ))

        return found


    def detect(self, SBdata: SpeechBubble | list[SpeechBubble]) -> list[TextCluster] | list[SpeechBubble]:
        if isinstance(SBdata, list):
            return self._detect_batch(SBdata)
        
        return self._detect_single(SBdata)


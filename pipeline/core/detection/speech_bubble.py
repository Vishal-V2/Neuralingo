from ultralytics import YOLO
import numpy as np
import cv2

from pipeline.core.box_data import SpeechBubble
import constants


# Lower thresholds tried, in order, when nothing passes the configured threshold.
RETRY_THRESHOLDS = (0.50, 0.40, 0.35, 0.30, 0.25)


def thresholds_to_try(threshold: float, retry_thresholds=RETRY_THRESHOLDS) -> list[float]:
    return [threshold] + [t for t in retry_thresholds if t < threshold]


# Boxes of *different* classes overlapping at least this much are the same bubble
# that the model couldn't decide how to classify (e.g. narration vs frameless).
DUPLICATE_IOU = 0.80


def merge_cross_class_duplicates(
    candidates: list[tuple[float, str, tuple[int, int, int, int]]]
) -> list[tuple[float, str, tuple[int, int, int, int]]]:
    """Merge near-identical boxes that YOLO reported under different classes.

    YOLO scores each class separately, so a bubble the model is unsure how to
    classify shows up as several weak boxes (e.g. narration 0.46 + frameless 0.45)
    that each fall under the threshold. Each group is merged into its strongest
    box/class, with the confidences combined as 1 - (1-a)(1-b).
    Takes and returns (confidence, class_name, (x_min, y_min, x_max, y_max)), with
    confidences rounded to 2 decimals and sorted strongest first.
    """
    merged: list[dict] = []
    for conf, class_name, xyxy in sorted(candidates, key=lambda c: c[0], reverse=True):
        for group in merged:
            if class_name not in group["classes"] and _iou(group["xyxy"], xyxy) >= DUPLICATE_IOU:
                group["conf"] = 1 - (1 - group["conf"]) * (1 - conf)
                group["classes"].add(class_name)
                break
        else:
            merged.append({"conf": conf, "class_name": class_name, "xyxy": xyxy, "classes": {class_name}})

    merged.sort(key=lambda g: g["conf"], reverse=True)
    return [(round(g["conf"], 2), g["class_name"], g["xyxy"]) for g in merged]


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = max(area_a + area_b - inter, 1)
    return inter / union


class SpeechBubbleDetection:
    def __init__(self, model: YOLO, threshold: float | None = None):
        self.model = model
        self.threshold = threshold if threshold is not None else constants.SPEECH_BUBBLE_THRESHOLD
    

    def detect(self, image: np.ndarray) -> list[SpeechBubble]:
        # Run YOLO once at the lowest threshold we might accept, then walk down the
        # threshold ladder by filtering that single result.
        SBresult = self.model(image, conf=min(self.threshold, *RETRY_THRESHOLDS))[0]
        candidates = merge_cross_class_duplicates([
            (box.conf.item(), self.model.names[int(box.cls.item())], tuple(map(int, box.xyxy.tolist()[0])))
            for box in (SBresult.boxes or [])
        ])

        for threshold in thresholds_to_try(self.threshold):
            SBdata: list[SpeechBubble] = []
            for SBconf, class_name, (SBx_min, SBy_min, SBx_max, SBy_max) in candidates:
                if SBconf < threshold:
                    continue
                bounding_image = image[SBy_min:SBy_max, SBx_min:SBx_max]
                SBdata.append(SpeechBubble(
                    position=(SBx_min, SBy_min, SBx_max, SBy_max),
                    confidence=SBconf,
                    class_name=class_name,
                    bounding_image=bounding_image,
                    text_clusters=None
                ))
            if SBdata:
                return SBdata

        # Classical-vision fallback: only run when YOLO found nothing even at the
        # lowest retry threshold, so it stays a rare-case supplement instead of
        # risking false-positive "bubbles" on plain artwork.
        SBdata = []

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Otsu threshold; bubbles are typically bright inside
        _, bin_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((7, 7), np.uint8)
        closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, ww, hh = cv2.boundingRect(cnt)
            area = ww * hh
            if area <= 0:
                continue
            ratio = area / float(w * h)
            # relax thresholds to catch tall/skinny bubbles too
            if ratio < 0.005 or ratio > 0.60:
                continue
            if ww < 24 or hh < 48:
                continue
            aspect = ww / float(hh)
            if aspect < 0.15 or aspect > 6.0:
                continue
            roi_mean = np.mean(gray[max(0, y):min(h, y + hh), max(0, x):min(w, x + ww)])
            if roi_mean < 160:  # bubbles tend to be light
                continue
            candidate = (x, y, x + ww, y + hh)
            # only add if not overlapping with existing bubbles too much
            if any(_iou(candidate, sb.position) > 0.15 for sb in SBdata):
                continue
            bounding_image = image[y:y + hh, x:x + ww]
            SBdata.append(SpeechBubble(
                position=candidate,
                confidence=0.30,
                class_name="standard",
                bounding_image=bounding_image,
                text_clusters=None
            ))

        return SBdata

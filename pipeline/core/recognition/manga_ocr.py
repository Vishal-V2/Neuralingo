from manga_ocr import MangaOcr
from PIL import Image
import cv2
import numpy as np

from pipeline.core.box_data import SpeechBubble, TextCluster
from pipeline.core.utility import convert_image
from pipeline.core.detection.captions import column_ocr_images


# Margin (px) kept around each line when reading paragraphs line by line.
LINE_PADDING = 4


class TextRecognition:
    def __init__(self, manga_ocr: MangaOcr):
        self.manga_ocr = manga_ocr
    

    def _text_recognition_single(self, text_cluster_data: TextCluster) -> str:
        if text_cluster_data.lines:
            return self._text_recognition_lines(text_cluster_data)

        text_cluster_image = text_cluster_data.bounding_image
        text_cluster_image_pil = convert_image.cv2_to_pil(text_cluster_image)

        ocr_results = self.manga_ocr(text_cluster_image_pil)
        print("OCR: ", ocr_results)
        return ocr_results


    def _text_recognition_lines(self, text_cluster_data: TextCluster) -> str:
        # Paragraphs on text pages: manga_ocr is trained on short snippets and garbles
        # or truncates long multi-line blocks, so read one line at a time.
        image = text_cluster_data.bounding_image
        h, w = image.shape[:2]
        cx, cy = text_cluster_data.position[:2]
        pad = LINE_PADDING

        parts = []
        for x_min, y_min, x_max, y_max in text_cluster_data.lines:
            crop = image[max(0, y_min - cy - pad):min(h, y_max - cy + pad),
                         max(0, x_min - cx - pad):min(w, x_max - cx + pad)]
            if not crop.size:
                continue
            if text_cluster_data.color is not None:
                # Outlined caption column: read its letter fills as clean text, in chunks.
                parts.extend(self.manga_ocr(Image.fromarray(chunk)) for chunk in column_ocr_images(crop))
            else:
                parts.append(self.manga_ocr(convert_image.cv2_to_pil(crop)))

        ocr_results = "".join(parts)
        print("OCR: ", ocr_results)
        return ocr_results

    
    def _text_recognition_batch(self, speech_bubbles_data: list[SpeechBubble]) -> list[SpeechBubble]:
        for speech_bubble_data in speech_bubbles_data:
            if speech_bubble_data.bounding_image is None:
                continue
            
            text_clusters_data = speech_bubble_data.text_clusters
            
            for text_cluster_data in text_clusters_data:
                if text_cluster_data.bounding_image is None:
                    continue
                jp_text = self._text_recognition_single(text_cluster_data)
                text_cluster_data.jp_text = jp_text
        
        return speech_bubbles_data
    

    def text_recognition(self, data: TextCluster | list[SpeechBubble]) -> str | list[SpeechBubble]:
        if isinstance(data, TextCluster):
            return self._text_recognition_single(data)
        elif isinstance(data, list):
            return self._text_recognition_batch(data)
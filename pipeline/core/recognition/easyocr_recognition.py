import easyocr
import cv2
import os

from pipeline.core.box_data import SpeechBubble, TextCluster



class EasyOcrRecognition:
    # Map config language names to EasyOCR language codes
    LANG_MAP = {
        "japanese": ["ja", "en"],
        "chinese_simplified": ["ch_sim", "en"],
        "chinese_traditional": ["ch_tra", "en"],
        "korean": ["ko", "en"],
    }

    def __init__(
        self,
        language: str = "japanese",
        easy_ocr: easyocr.Reader | None = None,
        gpu: bool | None = None,
    ):
        if easy_ocr is not None:
            self.easy_ocr = easy_ocr
            self.language = language
            return

        if gpu is None:
            try:
                import torch
                gpu = torch.cuda.is_available()
            except Exception:
                gpu = False

        # Prevent CPU overload when GPU is unavailable.
        if not gpu:
            try:
                import torch
                cpu_count = os.cpu_count() or 4
                torch.set_num_threads(max(1, min(4, cpu_count // 2)))
                torch.set_num_interop_threads(1)
            except Exception:
                pass

        lang_codes = self.LANG_MAP.get(language, ["ja", "en"])
        self.easy_ocr = easyocr.Reader(lang_codes, gpu=gpu)
        self.language = language
        print(f"EasyOCR initialized for language: {language} ({lang_codes})")


    @classmethod
    def create_reader(cls, language: str = "japanese", gpu: bool = True) -> easyocr.Reader:
        lang_codes = cls.LANG_MAP.get(language, ["ja", "en"])
        reader = easyocr.Reader(lang_codes, gpu=gpu)
        print(f"EasyOCR initialized for language: {language} ({lang_codes})")
        return reader


    @staticmethod
    def _normalize_text(parts: list[str]) -> str:
        return " ".join(part.strip() for part in parts if part and part.strip()).strip()


    def _readtext_best_effort(self, image) -> str:
        # Balanced OCR: keep coverage high without overwhelming CPU/GPU.
        results_line = self.easy_ocr.readtext(
            image,
            detail=0,
            paragraph=False,
            rotation_info=[90, 180, 270],
            decoder="greedy",
            batch_size=1,
            workers=0,
        )
        text_line = self._normalize_text(results_line)

        # Secondary paragraph pass can recover some split or vertical text layouts.
        results_para = self.easy_ocr.readtext(
            image,
            detail=0,
            paragraph=True,
            rotation_info=[90, 180, 270],
            decoder="greedy",
            batch_size=1,
            workers=0,
        )
        text_para = self._normalize_text(results_para)

        return text_para if len(text_para) > len(text_line) else text_line


    def _text_recognition_single(self, text_cluster_data: TextCluster) -> str:
        text_cluster_image = text_cluster_data.bounding_image
        if text_cluster_image is None:
            return ""

        # First pass: original crop (most accurate in most cases).
        ocr_text = self._readtext_best_effort(text_cluster_image)

        # Fallback pass 1: grayscale for low-contrast text.
        if len(text_cluster_image.shape) == 3:
            gray_image = cv2.cvtColor(text_cluster_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = text_cluster_image
        gray_text = self._readtext_best_effort(gray_image)
        if len(gray_text) > len(ocr_text):
            ocr_text = gray_text

        # Fallback pass 2 (only if still weak): thresholded image.
        if len(ocr_text) <= 1:
            _, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh_text = self._readtext_best_effort(thresh_image)
            if len(thresh_text) > len(ocr_text):
                ocr_text = thresh_text

        ocr_text = ocr_text.strip()

        print("OCR: ", ocr_text)
        return ocr_text


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


class TextRecognition(EasyOcrRecognition):
    def __init__(self, easy_ocr: easyocr.Reader):
        super().__init__(easy_ocr=easy_ocr)

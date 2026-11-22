from PySide6.QtCore import (
    QObject, Signal, Slot
)
import numpy as np
from PIL import Image

from pipeline.core.utility import convert_image
from pipeline.core.cleaning import text_masking
from pipeline.core.cleaning.base import CleaningBase
from pipeline.core.box_data import TextCluster



class CleaningWorker(QObject):
    finished = Signal(Image.Image)

    def __init__(self, inpainter, text_clusters_data: list[TextCluster], image_cv2: np.ndarray, image_pil: Image.Image):
        super().__init__()
        self.text_clusters_data = text_clusters_data
        self.image_cv2 = image_cv2
        self.image_pil = image_pil
        self.cleaning_process = CleaningBase(inpainter)

        self.image_mask = None
        self.image_cleaned = None


    @Slot()
    def run(self):
        try:
            self._text_masking()
            self._text_cleaning()
        except Exception as e:
            print(f"[CLEANING] Error: {e}")
            if self.image_cleaned is None:
                # Fall back to the uncleaned page so downstream stages still get a valid image.
                self.image_cleaned = self.image_pil

        self.finished.emit(self.image_cleaned)


    def _text_masking(self):
        self.mask_canvas = convert_image.blank_canvas(self.image_cv2)
        mask_np = self.mask_canvas.copy()
        for text_cluster_data in self.text_clusters_data:
            if text_cluster_data.bounding_image is None:
                continue
            mask_np = text_masking.apply_text_mask(mask_np, text_cluster_data)

        # apply_text_mask draws text black (0) on a white (255) canvas, but both
        # inpainters (SimpleLama, MiGAN ONNX) inpaint where the mask is white.
        # Invert so text = 255, background = 0. An untouched canvas becomes all
        # black, i.e. nothing is inpainted.
        mask_np = 255 - mask_np

        self.image_mask = Image.fromarray(mask_np)
        self.image_mask.save("output/image_mask.png")

    def _text_cleaning(self):
        self.image_cleaned = self.cleaning_process.batch_cleaning(self.image_pil, self.image_mask)
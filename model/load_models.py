from ultralytics import YOLO
import constants
import logging
import warnings
from config.base import ConfigBase

try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("manga_ocr").setLevel(logging.ERROR)
logging.getLogger("manga_ocr.ocr").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=UserWarning, message=".*unauthenticated requests to the HF Hub.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*tie.*word_embeddings.*")

try:
    from loguru import logger as _loguru_logger
    _loguru_logger.remove()
    _loguru_logger.add(lambda msg: None, level="ERROR")
except Exception:
    pass

MANGA_OCR_MODEL = "kha-white/manga-ocr-base"


def _manga_ocr_source() -> str:
    # Load manga_ocr from the local Hugging Face cache when it's there: by repo name,
    # transformers first asks the Hub whether the model changed, which costs ~4.5 s
    # per start. Falls back to the repo name (download) on a fresh machine.
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(MANGA_OCR_MODEL, local_files_only=True)
    except Exception:
        return MANGA_OCR_MODEL


speech_bubble_model = None
text_cluster_model = None
ocr_engine = None
cleaning_inpainter = None

def load_all_models():
    global speech_bubble_model, text_cluster_model, ocr_engine, cleaning_inpainter

    speech_bubble_model = YOLO(constants.SPEECH_BUBBLE_MODEL_PATH)
    text_cluster_model = YOLO(constants.TEXT_CLUSTER_MODEL_PATH)

    config = ConfigBase()
    recognition_config = config.recognition_config
    selected_engine = recognition_config.get("ocr_engine", "manga_ocr")
    ocr_language = recognition_config.get("ocr_language", "japanese")

    if selected_engine == "easyocr":
        from pipeline.core.recognition.easyocr_recognition import EasyOcrRecognition
        ocr_engine = EasyOcrRecognition(language=ocr_language)
    else:
        from manga_ocr import MangaOcr
        ocr_engine = MangaOcr(_manga_ocr_source())

    cleaning_config = config.cleaning_config
    selected_inpainter = cleaning_config.get("model", "lama")

    if selected_inpainter == "migan":
        from pipeline.core.cleaning.migan_inpainter import MiganInpainter
        cleaning_inpainter = MiganInpainter()
    else:
        from pipeline.core.cleaning.lama_inpainter import LamaInpainter
        cleaning_inpainter = LamaInpainter()

    return speech_bubble_model, text_cluster_model, ocr_engine, cleaning_inpainter
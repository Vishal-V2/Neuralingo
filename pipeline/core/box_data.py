from dataclasses import dataclass, field
import numpy as np
from PIL import Image

@dataclass
class TextCluster:
    position: tuple[int, int, int, int]
    confidence: float
    bounding_image: np.ndarray | None = None
    jp_text: str | None = None
    tr_text: str | None = None
    area: float | None = None
    # Absolute (x_min, y_min, x_max, y_max) of each text line, for paragraph regions
    # on text pages (page_mode: text); None for normal bubble text.
    lines: list[tuple[int, int, int, int]] | None = None
    # Fill colour (RGB) of an outlined caption's text: one per speaker on multi-speaker
    # captions (see detection/captions.py); None otherwise.
    color: tuple[int, int, int] | None = None
    # Outlined caption lettering with a light outline around darker fill (e.g. purple
    # letters outlined in white) instead of the usual dark outline around light fill.
    light_outline: bool = False


@dataclass
class SpeechBubble:
    position: tuple[int, int, int, int]
    confidence: float
    class_name: str
    bounding_image: np.ndarray | None = None
    text_clusters: list[TextCluster] = field(default_factory=list)


@dataclass
class TranslationPage:
    index: int
    file_name: str
    image_cv2: np.ndarray
    image_pil: Image.Image
    cleaned_image: Image.Image
    speech_bubbles: list[SpeechBubble] = field(default_factory=list)



@dataclass
class TranslationBatch:
    pages: list[TranslationPage] = field(default_factory=list)
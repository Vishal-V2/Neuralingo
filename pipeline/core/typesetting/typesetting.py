from PySide6.QtWidgets import (
    QGraphicsScene, QGraphicsPixmapItem, QGraphicsTextItem,
    QApplication
)
from PySide6.QtGui import (
    QPixmap, QFont, QFontDatabase, QFontMetricsF,
    QTextCursor, QTextBlockFormat, QImage, 
    QPainter, QTextOption, QPen
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
import os
from PIL import Image
from PIL.ImageQt import ImageQt

from pipeline.core.box_data import SpeechBubble
from pipeline.core.detection.text_blocks import PARAGRAPH_CLASS_NAME
from pipeline.core.detection.captions import CAPTION_CLASS_NAME, outline_padding
from pipeline.core.typesetting.text_color import text_color
import constants


# Space between speakers' paragraphs in a caption, as a fraction of the font size.
CAPTION_PARAGRAPH_GAP = 0.6


class OutlinedTextItem(QGraphicsTextItem):
    def __init__(self, text=""):
        super().__init__(text)
        self.outline_width = 0
        self.outline_color = Qt.black
        self.outline_enabled = False
        

    def setOutline(self, width, color):
        self.outline_width = width
        self.outline_color = color
        self.outline_enabled = width > 0
        self.update()
        

    def paint(self, painter, option, widget):
        if self.outline_enabled and self.outline_width > 0:
            painter.save()
            
            outline_pen = QPen(self.outline_color)
            outline_pen.setWidth(self.outline_width)
            painter.setPen(outline_pen)
            
            original_color = self.defaultTextColor()
            
            doc = self.document().clone()
            
            for dx in range(-self.outline_width, self.outline_width + 1, max(1, self.outline_width // 2)):
                for dy in range(-self.outline_width, self.outline_width + 1, max(1, self.outline_width // 2)):
                    if dx == 0 and dy == 0:
                        continue
                    
                    painter.translate(dx, dy)
                    self.setDefaultTextColor(self.outline_color)
                    super().paint(painter, option, widget)
                    painter.translate(-dx, -dy)
            
            self.setDefaultTextColor(original_color)
            
            # Draw the main text in the center
            painter.setPen(QPen(original_color))
            painter.translate(0, 0)
            super().paint(painter, option, widget)
            
            painter.restore()
        else:
            super().paint(painter, option, widget)



class TypesettingProcess:
    def __init__(self):
        self.clear_scene()
        self.outline_width = 2
        self.outline_color = Qt.white


    def convert_image(self, image_pil: Image.Image) -> QImage:
        image_qt = ImageQt(image_pil)

        return QImage(image_qt)


    def set_image(self, image_qt: QImage):
        self.scene = QGraphicsScene()
        pixmap = QPixmap.fromImage(image_qt)
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.image_item)


    def set_outline_properties(self, width=2, color=Qt.black):
        """Set the outline width and color for all text items"""
        self.outline_width = width
        self.outline_color = color
    

    def load_font(self):
        font_id = QFontDatabase.addApplicationFont(constants.CCVICTORYSPEECH_PATH)
        if font_id == -1:
            print("Failed to load custom font, trying alternative comic fonts...")
            # Try common comic/manga fonts that might be available
            available_fonts = QFontDatabase.families()
            comic_fonts = ["Comic Sans MS", "Anime Ace", "Manga Temple", "Arial Black", "Impact"]
            
            for font in comic_fonts:
                if font in available_fonts:
                    self.REGULAR_FONT = font
                    print(f"Using alternative font: {font}")
                    return
            
            # Fallback to system font
            self.REGULAR_FONT = QApplication.font().family()
            print(f"Using system font: {self.REGULAR_FONT}")
        else:
            self.REGULAR_FONT = QFontDatabase.applicationFontFamilies(font_id)[0]
            print(f"Successfully loaded custom font: {self.REGULAR_FONT}")
    

    def update_text_item(
        self, 
        font_size,
        text_width,
        text_item: OutlinedTextItem,
        alignment=Qt.AlignCenter
    ) -> OutlinedTextItem:
        selected_font = QFont(self.REGULAR_FONT)
        selected_font.setPixelSize(font_size)

        text_item.setDefaultTextColor(Qt.black)
        text_item.setFont(selected_font)
        text_item.setTextWidth(text_width)
        
        # Apply outline settings
        text_item.setOutline(self.outline_width, self.outline_color)

        text_option = QTextOption()
        text_option.setWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)

        text_document = text_item.document()
        text_document.setDefaultTextOption(text_option)

        text_cursor = QTextCursor(text_document)
        text_block = QTextBlockFormat()
        text_block.setAlignment(alignment)
        text_cursor.select(QTextCursor.Document)
        text_cursor.mergeBlockFormat(text_block)
        text_cursor.clearSelection()

        return text_item
    

    def text_typesetting(
        self, 
        speech_bubble_data: SpeechBubble,
        idx: int
    ):
        self.current_index = idx
        if speech_bubble_data.class_name == PARAGRAPH_CLASS_NAME:
            self.paragraph_typesetting(speech_bubble_data)
            return
        if speech_bubble_data.class_name == CAPTION_CLASS_NAME:
            self.caption_typesetting(speech_bubble_data)
            return

        is_frameless = (speech_bubble_data.class_name == constants.FRAMELESS_CLASS_NAME)
        SBx_min, SBy_min, SBx_max, SBy_max = speech_bubble_data.position
        SBx_length = (SBx_max - SBx_min)
        SBy_length = (SBy_max - SBy_min)

        if not is_frameless:
            SBx_length *= 0.65
            SBy_length *= 0.70
        elif is_frameless:
            if SBx_length < SBy_length:
                ratio = SBx_length / SBy_length
                if ratio < 0.15:
                    SBx_length *= 1.75
            elif SBy_length < SBx_length:
                ratio = SBy_length / SBx_length
                if ratio < 0.15:
                    SBy_length *= 1.75
                    

        text_clusters_data = speech_bubble_data.text_clusters

        if len(text_clusters_data) == 1:
            text_cluster_data = text_clusters_data[0]
            if text_cluster_data.tr_text:
                self.process_text_typesetting((SBx_min, SBx_max), (SBx_length, SBy_length), text_cluster_data.tr_text, text_cluster_data.position, is_side=False,
                                              color=text_color(text_cluster_data.bounding_image))
        else:
            for text_cluster_data in text_clusters_data:
                TCx_min, TCy_min, TCx_max, TCy_max = text_cluster_data.position
                TCx_length = (TCx_max - TCx_min)
                TCy_length = (TCy_max - TCy_min)

                TCarea = TCx_length * TCy_length
                text_cluster_data.area = TCarea
            
            TCsorted = sorted(text_clusters_data, key=lambda x: x.area, reverse=True)

            for i, td in enumerate(TCsorted):
                if not td.tr_text:
                    continue
                color = text_color(td.bounding_image)
                if i == 0:
                    self.process_text_typesetting((SBx_min, SBx_max), (SBx_length, SBy_length), td.tr_text, td.position, is_side=False, color=color)
                else:
                    self.process_text_typesetting((SBx_min, SBx_max), (SBx_length, SBy_length), td.tr_text, td.position, is_side=True, color=color)


    def paragraph_typesetting(self, speech_bubble_data: SpeechBubble):
        # Paragraphs on text pages: fill the block's rectangle with a left-aligned
        # paragraph starting at its top, like the original text.
        text_cluster_data = speech_bubble_data.text_clusters[0]
        if not text_cluster_data.tr_text:
            return

        x_min, y_min, x_max, y_max = speech_bubble_data.position
        width, height = x_max - x_min, y_max - y_min

        # Never larger than the original lines, so a short title doesn't balloon.
        lines = text_cluster_data.lines or []
        max_size = int(sorted(l[3] - l[1] for l in lines)[len(lines) // 2]) if lines else height

        text_item = OutlinedTextItem(text_cluster_data.tr_text)

        words = text_cluster_data.tr_text.split()

        def fits(size: int) -> bool:
            self.update_text_item(size, width, text_item, Qt.AlignLeft)
            document = text_item.document()
            # Qt breaks a word that doesn't fit mid-word, so check the longest word directly.
            font = QFont(self.REGULAR_FONT)
            font.setPixelSize(size)
            metrics = QFontMetricsF(font)
            usable_width = width - 2 * document.documentMargin()
            return (document.size().height() <= height
                    and max(metrics.horizontalAdvance(w) for w in words) <= usable_width)

        # Largest font size that fits. The bubble routine (start at 1000px, shrink by
        # the overflow ratio) overshoots to a tiny size on wide paragraph blocks.
        low, high = constants.MINIMUM_FONT_SIZE, max(constants.MINIMUM_FONT_SIZE, max_size)
        while low < high:
            mid = (low + high + 1) // 2
            if fits(mid):
                low = mid
            else:
                high = mid - 1
        self.update_text_item(low, width, text_item, Qt.AlignLeft)

        self.position_text_item(text_item, x_min, y_min)


    def caption_typesetting(self, speech_bubble_data: SpeechBubble):
        # Multi-speaker captions: one paragraph per speaker, stacked in reading order
        # over the whole caption area, each in the speaker's fill colour with an
        # outline like the original lettering (black, or white for light-outline captions).
        clusters = [c for c in speech_bubble_data.text_clusters if c.tr_text]
        if not clusters:
            return

        x_min, y_min, x_max, y_max = speech_bubble_data.position
        width, height = x_max - x_min, y_max - y_min
        column_widths = sorted(l[2] - l[0] - 2 * outline_padding(c.light_outline) for c in clusters for l in (c.lines or []))
        max_size = column_widths[len(column_widths) // 2] if column_widths else height

        items = [OutlinedTextItem(c.tr_text) for c in clusters]
        words = [w for c in clusters for w in c.tr_text.split()]

        def layout(size: int) -> float:
            """Lay the paragraphs out at this size; returns their total height."""
            total = 0.0
            for cluster, item in zip(clusters, items):
                self.update_text_item(size, width, item, Qt.AlignCenter)
                item.setDefaultTextColor(QColor(*cluster.color) if cluster.color else Qt.black)
                item.setOutline(max(2, size // 10), Qt.white if cluster.light_outline else Qt.black)
                total += item.document().size().height()
            return total + CAPTION_PARAGRAPH_GAP * size * (len(items) - 1)

        def fits(size: int) -> bool:
            font = QFont(self.REGULAR_FONT)
            font.setPixelSize(size)
            metrics = QFontMetricsF(font)
            usable_width = width - 2 * items[0].document().documentMargin()
            return (layout(size) <= height
                    and max(metrics.horizontalAdvance(w) for w in words) <= usable_width)

        low, high = constants.MINIMUM_FONT_SIZE, max(constants.MINIMUM_FONT_SIZE, max_size)
        while low < high:
            mid = (low + high + 1) // 2
            if fits(mid):
                low = mid
            else:
                high = mid - 1
        layout(low)

        y = y_min
        for item in items:
            self.position_text_item(item, x_min, y)
            y += item.document().size().height() + CAPTION_PARAGRAPH_GAP * low


    def process_text_typesetting(
        self, 
        speech_bubble_x: tuple[int, int], 
        speech_bubble_lengths: tuple[int, int], 
        tr_text: str, 
        position: tuple[int, int, int, int], 
        is_side=False,
        color: tuple[tuple[int, int, int], bool] | None = None):
        """color: (RGB, light) of the original lettering (see text_color); None draws the
        default black text with a white outline."""
        text_item = OutlinedTextItem(tr_text)
        font_size = constants.INITIAL_FONT_SIZE

        SBx_length, SBy_length = speech_bubble_lengths
        if is_side:
            TCx_min, TCy_min, TCx_max, TCy_max = position
            TCx_length = TCx_max - TCx_min
            TCy_length = TCy_max - TCy_min
            TCx_center = (TCx_min + TCx_max) / 2
            TCy_center = (TCy_min + TCy_max) / 2

            ratio = TCx_length / TCy_length

            if ratio < 0.50:
                TCx_length *= 1.5
            
            text_item, font_size = self.adjust_font_size(text_item, font_size, (TCx_length, TCy_length))

            text_item_x = TCx_center - (text_item.document().size().width() / 2)
            text_item_y =TCy_center - (text_item.document().size().height() / 2)

            SBx_min, SBx_max = speech_bubble_x 
            SBx_center = (SBx_min + SBx_max) / 2
            if TCx_center < SBx_center:
                dist = (SBx_center - TCx_center) * 0.30
                text_item_x -= dist
            elif TCx_center > SBx_center:
                dist = (TCx_center - SBx_center) * 0.30
                text_item_x += dist
            
            self._apply_text_color(text_item, color)
            self.position_text_item(text_item, text_item_x, text_item_y)
        else:
            text_item, font_size = self.adjust_font_size(text_item, font_size, (SBx_length, SBy_length))
            text_item = self.fine_tune_font_size(text_item, font_size, (SBx_length, SBy_length))

            TCx_min, TCy_min, TCx_max, TCy_max = position
            TCx_center = (TCx_min + TCx_max) / 2
            TCy_center = (TCy_min + TCy_max) / 2
            text_item_x = TCx_center - (text_item.document().size().width() / 2)
            text_item_y = TCy_center - (text_item.document().size().height() / 2)

            self._apply_text_color(text_item, color)
            self.position_text_item(text_item, text_item_x, text_item_y)
    

    def _apply_text_color(self, text_item: OutlinedTextItem, color) -> None:
        # Keep the original lettering's colour; the outline contrasts with it.
        if color is None:
            return
        rgb, light = color
        text_item.setDefaultTextColor(QColor(*rgb))
        text_item.setOutline(self.outline_width, Qt.black if light else Qt.white)


    def adjust_font_size(
        self,
        text_item: OutlinedTextItem,
        font_size: int,
        speech_bubble_lengths: tuple[int, int],
        alignment=Qt.AlignCenter
    ) -> tuple[OutlinedTextItem, int]:
        SBx_length, SBy_length = speech_bubble_lengths

        while True:
            text_item = self.update_text_item(font_size, SBx_length, text_item, alignment)
            text_document_height = text_item.document().size().height()

            if text_document_height > SBy_length:
                height_ratio = text_document_height / SBy_length
                font_size = font_size / height_ratio
            else:
                break
        text_item = self.update_text_item(font_size, SBx_length, text_item, alignment)

        return text_item, font_size


    def fine_tune_font_size(
        self,
        text_item: OutlinedTextItem,
        font_size: int,
        speech_bubble_lengths: tuple[int, int],
        alignment=Qt.AlignCenter
    ) -> OutlinedTextItem:
        SBx_length, SBy_length = speech_bubble_lengths
        i = 0
        while i < constants.MAX_ITERATIONS:
            text_document_height = text_item.document().size().height()
            speech_bubble_area = SBx_length * SBy_length
            text_document_area = SBx_length * text_document_height
            area_ratio = (text_document_area / speech_bubble_area) * 100

            if constants.MIN_AREA_THRESH < area_ratio < constants.MAX_AREA_THRESH:
                break
            elif area_ratio < constants.MIN_AREA_THRESH:
                font_size *= constants.INCREASE_VALUE
            elif area_ratio > constants.MAX_AREA_THRESH:
                font_size *= constants.DECREASE_VALUE
            
            i += 1
            text_item = self.update_text_item(font_size, SBx_length, text_item, alignment)
        text_item = self.update_text_item(font_size, SBx_length, text_item, alignment)

        return text_item
    

    def position_text_item(
        self, 
        text_item, 
        text_item_x, 
        text_item_y
    ):
        text_item.setPos(text_item_x, text_item_y)
        text_item.setParentItem(self.image_item)
    

    def save_image(self):
        rect = self.scene.itemsBoundingRect()
        image = QImage(rect.size().toSize(), QImage.Format_ARGB32)
        image.fill(Qt.transparent)

        painter = QPainter(image)
        self.scene.render(painter, target=rect, source=rect)
        painter.end()

        stem = os.path.splitext(os.path.basename(self.current_file_name))[0]
        output_path = f"output/{stem}.png"
        image.save(output_path)
        print(f"Saved image with text to {output_path}")


    def batch_text_typesetting(
        self, 
        speech_bubbles_data: list[SpeechBubble],
        idx: int,
        file_name: str
    ):
        self.current_index = idx
        self.current_file_name = file_name
        for speech_bubble_data in speech_bubbles_data:
            self.text_typesetting(speech_bubble_data, idx)
        self.save_image()
    

    def clear_scene(self):
        if hasattr(self, "scene"):
            self.scene.clear()
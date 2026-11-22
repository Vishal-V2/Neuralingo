from PySide6.QtWidgets import (
    QApplication, QMainWindow

)
import sys
import os
import re
from model.load_models import load_all_models
from pipeline.manager.base import AutomaticTranslationBase

def natural_sort_key(name: str):
    # "2.png" before "10.png": compare digit runs as numbers, the rest case-insensitively.
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Main Window")
        self.setMinimumSize(800, 600)

        sb, tc, ocr, inpainter = load_all_models()

        self.ye = AutomaticTranslationBase(sb, tc, ocr, inpainter)

        folder_path = "out"

        file_paths = [
            os.path.join(folder_path, f)
            for f in sorted(os.listdir(folder_path), key=natural_sort_key)
            if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg' , '.webp'))
        ]

        self.ye.receive_selected_files.emit(file_paths)


    

def main():
    app = QApplication(sys.argv)

    main_window = MainWindow()
    main_window.show()

    sys.exit(app.exec())
 

if __name__ == "__main__":
    main()  
    
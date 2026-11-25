from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
import sys
import os
import re
import traceback
from model.load_models import load_all_models
from pipeline.manager.base import AutomaticTranslationBase

INPUT_FOLDER = "out"


def natural_sort_key(name: str):
    # "2.png" before "10.png": compare digit runs as numbers, the rest case-insensitively.
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def main():
    # No window: the Qt application only drives the pipeline's signals and threads
    # (typesetting also needs it for fonts and painting). It quits when the batch is done.
    app = QApplication(sys.argv)

    def quit_on_error(exc_type, exc, tb):
        # An uncaught exception in a pipeline step means the next step never starts;
        # exit with an error instead of hanging.
        traceback.print_exception(exc_type, exc, tb)
        app.exit(1)

    sys.excepthook = quit_on_error

    file_paths = [
        os.path.join(INPUT_FOLDER, f)
        for f in sorted(os.listdir(INPUT_FOLDER), key=natural_sort_key)
        if os.path.isfile(os.path.join(INPUT_FOLDER, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    ]
    if not file_paths:
        sys.exit(f"No images in {INPUT_FOLDER}/")

    sb, tc, ocr, inpainter = load_all_models()
    translator = AutomaticTranslationBase(sb, tc, ocr, inpainter)
    translator.finished.connect(app.quit)
    # Start once the event loop is running, so a quit from a fast run isn't missed.
    QTimer.singleShot(0, lambda: translator.receive_selected_files.emit(file_paths))

    code = app.exec()
    # Everything is saved by now. Skip Python's teardown: it destroys Qt objects that
    # live on worker threads from the main thread, which can segfault after a good run.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()

"""Detection + OCR regression check over the pages in test/.

    python tools/regression.py snapshot [--mode comic|text]   # save current results
    python tools/regression.py compare  [--mode comic|text]   # diff against the saved results

Snapshots are written to test/snapshots/<mode>/<page>.json. Nothing is sent to any
translation API and no cleaning/typesetting is run; only detection and OCR.
"""
import argparse
import json
import logging
import os
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

TEST_DIR = os.path.join(ROOT, "test")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def load_pipeline(mode: str):
    from ultralytics import YOLO
    import constants
    from config.base import ConfigBase
    from pipeline.core.recognition.easyocr_recognition import EasyOcrRecognition
    from pipeline.core.recognition.manga_ocr import TextRecognition

    config = ConfigBase()
    detection_config = dict(config.detection_config)
    detection_config["page_mode"] = mode

    recognition_config = config.recognition_config
    if recognition_config.get("ocr_engine", "manga_ocr") == "easyocr":
        ocr = EasyOcrRecognition(language=recognition_config.get("ocr_language", "japanese"))
    else:
        from manga_ocr import MangaOcr
        ocr = TextRecognition(MangaOcr())

    models = (YOLO(constants.SPEECH_BUBBLE_MODEL_PATH), YOLO(constants.TEXT_CLUSTER_MODEL_PATH))
    return detection_config, models, ocr


def run_page(path: str, detection_config: dict, models, ocr) -> list[dict]:
    from pipeline.core.utility import convert_image
    from pipeline.worker.detection import DetectionWorker

    worker = DetectionWorker(detection_config, models[0], models[1], convert_image.to_cv2(path))
    worker._speech_bubble_detection()
    worker._text_cluster_detection()
    worker._uncovered_text_detection()
    if worker.page_mode == "text":
        worker._text_block_detection()
    worker._caption_detection()
    bubbles = ocr.text_recognition(worker.speech_bubbles_data)

    return [
        {
            "class": b.class_name,
            "position": list(b.position),
            "confidence": b.confidence,
            "clusters": [
                {"position": list(c.position), "confidence": c.confidence, "text": c.jp_text}
                for c in (b.text_clusters or [])
            ],
        }
        for b in sorted(bubbles, key=lambda b: tuple(b.position))
    ]


def describe(region: dict) -> str:
    texts = " | ".join(c["text"] or "" for c in region["clusters"])
    return f'{region["class"]:10s} {tuple(region["position"])} {region["confidence"]:.2f}  {texts}'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["snapshot", "compare"])
    parser.add_argument("--mode", choices=["comic", "text"], default="comic")
    args = parser.parse_args()

    pages = sorted(f for f in os.listdir(TEST_DIR) if f.lower().endswith(IMAGE_EXTENSIONS)) if os.path.isdir(TEST_DIR) else []
    if not pages:
        sys.exit(f"No test pages found in {TEST_DIR}")

    snapshot_dir = os.path.join(TEST_DIR, "snapshots", args.mode)
    os.makedirs(snapshot_dir, exist_ok=True)
    detection_config, models, ocr = load_pipeline(args.mode)

    changed = 0
    for page in pages:
        result = run_page(os.path.join(TEST_DIR, page), detection_config, models, ocr)
        snapshot_path = os.path.join(snapshot_dir, os.path.splitext(page)[0] + ".json")

        if args.action == "snapshot":
            with open(snapshot_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"[saved]     {page}: {len(result)} regions")
            continue

        if not os.path.exists(snapshot_path):
            print(f"[no snapshot] {page}")
            continue
        with open(snapshot_path, encoding="utf-8") as f:
            before = json.load(f)
        if before == result:
            print(f"[unchanged] {page}")
            continue

        changed += 1
        old, new = {describe(r) for r in before}, {describe(r) for r in result}
        print(f"[CHANGED]   {page}: {len(before)} -> {len(result)} regions")
        for line in sorted(old - new):
            print(f"    - {line}")
        for line in sorted(new - old):
            print(f"    + {line}")

    if args.action == "compare":
        print(f"\n{changed} of {len(pages)} pages changed")


if __name__ == "__main__":
    main()

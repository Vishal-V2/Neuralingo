"""Explain what the caption splitter decided for each region of a page.

    python tools/caption_diagnose.py <image> [--mode comic|text]

Runs detection the way the app does (up to, not including, the caption step), then
runs pipeline/core/detection/captions.split_caption on every region and prints its
reasoning: column spacing, the columns found and why any were dropped, or why the
region was rejected. Prints numbers only: no OCR, no text, no images.
"""
import argparse
import logging
import os
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image")
    parser.add_argument("--mode", choices=["comic", "text"], default=None,
                        help="page mode (default: the one in config.yaml)")
    parser.add_argument("--speaker-distance", type=float, default=None,
                        help="speaker colour split distance (default: the one in config.yaml)")
    args = parser.parse_args()

    from ultralytics import YOLO
    import constants
    from config.base import ConfigBase
    from pipeline.core.detection.captions import SPEAKER_SPLIT_DISTANCE, split_caption
    from pipeline.core.utility import convert_image
    from pipeline.worker.detection import DetectionWorker

    detection_config = dict(ConfigBase().detection_config)
    if args.mode:
        detection_config["page_mode"] = args.mode
    speaker_distance = args.speaker_distance
    if speaker_distance is None:
        speaker_distance = float((detection_config.get("caption") or {}).get("speaker_color_distance",
                                                                            SPEAKER_SPLIT_DISTANCE))
    image = convert_image.to_cv2(os.path.abspath(os.path.join(os.environ.get("PWD", ROOT), args.image))
                                 if not os.path.isabs(args.image) else args.image)
    if image is None:
        sys.exit(f"Could not read {args.image}")

    worker = DetectionWorker(detection_config, YOLO(constants.SPEECH_BUBBLE_MODEL_PATH),
                             YOLO(constants.TEXT_CLUSTER_MODEL_PATH), image)
    worker._speech_bubble_detection()
    worker._text_cluster_detection()
    worker._uncovered_text_detection()
    if worker.page_mode == "text":
        worker._text_block_detection()

    print(f"page {image.shape[1]}x{image.shape[0]}, mode {worker.page_mode}, {len(worker.speech_bubbles_data)} regions\n")
    for region in sorted(worker.speech_bubbles_data, key=lambda r: tuple(r.position)):
        print(f"{region.class_name} {tuple(region.position)} conf {region.confidence:.2f}, "
              f"{len(region.text_clusters or [])} text piece(s)")
        log: list[str] = []
        result = split_caption(image, region, log, speaker_distance=speaker_distance)
        for line in log:
            print(f"    {line}")
        if result is not None:
            print(f"    -> caption {tuple(result.position)} with {len(result.text_clusters)} speaker(s): "
                  + ", ".join(f"{len(c.lines)} column(s) rgb{c.color}" for c in result.text_clusters))
        print()


if __name__ == "__main__":
    main()

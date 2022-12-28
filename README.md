Detect missing critical items (owner's manual) attached to a lawn mower during assembly process.

# Detectors
[engine_gxv160_gcv170.pt](https://github.com/unitedtriangle/detector-of-missing-owners-manual-attached-to-lawn-mower/blob/main/detectors/engine_gxv160_gcv170.pt) is used to detect the engine type of the lawn mower ("gxv160" or "gcv170"). The detector was trained from the [YOLOv5n](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n.pt) model).

[critical_items.pt](https://github.com/unitedtriangle/detector-of-missing-owners-manual-attached-to-lawn-mower/blob/main/detectors/critical_items.pt) is used to detect critical items (owner's manual) attached to the lawn mower. The detector was trained from the [YOLOv5n](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n.pt) model).

[country_of_origin_label.pt](https://github.com/unitedtriangle/detector-of-missing-owners-manual-attached-to-lawn-mower/blob/main/detectors/country_of_origin_label.pt) is used to detect the country of origin label attached to the lawn mower. The detector was trained from the [YOLOv5n](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n.pt) model).

# References
[Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) and [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36) for training and running custom [YOLOv5](https://github.com/ultralytics/yolov5) detectors.

[YOLOv4 Object Detection on Webcam In Google Colab](https://github.com/theAIGuysCode/colab-webcam/blob/main/yolov4_webcam.ipynb) by [theAIGuysCode](https://github.com/theAIGuysCode).

[Pixabay](https://pixabay.com/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=music&amp;utm_content=26528) for sound effects.

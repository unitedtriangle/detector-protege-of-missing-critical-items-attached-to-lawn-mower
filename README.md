Detect missing owner's manual attached to the lawn mower during assembly process.

# Detectors
[detector_engine_gxv160_type_gcv170_type.pt](https://github.com/unitedtriangle/detector-of-missing-owners-manual-attached-to-lawn-mower/blob/main/detector_engine_gxv160_type_gcv170_type.pt) is used to detect the engine of the lawn mower ("gxv160 type" or "gcv170 type"). The detector was trained from the [YOLOv5n](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n.pt) model).

[detector_owners_manual_front_cover_back_cover.pt](https://github.com/unitedtriangle/detector-of-missing-owners-manual-attached-to-lawn-mower/blob/main/detector_owners_manual_front_cover_back_cover.pt) is used to detect the owner's manual attached to the lawn mower ("front cover" or "back cover"). The detector was trained from the [YOLOv5n](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n.pt) model).

[detector_country_of_origin_label.pt](https://github.com/unitedtriangle/detector-of-missing-owners-manual-attached-to-lawn-mower/blob/main/detector_country_of_origin_label.pt) is used to detect the country of origin label attached to the lawn mower. The detector was trained from the [YOLOv5n](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n.pt) model).

# References
[Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) and [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36) for training and running custom [YOLOv5](https://github.com/ultralytics/yolov5) detectors.

[YOLOv4 Object Detection on Webcam In Google Colab](https://github.com/theAIGuysCode/colab-webcam/blob/main/yolov4_webcam.ipynb) by [theAIGuysCode](https://github.com/theAIGuysCode).

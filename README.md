<div align="center">
  <p>
    <img src="https://github.com/unitedtriangle/detector-protege-of-missing-critical-items-attached-to-lawn-mower/raw/main/documentation/images/logo.jpg" width="180">
  </p>
  
  Protege detects missing critical items attached to a lawn mower at the final inspection stage of the assembly process. Critical items include an owner's manual.
</div>


# System layout
<div align="center">
  <p>
    <img src="https://github.com/unitedtriangle/detector-protege-of-missing-critical-items-attached-to-lawn-mower/raw/main/documentation/images/system-layout.jpg" width="540">
  </p>
</div>


# Installation
Clone the repository.
```bash
git clone https://github.com/unitedtriangle/detector-protege-of-missing-critical-items-attached-to-lawn-mower protege
cd protege
```

It is recommended to install the following packages in a Python>=3.7.0 [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment).

```bash
pip install easyocr  # to read texts on labels
pip uninstall opencv-python-headless  # to avoid conflict with opencv-python required for yolov5
pip install -r requirements.txt  # yolov5 and application-specific dependencies
```

# Detectors
[engine.pt](https://github.com/unitedtriangle/detector-protege-of-missing-critical-items-attached-to-lawn-mower/blob/main/detectors/engine.pt) trained from [YOLOv5n](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n.pt) is used to detect the engine of the lawn mower.

[items_critical.pt](https://github.com/unitedtriangle/detector-of-missing-owners-manual-attached-to-lawn-mower/blob/main/detectors/items_critical.pt) trained from [YOLOv5n](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n.pt) is used to detect critical items including an owner's manual attached to the lawn mower.

[label_origin.pt](https://github.com/unitedtriangle/detector-protege-of-missing-critical-items-attached-to-lawn-mower/blob/main/detectors/label_origin.pt) trained from [YOLOv5n](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n.pt) is used to detect the origin label attached to the lawn mower.

# References
[Ultralytics HUB](https://ultralytics.com/hub), [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) and [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36) for training and running custom [YOLOv5](https://github.com/ultralytics/yolov5) detectors.

[YOLOv4 Object Detection on Webcam In Google Colab](https://github.com/theAIGuysCode/colab-webcam/blob/main/yolov4_webcam.ipynb) by [theAIGuysCode](https://github.com/theAIGuysCode).

[Pixabay](https://pixabay.com/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=music&amp;utm_content=26528) for sound effects including [LOUD Smoke Alarm unplug](https://pixabay.com/sound-effects/loud-smoke-alarm-unplug-26528/), [Ding! idea](https://pixabay.com/sound-effects/ding-idea-40142/).

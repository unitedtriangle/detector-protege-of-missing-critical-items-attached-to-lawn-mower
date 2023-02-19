<div align="center">
  <p>
    <img src="https://github.com/unitedtriangle/detector-protege-of-missing-critical-items-attached-to-lawn-mower/raw/main/documentation/images/logo.jpg" width="180">
  </p>
  
  Protege detects missing critical items attached to a lawn mower at the final inspection stage of the assembly process. Critical items include an owner's manual.
</div>


# System layout
<div align="center">
  <p>
    <img src="https://github.com/unitedtriangle/detector-protege-of-missing-critical-items-attached-to-lawn-mower/raw/main/documentation/images/system-layout.jpg" width="720">
  </p>
</div>

Protege runs fastest on a computer with a graphics processing unit (GPU) from NVIDIA and a central processing unit (CPU) with multiple cores. The GPU if available speeds up the detection of critical items attached to the lawn mower and the reading of lawn mower identity number from the image of the country of origin label. The multi-core CPU supports the parallelism of the 2 processes.


# Installation
Clone the repository protege.
```bash
git clone https://github.com/unitedtriangle/detector-protege-of-missing-critical-items-attached-to-lawn-mower protege
cd protege
```

It is recommended to create and activate a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment) with Python 3.7.0 or later in the local repository to install all packages required for the application.
```bash
python3 -m venv env
source env/bin/activate
```

Clone the repository yolov5 to the directory site-packages.
```bash
YOLOV5_DIRPATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")/yolov5
git clone https://github.com/ultralytics/yolov5 $YOLOV5_DIRPATH
```

Install the following packages using [pip](https://pip.pypa.io/en/stable/).
```bash
pip install easyocr  # to read mower id from image of origin label
pip uninstall opencv-python-headless  # to avoid conflict with opencv-python required for yolov5
pip install -r $YOLOV5_DIRPATH/requirements.txt  # packages required for yolov5
pip install -r requirements.txt  # packages required for protege specifically
```


# Detectors
[engine.pt](https://github.com/unitedtriangle/detector-protege-of-missing-critical-items-attached-to-lawn-mower/blob/main/detectors/engine.pt) trained from [YOLOv5n](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n.pt) is used to detect the engine of the lawn mower.

[items_critical.pt](https://github.com/unitedtriangle/detector-of-missing-owners-manual-attached-to-lawn-mower/blob/main/detectors/items_critical.pt) trained from [YOLOv5n](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n.pt) is used to detect critical items including an owner's manual attached to the lawn mower.

[label_origin.pt](https://github.com/unitedtriangle/detector-protege-of-missing-critical-items-attached-to-lawn-mower/blob/main/detectors/label_origin.pt) trained from [YOLOv5n](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n.pt) is used to detect the origin label attached to the lawn mower.


# References
[Ultralytics HUB](https://ultralytics.com/hub), [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) and [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36) for training and running custom [YOLOv5](https://github.com/ultralytics/yolov5) detectors.

[YOLOv4 Object Detection on Webcam In Google Colab](https://github.com/theAIGuysCode/colab-webcam/blob/main/yolov4_webcam.ipynb) by [theAIGuysCode](https://github.com/theAIGuysCode).

[Pixabay](https://pixabay.com/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=music&amp;utm_content=26528) for sound effects including [LOUD Smoke Alarm unplug](https://pixabay.com/sound-effects/loud-smoke-alarm-unplug-26528/), [Ding! idea](https://pixabay.com/sound-effects/ding-idea-40142/).

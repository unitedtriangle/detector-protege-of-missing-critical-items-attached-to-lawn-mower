"""
detect, recognize things in images
"""


import torch  # faster on gpu device with cuda support greatly or mps support fairly
import easyocr  # reliant on torch thus faster on gpu device with cuda support greatly or mps support fairly
import cv2
from scipy import ndimage
import ray
import re
from datetime import datetime
from pytz import timezone
from collections import deque
import itertools
from pathlib import Path

from utilities.video import VideoCaptureBufferless
from utilities.parallel import Share


def detect_motion(image1=None, image2=None, min_different_area=1500):
  """
  detect motion or change between 2 images
  min_different_area: minimum different area bewteen 2 images to be considered motion or change
  """
  
  if image1 is None or image2 is None:
    return []
  
  # convert images to grayscale
  image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
  image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
  
  # smooth images with gaussian kernel
  image1_blur = cv2.GaussianBlur(image1_gray, ksize=(5, 5), sigmaX=0)
  image2_blur = cv2.GaussianBlur(image2_gray, ksize=(5, 5), sigmaX=0)
  
  diff_images_blur = cv2.absdiff(image1_blur, image2_blur)  # difference between 2 images in grayscale and smoothed
  ret, diff_images_threshold = cv2.threshold(diff_images_blur, thresh=20, maxval=255, type=cv2.THRESH_BINARY)  # set pixels > 20 to 255 (white) and others 0 (black)
  
  contours, hierarchy = cv2.findContours(diff_images_threshold, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)  # contours of motion areas
  
  return [contr for contr in contours if cv2.contourArea(contr) >= min_different_area]


@ray.remote
class ReaderIdMower:
  def __init__(self, share_states=None, x_text_init=0, y_text_init=30, offset_y_text=30, fontface_text=cv2.FONT_HERSHEY_SIMPLEX, fontscale_text=1, thickness_text=2):
    """
    stream video capturing origin label, read mower id in separate process, e.g. mower id MAYU-2023402
    
    share_states: instance of Share in separate process storing states of reader of mower id including
      is_stopped=True, if true read_id_mower_from_stream(...) called to stop, if false allowed to run; set to false automatically upon call of read_id_mower_from_stream(...)
      id_mower="",  # mower id read from origin label; if empty read_id_mower_from_stream(...) called to read mower id from origin label, otherwise consider mower id read already
      filepath_label="",  # path to origin label saved for record; if empty read_id_mower_from_stream(...) called to save origin label for record, otherwise consider origin label saved already;
    shareable with process calling read_id_mower_from_stream(...) for coordination between processes
    
    parameters to render texts written on images:
      x_text_init, y_text_init: position to write first line of text on image; passed to argument org of cv2.putText(...)
      offset_y_text: vertical move in pixels to write next line of text on image
      fontface_text: font or font type of text; passed to argument fontFace of cv2.putText(...)
      fontscale_text: factor to scale size of text from default size of given font; passed to argument fontScale of cv2.putText(...)
      thickness_text: thickness of strokes making up characters in text; passed to argument thickness of cv2.putText(...)
    """
    
    # states of reader of mower id shareable with process calling read_id_mower_from_stream(...) for coordination between processes
    if share_states is None:
      self.share_states = Share.remote(
        is_stopped=True,  # if true read_id_mower_from_stream(...) called to stop, if false allowed to run; set to true upon call of stop(), set to false automatically upon call of read_id_mower_from_stream(...)
        id_mower="",  # mower id read from origin label; if empty read_id_mower_from_stream(...) called to read mower id from origin label, otherwise consider mower id read already
        filepath_label="",  # path to origin label saved for record; if empty read_id_mower_from_stream(...) called to save origin label for record, otherwise consider origin label saved already
      )
    else:
      self.share_states = share_states
    
    # parameters to render texts written on images
    self.x_text_init, self.y_text_init = x_text_init, y_text_init
    self.offset_y_text = offset_y_text
    self.fontface_text = fontface_text
    self.fontscale_text = fontscale_text
    self.thickness_text = thickness_text

    # load detector of origin label
    # self.detector_label = torch.hub.load("ultralytics/yolov5", model="custom", path="https://github.com/unitedtriangle/detector-of-missing-owners-manual-attached-to-lawn-mower/raw/main/detectors/country_of_origin_label.pt")  # online
    self.detector_label = torch.hub.load(f"{Path.home()}/.cache/torch/hub/ultralytics_yolov5_master", source="local", model="custom", path="detectors/origin_label.pt")  # offline; path to repository must be string
    self.detector_label.max_det = 1  # maximum 1 detection of origin label per image
    self.detector_label.conf = 0.6  # confidence threshold
    
    self.reader_text = easyocr.Reader(["en"])  # load optical character recognition model to read mower id from origin label
  
  def read_id_mower_from_stream(self, source_label, dirpath_label=Path("records/temporary"), rotations=(15, -15), pattern_prefix=re.compile(r"[A-Z]{4}"), pattern_numeral=re.compile(r"\d{7,}"), min_different_area=3000, patience_id_mower=3):
    """
    stream video, detect, extract origin label from frame, read mower id from origin label, e.g. mower id MAYU-2023402; reliant on torch due to read_id_mower thus faster on gpu device with cuda support greatly or mps support fairly
    source_label: source of video capturing origin label in form of address of ip camera or index of device camera or path to video file; passed to argument source of VideoCaptureBufferless(...)
    dirpath_label: path to directory of origin labels saved for record
    rotations: rotate origin label by specified angles in degrees to find best reading of mower id; passed to argument rotations of read_id_mower(...)
    pattern_prefix: pattern object of prefix of mower id consisting of letters, e.g. prefix MAYU; passed to argument pattern_prefix of read_id_mower(...)
    pattern_numeral: pattern object of numeral of mower id consisting of digits, e.g. numeral 2023402; passed to argument pattern_numeral of read_id_mower(...)
    min_different_area: minimum different area bewteen 2 images to be considered motion or change; passed to argument min_different_area of detect_motion(...)
    patience_id_mower: maximum number of attempts at reading mower id, 1 attempt per image, before deeming mower id not readable
    """
    
    if not isinstance(dirpath_label, Path):
      dirpath_label = Path(dirpath_label)
    
    dirpath_label.mkdir(parents=True, exist_ok=True)  # create directory of origin labels if not existent

    capture = VideoCaptureBufferless(source=source_label).start()  # open source of video capturing origin label
    name_window = "protege reading mower id thru ip camera origin label"  # name of window to display video in
    
    self.share_states.set.remote("is_stopped", False)  # to allow read_id_mower_from_stream(...) to run
    
    # states of reader of mower id local
    frames_latest = deque(maxlen=2)  # last 2 frames
    patience_id_mower_remain = patience_id_mower  # remaining number of attempts at reading mower id, 1 attempt per image, before deeming mower id not readable
    id_mower = ""  # mower id read from origin label; if empty read_id_mower_from_stream(...) called to read mower id from origin label, otherwise consider mower id read already
    filepath_label = ""  # path to origin label saved for record; if empty read_id_mower_from_stream(...) called to save origin label for record, otherwise consider origin label saved already
    
    # stream video
    try:
      while True:
        if ray.get(self.share_states.get.remote("is_stopped")):  # read_id_mower_from_stream(...) called to stop
          break

        image = capture.read()  # image in bgr; convert image to rgb using image[:, :, ::-1] to suit yolov5 detectors

        datetime_capture = datetime.now(timezone('Australia/Melbourne')).strftime("%Y-%m-%d %H:%M:%S %Z")  # local datetime of image capture in proper format

        x_text, y_text = self.x_text_init, self.y_text_init  # position to write line of text on image

        frames_latest.append(image)  # update last 2 frames

        # detect origin label
        results_label = self.detector_label(image[:, :, ::-1])  # convert image from bgr to rgb to suit yolov5 detector
        detections_label = results_label.xyxy[0]  # detections of origin label for given single image

        # check for origin label in frame
        if not len(detections_label):  # no origin label detected
          # reset some of states of reader of mower id
          if patience_id_mower_remain != patience_id_mower:
            patience_id_mower_remain = patience_id_mower
          if id_mower:
            id_mower = ""
            self.share_states.set.remote("id_mower", id_mower)  # update mower id shareable with process calling read_id_mower_from_stream(...)
          if filepath_label:
            filepath_label = ""
            self.share_states.set.remote("filepath_label", filepath_label)  # update path to origin label shareable with process calling read_id_mower_from_stream(...)

          # write no origin label message on image
          text = "no origin label detected"
          cv2.putText(image, text=text, org=(x_text, y_text), fontFace=self.fontface_text, fontScale=self.fontscale_text, thickness=self.thickness_text, color=(255, 255, 255))  # message in white
          y_text += self.offset_y_text  # move to next line of text
        else:  # origin label in frame
          if filepath_label:  # origin label saved already
            # write origin label saved message on image
            text = "origin label saved already"
            cv2.putText(image, text=text, org=(x_text, y_text), fontFace=self.fontface_text, fontScale=self.fontscale_text, thickness=self.thickness_text, color=(0, 255, 0))  # message in green
            y_text += self.offset_y_text  # move to next line of text

            if id_mower:  # mower id readable fully or partly
              # write existent mower id on image
              text = f"mower id read already as {id_mower}"
              cv2.putText(image, text=text, org=(x_text, y_text), fontFace=self.fontface_text, fontScale=self.fontscale_text, thickness=self.thickness_text, color=(0, 255, 0))  # message in green
              y_text += self.offset_y_text  # move to next line of text
            else:  # not readable at all mower id
              # write no mower id message on image
              text = "not able to read mower id"
              cv2.putText(image, text=text, org=(x_text, y_text), fontFace=self.fontface_text, fontScale=self.fontscale_text, thickness=self.thickness_text, color=(255, 255, 255))  # message in white
              y_text += self.offset_y_text  # move to next line of text
          else:  # no origin label saved yet
            if detect_motion(*frames_latest, min_different_area=min_different_area):  # motion in frame
              # write motion caution on image
              text = "something moving. make sure everything still to read mower id"
              cv2.putText(image, text=text, org=(x_text, y_text), fontFace=self.fontface_text, fontScale=self.fontscale_text, thickness=self.thickness_text, color=(0, 192, 255))  # message in amber
              y_text += self.offset_y_text  # move to next line of text
            else:  # no motion detected
              image_label = results_label.crop(save=False)[0]["im"]  # extract origin label

              id_mower = self.read_id_mower(image_label, rotations=rotations, pattern_prefix=pattern_prefix, pattern_numeral=pattern_numeral)  # read mower id from origin label

              # check validity of mower id
              if id_mower:  # readable fully or partly
                # write mower id on image
                text = f"mower id {id_mower}"
                cv2.putText(image, text=text, org=(x_text, y_text), fontFace=self.fontface_text, fontScale=self.fontscale_text, thickness=self.thickness_text, color=(0, 255, 0))  # message in green
                y_text += self.offset_y_text  # move to next line of text

                # write local datetime of image capture on image
                text = datetime_capture
                cv2.putText(image, text=text, org=(x_text, y_text), fontFace=self.fontface_text, fontScale=self.fontscale_text, thickness=self.thickness_text, color=(255, 255, 255))  # message in white
                y_text += self.offset_y_text  # move to next line of text

                # save origin label for record
                filepath_label = Path(dirpath_label, f"{id_mower} {datetime_capture}.jpg")
                cv2.imwrite(filepath_label.as_posix(), image)  # save path of origin label must be string
                
                self.share_states.set.remote("id_mower", id_mower)  # update mower id shareable with process calling read_id_mower_from_stream(...)
                self.share_states.set.remote("filepath_label", filepath_label)  # update path to origin label shareable with process calling read_id_mower_from_stream(...)
              else:  # not readable at all
                if patience_id_mower_remain:  # attempt(s) at reading mower id allowed still
                  patience_id_mower_remain -= 1

                  # write read retry message on image
                  text = "not able to read mower id. trying again..."
                  cv2.putText(image, text=text, org=(x_text, y_text), fontFace=self.fontface_text, fontScale=self.fontscale_text, thickness=self.thickness_text, color=(255, 255, 255))  # message in white
                  y_text += self.offset_y_text  # move to next line of text
                else:  # no more attempts at reading mower id allowed
                  # write no mower id message on image
                  text = "not able to read mower id"
                  cv2.putText(image, text=text, org=(x_text, y_text), fontFace=self.fontface_text, fontScale=self.fontscale_text, thickness=self.thickness_text, color=(255, 255, 255))  # message in white
                  y_text += self.offset_y_text  # move to next line of text

                  # write local datetime of image capture on image
                  text = datetime_capture
                  cv2.putText(image, text=text, org=(x_text, y_text), fontFace=self.fontface_text, fontScale=self.fontscale_text, thickness=self.thickness_text, color=(255, 255, 255))  # message in white
                  y_text += self.offset_y_text  # move to next line of text

                  # save origin label for record
                  filepath_label = Path(dirpath_label, f"na {datetime_capture}.jpg")
                  cv2.imwrite(filepath_label.as_posix(), image)  # save path of origin label must be string
                  
                  self.share_states.set.remote("filepath_label", filepath_label)  # update path to origin label shareable with process calling read_id_mower_from_stream(...)
        
        cv2.imshow(name_window, image)  # show image
        cv2.waitKey(1)  # wait 1 ms for image to show unless key pressed
    except (KeyboardInterrupt, Exception) as error:
      # end video stream
      capture.stop()
      cv2.destroyAllWindows()
      cv2.waitKey(1)  # to fix bug on macos where window not close
      
      self.share_states.set.remote("is_stopped", True)  # to reflect read_id_mower_from_stream(...) stopped due to interruption
      
      display(error)

    # end video stream
    capture.stop()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # to fix bug on macos where window not close
    
    self.share_states.set.remote("is_stopped", True)  # to reflect read_id_mower_from_stream(...) stopped naturally
  
  def get_share_states(self):
    """
    get instance of Share storing states of reader of mower id
    """
    
    return self.share_states
  
  def read_id_mower(self, image_label, rotations=(15, -15), pattern_prefix=re.compile(r"[A-Z]{4}"), pattern_numeral=re.compile(r"\d{7,}")):
    """
    read mower id from origin label, e.g. MAYU-2023402; reliant on torch due to easyocr thus faster on gpu device with cuda support greatly or mps support fairly
    image_label (numpy array): origin label
    rotations: rotate origin label by specified angles in degrees to find best reading of mower id
    pattern_prefix: pattern object of prefix of mower id consisting of letters, e.g. prefix MAYU
    pattern_numeral: pattern object of numeral of mower id consisting of digits, e.g. numeral 2023402
    """
    
    prefix = ""  # prefix of mower id consisting of letters, e.g. MAYU
    numeral = ""  # numeral of mower id consisting of digits, e.g. 2023402

    images_label_checkable = itertools.chain([image_label], (ndimage.rotate(image_label, angle) for angle in rotations))  # checkable origin labels consisting of origin label as given and rotated by specified angles all evaluated lazily

    # find best reading of mower id from checkable origin labels
    for image in images_label_checkable:
      # read text from origin label
      words = self.reader_text.readtext(image, detail=False)
      text = " ".join(words)

      # read prefix of mower id from text
      match_prefix = pattern_prefix.search(text)
      if match_prefix:
        prefix = match_prefix.group()

      # read numeral of mower id from text
      match_numeral = pattern_numeral.search(text)
      if match_numeral:
        numeral = match_numeral.group()

      # stop when best reading of mower id found
      if prefix and numeral:  # readable fully
        break

    id_mower = str.lower(f"{prefix}-{numeral}" if prefix or numeral else "")  # mower id in proper format if readable fully or partly

    return id_mower
  
  def detect_extract_label(self, image):
    """
    detect, extract origin label from image; intended for testing purposes
    image: source of image in form of filepath or url or numpy array or other forms suitable for yolov5 detector; passed to first argument of detector_label(...)
    """
    
    results_label = self.detector_label(image)  # detect origin label
    
    image_bbox_label = results_label.render()[0]  # render bounding box around origin label
    
    image_label = results_label.crop(save=False)[0]["im"]  # extract origin label
    
    return image_bbox_label, image_label

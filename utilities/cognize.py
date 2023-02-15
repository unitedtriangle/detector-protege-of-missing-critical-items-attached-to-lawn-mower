"""
detect, recognize things in images
"""


import torch  # faster on gpu device via api compute unified device architecture greatly or api metal performance shaders fairly
import easyocr  # reliant on torch thus faster on gpu device via api compute unified device architecture greatly or api metal performance shaders fairly
import cv2
from scipy import ndimage
import ray
import re
from collections import deque
import itertools
import functools
import requests
from io import BytesIO
from pathlib import Path
from datetime import datetime
import pytz
from pygame import mixer
import traceback

from utilities.video import VideoCaptureBufferless
from utilities.parallel import Share


class Protege:
  def __init__(self, filepath_detector_engine=Path("detectors", "engine.pt"), filepath_detector_items=Path("detectors", "items_critical.pt"), url_detector_engine="https://github.com/unitedtriangle/detector-protege-of-missing-critical-items-attached-to-lawn-mower/raw/main/detectors/engine.pt", url_detector_items="https://github.com/unitedtriangle/detector-protege-of-missing-critical-items-attached-to-lawn-mower/raw/main/detectors/items_critical.pt", number_detections_max_engine=1, number_detections_max_items=1000, confidence_detector_engine=0.6, confidence_detector_items=0.6, is_on_gpu_mps_items=False, is_on_gpu_mps_label=False):
    """
    stream video 1 capturing critical items including owners manual
    detect missing critical items from frame of video 1
    save image capturing critical items temporarily for record
    run reader of mower id in process 2 to
      stream video 2 capturing origin label
      detect, extract image of origin label from frame of video 2
      read mower id from image of origin label, e.g. mower id MAYU-2023402
      save image capturing origin label for record
    coordinate with reader of mower id to
      raise warning when origin label missing
      move image capturing critical items, image capturing origin label saved temporarily for record to directory of record for permanent storage
    reliant on torch, ReaderIdMower thus faster on gpu device via api compute unified device architecture greatly or api metal performance shaders fairly

    filepath_detector_engine: path to model detector of engine
    if not existent download from specified url

    filepath_detector_items: path to model detector of critical items
    if not existent download from specified url

    url_detector_engine: url to download model detector of engine from

    url_detector_items: url to download model detector of critical items from

    number_detections_max_engine: limit number of detections of engine per image to specified maximum
    
    number_detections_max_items: limit number of detections of critical items per image to specified maximum
    
    confidence_detector_engine: confidence threshold of detector of engine
    
    confidence_detector_items: confidence threshold of detector of critical items

    is_on_gpu_mps_items: put detector of engine, detector of critical items on gpu device from apple via api metal performance shaders if possible or not
    
    is_on_gpu_mps_label: put detector of origin label on gpu device from apple via api metal performance shaders if possible or not
    """

    if not isinstance(filepath_detector_engine, Path):
      filepath_detector_engine = Path(filepath_detector_engine)

    if not isinstance(filepath_detector_items, Path):
      filepath_detector_items = Path(filepath_detector_items)

    # download model detector of engine if not existent
    if not filepath_detector_engine.is_file():
      torch.hub.download_url_to_file(url=url_detector_engine, dst=filepath_detector_engine)
    
    # load detector of engine
    self.detector_engine = torch.hub.load(repo_or_dir=f"{torch.hub.get_dir()}/ultralytics_yolov5_master", source="local", model="custom", path=filepath_detector_engine)  # offline using local repository yolov5, local engine.pt; path to local repository yolov5 must be string
    # self.detector_engine = torch.hub.load(repo_or_dir="ultralytics/yolov5", model="custom", path=url_detector_engine)  # online using remote repository yolov5, remote engine.pt
    self.detector_engine.max_det = number_detections_max_engine  # limit number of detections of engine per image to specified maximum
    self.detector_engine.conf = confidence_detector_engine  # confidence threshold
    
    # download model detector of critical items if not existent
    if not filepath_detector_items.is_file():
      torch.hub.download_url_to_file(url=url_detector_items, dst=filepath_detector_items)
    
    # load detector of critical items
    self.detector_items = torch.hub.load(repo_or_dir=f"{torch.hub.get_dir()}/ultralytics_yolov5_master", source="local", model="custom", path=filepath_detector_items)  # offline using local repository yolov5, local items_critical.pt; path to local repository yolov5 must be string
    # self.detector_items = torch.hub.load(repo_or_dir="ultralytics/yolov5", model="custom", path=url_detector_items)  # online using remote repository yolov5, remote items_critical.pt
    self.detector_items.max_det = number_detections_max_items  # limit number of detections of critical items per image to specified maximum
    self.detector_items.conf = confidence_detector_items  # confidence threshold
    
    # use gpu device if possible
    if torch.cuda.is_available():  # gpu device from nvidia available
      print(f"speed up detector of engine, detector of critical items greatly using gpu device via api compute unified device architecture {torch.cuda.get_device_properties(0).name}")
    else:  # not available gpu device from nvidia
      # caution: operations on images not implemented in api metal performance shaders of torch 1.13.1 stable
      if is_on_gpu_mps_items and torch.backends.mps.is_available() and torch.backends.mps.is_built():  # called to use gpu device from apple if available, enabled with current torch, macos
        device_mps = torch.device("mps")  # gpu device from apple via api metal performance shaders
        
        # put yolov5 detectors on gpu device from apple via api metal performance shaders
        self.detector_engine.to(device_mps)  # detector of engine
        self.detector_items.to(device_mps)  # detector of critical items

        print(f"speed up detector of engine, detector of critical items fairly using gpu device via api metal performance shaders")
      else:
        print("no speedup of detector of engine, detector of critical items due to no gpu device available")
    
    self.reader_id_mower = ReaderIdMower.remote(is_on_gpu_mps_label=is_on_gpu_mps_label)  # load reader of mower id in process 2
    self.share_states_reader = ray.get(self.reader_id_mower.get.remote("share_states_reader"))  # get instance of Share from other process storing states of reader of mower id including
      # is_stopped=True,  # if true read_from_video(...) called to stop; if false read_from_video(...) allowed to run; set to false automatically upon call of read_from_video(...)
      # id_mower="",  # mower id read from image of origin label; if empty read_from_video(...) called to read mower id from image of origin label; if populated consider mower id read already
      # filepath_label="",  # path to image capturing origin label saved for record; if evaluated to false read_from_video(...) called to save image capturing origin label for record; if evaluated to true consider image capturing origin label saved already
      # for coordination
    
    request_get = functools.lru_cache(requests.get)  # to keep recent results of requests.get(...) in memory
    
    mixer.init()  # to play sounds; run mixer.quit() to terminate mixer if needed
    
    # load sound go
    try:
      self.sound_go = mixer.Sound("sounds/go.wav")  # offline using local go.wav
    except FileNotFoundError:
      self.sound_go = mixer.Sound(BytesIO(request_get("https://github.com/unitedtriangle/detector-of-missing-critical-items-attached-to-lawn-mower/raw/main/sounds/go.wav").content))  # online using go.wav from remote repository protege
    
    # load sound warning
    try:
      self.sound_warning = mixer.Sound("sounds/warning.wav")  # offline using local warning.wav
    except FileNotFoundError:
      self.sound_warning = mixer.Sound(BytesIO(request_get("https://github.com/unitedtriangle/detector-of-missing-critical-items-attached-to-lawn-mower/raw/main/sounds/warning.wav").content))  # online using warning.wav from remote repository protege
  
  def inspect(self, source_items, source_label, rotations=(15, -15), pattern_prefix=re.compile(r"[A-Z]{4}"), pattern_numeral=re.compile(r"\d{7,}"), area_different_min=3000, patience_items=18, patience_id_mower=9, dirpath_records_master=Path("records"), dirpath_temporary=Path("records/temporary"),
              x_text_init=0, y_text_init=30, offset_y_text=30, fontface_text=cv2.FONT_HERSHEY_SIMPLEX, fontscale_text=1, thickness_text=2,
              name_timezone="Australia/Melbourne", format_datetime="%Y-%m-%d %H:%M:%S %Z", format_date="%Y-%m-%d",
              name_window_items="protege detecting missing critical items", name_window_label="protege reading mower id from image of origin label", scale_width_items=1, scale_height_items=1, scale_width_label=1, scale_height_label=1, x_window_items=0, y_window_items=0, x_window_label=0, y_window_label=0):
    """
    stream video 1 capturing critical items including owners manual
    detect missing critical items from frame of video 1
    save image capturing critical items temporarily for record
    run reader of mower id in process 2 to
      stream video 2 capturing origin label
      detect, extract image of origin label from frame of video 2
      read mower id from image of origin label, e.g. mower id MAYU-2023402
      save image capturing origin label for record
    coordinate with reader of mower id to
      raise warning when origin label missing
      move image capturing critical items, image capturing origin label saved temporarily for record to directory of record for permanent storage
    reliant on torch, ReaderIdMower thus faster on gpu device via api compute unified device architecture greatly or api metal performance shaders fairly
    
    source_items: source of video capturing critical items in form of address of ip camera or index of device camera or path to video file
    passed to argument source of VideoCaptureBufferless(...)
    
    source_label: source of video capturing origin label in form of address of ip camera or index of device camera or path to video file
    passed to argument source_label of reader_id_mower.read_from_video.remote(...)
    
    rotations: rotate image of origin label by specified angles in degrees to find best reading of mower id
    passed to argument rotations of reader_id_mower.read_from_video.remote(...)
    
    pattern_prefix: regular expression pattern of prefix of mower id consisting of letters, e.g. prefix MAYU
    passed to argument pattern_prefix of reader_id_mower.read_from_video.remote(...)
    
    pattern_numeral: regular expression pattern of numeral of mower id consisting of digits, e.g. numeral 2023402
    passed to argument pattern_numeral of reader_id_mower.read_from_video.remote(...)
    
    area_different_min: minimum different area bewteen 2 images to be considered motion or change
    passed to argument area_different_min of reader_id_mower.read_from_video.remote(...)
    
    patience_items: maximum number of attempts at detecting critical items, retrieving path to image capturing origin label saved for record, 1 attempt per frame, before raising warning critical item(s) or origin label missing
    
    patience_id_mower: maximum number of attempts at reading mower id from image of origin label, 1 attempt per frame, before raising warning mower id not readable
    passed to argument patience_id_mower of reader_id_mower.read_from_video.remote(...)
    
    dirpath_records_master: path to directory to store directory of record
    directory of record to store image capturing critical items, image capturing origin label permanently
    image capturing origin label moved from dirpath_temporary to directory of record for permanent storage
    
    dirpath_temporary: path to directory to save image capturing critical items, image capturing origin label temporarily for record to
    image capturing critical items, image capturing origin label moved to directory of record later for permanent storage
    passed to argument dirpath_label of reader_id_mower.read_from_video.remote(...)
    
    parameters to render texts written on images:
      x_text_init, y_text_init: position to write first line of text on image in
      passed to arguments x_text_init, y_text_init of reader_id_mower.read_from_video.remote(...)
      
      offset_y_text: vertical move in pixels to write next line of text on image
      passed to argument offset_y_text of reader_id_mower.read_from_video.remote(...)
      
      fontface_text: font or font type of text
      passed to argument
        fontFace of cv2.putText(...)
        fontface_text of reader_id_mower.read_from_video.remote(...)
      
      fontscale_text: factor to scale size of text from default size of given font
      passed to argument
        fontScale of cv2.putText(...)
        fontscale_text of reader_id_mower.read_from_video.remote(...)
      
      thickness_text: thickness of strokes making up characters in text
      passed to argument
        thickness of cv2.putText(...)
        thickness_text of reader_id_mower.read_from_video.remote(...)
    
    name_timezone: name of timezone to get datetime of image capture in
    passed to argument zone of pytz.timezone(...)
    
    format_datetime: format of datetime of image capture
    passed to argument format of datetime_capture.strftime(...)
    
    format_date: format of date of image capture
    passed to argument format of datetime_capture.strftime(...)
    
    name_window_items: name of window to display video capturing critical items in
    
    name_window_label: name of window to display video capturing origin label in
    passed to argument name_window_label of reader_id_mower.read_from_video.remote(...)
    
    scale_width_items: scale width of image capturing critical items by specified factor for display
    
    scale_height_items: scale height of image capturing critical items by specified factor for display
    
    scale_width_label: scale width of image capturing origin label by specified factor for display
    passed to argument scale_width_label of reader_id_mower.read_from_video.remote(...)
    
    scale_height_label: scale height of image capturing origin label by specified factor for display
    passed to argument scale_height_label of reader_id_mower.read_from_video.remote(...)
    
    x_window_items, y_window_items: position of window displaying video capturing critical items
    
    x_window_label, y_window_label: position of window displaying video capturing origin label
    passed to arguments x_window_label, y_window_label of reader_id_mower.read_from_video.remote(...)
    """
    
    assert isinstance(patience_items, int) and patience_items > 0, "argument patience_items must be positive integer"
    
    if not isinstance(dirpath_records_master, Path):
      dirpath_records_master = Path(dirpath_records_master)
    
    capture_items = VideoCaptureBufferless(source=source_items).start()  # open source of video capturing critical items
    
    # states of detector of critical items
    items_checklist = frozenset(["owners manual"])  # all required critical items
    items_missing = set() # missing critical items
    is_ok_mower = True  # all checks passed or not; checks saving image capturing critical items, saving image capturing origin label, reading mower id from image of origin label if possible
    patience_items_remain = patience_items  # remaining number of attempts at detecting critical items, retrieving path to image capturing origin label saved for record, 1 attempt per frame, before raising warning critical item(s) or origin label missing
    dirpath_record = ""  # path to directory of record to store image capturing critical items, image capturing origin label permanently
    filepath_items_temporary = ""  # path to image capturing critical items saved temporarily for record; image capturing critical items moved to directory of record later for permanent storage
    filepath_label_temporary = ""  # path to image capturing origin label saved for record retrieved from instance of Share from other process storing states of reader of mower id; image capturing origin label moved to directory of record later for permanent storage
    id_mower = ""  # mower id read from image of origin label retrieved from instance of Share from other process storing states of reader of mower id
    is_on_light_green = False  # light indicator green on or not
    is_on_light_red = False  # light indicator red on or not
    channel_go = None  # channel playing sound go; if not none sound go playing or played already
    channel_warning = None  # channel playing sound warning; if not none sound warning playing or played already
    
    dirpath_records_master.mkdir(parents=True, exist_ok=True)  # create directory to store directory of record if not existent
    
    dirpath_temporary.mkdir(parents=True, exist_ok=True)  # create directory to save image capturing critical items, image capturing origin label temporarily for record to if not existent
    
    # states of reader of mower id
    self.share_states_reader.set.remote("is_stopped", True)  # call reader_id_mower.read_from_video.remote(...) to stop if running
    self.share_states_reader.set.remote("id_mower", "")  # call reader_id_mower.read_from_video.remote(...) to read mower id from image of origin label when run
    self.share_states_reader.set.remote("filepath_label", "")  # call reader_id_mower.read_from_video.remote(...) to save image capturing origin label for record when run

    self.reader_id_mower.read_from_video.remote(
      source_label=source_label, rotations=rotations, pattern_prefix=pattern_prefix, pattern_numeral=pattern_numeral, area_different_min=area_different_min, patience_id_mower=patience_id_mower, dirpath_label=dirpath_temporary,
      x_text_init=x_text_init, y_text_init=y_text_init, offset_y_text=offset_y_text, fontface_text=fontface_text, fontscale_text=fontscale_text, thickness_text=thickness_text,
      format_datetime=format_datetime, name_timezone=name_timezone,
      name_window_label=name_window_label, scale_width_label=scale_width_label, scale_height_label=scale_height_label, x_window_label=x_window_label, y_window_label=y_window_label
    )  # stream video capturing origin label; detect, extract image of origin label from frame of video; read mower id from image of origin label, e.g. mower id MAYU-2023402; save image capturing origin label for record
    
    # prepare window to display video capturing critical items in
    cv2.namedWindow(winname=name_window_items)  # create window
    cv2.moveWindow(winname=name_window_items, x=x_window_items, y=y_window_items)  # position window
    cv2.setWindowProperty(winname=name_window_items, prop_id=cv2.WND_PROP_TOPMOST, prop_value=1)  # send window to top of other windows
    
    # stream video capturing critical items
    try:
      while True:
        image_items = capture_items.read()  # image in bgr; convert image to rgb using image[:, :, ::-1] to suit yolov5 detectors

        datetime_capture = datetime.now(pytz.timezone(name_timezone))  # datetime of image capture in specified timezone

        x_text, y_text = x_text_init, y_text_init  # position to write line of text on image in

        # detect engine
        results_engine = self.detector_engine(image_items[:, :, ::-1])  # convert image from bgr to rgb to suit yolov5 detector
        detections_engine = results_engine.xyxy[0]  # detections of engine for given single image
        
        # check for mower in frame
        if is_ok_mower:  # mower checked ok
          if not len(detections_engine):  # no mower detected
            # reset some of states of detector of critical items
            if patience_items_remain != patience_items:
              patience_items_remain = patience_items
            if dirpath_record:
              dirpath_record = ""
            if filepath_items_temporary:
              filepath_items_temporary = ""
            if filepath_label_temporary:
              filepath_label_temporary = ""
            if id_mower:
              id_mower = ""
            if is_on_light_green:
              is_on_light_green = False
            if is_on_light_red:
              is_on_light_red = False
            if channel_go:
              channel_go = None
            if channel_warning:
              channel_warning = None

            # reset some of states of reader of mower id
            if ray.get(self.share_states_reader.get.remote("id_mower")):
              self.share_states_reader.set.remote("id_mower", "")
            if ray.get(self.share_states_reader.get.remote("filepath_label")):
              self.share_states_reader.set.remote("filepath_label", "")
            
            # write message ready for mower on image
            text = "ready for mower"
            cv2.putText(image_items, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
            y_text += offset_y_text  # move to next line of text
          else:  # mower in frame
            if dirpath_record:  # directory of record existent storing image capturing critical items, image capturing origin label already; same mower in frame
              # write message found all required critical items already on image
              text = f"found all required critical items including {', '.join(items_checklist)} already"
              cv2.putText(image_items, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
              y_text += offset_y_text  # move to next line of text

              if id_mower:  # mower id readable fully or partly
                # write message read mower id on image
                text = f"read mower id as {id_mower} from source of video capturing origin label already"
                cv2.putText(image_items, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
                y_text += offset_y_text  # move to next line of text
              
              # write advice go ahead already on image
              text = "mower ok to go already"
              cv2.putText(image_items, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 255, 0))  # message in green
              y_text += offset_y_text  # move to next line of text
            else:  # no image capturing critical items, image capturing origin label stored in directory of record yet; new mower in frame
              # write message new mower in frame on image
              text = "new mower in view"
              cv2.putText(image_items, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
              y_text += offset_y_text  # move to next line of text
              
              # attempt to detect critical items, retrieve path to image capturing origin label saved for record
              
              items_missing.update(items_checklist)  # set missing critical items to all required critical items at start of check for missing critical items
              
              # detect critical items
              results_items = self.detector_items(image_items[:, :, ::-1])  # convert image from bgr to rgb to suit yolov5 detector
              detections_items = results_items.xyxy[0]  # detections of critical items for given single image

              # check for missing critical items
              items_missing.difference_update(results_items.names[det[5].item()] for det in detections_items)
              if not items_missing:  # found all required critical items
                # write message found all required critical items on image
                text = f"found all required critical items including {', '.join(items_checklist)}"
                cv2.putText(image_items, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
                y_text += offset_y_text  # move to next line of text

              # get mower id read from image of origin label retrieved from instance of Share from other process storing states of reader of mower id
              id_mower = ray.get(self.share_states_reader.get.remote("id_mower"))
              if id_mower:  # mower id readable fully or partly
                # write message read mower id on image
                text = f"read mower id as {id_mower} from source of video capturing origin label"
                cv2.putText(image_items, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
                y_text += offset_y_text  # move to next line of text
              
              filepath_label_temporary = ray.get(self.share_states_reader.get.remote("filepath_label"))  # get path to image capturing origin label saved for record retrieved from instance of Share from other process storing states of reader of mower id
              
              patience_items_remain -= 1  # deduct 1 attempt at detecting critical items, retrieving path to image capturing origin label saved for record
              
              # check for all required critical items, image capturing origin label
              if not items_missing and filepath_label_temporary:  # found, saved temporarily for record
                # write datetime of image capture on image
                text = datetime_capture.strftime(format_datetime)  # datetime of image capture in proper format
                cv2.putText(image_items, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
                y_text += offset_y_text  # move to next line of text
                
                # write advice go ahead on image
                text = "mower ok to go"
                cv2.putText(image_items, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 255, 0))  # message in green
                y_text += offset_y_text  # move to next line of text

                # save image capturing critical items temporarily for record
                filepath_items_temporary = Path(dirpath_temporary, f"critical items {datetime_capture.strftime(format_datetime)}.jpg")  # name image capturing critical items after datetime of image capture in proper format to save temporarily for record
                cv2.imwrite(filepath_items_temporary.as_posix(), image_items)  # save path of image capturing critical items must be string

                # move image capturing critical items, image capturing origin label to directory of record for permanent storage
                dirpath_record = Path(dirpath_records_master, datetime_capture.strftime(format_date), filepath_label_temporary.stem)  # name directory of record after date of image capture in proper format, name of image capturing origin label saved temporarily for record
                dirpath_record.mkdir(parents=True, exist_ok=True)  # create directory of record
                filepath_items_temporary.replace(Path(dirpath_record, filepath_items_temporary.name))  # move image capturing critical items to directory of record keeping same name
                filepath_label_temporary.replace(Path(dirpath_record, filepath_label_temporary.name))  # move image capturing origin label to directory of record keeping same name
                
                is_on_light_green = True  # turn on light indicator green

                channel_go = self.sound_go.play()  # play sound go
              else:  # not found or not saved temporarily for record
                is_ok_mower = False  # set status of mower to not ok
                
                if patience_items_remain:  # attempt(s) at detecting critical items, retrieving path to image capturing origin label saved for record allowed still
                  # write message remaining number of attempts to check for all required critical items, image capturing origin label on image
                  text = f"{patience_items_remain} attempt{'' if patience_items_remain == 1 else 's'} remaining to check for all required critical items, image capturing origin label"
                  cv2.putText(image_items, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
                  y_text += offset_y_text  # move to next line of text
                else:  # no more attempts at detecting critical items, retrieving path to image capturing origin label saved for record allowed
                  is_on_light_red = True  # turn on light indicator red

                  channel_warning = self.sound_warning.play(loops=-1)  # play sound warning on repeat
                  
                  if items_missing:  # missing critical items
                    # write message warning missing critical items on image
                    text = f"warning: rectify missing critical item{'' if len(items_missing) == 1 else 's'} {', '.join(items_missing)}"
                    cv2.putText(image_items, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 0, 255))  # message in red
                    y_text += offset_y_text  # move to next line of text
                  
                  if not filepath_label_temporary:  # not saved temporarily for record image capturing origin label
                    # write message warning missing origin label on image
                    text = "warning: rectify missing origin label"
                    cv2.putText(image_items, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 0, 255))  # message in red
                    y_text += offset_y_text  # move to next line of text
        else:  # not ok mower checked
          if not len(detections_engine):  # no mower detected
            # turn on light indicator red if not on yet
            if not is_on_light_red:
              is_on_light_red = True

            # play sound warning on repeat if not playing yet
            if channel_warning is None:
              channel_warning = self.sound_warning.play(loops=-1)

            if items_missing:  # missing critical items
              # write message warning missing critical items on image
              text = f"warning: put mower back in view to rectify missing critical item{'' if len(items_missing) == 1 else 's'} {', '.join(items_missing)}"
              cv2.putText(image_items, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 0, 255))  # message in red
              y_text += offset_y_text  # move to next line of text

            if not filepath_label_temporary:  # not saved temporarily for record image capturing origin label
              # write message warning missing origin label on image
              text = "warning: put mower back in view to rectify missing origin label"
              cv2.putText(image_items, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 0, 255))  # message in red
              y_text += offset_y_text  # move to next line of text
          else:  # mower in frame
            # write message mower in frame on image
            text = "mower in view"
            cv2.putText(image_items, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
            y_text += offset_y_text  # move to next line of text
            
            if patience_items_remain:  # attempt(s) at detecting critical items, retrieving path to image capturing origin label saved for record allowed still
              # turn off light indicator red if on
              if is_on_light_red:
                is_on_light_red = False

              # stop sound warning if playing
              if channel_warning is not None:
                channel_warning = channel_warning.stop()  # set channel warning to none after stop of sound warning

            # attempt to detect critical items, retrieve path to image capturing origin label saved for record
            
            items_missing.update(items_checklist)  # set missing critical items to all required critical items at start of check for missing critical items

            # detect critical items
            results_items = self.detector_items(image_items[:, :, ::-1])  # convert image from bgr to rgb to suit yolov5 detector
            detections_items = results_items.xyxy[0]  # detections of critical items for given single image

            # check for missing critical items
            items_missing.difference_update(results_items.names[det[5].item()] for det in detections_items)
            if not items_missing:  # found all required critical items
              # write message found all required critical items on image
              text = f"found all required critical items including {', '.join(items_checklist)}"
              cv2.putText(image_items, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
              y_text += offset_y_text  # move to next line of text

            # get mower id read from image of origin label retrieved from instance of Share from other process storing states of reader of mower id
            id_mower = ray.get(self.share_states_reader.get.remote("id_mower"))
            if id_mower:  # mower id readable fully or partly
              # write message read mower id on image
              text = f"read mower id as {id_mower} from source of video capturing origin label"
              cv2.putText(image_items, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
              y_text += offset_y_text  # move to next line of text
            
            filepath_label_temporary = ray.get(self.share_states_reader.get.remote("filepath_label"))  # get path to image capturing origin label saved for record retrieved from instance of Share from other process storing states of reader of mower id
            
            # deduct 1 attempt at detecting critical items, retrieving path to image capturing origin label saved for record but not allow patience_items_remain negative
            if patience_items_remain:  # attempt(s) at detecting critical items, retrieving path to image capturing origin label saved for record allowed still
              patience_items_remain -= 1

            # check for all required critical items, image capturing origin label
            if not items_missing and filepath_label_temporary:  # found, saved temporarily for record
              is_ok_mower = True  # set status of mower to ok

              # write datetime of image capture on image
              text = datetime_capture.strftime(format_datetime)  # datetime of image capture in proper format
              cv2.putText(image_items, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
              y_text += offset_y_text  # move to next line of text
              
              # write advice go ahead on image
              text = "mower ok to go"
              cv2.putText(image_items, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 255, 0))  # message in green
              y_text += offset_y_text  # move to next line of text

              # save image capturing critical items temporarily for record
              filepath_items_temporary = Path(dirpath_temporary, f"critical items {datetime_capture.strftime(format_datetime)}.jpg")  # name image capturing critical items after datetime of image capture in proper format to save temporarily for record
              cv2.imwrite(filepath_items_temporary.as_posix(), image_items)  # save path of image capturing critical items must be string

              # move image capturing critical items, image capturing origin label to directory of record for permanent storage
              dirpath_record = Path(dirpath_records_master, datetime_capture.strftime(format_date), filepath_label_temporary.stem)  # name directory of record after date of image capture in proper format, name of image capturing origin label saved temporarily for record
              dirpath_record.mkdir(parents=True, exist_ok=True)  # create directory of record
              filepath_items_temporary.replace(Path(dirpath_record, filepath_items_temporary.name))  # move image capturing critical items to directory of record keeping same name
              filepath_label_temporary.replace(Path(dirpath_record, filepath_label_temporary.name))  # move image capturing origin label to directory of record keeping same name
              
              is_on_light_green = True  # turn on light indicator green

              # turn off light indicator red if on
              if is_on_light_red:
                is_on_light_red = False

              channel_go = self.sound_go.play()  # play sound go

              # stop sound warning if playing
              if channel_warning is not None:
                channel_warning = channel_warning.stop()  # set channel warning to none after stop of sound warning

            else:  # not found or not saved temporarily for record
              if patience_items_remain:  # attempt(s) at detecting critical items, retrieving path to image capturing origin label saved for record allowed still
                # write message remaining number of attempts to check for all required critical items, image capturing origin label on image
                text = f"{patience_items_remain} attempt{'' if patience_items_remain == 1 else 's'} remaining to check for all required critical items, image capturing origin label"
                cv2.putText(image_items, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
                y_text += offset_y_text  # move to next line of text
              else:  # no more attempts at detecting critical items, retrieving path to image capturing origin label saved for record allowed
                if not is_on_light_red:
                  is_on_light_red = True

                # play sound warning on repeat if not playing yet
                if channel_warning is None:
                  channel_warning = self.sound_warning.play(loops=-1)
                
                if items_missing:  # missing critical items
                  # write message warning missing critical items on image
                  text = f"warning: rectify missing critical item{'' if len(items_missing) == 1 else 's'} {', '.join(items_missing)}"
                  cv2.putText(image_items, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 0, 255))  # message in red
                  y_text += offset_y_text  # move to next line of text
                
                if not filepath_label_temporary:  # not saved temporarily for record image capturing origin label
                  # write message warning missing origin label on image
                  text = "warning: rectify missing origin label"
                  cv2.putText(image_items, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 0, 255))  # message in red
                  y_text += offset_y_text  # move to next line of text

        # display statuses of light indicators
        if is_on_light_green:
          cv2.circle(image_items, center=(int(capture_items.stream.get(cv2.CAP_PROP_FRAME_WIDTH)) - 15, 15), radius=15, color=(0, 255, 0), thickness=-1)  # circle green in upper-right corner of image
        if is_on_light_red:
          cv2.circle(image_items, center=(int(capture_items.stream.get(cv2.CAP_PROP_FRAME_WIDTH)) - 45, 15), radius=15, color=(0, 0, 255), thickness=-1)  # circle red to left of circle green

        image_resized = image_items if scale_width_items == scale_height_items == 1 else cv2.resize(image_items, dsize=(0, 0), fx=scale_width_items, fy=scale_height_items)  # resize image capturing critical items for display
        
        cv2.imshow(name_window_items, image_resized)  # display image capturing critical items in specified window
        cv2.waitKey(1)  # wait 1 ms for image to show unless key pressed
    except (KeyboardInterrupt, Exception):
      self.share_states_reader.set.remote("is_stopped", True)  # call for reader_id_mower.read_from_video.remote(...) to stop

      # turn off light indicator green if on
      if is_on_light_green:
        is_on_light_green = False
      
      # turn off light indicator red if on
      if is_on_light_red:
        is_on_light_red = False
      
      mixer.stop()  # stop all sounds
      
      # stop video capturing critical items
      capture_items.stop()
      cv2.destroyAllWindows()
      cv2.waitKey(1)  # to fix bug on macos window displaying video not closed

      print(traceback.format_exc())
  
  def get(self, name):
    """
    get value of attribute
    
    name: name of attribute
    """
    
    return getattr(self, name)
  
  def set(self, name, value):
    """
    set attribute to specified value
    
    name: name of attribute
    
    value: specified value to set attribute to
    """
    
    setattr(self, name, value)
  
  def detect_engine(self, image_engine):
    """
    detect engine in image capturing engine
    reliant on torch thus faster on gpu device via api compute unified device architecture greatly or api metal performance shaders fairly
    intended for testing purposes
    
    image_engine: source of image capturing engine in form of filepath or url or numpy.array or other forms suitable for yolov5 detector
    passed to first argument of detector_engine(...)
    see details at https://github.com/ultralytics/yolov5/blob/30e4c4f09297b67afedf8b2bcd851833ddc9dead/models/common.py#L243-L252
    """

    # detect engine
    results_engine = self.detector_engine(image_engine)
    detections_engine = results_engine.xyxy[0]  # detections of engine for given single image

    classnames_engine = results_engine.names  # name of engine detectable by detector of engine

    image_engine_bbox = results_engine.render()[0]  # render bounding box around engine in rgb
    
    return detections_engine, classnames_engine, image_engine_bbox

  def detect_items(self, image_items):
    """
    detect critical items in image capturing critical items
    reliant on torch thus faster on gpu device via api compute unified device architecture greatly or api metal performance shaders fairly
    intended for testing purposes
    
    image_items: source of image capturing critical items in form of filepath or url or numpy.array or other forms suitable for yolov5 detector
    passed to first argument of detector_items(...)
    see details at https://github.com/ultralytics/yolov5/blob/30e4c4f09297b67afedf8b2bcd851833ddc9dead/models/common.py#L243-L252
    """

    # detect critical items
    results_items = self.detector_items(image_items)
    detections_items = results_items.xyxy[0]  # detections of critical items for given single image

    classnames_items = results_items.names  # names of critical items detectable by detector of critical items

    image_items_bbox = results_items.render()[0]  # render bounding boxes around critical items in rgb
    
    return detections_items, classnames_items, image_items_bbox


@ray.remote  # run instance of ReaderIdMower in separate process
class ReaderIdMower:
  def __init__(self, filepath_detector_label=Path("detectors", "label_origin.pt"), url_detector_label="https://github.com/unitedtriangle/detector-protege-of-missing-critical-items-attached-to-lawn-mower/raw/main/detectors/label_origin.pt", number_detections_max_label=1, confidence_detector_label=0.6, share_states_reader=None, is_on_gpu_mps_label=False):
    """
    stream video capturing origin label
    detect, extract image of origin label from image capturing origin label of video
    read mower id from image of origin label, e.g. mower id MAYU-2023402
    save image capturing origin label for record
    reliant on torch, easyocr thus faster on gpu device via api compute unified device architecture greatly or api metal performance shaders fairly
    
    filepath_detector_label: path to model detector of origin label
    if not existent download from specified url

    url_detector_label: url to download model detector of origin label from

    number_detections_max_label: limit number of detections of origin label per image to specified maximum

    confidence_detector_label: confidence threshold of detector of origin label
    
    share_states_reader: instance of Share from other process storing states of reader of mower id including
      is_stopped (bool): if true read_from_video(...) called to stop
      if false read_from_video(...) allowed to run uninterruptedly
      set to false automatically upon call of read_from_video(...)
      
      id_mower (str): mower id read from image of origin label
      if empty read_from_video(...) called to read mower id from image of origin label
      if populated consider mower id read already
      
      filepath_label (str, pathlib.Path): path to image capturing origin label saved for record
      if evaluated to false read_from_video(...) called to save image capturing origin label for record
      if evaluated to true consider image capturing origin label saved already
    shareable with process calling read_from_video(...) for coordination

    is_on_gpu_mps_label: put detector of origin label on gpu device from apple via api metal performance shaders if possible or not
    """
    
    if not isinstance(filepath_detector_label, Path):
      filepath_detector_label = Path(filepath_detector_label)

    # states of reader of mower id shareable with process calling read_from_video(...) for coordination
    if share_states_reader is None:
      self.share_states_reader = Share.remote(
        is_stopped=True,  # if true read_from_video(...) called to stop; if false read_from_video(...) allowed to run uninterruptedly; set to false automatically upon call of read_from_video(...)
        id_mower="",  # mower id read from image of origin label; if empty read_from_video(...) called to read mower id from image of origin label; if populated consider mower id read already
        filepath_label="",  # path to image capturing origin label saved for record; if evaluated to false read_from_video(...) called to save image capturing origin label for record; if evaluated to true consider image capturing origin label saved already
      )
    else:
      self.share_states_reader = share_states_reader
    
    # download model detector of origin label if not existent
    if not filepath_detector_label.is_file():
      torch.hub.download_url_to_file(url=url_detector_label, dst=filepath_detector_label)
    
    # load detector of origin label
    self.detector_label = torch.hub.load(repo_or_dir=f"{torch.hub.get_dir()}/ultralytics_yolov5_master", source="local", model="custom", path=filepath_detector_label)  # offline using local repository yolov5, local label_origin.pt; path to local repository yolov5 must be string
    # self.detector_label = torch.hub.load(repo_or_dir="ultralytics/yolov5", model="custom", path=url_detector_label)  # online using remote repository yolov5, remote label_origin.pt
    self.detector_label.max_det = number_detections_max_label  # limit number of detections of origin label per image to specified maximum
    self.detector_label.conf = confidence_detector_label  # confidence threshold
    
    self.reader_text = easyocr.Reader(["en"])  # load optical character recognition model to read mower id from image of origin label

    # use gpu device if possible
    if torch.cuda.is_available():  # gpu device from nvidia available
      print(f"speed up detector of origin label, optical character recognition model to read mower id greatly using gpu device via api compute unified device architecture {torch.cuda.get_device_properties(0).name}")
    else:  # not available gpu device from nvidia
      # caution: operations on images not implemented in api metal performance shaders of torch 1.13.1 stable
      if is_on_gpu_mps_label and torch.backends.mps.is_available() and torch.backends.mps.is_built():  # called to use gpu device from apple if available, enabled with current torch, macos
        device_mps = torch.device("mps")  # gpu device from apple via api metal performance shaders
      
        self.detector_label.to(device_mps)  # put detector of origin label on gpu device from apple via api metal performance shaders

        print(f"speed up detector of origin label fairly using gpu device via api metal performance shaders")
      else:
        print("no speedup of detector of origin label, optical character recognition model to read mower id due to no gpu device available")
  
  def read_from_video(self, source_label, rotations=(15, -15), pattern_prefix=re.compile(r"[A-Z]{4}"), pattern_numeral=re.compile(r"\d{7,}"), area_different_min=3000, patience_id_mower=9, dirpath_label=Path("records/label_origin"),
                      x_text_init=0, y_text_init=30, offset_y_text=30, fontface_text=cv2.FONT_HERSHEY_SIMPLEX, fontscale_text=1, thickness_text=2,
                      format_datetime="%Y-%m-%d %H:%M:%S %Z", name_timezone="Australia/Melbourne",
                      name_window_label="protege reading mower id from image of origin label", scale_width_label=1, scale_height_label=1, x_window_label=0, y_window_label=0):
    """
    stream video capturing origin label
    detect, extract image of origin label from frame of video
    read mower id from image of origin label, e.g. mower id MAYU-2023402
    save image capturing origin label for record
    reliant on torch, easyocr thus faster on gpu device via api compute unified device architecture greatly or api metal performance shaders fairly
    
    source_label: source of video capturing origin label in form of address of ip camera or index of device camera or path to video file
    passed to argument source of VideoCaptureBufferless(...)
    
    rotations: rotate image of origin label by specified angles in degrees to find best reading of mower id
    passed to argument rotations of read(...)
    
    pattern_prefix: regular expression pattern of prefix of mower id consisting of letters, e.g. prefix MAYU
    passed to argument pattern_prefix of read(...)
    
    pattern_numeral: regular expression pattern of numeral of mower id consisting of digits, e.g. numeral 2023402
    passed to argument pattern_numeral of read(...)
    
    area_different_min: minimum different area bewteen 2 images to be considered motion or change
    passed to argument area_different_min of detect_motion(...)
    
    patience_id_mower: maximum number of attempts at reading mower id from image of origin label, 1 attempt per frame, before raising warning mower id not readable
    
    dirpath_label: path to directory to save image capturing origin label for record to
    
    parameters to render texts written on images:
      x_text_init, y_text_init: position to write first line of text on image in
      
      offset_y_text: vertical move in pixels to write next line of text on image
      
      fontface_text: font or font type of text
      passed to argument fontFace of cv2.putText(...)
      
      fontscale_text: factor to scale size of text from default size of given font
      passed to argument fontScale of cv2.putText(...)
      
      thickness_text: thickness of strokes making up characters in text
      passed to argument thickness of cv2.putText(...)
    
    format_datetime: format of datetime of image capture
    passed to argument format of datetime_capture.strftime(...)
    
    name_timezone: name of timezone to get datetime of image capture in
    passed to argument zone of pytz.timezone(...)
    
    name_window_label: name of window to display video capturing origin label in
    
    scale_width_label: scale width of image capturing origin label by specified factor for display
    
    scale_height_label: scale height of image capturing origin label by specified factor for display
    
    x_window_label, y_window_label: position of window displaying video capturing origin label
    """
    
    assert isinstance(patience_id_mower, int) and patience_id_mower > 0, "argument patience_id_mower must be positive integer"
    
    if not isinstance(dirpath_label, Path):
      dirpath_label = Path(dirpath_label)
    
    capture_label = VideoCaptureBufferless(source=source_label).start()  # open source of video capturing origin label
    
    self.share_states_reader.set.remote("is_stopped", False)  # to allow read_from_video(...) to run uninterruptedly
    
    # states of reader of mower id local
    frames_latest = deque(maxlen=2)  # last 2 frames
    patience_id_mower_remain = patience_id_mower  # remaining number of attempts at reading mower id from image of origin label, 1 attempt per frame, before raising warning mower id not readable
    
    dirpath_label.mkdir(parents=True, exist_ok=True)  # create directory to save image capturing origin label for record to if not existent
    
    # prepare window to display video capturing origin label in
    cv2.namedWindow(winname=name_window_label)  # create window
    cv2.moveWindow(winname=name_window_label, x=x_window_label, y=y_window_label)  # position window
    cv2.setWindowProperty(winname=name_window_label, prop_id=cv2.WND_PROP_TOPMOST, prop_value=1)  # send window to top of other windows
    
    # stream video capturing origin label
    try:
      while True:
        if ray.get(self.share_states_reader.get.remote("is_stopped")):  # read_from_video(...) called to stop
          break

        image_label = capture_label.read()  # image in bgr; convert image to rgb using image[:, :, ::-1] to suit yolov5 detectors

        datetime_capture = datetime.now(pytz.timezone(name_timezone))  # datetime of image capture in specified timezone
        
        x_text, y_text = x_text_init, y_text_init  # position to write line of text on image in
        
        frames_latest.append(image_label)  # update last 2 frames

        # detect origin label
        results_label = self.detector_label(image_label[:, :, ::-1])  # convert image from bgr to rgb to suit yolov5 detector
        detections_label = results_label.xyxy[0]  # detections of origin label for given single image

        # check for origin label in frame
        if not len(detections_label):  # no origin label detected
          # reset some of states of reader of mower id
          if patience_id_mower_remain != patience_id_mower:
            patience_id_mower_remain = patience_id_mower
          if ray.get(self.share_states_reader.get.remote("id_mower")):
            self.share_states_reader.set.remote("id_mower", "")
          if ray.get(self.share_states_reader.get.remote("filepath_label")):
            self.share_states_reader.set.remote("filepath_label", "")
          
          # write message no origin label on image
          text = "no origin label detected"
          cv2.putText(image_label, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
          y_text += offset_y_text  # move to next line of text
        else:  # origin label in frame
          if ray.get(self.share_states_reader.get.remote("filepath_label")):  # image capturing origin label saved already
            if ray.get(self.share_states_reader.get.remote("id_mower")):  # mower id readable fully or partly
              # write message read mower id already on image
              text = f"read mower id as {ray.get(self.share_states_reader.get.remote('id_mower'))} already"
              cv2.putText(image_label, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
              y_text += offset_y_text  # move to next line of text
            
            # write message image capturing origin label saved already on image
            text = "image capturing origin label saved already"
            cv2.putText(image_label, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 255, 0))  # message in green
            y_text += offset_y_text  # move to next line of text
          else:  # no image capturing origin label saved yet
            if detect_motion(*frames_latest, area_different_min=area_different_min):  # motion in frame
              # write caution motion on image
              text = "caution: stop motion to read mower id"
              cv2.putText(image_label, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 192, 255))  # message in amber
              y_text += offset_y_text  # move to next line of text
            else:  # no motion detected
              image_cropped_label = results_label.crop(save=False)[0]["im"]  # extract image of origin label

              id_mower = self.read(image_cropped_label, rotations=rotations, pattern_prefix=pattern_prefix, pattern_numeral=pattern_numeral)  # read mower id from image of origin label
              
              patience_id_mower_remain -= 1  # deduct 1 attempt at reading mower id from image of origin label

              # check validity of mower id
              if id_mower:  # readable fully or partly
                # write message read mower id on image
                text = f"read mower id as {id_mower}"
                cv2.putText(image_label, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 255, 0))  # message in green
                y_text += offset_y_text  # move to next line of text

                # write datetime of image capture on image
                text = datetime_capture.strftime(format_datetime)  # datetime of image capture in proper format
                cv2.putText(image_label, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
                y_text += offset_y_text  # move to next line of text

                # save image capturing origin label for record
                filepath_label = Path(dirpath_label, f"mower id {id_mower} {datetime_capture.strftime(format_datetime)}.jpg")  # name image capturing origin label after mower id, datetime of image capture in proper format
                cv2.imwrite(filepath_label.as_posix(), image_label)  # save path of image capturing origin label must be string
                
                self.share_states_reader.set.remote("id_mower", id_mower)  # update mower id shareable with process calling read_from_video(...)
                self.share_states_reader.set.remote("filepath_label", filepath_label)  # update path to image capturing origin label shareable with process calling read_from_video(...)
              else:  # not readable at all
                if patience_id_mower_remain:  # attempt(s) at reading mower id from image of origin label allowed still
                  # write message remaining number of attempts to read mower id from image of origin label on image
                  text = f"{patience_id_mower_remain} attempt{'' if patience_id_mower_remain == 1 else 's'} remaining to read mower id from image of origin label"
                  cv2.putText(image_label, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
                  y_text += offset_y_text  # move to next line of text
                else:  # no more attempts at reading mower id from image of origin label allowed
                  # write datetime of image capture on image
                  text = datetime_capture.strftime(format_datetime)  # datetime of image capture in proper format
                  cv2.putText(image_label, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
                  y_text += offset_y_text  # move to next line of text

                  # save image capturing origin label for record
                  filepath_label = Path(dirpath_label, f"mower id na {datetime_capture.strftime(format_datetime)}.jpg")  # name image capturing origin label after mower id, datetime of image capture in proper format
                  cv2.imwrite(filepath_label.as_posix(), image_label)  # save path of image capturing origin label must be string
                  
                  self.share_states_reader.set.remote("filepath_label", filepath_label)  # update path to image capturing origin label shareable with process calling read_from_video(...)
        
        image_resized = image_label if scale_width_label == scale_height_label == 1 else cv2.resize(image_label, dsize=(0, 0), fx=scale_width_label, fy=scale_height_label)  # resize image capturing origin label for display
        
        cv2.imshow(name_window_label, image_resized)  # display image capturing origin label in specified window
        cv2.waitKey(1)  # wait 1 ms for image to show unless key pressed
    except (KeyboardInterrupt, Exception):
      self.share_states_reader.set.remote("is_stopped", True)  # to reflect read_from_video(...) stopped due to interruption
      
      # stop video capturing origin label
      capture_label.stop()
      cv2.destroyAllWindows()
      cv2.waitKey(1)  # to fix bug on macos window displaying video not closed
      
      print(traceback.format_exc())

    # stop video capturing origin label
    capture_label.stop()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # to fix bug on macos window displaying video not closed
  
  def get(self, name):
    """
    get value of attribute
    
    name: name of attribute
    """
    
    return getattr(self, name)
  
  def set(self, name, value):
    """
    set value of attribute to specified value
    
    name: name of attribute
    
    value: specified value to assign to attribute
    """
    
    setattr(self, name, value)
  
  def read(self, image_cropped_label, rotations=(15, -15), pattern_prefix=re.compile(r"[A-Z]{4}"), pattern_numeral=re.compile(r"\d{7,}")):
    """
    read mower id from image of origin label, e.g. MAYU-2023402
    reliant on torch due to easyocr thus faster on gpu device via api compute unified device architecture greatly or api metal performance shaders fairly
    
    image_cropped_label (numpy.array): image of origin label
    
    rotations: rotate image of origin label by specified angles in degrees to find best reading of mower id
    
    pattern_prefix: pattern object of prefix of mower id consisting of letters, e.g. prefix MAYU
    
    pattern_numeral: pattern object of numeral of mower id consisting of digits, e.g. numeral 2023402
    """
    
    prefix = ""  # prefix of mower id consisting of letters, e.g. MAYU
    numeral = ""  # numeral of mower id consisting of digits, e.g. 2023402

    images_label_checkable = itertools.chain([image_cropped_label], (ndimage.rotate(image_cropped_label, angle) for angle in rotations))  # checkable images of origin label including image of origin label as given and rotated by specified angles all evaluated lazily

    # find best reading of mower id from checkable images of origin label
    for image in images_label_checkable:
      # read text from image of origin label
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
  
  def detect_extract_label(self, image_label):
    """
    detect, extract image of origin label from image capturing origin label
    reliant on torch thus faster on gpu device via api compute unified device architecture greatly or api metal performance shaders fairly
    intended for testing purposes
    note: not able to pass object of results of detector_label(...) in process 2 to other process calling detect_extract_label(...) due to location of yolov5 model to load object of results in process 2 not appropriate for other process
    
    image_label: source of image capturing origin label in form of filepath or url or numpy.array or other forms suitable for yolov5 detector
    passed to first argument of detector_label(...)
    see details at https://github.com/ultralytics/yolov5/blob/30e4c4f09297b67afedf8b2bcd851833ddc9dead/models/common.py#L243-L252
    """

    # detect origin label
    results_label = self.detector_label(image_label)
    detections_label = results_label.xyxy[0]  # detections of origin label for given single image

    classnames_label = results_label.names  # name of origin label detectable by detector of origin label

    image_label_bbox = results_label.render()[0]  # render bounding box around origin label in rgb for given single image

    crops_label = results_label.crop(save=False)  # extract images of origin label
    
    return detections_label, classnames_label, image_label_bbox, crops_label


def detect_motion(image1=None, image2=None, area_different_min=1500):
  """
  detect motion or change between 2 images
  
  area_different_min: minimum different area bewteen 2 images to be considered motion or change
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
  
  return [contr for contr in contours if cv2.contourArea(contr) >= area_different_min]

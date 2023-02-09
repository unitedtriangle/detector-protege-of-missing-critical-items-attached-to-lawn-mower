"""
detect, recognize things in images
"""


import torch  # faster on gpu device with cuda support greatly or mps support fairly
import easyocr  # reliant on torch thus faster on gpu device with cuda support greatly or mps support fairly
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
  def __init__(self):
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
    reliant on torch, ReaderIdMower thus faster on gpu device with cuda support greatly or mps support fairly
    """
    
    # load detector of engine
    try:
      self.detector_engine = torch.hub.load(repo_or_dir=f"{Path.home()}/.cache/torch/hub/ultralytics_yolov5_master", source="local", model="custom", path="detectors/engine.pt")  # offline using local repository yolov5, local engine.pt; path to local repository yolov5 must be string
      
      print("loaded detector of engine offline using local repository yolov5, local engine.pt")
    except Exception:
      self.detector_engine = torch.hub.load(repo_or_dir="ultralytics/yolov5", model="custom", path="https://github.com/unitedtriangle/detector-protege-of-missing-critical-items-attached-to-lawn-mower/raw/main/detectors/engine.pt")  # online using remote repository yolov5, engine.pt from remote repository protege
      
      print("not able to load detector of engine offline using local repository yolov5, local engine.pt\nloaded detector of engine online using remote repository yolov5, engine.pt from remote repository protege")
    self.detector_engine.max_det = 1  # maximum 1 detection of engine per image
    self.detector_engine.conf = 0.6  # confidence threshold
    
    # load detector of critical items
    try:
      self.detector_items = torch.hub.load(repo_or_dir=f"{Path.home()}/.cache/torch/hub/ultralytics_yolov5_master", source="local", model="custom", path="detectors/items_critical.pt")  # offline using local repository yolov5, local items_critical.pt; path to local repository yolov5 must be string
      
      print("loaded detector of critical items offline using local repository yolov5, local items_critical.pt")
    except Exception:
      self.detector_items = torch.hub.load(repo_or_dir="ultralytics/yolov5", model="custom", path="https://github.com/unitedtriangle/detector-protege-of-missing-critical-items-attached-to-lawn-mower/raw/main/detectors/items_critical.pt")  # online using remote repository yolov5, items_critical.pt from remote repository protege
      
      print("not able to load detector of critical items offline using local repository yolov5, local items_critical.pt\nloaded detector of critical items online using remote repository yolov5, items_critical.pt from remote repository protege")
    self.detector_items.conf = 0.6  # confidence threshold
    
    self.reader_id_mower = ReaderIdMower.remote()  # load reader of mower id in process 2
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
  
  def detect_from_video(self, source_items, source_label, rotations=(15, -15), pattern_prefix=re.compile(r"[A-Z]{4}"), pattern_numeral=re.compile(r"\d{7,}"), min_different_area=3000, patience_items=9, patience_id_mower=3, dirpath_records_master=Path("records"), dirpath_temporary=Path("records/temporary"),
                         x_text_init=0, y_text_init=30, offset_y_text=30, fontface_text=cv2.FONT_HERSHEY_SIMPLEX, fontscale_text=1, thickness_text=2,
                         format_datetime="%Y-%m-%d %H:%M:%S %Z", format_date="%Y-%m-%d", name_timezone="Australia/Melbourne",
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
    reliant on torch, ReaderIdMower thus faster on gpu device with cuda support greatly or mps support fairly
    
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
    
    min_different_area: minimum different area bewteen 2 images to be considered motion or change
    passed to argument min_different_area of reader_id_mower.read_from_video.remote(...)
    
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
    
    format_datetime: format of datetime of image capture
    passed to argument format of datetime_capture.strftime(...)
    
    format_date: format of date of image capture
    passed to argument format of datetime_capture.strftime(...)
    
    name_timezone: name of timezone to get datetime of image capture in
    passed to argument zone of pytz.timezone(...)
    
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
    
    self.reader_id_mower.read_from_video.remote(source_label=source_label, rotations=rotations, pattern_prefix=pattern_prefix, pattern_numeral=pattern_numeral, min_different_area=min_different_area, patience_id_mower=patience_id_mower, dirpath_label=dirpath_temporary,
                                                 x_text_init=x_text_init, y_text_init=y_text_init, offset_y_text=offset_y_text, fontface_text=fontface_text, fontscale_text=fontscale_text, thickness_text=thickness_text,
                                                 format_datetime=format_datetime, name_timezone=name_timezone,
                                                 name_window_label=name_window_label, scale_width_label=scale_width_label, scale_height_label=scale_height_label, x_window_label=x_window_label, y_window_label=y_window_label)  # stream video capturing origin label; detect, extract image of origin label from frame of video; read mower id from image of origin label, e.g. mower id MAYU-2023402; save image capturing origin label for record
    
    # prepare window to display video capturing critical items in
    cv2.namedWindow(winname=name_window_items)  # create window
    cv2.moveWindow(winname=name_window_items, x=x_window_items, y=y_window_items)  # position window
    cv2.setWindowProperty(winname=name_window_items, prop_id=cv2.WND_PROP_TOPMOST, prop_value=1)  # send window to top of other windows
    
    # stream video capturing critical items
    try:
      while True:
        image = capture_items.read()  # image in bgr; convert image to rgb using image[:, :, ::-1] to suit yolov5 detectors

        datetime_capture = datetime.now(pytz.timezone(name_timezone))  # datetime of image capture in specified timezone

        x_text, y_text = x_text_init, y_text_init  # position to write line of text on image in

        # detect engine
        results_engine = self.detector_engine(image[:, :, ::-1])  # convert image from bgr to rgb to suit yolov5 detector
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
            
            # write message ready for mower on image
            text = "ready for mower"
            cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
            y_text += offset_y_text  # move to next line of text
          else:  # mower in frame
            if dirpath_record:  # directory of record existent storing image capturing critical items, image capturing origin label already; same mower in frame
              # write message found all required critical items already on image
              text = f"found all required critical items including {', '.join(items_checklist)} already"
              cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 255, 0))  # message in green
              y_text += offset_y_text  # move to next line of text
              
              if id_mower:  # mower id readable fully or partly
                # write message read mower id already on image
                text = f"read mower id as {id_mower} already"
                cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 255, 0))  # message in green
                y_text += offset_y_text  # move to next line of text
              else:  # not readable at all mower id
                # write message not readable mower id already on image
                text = "not able to read mower id already"
                cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
                y_text += offset_y_text  # move to next line of text
              
              # write advice go ahead already on image
              text = "mower ok to go already"
              cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 255, 0))  # message in green
              y_text += offset_y_text  # move to next line of text
            else:  # no image capturing critical items, image capturing origin label stored in directory of record yet; new mower in frame
              # write message new mower in frame on image
              text = "new mower in view"
              cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
              y_text += offset_y_text  # move to next line of text
              
              # attempt to detect critical items, retrieve path to image capturing origin label saved for record
              
              # no image capturing critical items saved temporarily for record
              
              items_missing.update(items_checklist)  # set missing critical items to all required critical items if not equal already at start of check for missing critical items
              
              # detect critical items
              results_items = self.detector_items(image[:, :, ::-1])  # convert image from bgr to rgb to suit yolov5 detector
              detections_items = results_items.xyxy[0]  # detections of critical items for given single image

              # check for missing critical items
              items_missing.difference_update(results_items.names[det[5].item()] for det in detections_items)
              if not items_missing:  # found all required critical items
                # write message found all required critical items on image
                text = f"found all required critical items including {', '.join(items_checklist)}"
                cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 255, 0))  # message in green
                y_text += offset_y_text  # move to next line of text

                # write datetime of image capture on image
                text = datetime_capture.strftime(format_datetime)  # datetime of image capture in proper format
                cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
                y_text += offset_y_text  # move to next line of text

                # save image capturing critical items temporarily for record
                filepath_items_temporary = Path(dirpath_temporary, f"critical items {datetime_capture.strftime(format_datetime)}.jpg")  # name image capturing critical items after datetime of image capture in proper format to save temporarily for record
                cv2.imwrite(filepath_items_temporary.as_posix(), image)  # save path of image capturing critical items must be string

              # no image capturing origin label saved temporarily for record
              
              filepath_label_temporary = ray.get(self.share_states_reader.get.remote("filepath_label"))  # get path to image capturing origin label saved for record retrieved from instance of Share from other process storing states of reader of mower id
              
              id_mower = ray.get(self.share_states_reader.get.remote("id_mower"))  # get mower id read from image of origin label retrieved from instance of Share from other process storing states of reader of mower id
                
              patience_items_remain -= 1  # deduct 1 attempt at detecting critical items, retrieving path to image capturing origin label saved for record
              
              # check image capturing critical items, image capturing origin label
              if filepath_items_temporary and filepath_label_temporary:  # saved temporarily for record
                # move image capturing critical items, image capturing origin label to directory of record for permanent storage
                dirpath_record = Path(dirpath_records_master, datetime_capture.strftime(format_date), filepath_label_temporary.stem)  # name directory of record after date of image capture in proper format, name of image capturing origin label saved temporarily for record
                dirpath_record.mkdir(parents=True, exist_ok=True)  # create directory of record
                filepath_items_temporary.replace(Path(dirpath_record, filepath_items_temporary.name))  # move image capturing critical items to directory of record keeping same name
                filepath_label_temporary.replace(Path(dirpath_record, filepath_label_temporary.name))  # move image capturing origin label to directory of record keeping same name
                
                is_on_light_green = True  # turn on light indicator green

                channel_go = self.sound_go.play()  # play sound go
                
                if id_mower:  # mower id readable fully or partly
                  # write message read mower id on image
                  text = f"read mower id as {id_mower}"
                  cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 255, 0))  # message in green
                  y_text += offset_y_text  # move to next line of text
                else:  # not readable at all mower id
                  # write message not readable mower id on image
                  text = "not able to read mower id"
                  cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
                  y_text += offset_y_text  # move to next line of text
                
                # write advice go ahead on image
                text = "mower ok to go"
                cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 255, 0))  # message in green
                y_text += offset_y_text  # move to next line of text
              else:  # not saved temporarily for record
                is_ok_mower = False  # set status of mower to not ok
                
                if patience_items_remain:  # attempt(s) at detecting critical items, retrieving path to image capturing origin label saved for record allowed still
                  if not filepath_items_temporary:  # not saved temporarily for record image capturing critical items
                    # write message remaining number of attempts to detect critical items on image
                    text = f"not able to detect {', '.join(items_missing)}. {patience_items_remain} attempt{'' if patience_items_remain == 1 else 's'} remaining"
                    cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
                    y_text += offset_y_text  # move to next line of text
                  
                  if not filepath_label_temporary:  # not saved temporarily for record image capturing origin label
                    # write message remaining number of attempts to retrieve path to image capturing origin label saved for record on image
                    text = f"no image capturing origin label saved for record. {patience_items_remain} attempt{'' if patience_items_remain == 1 else 's'} remaining"
                    cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
                    y_text += offset_y_text  # move to next line of text
                else:  # no more attempts at detecting critical items, retrieving path to image capturing origin label saved for record allowed
                  is_on_light_red = True  # turn on light indicator red

                  channel_warning = self.sound_warning.play(loops=-1)  # play sound warning on repeat
                  
                  if not filepath_items_temporary:  # not saved temporarily for record image capturing critical items
                    # write message warning missing critical items on image
                    text = f"warning: rectify missing critical item{'' if len(items_missing) == 1 else 's'} {', '.join(items_missing)}"
                    cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 0, 255))  # message in red
                    y_text += offset_y_text  # move to next line of text
                  
                  if not filepath_label_temporary:  # not saved temporarily for record image capturing origin label
                    # write message warning missing origin label on image
                    text = "warning: rectify missing origin label"
                    cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 0, 255))  # message in red
                    y_text += offset_y_text  # move to next line of text
        else:  # not ok mower checked
          if not len(detections_engine):  # no mower detected
            # turn on light indicator red if not on yet
            if not is_on_light_red:
              is_on_light_red = True

            # play sound warning on repeat if not playing yet
            if channel_warning is None:
              channel_warning = self.sound_warning.play(loops=-1)

            if not filepath_items_temporary:  # not saved temporarily for record image capturing critical items
              # write message warning missing critical items on image
              text = f"warning: put mower back in view to rectify missing critical item{'' if len(items_missing) == 1 else 's'} {', '.join(items_missing)}"
              cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 0, 255))  # message in red
              y_text += offset_y_text  # move to next line of text

            if not filepath_label_temporary:  # not saved temporarily for record image capturing origin label
              # write message warning missing origin label on image
              text = "warning: put mower back in view to rectify missing origin label"
              cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 0, 255))  # message in red
              y_text += offset_y_text  # move to next line of text
          else:  # mower in frame
            # write message mower in frame on image
            text = "mower in view"
            cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
            y_text += offset_y_text  # move to next line of text
            
            if patience_items_remain:  # attempt(s) at detecting critical items, retrieving path to image capturing origin label saved for record allowed still
              # turn off light indicator red if on
              if is_on_light_red:
                is_on_light_red = False

              # stop sound warning if playing
              if channel_warning is not None:
                channel_warning = channel_warning.stop()  # set channel warning to none after stop of sound warning

            # attempt to detect critical items, retrieve path to image capturing origin label saved for record
            
            if not filepath_items_temporary:  # no image capturing critical items saved temporarily for record
              # set missing critical items to all required critical items if not equal already at start of check for missing critical items
              if len(items_missing) != len(items_checklist):  # not equal missing critical items, all required critical items
                items_missing.update(items_checklist)

              # detect critical items
              results_items = self.detector_items(image[:, :, ::-1])  # convert image from bgr to rgb to suit yolov5 detector
              detections_items = results_items.xyxy[0]  # detections of critical items for given single image

              # check for missing critical items
              items_missing.difference_update(results_items.names[det[5].item()] for det in detections_items)
              if not items_missing:  # found all required critical items
                # write message found all required critical items on image
                text = f"found all required critical items including {', '.join(items_checklist)}"
                cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 255, 0))  # message in green
                y_text += offset_y_text  # move to next line of text

                # write datetime of image capture on image
                text = datetime_capture.strftime(format_datetime)  # datetime of image capture in proper format
                cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
                y_text += offset_y_text  # move to next line of text

                # save image capturing critical items temporarily for record
                filepath_items_temporary = Path(dirpath_temporary, f"critical items {datetime_capture.strftime(format_datetime)}.jpg")  # name image capturing critical items after datetime of image capture in proper format to save temporarily for record
                cv2.imwrite(filepath_items_temporary.as_posix(), image)  # save path of image capturing critical items must be string

            if not filepath_label_temporary:  # no image capturing origin label saved temporarily for record
              filepath_label_temporary = ray.get(self.share_states_reader.get.remote("filepath_label"))  # get path to image capturing origin label saved for record retrieved from instance of Share from other process storing states of reader of mower id
              
              id_mower = ray.get(self.share_states_reader.get.remote("id_mower"))  # get mower id read from image of origin label retrieved from instance of Share from other process storing states of reader of mower id
            
            # deduct 1 attempt at detecting critical items, retrieving path to image capturing origin label saved for record but not allow patience_items_remain negative
            if patience_items_remain:  # attempt(s) at detecting critical items, retrieving path to image capturing origin label saved for record allowed still
              patience_items_remain -= 1

            # check image capturing critical items, image capturing origin label
            if filepath_items_temporary and filepath_label_temporary:  # saved temporarily for record
              is_ok_mower = True  # set status of mower to ok
              
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

              # write message found all required critical items on image
              text = f"found all required critical items including {', '.join(items_checklist)}"
              cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 255, 0))  # message in green
              y_text += offset_y_text  # move to next line of text

              if id_mower:  # mower id readable fully or partly
                # write message read mower id on image
                text = f"read mower id as {id_mower}"
                cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 255, 0))  # message in green
                y_text += offset_y_text  # move to next line of text
              else:  # not readable at all mower id
                # write message not readable mower id on image
                text = "not able to read mower id"
                cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
                y_text += offset_y_text  # move to next line of text

              # write advice go ahead on image
              text = "mower ok to go"
              cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 255, 0))  # message in green
              y_text += offset_y_text  # move to next line of text
            else:  # not saved temporarily for record
              if patience_items_remain:  # attempt(s) at detecting critical items, retrieving path to image capturing origin label saved for record allowed still
                if not filepath_items_temporary:  # not saved temporarily for record image capturing critical items
                  # write message remaining number of attempts to detect critical items on image
                  text = f"not able to detect {', '.join(items_missing)}. {patience_items_remain} attempt{'' if patience_items_remain == 1 else 's'} remaining"
                  cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
                  y_text += offset_y_text  # move to next line of text

                if not filepath_label_temporary:  # not saved temporarily for record image capturing origin label
                  # write message remaining number of attempts to retrieve path to image capturing origin label saved for record on image
                  text = f"no image capturing origin label saved for record. {patience_items_remain} attempt{'' if patience_items_remain == 1 else 's'} remaining"
                  cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
                  y_text += offset_y_text  # move to next line of text
              else:  # no more attempts at detecting critical items, retrieving path to image capturing origin label saved for record allowed
                # turn on light indicator red if not on yet
                if not is_on_light_red:
                  is_on_light_red = True

                # play sound warning on repeat if not playing yet
                if channel_warning is None:
                  channel_warning = self.sound_warning.play(loops=-1)

                if not filepath_items_temporary:  # not saved temporarily for record image capturing critical items
                  # write message warning missing critical items on image
                  text = f"warning: rectify missing critical item{'' if len(items_missing) == 1 else 's'} {', '.join(items_missing)}"
                  cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 0, 255))  # message in red
                  y_text += offset_y_text  # move to next line of text

                if not filepath_label_temporary:  # not saved temporarily for record image capturing origin label
                  # write message warning missing origin label on image
                  text = "warning: rectify missing origin label"
                  cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 0, 255))  # message in red
                  y_text += offset_y_text  # move to next line of text
        
        # display statuses of light indicators
        if is_on_light_green:
          cv2.circle(image, center=(int(capture_items.stream.get(cv2.CAP_PROP_FRAME_WIDTH)) - 15, 15), radius=15, color=(0, 255, 0), thickness=-1)  # circle green in upper-right corner of image
        if is_on_light_red:
          cv2.circle(image, center=(int(capture_items.stream.get(cv2.CAP_PROP_FRAME_WIDTH)) - 45, 15), radius=15, color=(0, 0, 255), thickness=-1)  # circle red to left of circle green

        image_resized = image if scale_width_items == scale_height_items == 1 else cv2.resize(image, dsize=(0, 0), fx=scale_width_items, fy=scale_height_items)  # resize image for display
        
        cv2.imshow(name_window_items, image_resized)  # display image in specified window
        cv2.waitKey(1)  # wait 1 ms for image to show unless key pressed
    except (KeyboardInterrupt, Exception):
      # turn off light indicator green if on
      if is_on_light_green:
        is_on_light_green = False
      
      # turn off light indicator red if on
      if is_on_light_red:
        is_on_light_red = False
      
      mixer.stop()  # stop all sounds
      
      self.share_states_reader.set.remote("is_stopped", True)  # call for reader_id_mower.read_from_video.remote(...) to stop

      # stop video capturing critical items
      capture_items.stop()
      cv2.destroyAllWindows()
      cv2.waitKey(1)  # to fix bug on macos where window displaying video not close

      print(traceback.format_exc())


@ray.remote  # run instance of ReaderIdMower in separate process
class ReaderIdMower:
  def __init__(self, share_states_reader=None):
    """
    stream video capturing origin label
    detect, extract image of origin label from image capturing origin label of video
    read mower id from image of origin label, e.g. mower id MAYU-2023402
    save image capturing origin label for record
    reliant on torch due to easyocr thus faster on gpu device with cuda support greatly or mps support fairly
    
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
    """
    
    # states of reader of mower id shareable with process calling read_from_video(...) for coordination
    if share_states_reader is None:
      self.share_states_reader = Share.remote(
        is_stopped=True,  # if true read_from_video(...) called to stop; if false read_from_video(...) allowed to run uninterruptedly; set to false automatically upon call of read_from_video(...)
        id_mower="",  # mower id read from image of origin label; if empty read_from_video(...) called to read mower id from image of origin label; if populated consider mower id read already
        filepath_label="",  # path to image capturing origin label saved for record; if evaluated to false read_from_video(...) called to save image capturing origin label for record; if evaluated to true consider image capturing origin label saved already
      )
    else:
      self.share_states_reader = share_states_reader
    
    # load detector of origin label
    try:
      self.detector_label = torch.hub.load(repo_or_dir=f"{Path.home()}/.cache/torch/hub/ultralytics_yolov5_master", source="local", model="custom", path="detectors/label_origin.pt")  # offline using local repository yolov5, local label_origin.pt; path to local repository yolov5 must be string
      
      print("loaded detector of origin label offline using local repository yolov5, local label_origin.pt")
    except Exception:
      self.detector_label = torch.hub.load(repo_or_dir="ultralytics/yolov5", model="custom", path="https://github.com/unitedtriangle/detector-protege-of-missing-critical-items-attached-to-lawn-mower/raw/main/detectors/label_origin.pt")  # online using remote repository yolov5, label_origin.pt from remote repository protege
      
      print("not able to load detector of origin label offline using local repository yolov5, local label_origin.pt\nloaded detector of origin label online using remote repository yolov5, label_origin.pt from remote repository protege")
    self.detector_label.max_det = 1  # maximum 1 detection of origin label per image
    self.detector_label.conf = 0.6  # confidence threshold
    
    self.reader_text = easyocr.Reader(["en"])  # load optical character recognition model to read mower id from image of origin label
  
  def read_from_video(self, source_label, rotations=(15, -15), pattern_prefix=re.compile(r"[A-Z]{4}"), pattern_numeral=re.compile(r"\d{7,}"), min_different_area=3000, patience_id_mower=3, dirpath_label=Path("records/label_origin"),
                       x_text_init=0, y_text_init=30, offset_y_text=30, fontface_text=cv2.FONT_HERSHEY_SIMPLEX, fontscale_text=1, thickness_text=2,
                       format_datetime="%Y-%m-%d %H:%M:%S %Z", name_timezone="Australia/Melbourne",
                       name_window_label="protege reading mower id from image of origin label", scale_width_label=1, scale_height_label=1, x_window_label=0, y_window_label=0):
    """
    stream video capturing origin label
    detect, extract image of origin label from frame of video
    read mower id from image of origin label, e.g. mower id MAYU-2023402
    save image capturing origin label for record
    reliant on torch, easyocr thus faster on gpu device with cuda support greatly or mps support fairly
    
    source_label: source of video capturing origin label in form of address of ip camera or index of device camera or path to video file
    passed to argument source of VideoCaptureBufferless(...)
    
    rotations: rotate image of origin label by specified angles in degrees to find best reading of mower id
    passed to argument rotations of read(...)
    
    pattern_prefix: regular expression pattern of prefix of mower id consisting of letters, e.g. prefix MAYU
    passed to argument pattern_prefix of read(...)
    
    pattern_numeral: regular expression pattern of numeral of mower id consisting of digits, e.g. numeral 2023402
    passed to argument pattern_numeral of read(...)
    
    min_different_area: minimum different area bewteen 2 images to be considered motion or change
    passed to argument min_different_area of detect_motion(...)
    
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
    id_mower = ""  # mower id read from image of origin label; if empty read_from_video(...) called to read mower id from image of origin label; if populated consider mower id read already
    filepath_label = ""  # path to image capturing origin label saved for record; if evaluated to false read_from_video(...) called to save image capturing origin label for record; if evaluated to true consider image capturing origin label saved already
    
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

        image = capture_label.read()  # image in bgr; convert image to rgb using image[:, :, ::-1] to suit yolov5 detectors

        datetime_capture = datetime.now(pytz.timezone(name_timezone))  # datetime of image capture in specified timezone
        
        x_text, y_text = x_text_init, y_text_init  # position to write line of text on image in
        
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
            self.share_states_reader.set.remote("id_mower", id_mower)  # update mower id shareable with process calling read_from_video(...)
          if filepath_label:
            filepath_label = ""
            self.share_states_reader.set.remote("filepath_label", filepath_label)  # update path to image capturing origin label saved for record shareable with process calling read_from_video(...)
          
          # write message no origin label on image
          text = "no origin label detected"
          cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
          y_text += offset_y_text  # move to next line of text
        else:  # origin label in frame
          if filepath_label:  # image capturing origin label saved already
            # write message image capturing origin label saved already on image
            text = "image capturing origin label saved already"
            cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 255, 0))  # message in green
            y_text += offset_y_text  # move to next line of text

            if id_mower:  # mower id readable fully or partly
              # write message read mower id already on image
              text = f"read mower id as {id_mower} already"
              cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 255, 0))  # message in green
              y_text += offset_y_text  # move to next line of text
            else:  # not readable at all mower id
              # write message not readable mower id already on image
              text = "not able to read mower id already"
              cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
              y_text += offset_y_text  # move to next line of text
          else:  # no image capturing origin label saved yet
            if detect_motion(*frames_latest, min_different_area=min_different_area):  # motion in frame
              # write caution motion on image
              text = "caution: stop motion to read mower id"
              cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 192, 255))  # message in amber
              y_text += offset_y_text  # move to next line of text
            else:  # no motion detected
              image_label = results_label.crop(save=False)[0]["im"]  # extract image of origin label

              id_mower = self.read(image_label, rotations=rotations, pattern_prefix=pattern_prefix, pattern_numeral=pattern_numeral)  # read mower id from image of origin label
              
              patience_id_mower_remain -= 1  # deduct 1 attempt at reading mower id from image of origin label

              # check validity of mower id
              if id_mower:  # readable fully or partly
                # write message read mower id on image
                text = f"read mower id as {id_mower}"
                cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(0, 255, 0))  # message in green
                y_text += offset_y_text  # move to next line of text

                # write datetime of image capture on image
                text = datetime_capture.strftime(format_datetime)  # datetime of image capture in proper format
                cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
                y_text += offset_y_text  # move to next line of text

                # save image capturing origin label for record
                filepath_label = Path(dirpath_label, f"mower id {id_mower} {datetime_capture.strftime(format_datetime)}.jpg")  # name image capturing origin label after mower id, datetime of image capture in proper format
                cv2.imwrite(filepath_label.as_posix(), image)  # save path of image capturing origin label must be string
                
                self.share_states_reader.set.remote("id_mower", id_mower)  # update mower id shareable with process calling read_from_video(...)
                self.share_states_reader.set.remote("filepath_label", filepath_label)  # update path to image capturing origin label shareable with process calling read_from_video(...)
              else:  # not readable at all
                if patience_id_mower_remain:  # attempt(s) at reading mower id from image of origin label allowed still
                  
                  # write message remaining number of attempts to read mower id from image of origin label on image
                  text = f"not able to read mower id from image of origin label. {patience_id_mower_remain} attempt{'' if patience_id_mower_remain == 1 else 's'} remaining"
                  cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
                  y_text += offset_y_text  # move to next line of text
                else:  # no more attempts at reading mower id from image of origin label allowed
                  # write message not readable mower id on image
                  text = "not able to read mower id"
                  cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
                  y_text += offset_y_text  # move to next line of text

                  # write datetime of image capture on image
                  text = datetime_capture.strftime(format_datetime)  # datetime of image capture in proper format
                  cv2.putText(image, text=text, org=(x_text, y_text), fontFace=fontface_text, fontScale=fontscale_text, thickness=thickness_text, color=(255, 255, 255))  # message in white
                  y_text += offset_y_text  # move to next line of text

                  # save image capturing origin label for record
                  filepath_label = Path(dirpath_label, f"mower id na {datetime_capture.strftime(format_datetime)}.jpg")  # name image capturing origin label after mower id, datetime of image capture in proper format
                  cv2.imwrite(filepath_label.as_posix(), image)  # save path of image capturing origin label must be string
                  
                  self.share_states_reader.set.remote("filepath_label", filepath_label)  # update path to image capturing origin label shareable with process calling read_from_video(...)
        
        image_resized = image if scale_width_label == scale_height_label == 1 else cv2.resize(image, dsize=(0, 0), fx=scale_width_label, fy=scale_height_label)  # resize image for display
        
        cv2.imshow(name_window_label, image_resized)  # display image in specified window
        cv2.waitKey(1)  # wait 1 ms for image to show unless key pressed
    except (KeyboardInterrupt, Exception):
      self.share_states_reader.set.remote("is_stopped", True)  # to reflect read_from_video(...) stopped due to interruption
      
      # stop video capturing origin label
      capture_label.stop()
      cv2.destroyAllWindows()
      cv2.waitKey(1)  # to fix bug on macos where window displaying video not close
      
      print(traceback.format_exc())

    # stop video capturing origin label
    capture_label.stop()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # to fix bug on macos where window displaying video not close
  
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
  
  def read(self, image_label, rotations=(15, -15), pattern_prefix=re.compile(r"[A-Z]{4}"), pattern_numeral=re.compile(r"\d{7,}")):
    """
    read mower id from image of origin label, e.g. MAYU-2023402
    reliant on torch due to easyocr thus faster on gpu device with cuda support greatly or mps support fairly
    
    image_label (numpy.array): image of origin label
    
    rotations: rotate image of origin label by specified angles in degrees to find best reading of mower id
    
    pattern_prefix: pattern object of prefix of mower id consisting of letters, e.g. prefix MAYU
    
    pattern_numeral: pattern object of numeral of mower id consisting of digits, e.g. numeral 2023402
    """
    
    prefix = ""  # prefix of mower id consisting of letters, e.g. MAYU
    numeral = ""  # numeral of mower id consisting of digits, e.g. 2023402

    images_label_checkable = itertools.chain([image_label], (ndimage.rotate(image_label, angle) for angle in rotations))  # checkable images of origin label including image of origin label as given and rotated by specified angles all evaluated lazily

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
  
  def detect_extract_label(self, image):
    """
    detect, extract image of origin label from image; intended for testing purposes
    
    image: source of image in form of filepath or url or numpy.array or other forms suitable for yolov5 detector
    passed to first argument of detector_label(...)
    """
    
    results_label = self.detector_label(image)  # detect origin label
    
    image_bbox_label = results_label.render()[0]  # render bounding box around origin label
    
    image_label = results_label.crop(save=False)[0]["im"]  # extract image of origin label
    
    return image_bbox_label, image_label


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

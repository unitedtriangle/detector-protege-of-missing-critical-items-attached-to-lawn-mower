"""
stream, process videos
"""


import cv2
from threading import Thread
from collections import deque


class VideoCaptureBufferless:
  """
  image processing speed lower than capturing speed of IP camera results in frames accumulated in buffer. Since capture.set(cv2.CAP_PROP_BUFFERSIZE) not work, run separate thread to flush buffer to make sure latest frame always read.
  implement based on answer https://stackoverflow.com/questions/43665208/how-to-get-the-latest-frame-from-capture-device-camera-in-opencv/63057626#63057626 and implementation https://github.com/PyImageSearch/imutils/blob/9f740a53bcc2ed7eba2558afed8b4c17fd8a1d4c/imutils/video/filevideostream.py#L16
  """
  
  def __init__(self, source):
    """
    read frames from source without buffer to reduce latency
    source: source of video in form of ip camera address or index of device camera or path to video file; passed to first argument of cv2.VideoCapture(...)
    """
    
    self.stream = cv2.VideoCapture(source)
    self.frames_latest = deque(maxlen=1)  # latest frame
    self.is_stopped = False
    self.thread = Thread(target=self._grab_frames_latest, daemon=True)
  
  def start(self):
    self.thread.start()  # start thread to continuously read and store latest frame from video stream
    
    return self
  
  def stop(self):
    self.is_stopped = True  # call for stop of thread
    
    self.thread.join()  # wait for termination of thread
  
  def read(self):
    """
    retrieve latest frame from storage
    """
    
    while True:
      if self.frames_latest:
        return self.frames_latest.popleft()  # to never read frame twice
  
  def _grab_frames_latest(self):
    """
    continuously read and store latest frame from video stream
    """
    
    while True:
      if self.is_stopped:
        break
      
      ret, image = self.stream.read()
      
      self.frames_latest.append(image)  # update latest frame
    
    self.stream.release()

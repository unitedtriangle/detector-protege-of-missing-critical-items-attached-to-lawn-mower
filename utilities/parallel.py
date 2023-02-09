"""
support parallel computing
"""


import ray


@ray.remote
class Share:
  def __init__(self, **shareables):
    """
    store shareables between processes in separate process
    
    shareables (dict): pairs of name and associated value intended for share
    """
    
    for name, value in shareables.items():
      setattr(self, name, value)
  
  def get(self, name):
    """
    get value of shareable
    
    name: name of shareable
    """
    
    return getattr(self, name)
  
  def set(self, name, value):
    """
    set value of shareable to specified value
    
    name: name of shareable
    
    value: specified value to assign to shareable
    """
    
    setattr(self, name, value)

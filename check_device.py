"""
def is_gpu_available(cuda_only=True):
  from tensorflow.python.client import device_lib as _device_lib
 
  if cuda_only:
    return any((x.device_type == 'GPU')
               for x in _device_lib.list_local_devices())
  else:
    return any((x.device_type == 'GPU' or x.device_type == 'SYCL')
               for x in _device_lib.list_local_devices())
"""
import os

from tensorflow.python.client import device_lib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "99"
 

if __name__ == "__main__":

    print(device_lib.list_local_devices())

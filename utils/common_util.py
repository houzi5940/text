import cv2
import numpy as np
import base64

def base64_2_array(base64_data):
    im_data=base64.b64decode(base64_data)
    im_array=np.frombuffer(im_data,np.uint8)
    im_array=cv2.imdecode(im_array,cv2.COLOR_RGB2BGR)

    return im_array
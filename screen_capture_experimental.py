import cv2
from PIL import ImageGrab
import numpy as np

while True:
    img = ImageGrab(bbox = (0,0,128,128))

    np_img = np.array(img)
    cv2.imshow("Video capture",  )
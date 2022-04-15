import numpy as np
import os
from os import listdir
import glob
import cv2
count = 0
path = '/media/risc/8gb_usb/rp2/a'
for image in glob.glob(path + "/*.png"):
	count = count+1
	frame = cv2.imread(image)
	#print(type(frame))
	resized = cv2.resize(frame, (32, 32), interpolation = cv2.INTER_NEAREST)
	cv2.imwrite("%i.png"%count,resized)

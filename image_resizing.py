import numpy as np
import cv2
import glob


path_read = '/media/risc/8gb_usb/rp2/a'
path_write = '/media/risc/8gb_usb/rp2/resized/'
count = 0
for image in glob.glob(path_read + "/*.png"):
	count = count+1
	frame = cv2.imread(image)
	#print(type(frame))
	resized = cv2.resize(frame, (32, 32), interpolation = cv2.INTER_NEAREST)
	cv2.imwrite(path_write + "%i.png"%count,resized)
print("Done Resizing")

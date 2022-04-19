import cv2
import numpy as np
import glob
import re

img_array = []
numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

img_array = []
for filename in sorted(glob.glob('/home/risc/Downloads/Awais/yolo_2nd/yolov5/runs/detect/exp4/*.jpg'), key=numericalSort):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter('project1.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

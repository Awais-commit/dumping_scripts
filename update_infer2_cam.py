#/home/mendel/motorbike# Now we do evaluation on the tflite model.
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
from PIL import Image
from PIL import ImageDraw
#%matplotlib inline
import os
import numpy as np

#PATH_TO_MODEL = '/home/mendel/mask/output_model/ssdlite_mobiledet_mask.tflite'
PATH_TO_MODEL = '/home/mendel/mask/output_model/ssdlite_mobiledet_mask_edgetpu.tflite'
labels = {0:'MotorBike'}
#interpreter = Interpreter(PATH_TO_MODEL)
interpreter = Interpreter(PATH_TO_MODEL , experimental_delegates=[load_delegate('libedgetpu.so.1')])

interpreter.allocate_tensors()
interpreter.invoke() # warmup
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
width = input_details[0]['shape'][2]
height = input_details[0]['shape'][1]

PATH_TO_TEST_IMAGES_DIR = '/home/mendel/mask/test'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'maksssksksss{}.png'.format(i)) for i in range(0, 100)]

def run_inference_for_single_image(image, interpreter):
  interpreter.set_tensor(input_details[0]['index'], image)
  interpreter.invoke()
  boxes = interpreter.get_tensor(output_details[0]['index'])[0]
  classes = interpreter.get_tensor(output_details[1]['index'])[0]
  scores = interpreter.get_tensor(output_details[2]['index'])[0]
  
  return boxes, classes, scores

colors = {0:(128, 255, 102), 1:(102, 255, 255), 2:(232, 123, 212)}

import cv2
import time


cap = cv2.VideoCapture('/dev/video0')
image_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
image_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

out = cv2.VideoWriter('cam_det_fps2.avi', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 30, (int(cap.get(3)), int(cap.get(4))))
    
list_fps = []
#fps=0
x = 0.5 # displays the frame rate every 1 second
counter = 0

while(cap.isOpened()):
    # Read the frame    
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    print('image width: ', width)
    print('image height: ', height)

    start_time = time.time()
    # Actual detection.
    boxes, classes, scores = run_inference_for_single_image(input_data , interpreter)
    end_time = time.time()
    time_taken = (end_time - start_time)
    print('fps: ', 1/(time_taken))
    print('Time Taken: ', (time_taken))
    print('---end detecyion--- ')

    # Visualization of the results of a detection.
    for i in range(len(boxes)):
    	if scores[i] > 0.5:
            ymin = int(max(1, (boxes[i][0] * image_height)))
            xmin = int(max(1, (boxes[i][1] * image_width)))
            ymax = int(min(image_height, (boxes[i][2] * image_height)))
            xmax = int(min(image_width, (boxes[i][3] * image_width)))
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 4)
            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i]*100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), 
                ( xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imshow('Object detector', frame)
    if cv2.waitKey(1) == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()

    

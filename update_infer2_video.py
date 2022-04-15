# Now we do evaluation on the tflite model.
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
from PIL import Image
from PIL import ImageDraw
#%matplotlib inline
import os
import numpy as np

#PATH_TO_MODEL = '/home/mendel/mask/output_model/ssdlite_mobiledet_motorbike.tflite'
PATH_TO_MODEL = '/home/mendel/motorbike/output_model/ssdlite_mobiledet_motorbike_edgetpu.tflite'
labels = {0:'MotorBike'}
#interpreter = Interpreter(PATH_TO_MODEL)
interpreter = Interpreter(PATH_TO_MODEL , experimental_delegates=[load_delegate('libedgetpu.so.1')])

interpreter.allocate_tensors()
interpreter.invoke() # warmup
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
width = input_details[0]['shape'][2]
height = input_details[0]['shape'][1]

PATH_TO_TEST_IMAGES_DIR = '/home/mendel/motorbike/test'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, '00463{}.jpg'.format(i)) for i in range(0, 10)]

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

#cap = cv2.VideoCapture('/home/mendel/mask/Dron7500.avi')
cap = cv2.VideoCapture('/dev/video0')

#out = cv2.VideoWriter('cctv2_det.mp4', cv2.VideoWriter_fourcc(
#        'M', 'J', 'P', 'G'), 15, (640, 364))
out = cv2.VideoWriter('Dron7500_det_fps2.avi', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 30, (int(cap.get(3)), int(cap.get(4))))
    
list_fps = []
#fps=0
x = 0.5 # displays the frame rate every 1 second
counter = 0

while(cap.isOpened()):
    # Read the frame
    
    ret, image = cap.read()
    cv2.imwrite("imagex.jpg", image)
    image = "imagex.jpg"


    image = Image.open(image)
    image_width, image_height = image.size
    print('image width: ', image_width)
    print('image height: ', image_height)
    draw = ImageDraw.Draw(image)
    print('evalutaing')
    # Removing the Alpha Channel
    if image.mode == 'RGBA':
    	image.load()
    	removed_A = Image.new('RGB', image.size, (255,255,255))
    	removed_A.paste(image, mask=image.split()[3])
    	resized_image = removed_A.resize((width, height))
    else:  
    	resized_image = image.resize((width, height))

    np_image = np.asarray(resized_image)
    image_np_expanded = np.expand_dims(np_image, axis=0)


    print('---detecting Image---')
    start_time = time.time()
    # Actual detection.
    boxes, classes, scores = run_inference_for_single_image(image_np_expanded, interpreter)
    end_time = time.time()
 
    time_taken = (end_time - start_time)
    fps = 1/(time_taken)
    print('fps: ', fps)
    print('Time Taken: ', (time_taken))
    print('---end detecyion--- ')
    list_fps.append(fps)
    print("Average fps: ", sum(list_fps)/len(list_fps))


    # Visualization of the results of a detection.
    for i in range(len(boxes)):
    	if scores[i] > 0.5:
    		ymin = int(max(1, (boxes[i][0] * image_height)))
    		xmin = int(max(1, (boxes[i][1] * image_width)))
    		ymax = int(min(image_height, (boxes[i][2] * image_height)))
    		xmax = int(min(image_width, (boxes[i][3] * image_width)))
    		draw.rectangle((xmin, ymin, xmax, ymax), width=3, outline=colors[int(classes[i])])
    		draw.rectangle((xmin, ymin, xmax, ymin-10), fill=colors[int(classes[i])])
    		text = labels[int(classes[i])] + ' ' + str(scores[i]*100) + '%'
    		draw.text((xmin+2, ymin-10), text, fill=(0,0,0), width=2)

    image = np.asarray(image)
    #out.write(image) #Uncomment after visit to save the video
    cv2.imshow('bike_det',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()

print("FPS List: ", list_fps)
print("Average fps: ", sum(list_fps)/len(list_fps))


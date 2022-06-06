import cv2

img_counter = 300
dispW=1280
dispH=960
flip=2



#cv2.namedWindow("test")


#Uncomment These next Two Line for Pi Camera
#camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'

#camSet="nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)NV12, framerate=(fraction)24/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

#camSet='nvarguscamerasrc ! video/x-raw(memory:NVMM),width=640, height=480, framerate=30/1, format=NV12 ! nvvidconv flip-method=2 ! nvegltransform ! nveglglessink -e'

#camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'

#cam= cv2.VideoCapture(camSet)

cam = cv2.VideoCapture(0)
#cam = cv2.VideoCapture('Test-videos/Earth 101 _ National Geographic.mp4')

while True:
    ret, frame = cam.read()
    # 0-vert-picam, 1-horz-webcam
    #frame = cv2.flip(frame, 0)
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 ==ord('q'):
        break
    elif k&0xFF == ord(' '):
        # SPACE pressed
        img_name = "data2/shahid/opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()
cv2.destroyAllWindows()

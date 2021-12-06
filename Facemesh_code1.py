import cv2
import mediapipe as mp
import time

#camera on
capture = cv2.VideoCapture(0)

pTime = 0 #defining previous time

#below code are prewritten from their the lib mediapipe
mpdraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)        #create our object

drawspec = mpdraw.DrawingSpec(thickness=1,circle_radius=1)

while True:
    success, img = capture.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB )   #convert to bgr
    results = faceMesh.process(imgRGB)

#display the mesh on face

    if results.multi_face_landmarks:
        for faceLns in results.multi_face_landmarks:    #for multiple faces
            mpdraw.draw_landmarks(img, faceLns, mpFaceMesh.FACE_CONNECTIONS,
                                  drawspec,drawspec)

            #get the landmark xyz positions
            for id,ln in enumerate(faceLns.landmark):
                #print(ln)
                ih , iw, ic = img.shape
                x,y = int(ln.x*ih), int(ln.y*ih)
                print(id,x,y)


    #FPS

    cTime = time.time()
    fps = 1/(cTime - pTime)  #comaparing previous and current time to get the FPS
    pTime = cTime
    cv2.putText(img,f'FPS: {int(fps)}',(20,70), cv2.FONT_HERSHEY_PLAIN, 3,(0,255,0),3) #on screen FPS display




    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
# Release the VideoCapture object
#cap.release()



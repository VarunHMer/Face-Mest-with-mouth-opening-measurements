import cv2
import mediapipe as mp
import time

class FaceMeshDtector():
    def __init__(self,staicMode =False, maxfaces=2,minDetection_confidence=0.5, mintracking_confidence=0.5):
        self.staicMode = staicMode
        self.maxfaces = maxfaces
        self.minDetection_confidence = minDetection_confidence
        self.mintracking_confidence = mintracking_confidence

        #below code are prewritten from their the lib mediapipe
        self.mpdraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staicMode,self.maxfaces,
                                                 self.minDetection_confidence,
                                                 self.mintracking_confidence)        #create our object

        self.drawspec = self.mpdraw.DrawingSpec(thickness=1 , circle_radius=1)

    def findFaceMesh(self,img, draw = True):
        self.imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB )   #convert to bgr
        self.results = self.faceMesh.process(self.imgRGB)

        #display the mesh on face

        faces = [] #to get the face and store the x and y markinbgs after append

        if self.results.multi_face_landmarks:

            for faceLns in self.results.multi_face_landmarks:    #for multiple faces
                if draw:
                    self.mpdraw.draw_landmarks(img, faceLns, self.mpFaceMesh.FACE_CONNECTIONS,
                                      self.drawspec,self.drawspec)
                #store the face points
                face =[]

                #get the landmark xyz positions
                for id,ln in enumerate(faceLns.landmark):
                    #print(ln)
                    ih , iw, ic = img.shape
                    x,y = int(ln.x*ih), int(ln.y*ih)

                    #what point number shown , printing ID number
                    # cv2.putText(img, str(id),(x,y), cv2.FONT_HERSHEY_PLAIN,
                    #             0.7, (0, 255, 0),1)  # on screen FPS display



                    #print(id,x,y)
                    face.append([x,y])

                faces.append(face)

        return img, faces


# creating a module to have the facemesh numbers
def main():

    # camera on
    capture = cv2.VideoCapture(0)

    pTime = 0  # defining previous time

    detector=FaceMeshDtector()

    while True:
        success, img = capture.read()
        img, faces = detector.findFaceMesh(img)
        if len(faces)!= 0:
            print(faces[0])
        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)  # comaparing previous and current time to get the FPS
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0),
                    3)  # on screen FPS display
        # Display
        cv2.imshow('img', img)
        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break


# Release the VideoCapture object
# cap.release()


if __name__=="__main__":
    main()

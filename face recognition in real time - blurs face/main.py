import cv2 

video_capture = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def face_manipulation(frame):
    face = classifier.detectMultiScale(frame)
    
    if len(face) != 0: 
        for x, y, w, h in face:
            image = cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 0), 3)
            image[y: y+h, x: x+w] = cv2.blur(image[y: y+h, x: x+w], (30,30))
        return frame
    else:
        return frame
    
while True: 
    check, frame = video_capture.read()
    cv2.imshow("Face Manipulation", face_manipulation(frame))
         
    key = cv2.waitKey(10) 
    if key == ord("q") or key == 32: 
        break 

cv2.destroyAllWindows()
video_capture.release()
import cv2 
from simple_facerec import SimpleFacerec
#face_loc = face locations

#encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

video = cv2.VideoCapture(0)

def face_manipulation(frame): 
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y, w, h, x = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)
        
        if name == "Unknown":
            image = cv2.rectangle(frame, (x, y), (w, h), (0,0, 200), 4)
            image[y: h, x: w] = cv2.blur(image[y: h, x: w], (30,30))
    return frame

while True: 
    check, frame = video.read()
    
    #detect faces 
    
    cv2.imshow("Frame", face_manipulation(frame))
    
    key = cv2.waitKey(1)
    if key == ord("q") or key == 32: 
        break 

video.release()
cv2.destroyAllWindows()
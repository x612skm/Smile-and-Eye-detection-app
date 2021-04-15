import cv2
face_detector = cv2.CascadeClassifier('harcascade_face.xml')
smile_detector = cv2.CascadeClassifier('smile.xml')
eye_detector = cv2.CascadeClassifier('eye.xml')
#taking webcam feed
webcam = cv2.VideoCapture(0)

#show the current frame
while True:
    #read the current frame from the webcam video stream
    successful_frame_read, frame = webcam.read() #reading the frame details around in clicks

    #if theres an error then abort
    if not successful_frame_read:
        break
    #change to grayscale so that harr features applied
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detect only faces
    faces=face_detector.detectMultiScale(frame_grayscale)
    #run smile detection harr in each of the faces(run face detection in each of those faces in front)
    for(x, y, w, h) in faces:
        #drawing rectangle around the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)
        #get the sub frame(using numpy N-dimensional array )
        the_face = frame[y:y+h, x:x+w]
        #change too grayscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)

        eyes=eye_detector.detectMultiScale(face_grayscale, scaleFactor=1.1, minNeighbors=20)

        #find all the smile in face
        ##for(x_, y_, w_, h_) in smiles:
          #  cv2.rectangle(the_face, (x_,y_), (x_ + y_ + w_ + h_), (50, 50, 200), 4)

        #find all the smiles in the face
        #for(x_, y_, w_, h_) in eyes:
            #draw a rectangle around the smile
           # cv2.rectangle(the_face, (x_,y_), (x_ + y_ + w_ + h_), (255,255,255), 4)

        #find all the smiles in the face
        #for (x_,y_,w_,h_) in smiles:
            #draw a rectangle around the smile
            #cv2.rectangle(the_face, (x_,y_), (x_+y_+w_+h_), (50,50,200), 4)

        #label this face as smiling
    if len(smiles) > 0:
        cv2.putText(frame,'keep smiling you look preety', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,255,255))

    if len(eyes) > 0:
        cv2.putText(frame,'your eyes are nice', (x, y+h+90), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,255,255))

    #show the current frame
    cv2.imshow('smile', frame)
    #display
    cv2.waitKey(1)

#cleanup
webcam.release()
cv2.destroyAllWindows()

print("whats up")
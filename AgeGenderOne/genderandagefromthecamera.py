import cv2
import numpy as np
import cvlib as cv

from keras.models import load_model

model = load_model("model1.h5")
isim={0:"Arda",1:"Belhanda",2:"Taner"}

kamera=cv2.VideoCapture(0)
kamera.set(3, 1920)
kamera.set(4, 1080)

while kamera.isOpened():
    ret,frame=kamera.read()

    yuz, confidence = cv.detect_face(frame)
    tonlama = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for idx, face in enumerate(yuz):

        (x, y) = face[0], face[1]
        (w, h) = face[2], face[3]

        cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 3)

        image_c = np.copy(tonlama[y:h, x:w])
        resized=cv2.resize(image_c, (224,224))
        normalized=resized/255.
        reshape=np.reshape(normalized, (1,224,224,1) )

        result=model.predict(reshape)
        label=np.argmax(result, axis=1)[0]
        ad=isim[label]
        oran=np.max(result)*100
        cv2.putText(frame,ad+":%"+str(round(oran,2)), (x, y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 3)


    cv2.imshow("Webcam",frame)

    if (cv2.waitKey(30) & 0xFF == ord('q')):
        break

kamera.release()
cv2.destroyAllWindows()







import cv2
import numpy as np
import cvlib as cv
import pafy

from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

models = load_model("AgeGenderModel1.h5")

font = cv2.FONT_HERSHEY_COMPLEX
sex_F=["male","female"]

url = 'https://www.youtube.com/watch?v=pPMQ5nGxOiQ'
vPafy = pafy.new(url)

play = vPafy.getbest(preftype="mp4")
cap = cv2.VideoCapture(play.url)

while cap.isOpened():
    ret, goruntu = cap.read()
    tonlama = cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB)  # tonlama ayarı

    yuz, confidence = cv.detect_face(goruntu)

    for idx, face in enumerate(yuz):
        (startX, startY) = face[0], face[1]
        (endX, endY) = face[2], face[3]

        cv2.rectangle(goruntu, (startX, startY), (endX, endY), (0, 0, 255), 3)

        image_c = np.copy(tonlama[startY:endY, startX:endX])

        if (image_c.shape[0]) < 10 or (image_c.shape[1]) < 10:
            continue

        img = cv2.resize(image_c, (64, 64))
        face_crop = img.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)
        a = face_crop

        conf_age = models.predict(a)[1][0]
        idx =int(np.round(conf_age))
        label1 = idx

        conf_gender = models.predict(a)[0][0]
        idx2 = int(np.round(conf_gender))
        label2 = sex_F[idx2]
        cv2.putText(goruntu,label2+"/"+str(label1), (startX,startY), font, 0.7, (0, 255, 255), 2)

    cv2.imshow('Goruntu', goruntu)

    if cv2.waitKey(13) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

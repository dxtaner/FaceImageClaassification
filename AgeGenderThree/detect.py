import cvlib as cv
import cv2
import numpy as np
import pafy

from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

gender_model = load_model("./GenderModel3.h5")
age_model = load_model("./ageModel1.h5")
font = cv2.FONT_HERSHEY_COMPLEX

gender_dict = {
    0:"Erkek",
    1:"Kadin"
}
age_dict = {
    0: "(0-15)",
    1: "(20-65)",
    2: "65++"
}

url = 'https://www.youtube.com/watch?v=3fkEk8Qib1M'
vPafy = pafy.new(url)

play = vPafy.getbest(preftype="mp4")
cap = cv2.VideoCapture(play.url)

while cap.isOpened():
    ret, goruntu = cap.read()
    yuz, confidence = cv.detect_face(goruntu)

    tonlama = cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB)  # tonlama ayarı

    for idx, face in enumerate(yuz):

        (x, y) = face[0], face[1]
        (w, h) = face[2], face[3]

        cv2.rectangle(goruntu, (x, y), (w, h), (0, 0, 255), 3)
        image_c = np.copy(tonlama[y:h, x:w])

        if (image_c.shape[0]) < 10 or (image_c.shape[1]) < 10:
            continue

        img = cv2.resize(image_c, (64, 64))
        img = img.astype("float") / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        a = img

        conf_gender = gender_model.predict(a)[0]
        idx = np.argmax(conf_gender)
        gender = gender_dict[idx]
        gender = "{}:{:.2f}%".format(gender, conf_gender[idx] * 100)

        conf_age = age_model.predict(a)[0]
        idx = np.argmax(conf_age)
        age = age_dict[idx]
        age = "{}:{:.2f}%".format(age, conf_age[idx] * 100)

        cv2.putText(goruntu, gender+"/"+age, (x, y-8), font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Goruntu Tespiti', goruntu)

    if cv2.waitKey(13) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

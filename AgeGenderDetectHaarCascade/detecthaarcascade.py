import cv2
import numpy as np

from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

gender_model = load_model("./GenderModel3.h5")
age_model_1 = load_model("./ageModel1.h5")
#age_model_2 = load_model("AgeModel3.h5")

font=cv2.FONT_HERSHEY_TRIPLEX

gender_dict = {
    0:"Erkek",
    1:"Kadin"
}

age_dict_1 = {
    0:"0-20",
    1:"20-65",
    2:"65++"
}

"""
age_dict_2 = {
    0:"(0-15)",
    1:"(15-30)",
    2:"(30-48)",
    3:"(48-65)",
    4:"65++"
}
"""

yuz_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
kamera = cv2.VideoCapture(0)
kamera.set(3,1280)
kamera.set(4,1024)

while kamera.isOpened():
    ret, goruntu = kamera.read()

    griton = cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB)  # tonlama ayarı
    yuzler = yuz_cascade.detectMultiScale(griton, 1.3, 5, minSize=(40,40), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in yuzler:

        roi_gray = griton[y:y + w, x:x + h]
        img = cv2.resize(roi_gray, (64, 64))

        face_crop = img.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        """test_pic = img.reshape((1, 64, 64, 3))
        a = test_pic"""
        a = face_crop

        conf_gender = gender_model.predict(a)[0]
        idx1 = np.argmax(conf_gender)
        label1 = gender_dict[idx1]
        label1 = "{}:{:.2f}%".format(label1, conf_gender[idx1] * 100)

        conf_age = age_model_1.predict(a)[0]
        idx2 = np.argmax(conf_age)
        label2 = age_dict_1[idx2]
        label2 = "{}:{:.2f}%".format(label2, conf_age[idx2] * 100)

        cv2.rectangle(goruntu, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(goruntu, label1+"/"+label2, (x, y - 10), font,0.7, (0, 255, 255), 2)

    cv2.imshow('Goruntu', goruntu)

    if cv2.waitKey(13) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()

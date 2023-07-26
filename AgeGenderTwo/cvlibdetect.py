import cv2
import numpy as np
import cvlib as cv

from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

gender_model = load_model("GenderModel3.h5")
age_model_1 = load_model("ageModel1.h5")
font = cv2.FONT_HERSHEY_TRIPLEX

gender_dict = {
    0: "Erkek",
    1: "Kadin"
}
age_dict_1 = {
    0: "(0-20)",
    1: "(20-65)",
    2: "(65++)"
}

kamera = cv2.VideoCapture(0)
kamera.set(3, 1280)
kamera.set(4, 1024)

while kamera.isOpened():
    ret, goruntu = kamera.read()

    tonlama = cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB)  # tonlama ayarı

    yuz, confidence = cv.detect_face(goruntu)

    for ax, face in enumerate(yuz):

        (startX, startY) = face[0], face[1]
        (endX, endY) = face[2], face[3]

        cv2.rectangle(goruntu, (startX, startY), (endX, endY), (0, 0, 255), 3)

        image_c = np.copy(tonlama[startY:endY, startX:endX])

        if (image_c.shape[0]) < 10 or (image_c.shape[1]) < 10:
            continue

        image_c = cv2.resize(image_c, (64, 64))
        image_c = image_c.astype("float") / 255.0
        image_c = img_to_array(image_c)
        image_c = np.expand_dims(image_c, axis=0)
        test_pic = image_c.reshape((1, 64, 64, 3))
        a=test_pic


        conf = gender_model.predict(a)[0]
        ax = np.argmax(conf)
        label = gender_dict[ax]
        label="{}:{:.2f}%".format(label, conf[ax] * 100)

        conf_age = age_model_1.predict(a)[0]
        ax = np.argmax(conf_age)
        label2 = age_dict_1[ax]
        label2 = "{}:{:.2f}%".format(label2, conf_age[ax] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(goruntu, label+"/"+label2, (startX, Y), font, 0.7, (0, 255, 255), 2)

    cv2.imshow('Goruntu', goruntu)

    if cv2.waitKey(13) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()

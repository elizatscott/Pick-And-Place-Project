# NewStar Technology LLC Training Set Preparation.
# Author : Kamesh Arvind Sarangan
import cv2
from keras.models import load_model
import numpy as np
categories = ["Screw Driver", "Ball"]
font = cv2.FONT_HERSHEY_COMPLEX
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 30.0, (1280,720))
model = load_model("CNN.model")
cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()
    img_size = 60
    array_img = np.array(img)
    resized_img = cv2.resize(array_img, (img_size, img_size))
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    normalized_img = gray_img / 255
    array = normalized_img.reshape(-1, img_size, img_size, 1)
    prediction = model.predict(array)
    prediction = round(prediction[0][0])
    print(prediction)
    if (prediction == 1):
        cv2.putText(img," BALL" , (180, 75), font, 0.75, (0, 0, 255), 2,
                    cv2.LINE_AA)
    elif prediction == 0:
        cv2.putText(img," Screw Driver" , (180, 75), font, 0.75, (0, 0, 255), 2,
                    cv2.LINE_AA)
    cv2.imshow("image", img)
    b = cv2.resize(img, (1280, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    out.write(b)
    cv2.waitKey(1)
#print(prediction)


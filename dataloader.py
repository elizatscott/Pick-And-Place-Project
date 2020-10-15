# NewStar Technology LLC Training Set Preparation.
# Author : Kamesh Arvind Sarangan
import cv2
import os
import random
import numpy as np
import pickle

#READ THE DATASET
datadir = "C:/Users/kames/PycharmProjects/pythonProject/data"
categories = ["screw driver", "ball"]

#RESIZE & CONVERT THE IMAGE TO GRAY SCALE AND PREPARE THE TRAINING DATA
img_size = 60
training_data = []
def create_training_data():
    for category in categories:
        path = os.path.join(datadir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (img_size, img_size))
            training_data.append([new_array, class_num])
create_training_data()
#print(len(training_data))

random.shuffle(training_data)
#Check whether the targets are correct
#for sample in training_data:
#    print(sample[1])

X= []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)

#Convert the list to array
X = np.array(X).reshape(-1, img_size, img_size, 1)
y = np.array(y)

#STORE THE FEATURES 'X' AND LABEL 'Y' AS SEPERATE PICKLE FILES
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()



# NewStar Technology LLC Training Set Preparation.
# Author : Kamesh Arvind Sarangan
import pickle
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

file = open("X.pickle", "rb")
X = pickle.load(file)
file.close()

file = open("y.pickle", "rb")
y = pickle.load(file)
file.close()

#Nomralize the feature data
X = X / 255.0

#print(type(X))
#print(type(y))

#Create a 3 layered Convolution Neural Network
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape= X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(12))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(1))
model.add(Activation("sigmoid"))
opt = Adamax(lr=0.003)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['accuracy'])

#Gives the accuracy of the model and information about the model
model.fit(X, y, batch_size=22, epochs=30, validation_split=0.1)
print(model.summary())

### SAVE THE TRAINED MODEL
model.save('CNN.model')
print("Trained Model Saved Perfectly!!!")
# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

## Neural Network Model

![image](https://github.com/aldrinlijo04/mnist-classification/assets/118544279/eef099d4-ccf0-4148-8d61-3cbe8c06ac37)

## DESIGN STEPS

### STEP 1:
Start by importing all the necessary libraries. And load the Data into Test sets and Training sets.

### STEP 2:
Then we move to normalization and encoding of the data.

### STEP 3:
The Model is then built using a Conv2D layer, MaxPool2D layer, Flatten layer, and 2 Dense layers of 16 and 10 neurons respectively.

### STEP 4:
Finally, we pass handwritten digits to the model for prediction.

## PROGRAM

### Name: Dhiyaneshwar
### Register Number: 212222110009
```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape

X_test.shape

single_image= X_train[0]

single_image.shape

plt.imshow(single_image,cmap='gray')

y_train.shape

X_train.min()

X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()

model.add(layers.Input(shape=(28,28,1)))

model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))

model.add(layers.MaxPool2D(pool_size=(2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(32,activation='relu'))

model.add(layers.Dense(64,activation='relu'))

model.add(layers.Dense(10,activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)

print('Dhiyaneshwar P -212222110009')
metrics.head()

print('Dhiyaneshwar P -212222110009')
metrics[['accuracy','val_accuracy']].plot()

print('Dhiyaneshwar P -212222110009')
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))
img = image.load_img('NEWIMG.jpg')
type(img)

img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
x_single_prediction = np.argmax(model.predict(img_28_gray_scaled.reshape(1,28,28,1)),axis=1)

print(x_single_prediction)
print('Dhiyaneshwar P -212222110009')
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
print(x_single_prediction)

img1 = image.load_img('3.jpg')
img_tensor_1= tf.convert_to_tensor(np.asarray(img1))
img_28 = tf.image.resize(img_tensor_1,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction1 = np.argmax(model.predict(img_28_gray_scaled.reshape(1,28,28,1)),axis=1)
print(x_single_prediction1)
print("Dhiyaneshwar P -212222110009")

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
print(x_single_prediction1)
````
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/user-attachments/assets/5ffdf2ca-64fa-4eaf-8e1a-75764e2599af)

![image](https://github.com/user-attachments/assets/07e4ed06-763f-4ca9-bc85-0336d545b9d4)


### Classification Report
![image](https://github.com/user-attachments/assets/ce38d327-fc88-4fe3-806d-98597ec8f9d7)


### Confusion Matrix
![image](https://github.com/user-attachments/assets/ef3a809b-3631-45fa-ad1d-24c6d03bb8a0)


### New Sample Data Prediction
#### INPUT
<img src="https://github.com/user-attachments/assets/022e4cb3-ac9f-4424-bfdd-32645371e211" width="300" height="200" alt="Reduced Image">

#### OUTPUT
![image](https://github.com/user-attachments/assets/a612b238-0a2b-4d07-af58-33ddc89e16c4)

## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.

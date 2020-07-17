import numpy as np
from keras.datasets import mnist
from keras import Sequential, optimizers
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# raw data set
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
images = (np.concatenate((train_images, test_images)) > 0) * 1
labels = np.concatenate((train_labels, test_labels)).reshape(-1, 1)
print("Raw Image count : " + str(images.shape[0]))
print("Raw Label count : " + str(labels.shape[0]))
print("Image dimension : " + str(images.shape[1]) + "x" + str(images.shape[2]))

# flattening images
flattened_images = images.reshape(images.shape[0], (images.shape[1] * images.shape[2]))
print("Flattened images' shape : " + str(flattened_images.shape))

# one hot encoding
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(labels)
categorical_labels = encoder.transform(labels).toarray()
print("One hot encoded labels' shape : " + str(categorical_labels.shape))

# train/dev set division
x_train, x_test, y_train, y_test = train_test_split(flattened_images, categorical_labels, test_size=0.1,
                                                    random_state=42)
print("#" * 50)
print("Train image set shape : " + str(x_train.shape))
print("Train label set shape : " + str(y_train.shape))
print("Test image set shape : " + str(x_test.shape))
print("Test label set shape : " + str(y_test.shape))
print("#" * 50)

# model building
model = Sequential()
model.add(Dense(90, activation='relu', input_dim=flattened_images.shape[1]))
model.add(Dense(10, activation='sigmoid'))
model.summary()

sgd = optimizers.SGD(lr=0.1)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, verbose=2)

# model evaluation
score = model.evaluate(x_test, y_test, verbose=0)
print('Test set loss : ', score[0])
print('Test set accuracy : ', score[1])
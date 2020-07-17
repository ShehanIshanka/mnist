import numpy as np
from keras.datasets import mnist
from keras import Sequential, optimizers
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

for noise_factor in np.arange(0.1, 1, 0.1):
    print("Noise Factor : " + str(noise_factor))

    # raw data set
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    images = (np.concatenate((train_images, test_images)) > 0) * 1
    labels = np.concatenate((train_labels, test_labels)).reshape(-1, 1)

    # flattening images
    flattened_images = images.reshape(images.shape[0], (images.shape[1] * images.shape[2]))

    # one hot encoding
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(labels)
    categorical_labels = encoder.transform(labels).toarray()

    # train/dev set division
    x_train, x_test, y_train, y_test = train_test_split(flattened_images, categorical_labels, test_size=0.1,
                                                        random_state=42)

    # noise addition
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    # model building
    model = Sequential()
    model.add(Dense(90, activation='relu', input_dim=flattened_images.shape[1]))
    model.add(Dense(10, activation='sigmoid'))

    sgd = optimizers.SGD(lr=0.1)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train_noisy, y_train, epochs=10, verbose=0)

    # model evaluation
    score = model.evaluate(x_test_noisy, y_test, verbose=0)
    print('Test set loss : ', score[0])
    print('Test set accuracy : ', score[1])
    print("#" * 50)

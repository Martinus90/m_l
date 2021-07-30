import tensorflow as tf
import matplotlib.pyplot as plt
import random as rnd
import numpy as np
import cv2

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))


load_model = False

if load_model == True:
    try:
        new_model = tf.keras.models.load_model('read_numb2.model')
        print("Successfully load model")
    except Exception as e:
        print(e)
        if e == FileNotFoundError:
            print("Trying to generate model.")
else:
    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    model.fit(x_train,y_train,epochs=3,steps_per_epoch=3000)
    model.save('read_numb2.model')
    new_model = tf.keras.models.load_model('read_numb2.model')
    print("Successfully creating model")


path = 'numbers/'
"""
img = cv2.imread(path + 'number.bmp')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
resized = cv2.resize(gray, (28,28), interpolation = cv2.INTER_AREA)
resized = cv2.bitwise_not(resized)


rgb_tensor = tf.convert_to_tensor(resized, dtype=tf.float32)
rgb_tensor = tf.expand_dims(rgb_tensor , 0)
"""
nr = 1


while True:

    img = cv2.imread(path + 'number.bmp')
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (28,28), interpolation = cv2.INTER_AREA)
    resized = cv2.bitwise_not(resized)
    rgb_tensor = tf.convert_to_tensor(resized, dtype=tf.float32)
    rgb_tensor = tf.expand_dims(rgb_tensor , 0)
    predictions = new_model.predict(rgb_tensor)


    #predictions = new_model.predict(x_test)
    print("Prediction = {}".format(np.argmax(predictions)))
    print("Print all {}".format(predictions))
    thismanager = plt.get_current_fig_manager()
    thismanager.window.wm_geometry("+2000+100")
    plt.imshow(resized, cmap = plt.cm.binary)
    #plt.imshow(x_test[nr], cmap = plt.cm.binary)
    plt.show()
    nr += 1
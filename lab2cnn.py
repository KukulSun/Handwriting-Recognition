import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = (np.expand_dims(train_images, axis = -1) / 255. ).astype(np.float32)
train_labels = (train_labels).astype(np.int64)
test_images = (np.expand_dims(test_images, axis = -1) / 255. ).astype(np.float32)
test_labels = (test_labels).astype(np.int64)

# The model function
def build_cnn_model():
    cnn_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters = 24, kernel_size = (3, 3), activation = tf.nn.relu),
        tf.keras.layers.MaxPool2D(pool_size = (2, 2)),
        tf.keras.layers.Conv2D(filters = 36, kernel_size = (3, 3), activation = tf.nn.relu),
        tf.keras.layers.MaxPool2D(pool_size = (2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation = tf.nn.relu),
        tf.keras.layers.Dense(10, activation = tf.nn.softmax)
    ])
    return cnn_model

cnn_model = build_cnn_model()
cnn_model.predict(train_images[[0]])
print(cnn_model.summary())

# parameters
cnn_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# training
BATCH_SIZE = 64
EPOCHS = 5
cnn_model.fit(train_images, train_labels, batch_size = BATCH_SIZE, epochs = EPOCHS)

'''TODO: Use the evaluate method to test the model!'''
test_loss, test_acc = cnn_model.evaluate(test_images, test_labels)
# test_loss, test_acc = # TODO

print('Test accuracy:', test_acc)

predictions = cnn_model.predict(test_images)
print(predictions[0])

print("Label of this digit is:", test_labels[0])
plt.imshow(test_images[0,:,:,0], cmap=plt.cm.binary)


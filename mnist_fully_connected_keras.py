import keras
from keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Dense # Dense layers are "fully connected" layers
from keras.models import Sequential # Documentation: https://keras.io/models/sequential/
from keras.callbacks import CSVLogger

# Setup train and test splits
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Training data shape: ", x_train.shape) # (60000, 28, 28) -- 60000 images, each 28x28 pixels
print("Test data shape", x_test.shape) # (10000, 28, 28) -- 10000 images, each 28x28

# Flatten the images
image_vector_size = 28*28
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)

# Convert to "one-hot" vectors using the to_categorical function
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



image_size = 784 # 28*28
num_classes = 10 # ten unique digits

model = Sequential()
# The input layer requires the special input_shape parameter which should match
# the shape of our training data.
model.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
model.add(Dense(units=64, activation='sigmoid'))
model.add(Dense(units=num_classes, activation='softmax'))
model.summary()
model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
csv_logger = CSVLogger("mnist_fully_connected_history_log.csv", append=True)
history = model.fit(x_train, y_train, batch_size=32, epochs=50, verbose=False, validation_split=.1, callbacks=[csv_logger])

loss, accuracy  = model.evaluate(x_test, y_test, verbose=False)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
#plt.show()


print(f'Test loss: {loss:.3}')
print(f'Test accuracy: {accuracy:.3}')
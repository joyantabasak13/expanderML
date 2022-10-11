import keras
from keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Dense  # Dense layers are "fully connected" layers
from keras.models import Sequential  # Documentation: https://keras.io/models/sequential/
from keras.callbacks import CSVLogger
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Concatenate, Lambda
import keras.backend as K
import random
import os

### Set Seeds
os.environ['PYTHONHASHSEED']=str(1)
random.seed(1)
np.random.seed(1)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


class CustomConnected(Dense):

    def __init__(self, units, connections, **kwargs):
        # this is matrix A
        self.connections = connections

        # initalize the original Dense with all the usual arguments
        super(CustomConnected, self).__init__(units, **kwargs)

    def call(self, inputs):
        output = K.dot(inputs, self.kernel * self.connections)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output


def get_adjacency_matrix(matrix, dim_left, dim_right):
    total_nodes = dim_left, dim_right
    adjacency_mat = np.zeros((dim_left, dim_right), dtype=float)
    rows = matrix.shape[0]
    columns = matrix.shape[1]
    print("Rows: " + str(rows) + " Columns: "+str(columns))
    for i in range(0, rows):
        for j in range(0, columns):
            adjacency_mat[i][matrix[i][j]] = 1.0
    print(adjacency_mat)
    return adjacency_mat


def get_random_layer(num_neurons_left, num_neurons_right):
    degree = num_neurons_right//2
    rnd_mat = []
    for i in range(0, num_neurons_left):
        rnd_array_row = np.array(random.sample(range(0, num_neurons_right), degree))
        rnd_mat.append(rnd_array_row)
    rnd_mat = np.array(rnd_mat)
    print("Information")
    # print(input_tensor.shape)
    print(rnd_mat.shape)
    print(rnd_mat)
    adj_mat = get_adjacency_matrix(rnd_mat, num_neurons_left, num_neurons_right)
    return adj_mat


def build_model(nodes):
    image_size = 784  # 28*28
    num_classes = 10  # ten unique digits
    first_layer = Dense(units=nodes[0], activation='sigmoid', input_shape=(image_size,))
    model.add(first_layer)
    for i in range(1, len(nodes)):
        #adj_matrix = get_random_layer(first_layer, 100, 200)
        #adjacency_mat = np.zeros((100, 200), dtype=float)
        #hidden_layer_1 = CustomConnected(200, adj_matrix, activation='sigmoid', use_bias=False)
        #model.add(hidden_layer_1)
        model.add(Dense(units=nodes[i], activation='sigmoid'))

    final_layer = Dense(units=num_classes, activation='softmax')
    model.add(final_layer)
    model.summary()
    model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])


def build_randomized_model(nodes):
    image_size = 784  # 28*28
    num_classes = 10  # ten unique digits
    first_layer = Dense(units=nodes[0], activation='sigmoid', input_shape=(image_size,), use_bias=False)
    model.add(first_layer)
    for i in range(1, len(nodes)):
        adj_matrix = get_random_layer(nodes[i-1], nodes[i])
        # adjacency_mat = np.zeros((100, 200), dtype=float)
        hidden_layer = CustomConnected(nodes[i], adj_matrix, activation='sigmoid', use_bias=False)
        model.add(hidden_layer)
        #model.add(Dense(units=nodes[i], activation='sigmoid', use_bias=False))

    final_layer = Dense(units=num_classes, activation='softmax', use_bias=False)
    model.add(final_layer)
    model.summary()
    model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])

# Setup train and test splits
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Training data shape: ", x_train.shape)  # (60000, 28, 28) -- 60000 images, each 28x28 pixels
print("Test data shape", x_test.shape)  # (10000, 28, 28) -- 10000 images, each 28x28

# Flatten the images
image_vector_size = 28 * 28
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)

# Convert to "one-hot" vectors using the to_categorical function
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
node_list = [352, 384]
# initial_layer_size = 128
# cur_acc = 0.96
# prev_acc = 0.95
# layer_stagnation = 0
# node_increment = 32
# is_last_step_layer_add = False
while True:
    model = Sequential()
    # print(f"Current Acc: {cur_acc} and Prev Acc: {prev_acc}")
    # print(f"Difference: {cur_acc - prev_acc}")
    # if cur_acc < prev_acc:
    #     if is_last_step_layer_add == True:
    #         print("layer added in last gen")
    #         # Give new layer an extra chance
    #         node_list[-1] = node_list[-1] + node_increment * 2
    #         layer_stagnation = layer_stagnation + 1
    #         is_last_step_layer_add = False
    #     else:
    #         # Reverse last change, Add layer
    #         print("Need to add new layer")
    #         node_list[-1] = node_list[-1] - node_increment
    #         node_list.append(initial_layer_size)
    #         is_last_step_layer_add = True
    #         layer_stagnation = 0
    # else:
    #     if (cur_acc - prev_acc) > 0.001:
    #         print("Doing good")
    #         node_list[-1] = node_list[-1] + node_increment
    #         layer_stagnation = 0
    #         is_last_step_layer_add = False
    #     else:
    #         print("Doing NOT so good")
    #         node_list[-1] = node_list[-1] + node_increment
    #         layer_stagnation = layer_stagnation + 1
    #         is_last_step_layer_add = False
    #     if layer_stagnation > 3:
    #         # Add layer
    #         print("Lets add layers")
    #         node_list.append(initial_layer_size)
    #         is_last_step_layer_add = True
    #         layer_stagnation = 0
    # print("NODE LIST")
    # print(node_list)
    # print(f"LAYER STAGNATION After: {layer_stagnation}")
    #build_model(node_list)
    build_randomized_model(node_list)
    csv_logger = CSVLogger("mnist_random_connected_history_log.csv", append=True)
    history = model.fit(x_train, y_train, batch_size=32, epochs=16, verbose=False, validation_split=.1,
                        callbacks=[csv_logger])
    loss, accuracy = model.evaluate(x_test, y_test, verbose=True)
    # prev_acc = cur_acc
    # cur_acc = accuracy
    print(f'Test loss: {loss:.3}')
    print(f'Test accuracy: {accuracy:.3}')
    break
    # if cur_acc > 0.97:
    #     print("Architecture Found")
    #     print(node_list)
    #     print(f'Test loss: {loss:.3} And Test Accuracy: {accuracy:.3}')
    #     break


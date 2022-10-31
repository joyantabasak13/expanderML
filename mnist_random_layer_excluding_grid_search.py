import math

import keras
from keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Dense  # Dense layers are "fully connected" layers
from keras.models import Sequential  # Documentation: https://keras.io/models/sequential/
from keras.callbacks import CSVLogger
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Concatenate, Lambda, Dropout
import keras.backend as K
import random
import os

### Set Seeds
# os.environ['PYTHONHASHSEED']=str(1)
# random.seed(1)
# np.random.seed(1)
# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# tf.compat.v1.keras.backend.set_session(sess)


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


def get_random_layer(num_neurons_left, num_neurons_right, rate):
    degree = math.ceil(num_neurons_right*rate)
    print(f"{num_neurons_left}, {num_neurons_right}, {degree}")
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
        model.add(Dense(units=nodes[i], activation='sigmoid'))

    final_layer = Dense(units=num_classes, activation='softmax')
    model.add(final_layer)
    model.summary()
    model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])


def build_model_with_dropout(nodes, rate):
    image_size = 784  # 28*28
    num_classes = 10  # ten unique digits
    first_layer = Dense(units=nodes[0], activation='sigmoid', input_shape=(image_size,))
    model.add(first_layer)
    model.add(Dropout(rate))
    for i in range(1, len(nodes)):
        model.add(Dense(units=nodes[i], activation='sigmoid'))
        model.add(Dropout(rate))

    final_layer = Dense(units=num_classes, activation='softmax')
    model.add(final_layer)
    model.summary()
    model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])


def build_randomized_model(nodes, rate):
    image_size = 784  # 28*28
    num_classes = 10  # ten unique digits
    first_layer = Dense(units=nodes[0], activation='sigmoid', input_shape=(image_size,))
    model.add(first_layer)
    for i in range(1, len(nodes)):
        adj_matrix = get_random_layer(nodes[i-1], nodes[i], rate)
        # adjacency_mat = np.zeros((100, 200), dtype=float)
        hidden_layer = CustomConnected(nodes[i], adj_matrix, activation='sigmoid')
        model.add(hidden_layer)
        #model.add(Dense(units=nodes[i], activation='sigmoid', use_bias=False))

    final_layer = Dense(units=num_classes, activation='softmax')
    model.add(final_layer)
    model.summary()
    model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])


def build_randomized_model_with_random_input_layer(nodes, rate):
    image_size = 784  # 28*28
    num_classes = 10  # ten unique digits
    adj_matrix = get_random_layer(image_size, nodes[0], rate)
    first_layer = CustomConnected(nodes[0], adj_matrix, activation='sigmoid', input_shape=(image_size,))
    model.add(first_layer)
    for i in range(1, len(nodes)):
        adj_matrix = get_random_layer(nodes[i-1], nodes[i], rate)
        hidden_layer = CustomConnected(nodes[i], adj_matrix, activation='sigmoid')
        model.add(hidden_layer)

    final_layer = Dense(units=num_classes, activation='softmax')
    model.add(final_layer)
    model.summary()
    model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])


def build_randomized_model_with_random_input_output_layer(nodes, rate):
    image_size = 784  # 28*28
    num_classes = 10  # ten unique digits
    adj_matrix = get_random_layer(image_size, nodes[0], rate)
    first_layer = CustomConnected(nodes[0], adj_matrix, activation='sigmoid', input_shape=(image_size,))
    model.add(first_layer)
    for i in range(1, len(nodes)):
        adj_matrix = get_random_layer(nodes[i-1], nodes[i], rate)
        hidden_layer = CustomConnected(nodes[i], adj_matrix, activation='sigmoid')
        model.add(hidden_layer)

    adj_matrix = get_random_layer(nodes[-1], num_classes, rate)
    final_layer = CustomConnected(num_classes, adj_matrix, activation='softmax')
    model.add(final_layer)
    model.summary()
    model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])


def get_vertex_double_cover(mat):
    n = len(mat[0])
    bi_partite_adj_mat = np.zeros(shape=(2*n, 2*n), dtype=float)
    for x in range (0,2*n):
        if x<n:
            for y in range(0,2*n):
                if y<n:
                    bi_partite_adj_mat[x][y] = 0
                else:
                    bi_partite_adj_mat[x][y] = mat[x%n][y%n]
        else:
            for y in range(0,2*n):
                if y<n:
                    bi_partite_adj_mat[x][y] = mat[y][x%n]
                else:
                    bi_partite_adj_mat[x][y] = 0
    print(f"Printing matrix with shape {bi_partite_adj_mat.shape}")
    # print_square_mat(bi_partite_adj_mat)
    return bi_partite_adj_mat


def print_square_mat(mat):
    n = len(mat[0])
    for i in range(n):
        row = ""
        for j in range(n):
            row = row + str(mat[i][j]) + " "
        print(row)


def generate_random_undirected_matrix(n):
    mat = np.zeros(shape=(n, n), dtype=float)
    for i in range(n):
        for j in range(i+1):
            rand_num = random.randint(0, 2)
            if rand_num == 1:
                mat[i][j] = 1
                mat[j][i] = 1
            else:
                mat[i][j] = 0
                mat[j][i] = 0
    return mat


# Returns the Margulis-Gabber-Galil undirected MultiGraph on `n^2` nodes.
# The undirected MultiGraph is regular with degree `8`.
# The second-largest eigenvalue of the adjacency matrix of the graph
# is at most `5 \sqrt{2}`, regardless of `n`.
def generate_margulis_gabber_galil_graph(n):
    mat = np.zeros(shape=(n*n, n*n), dtype=float)
    for x in range(n):
        for y in range(n):
            # vertex x*n + y
            conn_1_x = (x+y) % n
            conn_2_x = (x-y) % n
            conn_3_x = (x+y+1) % n
            conn_4_x = (x-y+1) % n
            conn_1234_y = y
            conn_5678_x = x
            conn_5_y = (y+x) % n
            conn_6_y = (y-x) % n
            conn_7_y = (y+x+1) % n
            conn_8_y = (y-x+1) % n

            print(f"Vertex: {x*n+y} is connected to: ")
            print(f"{conn_1_x * n + conn_1234_y}")
            print(f"{conn_2_x * n + conn_1234_y}")
            print(f"{conn_3_x * n + conn_1234_y}")
            print(f"{conn_4_x * n + conn_1234_y}")
            print(f"{conn_5678_x * n + conn_5_y}")
            print(f"{conn_5678_x * n + conn_6_y}")
            print(f"{conn_5678_x * n + conn_7_y}")
            print(f"{conn_5678_x * n + conn_8_y}")

            mat[x*n+y][conn_1_x*n+conn_1234_y] += 1
            mat[conn_1_x*n+conn_1234_y][x*n+y] += 1

            mat[x * n + y][conn_2_x * n + conn_1234_y] += 1
            mat[conn_2_x * n + conn_1234_y][x * n + y] += 1

            mat[x * n + y][conn_3_x * n + conn_1234_y] += 1
            mat[conn_3_x * n + conn_1234_y][x * n + y] += 1

            mat[x * n + y][conn_4_x * n + conn_1234_y] += 1
            mat[conn_4_x * n + conn_1234_y][x * n + y] += 1

            mat[x * n + y][conn_5678_x * n + conn_5_y] += 1
            mat[conn_5678_x * n + conn_5_y][x * n + y] += 1

            mat[x * n + y][conn_5678_x * n + conn_6_y] += 1
            mat[conn_5678_x * n + conn_6_y][x * n + y] += 1

            mat[x * n + y][conn_5678_x * n + conn_7_y] += 1
            mat[conn_5678_x * n + conn_7_y][x * n + y] += 1

            mat[x * n + y][conn_5678_x * n + conn_8_y] += 1
            mat[conn_5678_x * n + conn_8_y][x * n + y] += 1

    # Divide by two due to over counting
    for i in range(n*n):
        for j in range(n*n):
            mat[i][j] = mat[i][j] / 2.0
    return mat


def generate_random_d_regular_graph(n, d):
    mat = np.zeros(shape=(n, n), dtype=float)
    degree_list = np.zeros(shape=n, dtype=int)
    for i in range(0, n):
        while degree_list[i] < d:
            rand_num = random.randint(0, n-1)
            if mat[i][rand_num] == 0 and degree_list[rand_num] < d:
                mat[i][rand_num] += 1
                mat[rand_num][i] += 1

                degree_list[rand_num] += 1
                degree_list[i] += 1
        print(f"i= {i} Done with {mat[i]}")
    # print_square_mat(mat)
    return mat


def build_zigzag_graph_with_d_reg_randomized_hidden_layer_model(nodes):
    image_size = 784  # 28*28
    num_classes = 10  # ten unique digits
    first_layer = Dense(units=nodes[0], activation='sigmoid', input_shape=(image_size,))
    model.add(first_layer)
    for i in range(1, len(nodes)):
        g = generate_random_d_regular_graph(128, 8)
        h = generate_random_d_regular_graph(8, 4)
        adj_matrix = generate_zigzag_2d_regular_graph(g,h)
        print(adj_matrix.shape)
        # adjacency_mat = np.zeros((100, 200), dtype=float)
        hidden_layer = CustomConnected(nodes[i], adj_matrix, activation='sigmoid')
        model.add(hidden_layer)
        #model.add(Dense(units=nodes[i], activation='sigmoid', use_bias=False))

    final_layer = Dense(units=num_classes, activation='softmax')
    model.add(final_layer)
    model.summary()
    model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])


def build_d_reg_randomized_hidden_layer_model(nodes, degree):
    image_size = 784  # 28*28
    num_classes = 10  # ten unique digits
    first_layer = Dense(units=nodes[0], activation='sigmoid', input_shape=(image_size,))
    model.add(first_layer)
    for i in range(1, len(nodes)):
        adj_matrix = generate_random_d_regular_graph(1024, 16)
        # adjacency_mat = np.zeros((100, 200), dtype=float)
        hidden_layer = CustomConnected(nodes[i], adj_matrix, activation='sigmoid')
        model.add(hidden_layer)
        # model.add(Dense(units=nodes[i], activation='sigmoid', use_bias=False))

    final_layer = Dense(units=num_classes, activation='softmax')
    model.add(final_layer)
    model.summary()
    model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])


def build_mgg_hidden_layer_model(nodes):
    image_size = 784  # 28*28
    num_classes = 10  # ten unique digits
    first_layer = Dense(units=nodes[0], activation='sigmoid', input_shape=(image_size,))
    model.add(first_layer)
    for i in range(1, len(nodes)):
        adj_matrix = generate_margulis_gabber_galil_graph(int(math.sqrt(nodes[i])))
        # adjacency_mat = np.zeros((100, 200), dtype=float)
        hidden_layer = CustomConnected(nodes[i], adj_matrix, activation='sigmoid')
        model.add(hidden_layer)
        #model.add(Dense(units=nodes[i], activation='sigmoid', use_bias=False))

    final_layer = Dense(units=num_classes, activation='softmax')
    model.add(final_layer)
    model.summary()
    model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])


def generate_zigzag_2d_regular_graph(g, h):
    n = len(g[0])
    m = len(h[0])
    mat = np.zeros(shape=(m*n, m*n), dtype=float)
    for i in range(0, n):
        for j in range(0, m):
            for k in range(0, m):
                mat[i*m+j][i*m+k] = h[j][k]

    d = 0
    for i in range(0,m):
        d = d + h[0][i]

    for i in range(0,n):
        k = 0
        for j in range(0,n):
            if g[i][j] > 0:
                mat[i*m+k][j*m+k] = g[i][j] * d
                k = k+1

    # print_square_mat(mat)
    return mat


def build_mgg_hidden_layer_model(nodes):
    image_size = 784  # 28*28
    num_classes = 10  # ten unique digits
    first_layer = Dense(units=nodes[0], activation='sigmoid', input_shape=(image_size,))
    model.add(first_layer)
    for i in range(1, len(nodes)):
        adj_matrix = generate_margulis_gabber_galil_graph(int(math.sqrt(nodes[i])))
        # adjacency_mat = np.zeros((100, 200), dtype=float)
        hidden_layer = CustomConnected(nodes[i], adj_matrix, activation='sigmoid')
        model.add(hidden_layer)
        #model.add(Dense(units=nodes[i], activation='sigmoid', use_bias=False))

    final_layer = Dense(units=num_classes, activation='softmax')
    model.add(final_layer)
    model.summary()
    model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])

# generate_margulis_gabber_galil_graph(4)

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
node_list = [1024, 1024]
# rate = 0.25
model = Sequential()
# build_model(node_list)
# build_model_with_dropout(node_list, 0.1)
# build_randomized_model(node_list, rate)
# build_mgg_hidden_layer_model(node_list)
# build_zigzag_graph_with_d_reg_randomized_hidden_layer_model(node_list)
build_d_reg_randomized_hidden_layer_model(node_list,8)
# build_randomized_model_with_random_input_layer(node_list, rate)
# build_randomized_model_with_random_input_output_layer(node_list, rate)
csv_logger = CSVLogger("mnist_random_connected_history_log.csv", append=True)
history = model.fit(x_train, y_train, batch_size=32, epochs=16, verbose=False, validation_split=.1,
                        callbacks=[csv_logger])
loss, accuracy = model.evaluate(x_test, y_test, verbose=True)
print(f'Test loss: {loss:.3}')
print(f'Test accuracy: {accuracy:.3}')



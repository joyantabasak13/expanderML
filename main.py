import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Reading the data...")
    data = pd.read_csv('./data/train.csv', sep=",")
    test_data = pd.read_csv('./data/test.csv', sep=",")

    print("Reshaping the data...")
    data_1 = data.drop('label', axis=1)
    labels_1 = data['label']

    data_2 = test_data.drop('label', axis=1)
    labels_2 = test_data['label']

    data_comb = pd.concat([data_1,data_2])
    label_comb = pd.concat([labels_1, labels_2])

    print(data_comb.shape)



    # dataNp = dataFinal.values
    # labelsNp = labels.values
    # test_dataNp = test_data.values

    print("Data is ready")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

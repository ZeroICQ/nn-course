import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import tensorflow as tf
import sys
from tensorflow import keras


def source_function(x):
    y1 = 1 - 0.1*x
    y2 = -0.5 + 1/(1 + math.exp(-75 - 40*x))
    y3 = -0.5 + 1/(1 + math.exp(-2*x))
    y4 = -1 + 1/(1 + np.exp(-170 - 85*x))
    return y1 + y2 + y3 + y4


def main():
    model_path = sys.argv[1]
    model = keras.models.load_model(model_path)
    start_x = -3
    end_x = 3
    step_x = .01
    x = np.arange(start_x, end_x, step_x)
    y = [source_function(x) for x in x]
    ax = sns.lineplot(x, y, label="source function", color="red")
    prediction_y = model.predict(x).flatten()
    sns.lineplot(x, prediction_y, label="prediction", color="blue")
    plt.show()


if __name__ == '__main__':
    main()
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
sns.set()


def source_function(x):
    y1 = 1 - 0.1*x
    y2 = -0.5 + 1/(1 + math.exp(-75 - 40*x))
    y3 = -0.5 + 1/(1 + math.exp(-2*x))
    y4 = -1 + 1/(1 + np.exp(-170 + 85*x))
    return y1 + y2 + y3 + y4


def learn(x, y):

    model = keras.Sequential([
        # keras.layers.InputLayer(input_shape=1),
        keras.layers.Dense(128, activation='softmax', input_shape=(1,)),
        keras.layers.Dense(1, activation='linear'),
    ])
    lr = 1e-2
    epochs = 5000
    sgd = keras.optimizers.SGD(lr=lr, decay=lr/epochs, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd,
                  loss='mean_absolute_error',
                  metrics=['accuracy'])
    #'./models/weights{epoch:08d}_loss{loss:.5f}.h5'
    mc = keras.callbacks.ModelCheckpoint('./models/best_loss.h5',
                                         save_weights_only=False, period=500,
                                         save_best_only=True, monitor='loss')
    model.fit(x, y, epochs=epochs, callbacks=[mc])
    test_loss, test_acc = model.evaluate(x,  y, verbose=2)
    # model.save("model"+epochs+".save")
    print('\nTest accuracy:', test_acc)
    prediction_y = model.predict(x).flatten()
    sns.lineplot(x, prediction_y, label="prediction", color="blue")


def main():
    start_x = -3
    end_x = 3
    step_x = .01
    x = np.arange(start_x, end_x, step_x)
    y = [source_function(x) for x in x]
    ax = sns.lineplot(x, y, label="source function", color="red")
    ax.set(xlabel="x", ylabel="y")
    learn(x, y)
    plt.show()

if __name__ == '__main__':
    main()
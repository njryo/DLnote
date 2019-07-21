import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorboard.plugins.hparams import api as hp


def make_toy_dataset(num_data):
    X = np.random.randn(num_data, 3)
    y = 3 * X[:, 0] - 2 * X[:, 1]**3 + 2 * \
        X[:, 2]**2 + 0.5 * np.random.randn(num_data)
    y = y[:, np.newaxis]
    return X, y


def run_one_experiment(X_train, y_train, validation_data, hparams, logdir):
    model = keras.models.Sequential([
        keras.layers.Dense(hparams["num_units"],
                           activation="relu", input_dim=3),
        keras.layers.Dense(1)
    ])

    optimizer = keras.optimizers.SGD(
        hparams["learning_rate"], momentum=hparams["momentum"])

    model.compile(optimizer, loss="mse", metrics=["mae"])

    tb_callback = keras.callbacks.TensorBoard(logdir)
    hp_callback = hp.KerasCallback(logdir, hparams)

    model.fit(X_train, y_train, batch_size=32, epochs=30,
              callbacks=[tb_callback, hp_callback],
              validation_data=validation_data)


def run_all_experiments():
    X_train, y_train = make_toy_dataset(1000)
    X_test, y_test = make_toy_dataset(200)

    experiment_num = 0

    for num_units in [16, 32]:
        for learning_rate in [0.01, 0.05]:
            for momentum in [0.0, 0.1, 0.2]:

                logdir = os.path.join("logs", "hptuning", "run{}".format(experiment_num))
                hparams = {
                    "num_units": num_units,
                    "learning_rate": learning_rate,
                    "momentum": momentum
                }

                run_one_experiment(X_train, y_train, (X_test, y_test), hparams, logdir)

                experiment_num += 1

if __name__ == '__main__':
    run_all_experiments()

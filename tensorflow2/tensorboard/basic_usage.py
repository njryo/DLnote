import numpy as np
from tensorflow import keras


def make_toy_dataset(num_data):
    X = np.random.randn(num_data, 3)
    y = 3 * X[:, 0] - 2 * X[:, 1]**3 + 2 * X[:, 2]**2 + 0.5 * np.random.randn(num_data)
    y = y[:, np.newaxis]
    return X, y

def build_model():
    model = keras.models.Sequential([
        keras.layers.Dense(10, activation="relu", input_dim=3),
        keras.layers.Dense(10, activation="relu"),
        keras.layers.Dense(1)
    ])

    return model


if __name__ == '__main__':
    N_train = 1000
    N_test = 200
    num_epochs = 30
    learning_rate = 0.01
    batch_size = 32
    logdir = "logs\\basic"

    X_train, y_train = make_toy_dataset(N_train)
    X_test, y_test = make_toy_dataset(N_test)

    model = build_model()
    model.compile(
        keras.optimizers.SGD(learning_rate),
        loss="mse",
        metrics=["mse", "mae"]
        )

    tb_callback = keras.callbacks.TensorBoard(logdir)

    model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        verbose=1,
        callbacks=[tb_callback],
        validation_data=(X_test, y_test)
    )

# このスクリプトを実行で、tensorboardのscalarsの項にtrain, validationそれぞれの
# epoch_loss, epoch_mae, epoch_mse(今回の場合は==epoch_loss)が記録される（名前に注意）。
# 加えて、graphとprofileも記録される。

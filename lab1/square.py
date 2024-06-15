from impl import MLP, Layer, Normalizer, linear, sigmoid
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Square simple


df_train = pd.read_csv("mio1/regression/square-simple-training.csv")
df_test = pd.read_csv("mio1/regression/square-simple-test.csv")

X_train = df_train["x"].values
y_train = df_train["y"].values

X_test = df_test["x"].values
y_test = df_test["y"].values


def build_model(b1=-1.4, b2=2, w3=15, w4=-15, b3=9):
    W1 = np.array([1, 1, 0, 0, 0]).reshape(5, 1)
    B1 = np.array([b1, b2, 0, 0, 0])

    W2 = np.array([[w3, w4, 0, 0, 0]])
    B2 = np.array([b3])

    model = MLP(
        [
            Layer(1, 5, weights=W1, bias=B1, activation_function=sigmoid),
            Layer(5, 1, weights=W2, bias=B2, activation_function=linear),
        ]
    )
    return model


if __name__ == "__main__":
    print("Optimized model with 1 hidden layer 5 neurons")
    # 1 hidden 5 neurons
    x_normalizer = Normalizer()
    x = x_normalizer.fit_transform(X_train)

    # Params found with grid search
    params = {
        "b1": -2.0,
        "b2": 2.5555555555555554,
        "w3": 1415.7894736842106,
        "w4": -1305.2631578947369,
        "b3": 919.1919191919192,
    }
    model = build_model(**params)
    train_loss = model.forward_loss(x, y_train)
    print(f"MSE on training dataset: {train_loss}")
    test_loss = model.forward_loss(x_normalizer.transform(X_test), y_test)
    print(f"MSE on test dataset: {test_loss}")

    
    X_test_scaled = x_normalizer.transform(X_test)
    plt.plot(X_test_scaled, y_test, "ro", label = "Dane testowe")
    plt.plot(X_test_scaled, model.forward(X_test_scaled), "bo", label = "Predykcje")
    plt.legend()
    plt.title("Predykcje modelu na danych testowych dla square-simple")
    plt.show()

    # 1 hidden 10 neurons
    print("Raw model with 1 hidden layer 10 neurons")
    mlp = MLP(
        [
            Layer(1, 10, activation_function=sigmoid),
            Layer(10, 1, activation_function=linear),
        ]
    )
    print(mlp.forward_loss(X_test, y_test))

    print("Row model with 2 hidden layer 5 neurons each")
    mlp = MLP(
        [
            Layer(1, 5, activation_function=sigmoid),
            Layer(5, 5, activation_function=sigmoid),
            Layer(5, 1, activation_function=linear),
        ]
    )
    print(mlp.forward_loss(X_test, y_test))


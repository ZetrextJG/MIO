from impl import MLP, Layer, linear, sigmoid
import pandas as pd
import numpy as np

# Steps large
df_train = pd.read_csv("../mio1/regression/steps-large-training.csv")
df_test = pd.read_csv("../mio1/regression/steps-large-test.csv")

X_train = df_train["x"].values
y_train = df_train["y"].values

X_test = df_test["x"].values
y_test = df_test["y"].values


def build_model(b1=0, b2=0, b3=0, w1=1, w2=1, w3=1, b4=0):
    W1 = np.array([1000, 1000, 1000, 0, 0]).reshape(5, 1)
    B1 = np.array([b1, b2, b3, 0, 0])

    W2 = np.array([[w1, w2, w3, 0, 0]])
    B2 = np.array([b4])

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

    # Params found with grid search
    params = {
        "b1": -499.44444444444446,
        "b2": -1500.5555555555557,
        "b3": 500.55555555555554,
        "b4": -80,
        "w1": 80.55555555555556,
        "w2": 80.55555555555556,
        "w3": 79.44444444444444,
    }

    model = build_model(**params)
    train_loss = model.forward_loss(X_train, y_train)
    print(f"MSE on training dataset: {train_loss}")
    test_loss = model.forward_loss(X_test, y_test)
    print(f"MSE on test dataset: {test_loss}")

    # 1 hidden 10 neurons
    print("Raw model with 1 hidden layer 10 neurons")
    mlp = MLP(
        [
            Layer(1, 10, activation_function=sigmoid),
            Layer(10, 1, activation_function=linear),
        ]
    )
    print(mlp.forward_loss(X_test, y_test))

    print("Raw model with 2 hidden layer 5 neurons each")
    mlp = MLP(
        [
            Layer(1, 5, activation_function=sigmoid),
            Layer(5, 5, activation_function=sigmoid),
            Layer(5, 1, activation_function=linear),
        ]
    )
    print(mlp.forward_loss(X_test, y_test))

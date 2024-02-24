import matplotlib.pyplot as plt
import numpy as np
from shiny.express import ui, input, render
from shiny import run_app
import pandas as pd
from impl import Layer, MLP, linear, sigmoid

df_train = pd.read_csv("../mio1/regression/square-simple-training.csv")
df_test = pd.read_csv("../mio1/regression/square-simple-test.csv")

X_train = df_train["x"].values
y_train = df_train["y"].values


with ui.sidebar():
    ui.input_slider("w1", "w1", -10, 10, 0, step=0.1)
    ui.input_slider("w2", "w2", -10, 10, 0, step=0.1)
    ui.input_slider("b1", "b1", -10, 10, 0, step=0.1)
    ui.input_slider("b2", "b2", -10, 10, 0, step=0.1)

    ui.input_slider("w3", "w3", -20, 20, 0, step=0.1)
    ui.input_slider("w4", "w4", -20, 20, 0, step=0.1)
    ui.input_slider("b3", "b3", -10, 30, 0, step=0.1)


def model():
    weights1 = np.array([input.w1(), input.w2(), 0, 0, 0]).reshape(5, 1)
    bias1 = np.array([input.b1(), input.b2(), 0, 0, 0])
    layer1 = Layer(1, 5, weights=weights1, bias=bias1, activation_function=sigmoid)

    weights2 = np.array([[input.w3(), input.w4(), 0, 0, 0]])
    bias2 = np.array([input.b3()])
    layer2 = Layer(5, 1, weights=weights2, bias=bias2, activation_function=linear)

    mlp = MLP([layer1, layer2])
    return mlp


@render.plot(alt="Training plot")
def plot():
    print("replot")
    mod = model()
    normalize_x = (X_train - X_train.mean()) / X_train.std()
    normlize_y = (y_train - y_train.mean()) / y_train.std()
    plt.plot(normalize_x, normlize_y, "go")
    plt.plot(normalize_x, mod.forward(normalize_x), "r+")
    # print(mod.forward_loss(X_train, y_train))


if __name__ == "__main__":
    run_app()

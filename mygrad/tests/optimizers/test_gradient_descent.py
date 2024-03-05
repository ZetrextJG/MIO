import numpy as np

from mygrad.optimizers.sgd import GradientDescent
from mygrad.losses import AbsoluteDifferenceLoss
import mygrad.components as mc


def test_gradient_descent_optimizer_pass():
    model = mc.Sequential(
        mc.Linear(
            2,
            2,
            weights=np.array([[1, 1], [1, 1]], dtype=np.float64),
            bias=np.array([-1.5, 0], dtype=np.float64),
        ),
        mc.ThresholdJump(),
        mc.Linear(
            2,
            1,
            weights=np.array([[1, -1.5]], dtype=np.float64),
            bias=np.array([1], dtype=np.float64),
        ),
        mc.ThresholdJump(),
    )
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[1], [0], [0], [1]])
    loss = AbsoluteDifferenceLoss()
    optimizer = GradientDescent(model.parameters(), learning_rate=0.01)

    for _ in range(100):
        y_pred = model.forward(x)
        loss_value, loss_grad = loss.both(y_pred, y)
        model.backward(loss_grad)
        optimizer.step()
        optimizer.zero_grad()

    assert np.allclose(y_pred, y, atol=1e-3)

import numpy as np

x = np.array([1, 2, 3])
W = np.array(
    [
        [1, 2, 3],
        [-1, -2, -3],
    ]
)

b = np.array([3, 3])

print(x)
print(W)
print(b)

y = x @ W.T + b
print(y)


X = np.stack([x, x])
Y = X @ W.T + b
print(Y)


a = np.ones((3, 3))
b = np.array([1, 2, 3])
print(a + b)

# This is why I use the row vector as the default vector

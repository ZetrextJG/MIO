from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


# x = np.arange(0, 1, 0.01)
# y = np.cos(x)
# y2 = np.cos(np.pi / 2 * x)
# y3 = np.exp(-(x**2) / 2)
#
# plt.title("Decay functions used in this project")
# plt.xlabel("Fraction of completed epochs")
# plt.ylabel("Decay factor")
#
# plt.plot(x, y, label="cosine")
# plt.plot(x, y2, label="full_cosine")
# plt.plot(x, y3, label="exponential")
# plt.legend()
# plt.show()
#
#
# d = np.arange(0, 5, 0.01)
# RADIUS = 1
# y_c = (d <= RADIUS * np.exp(-1 / 50)) * 1
# y_g = np.exp(-((d / RADIUS) ** 2))
# y_m = (2 - 4 * (d / RADIUS) ** 2) * np.exp(-((d / RADIUS) ** 2))
#
# plt.title("Neighborhood used in this project")
# plt.xlabel("Distance from the winner")
# plt.ylabel("Neighborhood factor")
#
# plt.plot(d, y_c, label="circle")
# plt.plot(d, y_g, label="gaussian")
# plt.plot(d, y_m, label="mexican hat")
#
# plt.legend()
#
# plt.show()


df = pd.read_json("./hexagon.json", lines=True, orient="records")
print(df.head())

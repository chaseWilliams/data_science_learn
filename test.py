import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
my_data = np.genfromtxt('Iris.csv', delimiter=',')
print(my_data)
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
# plt.show()
"""
Make two sub-plots, one with sepal length/width, other with
petal length/width. Than, compare them both on one plot. Colorful, clear plots.
"""

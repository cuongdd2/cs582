import numpy as np
import matplotlib.pyplot as plt

# height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

# plt.show()

one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)

w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)

w1 = w[0][0]
w2 = w[1][0]

x0 = np.linspace(145, 185)
y0 = w1 + w2 * x0

plt.plot(X.T, y.T, "ro")
plt.plot(x0, y0)
plt.axis([140, 190, 45, 75])
plt.xlabel("Height")
plt.ylabel("Weight")

from sklearn.linear_model import LinearRegression

regr = LinearRegression(fit_intercept=False)
regr.fit(Xbar, y)
print(regr.coef_)
print(w.T)
plt.show()

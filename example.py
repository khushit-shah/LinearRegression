import LinearRegression
import numpy as np

X = np.array([1,2,3,4,5,6,7,8,9,10])

# Y = 0 + 1X
Y = np.array([1,2,3,4,5,6,7,8,9,10])

modal = LinearRegression.LinearRegression()

modal.train(X,Y)

print(modal.predict(14))

import matplotlib.pyplot as plt
import numpy as np

def f(x1,x2):
    return np.power(np.subtract(x2,x1),4) + 8*np.multiply(x1,x2) - x1 + x2 +3

x1 = np.linspace(-1,1,60,endpoint=True).reshape((60,1))

y11 = [f(i,-1) for i in x1]
y12 = [f(i,1) for i in x1]

plt.plot(x1,y11,label="x2 = -1")
plt.plot(x1,y12,label="x2 = 1")
plt.xlabel("x1")
plt.ylabel("Test function")
plt.title("Test function vs x1 for a fixed x2")
plt.legend()
plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as lr

xs = pd.read_csv("F:\Ishu\Modern Physics\pc.csv")

x = np.array(xs["x"]).reshape(-1,1)
y = np.array(xs["y"])
y_cal = []

ols = lr().fit(x, y)
ols_slope = float(ols.coef_)
ols_intc = ols.intercept_

for i in range(len(x)):
    y_cal.append(ols_slope*x[i] + ols_intc)

plt.plot(x,y_cal,c = "blue",label = "Fitted line")
plt.scatter(x,y,c = "maroon",label = "Reading points")
plt.ylabel("Kinetic Energy (in eV)")
plt.xlabel("Frequency $\\nu$ (in Hz)")
plt.grid(ls = "--")
plt.legend()
plt.show()
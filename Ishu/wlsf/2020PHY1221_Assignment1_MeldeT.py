import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def input_table():
    xs = pd.read_csv("/home/ishmeet/BSc physics/Waves and optics/Lab/Melde Experiment/Melde(T).csv")
    xs["XY"] = xs["X"] * xs["Y"]
    xs["X^2"] = xs["X"].apply(np.square)
    xs.head()
    # print(xs)
    return xs

def calculations(xs, n):
    # SLOPE
    m = ((xs["X"].sum() * xs["Y"].sum()) - (n * xs["XY"].sum())) / (((xs["X"].sum()) ** 2) - (n * xs["X^2"].sum()))
    print("\nThe value of the slope is", m)
    # INTERCEPT
    c = ((xs["X^2"].sum() * xs["Y"].sum()) - (xs["X"].sum() * xs["XY"].sum())) / ((n * xs["X^2"].sum()) - ((xs["X"].sum()) ** 2))       
    print("\nThe value of the intercept is", c)
    # Y Calculated
    list_1 = []
    x = np.array(xs["X"])
    for i in range(n):
        y_cal = m * x[i] + c
        list_1.append(y_cal)
    print("\nCalculated Y is:\n", list_1, "\n")
    # ERROR IN Y_CAL/ RESIDUAL SUM
    list_2 = []
    y = np.array(xs["Y"])
    for j in range(n):
        e = y[j] - list_1[j]
        list_2.append(e)
    print("\nErrors in y_cal are:\n", list_2)
    # ERROR SUM
    error = sum(list_2)
    print("\nResidual sum is:", error)
    # ERROR SUM SQUARE
    list_3 = []
    for j in range(n):
        e = np.square(y[j] - list_1[j])
        list_3.append(e)
    error_square = sum(list_3)
    print("\nResidual sum squares is:", error_square)
    # Standard error in slope
    s = np.sqrt(error_square / (n - 2))
    SS_xx = ((xs["X"].apply(np.square)).sum() - ((np.square(xs["X"].sum()))/n))
    S_m = s / (np.sqrt(SS_xx))
    print("\nStandard Error in slope(m) is",S_m)
    # Standard error in intercept
    S_c = (S_m * ((np.sqrt(xs["X"].apply(np.square)).sum())/n))
    print("\nStandard Error in intercept(c) is",S_c)
    # Cofficent of determination
    list_4 = []
    y_mean = float(xs["Y"].mean())
    for k in range(n):
        diff = np.square(y[k] - y_mean)
        list_4.append(diff)
    tss = sum(list_4)
    print("\nTotal sum of squares(tss):",tss)
    R_sqr = (tss - error_square) / tss
    print("\nCofficent of determination (R_sqr) for the given data is:", R_sqr)
    # Cofficient of correlation
    print("\nCorrelation coefficient for the given data is:", np.sqrt(R_sqr))
    return list_1

def graph(cal, xs):
    x = np.array(xs["X"])
    y = np.array(xs["Y"])
    plt.style.use("dark_background")
    plt.grid(True)
    plt.plot(x,cal,c = "blue")
    plt.scatter(x,y,c = "red")
    plt.xlabel("Tension ($T$)")
    plt.ylabel("Wavelength square ($\lambda^2$)")
    plt.title("Ishmeet Singh\n2020PHY1221\nTransverse Waves")
    plt.show()

if __name__ == "__main__":
    n = int(input("\nEnter the number of readings (Enter 7): "))
    xs = input_table()
    print("\nInput table is:\n", xs, "\n")
    cal = calculations(xs, n)
    graph(cal, xs)

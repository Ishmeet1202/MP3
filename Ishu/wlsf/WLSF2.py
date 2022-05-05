from Fitting import *
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    x_vals = pd.read_csv(r"C:\Users\parmm\OneDrive\Desktop\wlsf\data.csv", usecols = [1])
    x_vals = (x_vals.to_numpy()).flatten()
    df = pd.read_csv(r"C:\Users\parmm\OneDrive\Desktop\wlsf\data.csv", usecols = range(2, 12))
    df = df.to_numpy()

    y_mean = np.array([])
    y_std_error = np.array([])
    for i in range(len(df[0])):
        mean = np.mean(df[i])
        y_mean = np.append(y_mean, mean)
        var = np.var(df[i])
        std_error = (4*mean**2)*var/len(df[0])
        # print(std_error)
        y_std_error = np.append(y_std_error, std_error)

    x = x_vals.reshape(-1, 1)
    y = (y_mean**2).reshape(-1, 1)
    weights = 1/y_std_error

    xx = np.linspace(8*min(x)/10, 11*max(x)/10, 100)

    y_ols, ols_slope, ols_intc, ols = lsf(x, y, xx)
    y_wls, wls_slope, wls_intc, wls = wlsf(x, y, weights, xx)
    
    errors_ols = errors(ols, x, y, ols_slope, ols_intc)
    errors_wls = errors(wls, x, y, wls_slope, wls_intc, weights)
    
    print("\n\nFitting Parameters")
    print("\nSlope of OLS Fitted Line\t\t\t=\t{:.7e}\nStandard Error of Slope in OLS Fitted Line\t=\t{:.7e}\nIntercept of OLS Fitted Line\t\t\t=\t{:.7e}\nStandard Error of Intercept in OLS Fitted Line\t=\t{:.7e}\nSum of Residuals for OLS Fitted Line\t\t=\t{:.7}\nSum of Residuals Square for OLS Fitted Line\t=\t{:.7}\nCoefficient of Determination of OLS Fitted Line\t=\t{:.7}\nCorrelation Coefficient of OLS Fitted Line\t=\t{:.7}\n".format(float(ols_slope), float(errors_ols[0]), float(ols_intc), float(errors_ols[1]), float(errors_ols[2]), float(errors_ols[3]), float(errors_ols[4]), float(errors_ols[4]**0.5)))
    print("\nSlope of WLS Fitted Line\t\t\t=\t{:.7e}\nStandard Error of Slope in WLS Fitted Line\t=\t{:.7e}\nIntercept of WLS Fitted Line\t\t\t=\t{:.7e}\nStandard Error of Intercept in WLS Fitted Line\t=\t{:.7e}\nSum of Residuals for WLS Fitted Line\t\t=\t{:.7}\nSum of Residuals Square for WLS Fitted Line\t=\t{:.7}\nCoefficient of Determination of WLS Fitted Line\t=\t{:.7}\nCorrelation Coefficient of WLS Fitted Line\t=\t{:.7}\nChi Square\t\t\t\t\t=\t{:.5e}\n".format(float(wls_slope), float(errors_wls[0]), float(wls_intc), float(errors_wls[1]), float(errors_wls[2]),  float(errors_wls[3]), float(errors_wls[4]), float(errors_wls[4]**0.5), float(errors_wls[5])))

    k = (4*np.pi**2)/wls_slope
    m = wls_intc*k/(4*np.pi**2)
    error_k = errors_wls[0]*k/wls_slope
    error_m = (errors_wls[1]/wls_intc + error_k/k)*m
    print("\nm\t\t=\t{:.5e} g\nError in m\t=\t{:.5e} g\nk\t\t=\t{:.5e} N/m\nError in k\t=\t{:.5e} N/m\n".format(float(m), float(error_m), float(k), float(error_k)))
    
    # Plots and Scatters
    plt.scatter(x, y, marker = 'o', color = "xkcd:sky blue")    
    plt.plot(xx, y_ols, color = "xkcd:purple", linestyle = '--', linewidth = 1, label = "OLS Fitted Line")
    plt.plot(xx, y_wls, color = "xkcd:green", linewidth = 1, label = "WLS Fitted Line")
    print(y_std_error)
    plt.title("Samarth Jain and Jagjyot Singh\nLinear Regression for Spring Constant")       #NAME OF EXPERIMENT
    plt.ylabel("Time Period - $T^2$ ($s^2$)\n")         #Y LABEL
    plt.xlabel("Mass (g.)\n")                           #X LABEL
    plt.grid(True)

    plt.legend()
    plt.show()
import numpy as np
from sklearn.linear_model import LinearRegression as lr

def lsf(x, y, xx):
    ols = lr().fit(x, y)
    ols_slope = float(ols.coef_)
    ols_intc = ols.intercept_
    y_ols = ols_slope * xx + ols_intc
    return(y_ols, ols_slope, ols_intc, ols)

def wlsf(x, y, weights, xx):
    wls = lr().fit(x, y, sample_weight = weights)
    wls_slope = wls.coef_
    wls_intc = wls.intercept_
    y_wls = wls_slope*xx + wls_intc
    return(y_wls, wls_slope, wls_intc, wls)

def sum(x, y, m, c):    # Sum of Residuals/Sum of Res sq./Sum x**2
    sum_res = 0
    sum_sq_res = 0
    sum_xsq = 0			
    for i in range(len(x)):
        z = m*x[i] + c
        d = (y[i] - z)			# Residuals
        d2 = d**2               # Residuals Sq.
        sum_res += d
        sum_sq_res += d2
        sum_xsq += (x[i])**2
    sum_m = sum_xsq**0.5
    return(sum_res, sum_sq_res, sum_m)

def exp_sum_sq(x):    # Explained Sum Squares
    t = 0
    standard_error = 0
    mean_x = np.mean(x)
    while(t < len(x)):
        standard_error += (x[t] - mean_x)**2
        t += 1
    ess = np.sqrt(standard_error)
    return(ess)

def errors(fit, x, y, slope, intercept, weight = None):
    sum_res, sum_sq_res, sum_xsq = sum(x, y, slope, intercept)
    A = np.sqrt((sum_sq_res)/(len(x)-2))
    B = exp_sum_sq(x)

    err_slope = A/B
    err_intc = A * (sum_xsq)/((len(x)**0.5) * B)
    r_sq = fit.score(x, y)
    chi_sq = None
    error_bars = None 

    # Calculating Chi^2 and Error Bars
    if (weight is not None):
        yerr = []
        yf = [j*slope + intercept for j in x]
        for i in range(len(y)):
            yerr.append((float(yf[i] - y[i])))
        sum_y = 0
        for j in range(len(yerr)):
            sum_y += float(yerr[j]**2) * weight[j]
        chi_sq = sum_y
        error_bars = weight/np.sqrt(len(x))

    return(err_slope, err_intc, sum_res, sum_sq_res, r_sq, chi_sq, error_bars)

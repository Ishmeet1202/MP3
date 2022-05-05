import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rc = pd.read_csv("F:\Ishu\Modern Physics\\rc.csv")

nu = np.array(rc["nu"])
vc = np.array(rc["vc"])
vr = np.array(rc["vr"])

plt.title("RC circuit experiment\n$\\nu$ Vs $V_c$\nR = 1k $\Omega$ , C = 1 $\mu$F")
plt.xlabel("Frequency $\\nu$ (in Hz)")
plt.ylabel("Voltage across capacitor (in Volt)")
plt.plot(nu,vc)
plt.scatter(nu,vc,marker = ".")
plt.grid(ls = "--")
plt.show()
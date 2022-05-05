import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

yellow = pd.read_csv("F:\Ishu\Modern Physics\yellow.csv")
blue = pd.read_csv("F:\Ishu\Modern Physics\\blue.csv")

# HIGH INTENSITY
v1_y = np.array(yellow["v1"])
i1_y = np.array(yellow["i1"])
v1_b = np.array(blue["v1"])
i1_b = np.array(blue["i1"])

# MEDIUM INTENSITY
v2_y = np.array(yellow["v2"])
i2_y = np.array(yellow["i2"])
v2_b= np.array(blue["v2"])
i2_b = np.array(blue["i2"])

# LOW INTENSITY
v3_y = np.array(yellow["v3"])
i3_y = np.array(yellow["i3"])
v3_b= np.array(blue["v3"])
i3_b = np.array(blue["i3"])

plt.title("Light Yellow Colour (V Vs I)")
plt.ylabel("Voltage (V) (in Volts)")
plt.xlabel("Current (I) (in mA)")
plt.plot(v1_y,i1_y,label = "High Intensity")
plt.plot(v2_y,i2_y,label = "Medium Intensity")
plt.plot(v3_y,i3_y,label = "Low Intensity")
plt.grid(ls = "--")
plt.legend()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv("data-lsf.csv")

T=['T1(s)','T2(s)','T3(s)','T4(s)','T5(s)','T6(s)','T7(s)','T8(s)','T9(s)','T10(s)']
df['y_mean']=df[T].mean(axis=1)


L1=df['Mass (g)'].values.tolist()
x=np.array(L1)

L2=df['y_mean'].values.tolist()
y=np.array(L2)**2           #T**2------This is your y

df['y**2']=list(y)
df['y_sqr_var']=df['y**2'].var(axis=0)

ymean=sum(y)/(len(y))       #ymean
var=(np.sum(y-ymean)**2)/(len(y)-1)     #complete variance
print("var", var)

'''L3=df['variance'].values.tolist()
var=np.array(L3)'''

print(df.loc[:,['Mass (g)','y_mean']])


w=1/(np.array(L1))

slope=(np.sum(w*x*y)*np.sum(w) - np.sum(w*x)*np.sum(w*y))/(np.sum(w)*np.sum(w*(x**2)) - np.sum(w*x)**2)
print("SLOPE: ",slope)

intercept=(np.sum(w*(x**2))*np.sum(w*y) - np.sum(w*x)*np.sum(w*x*y))/(np.sum(w)*np.sum(w*(x**2)) - np.sum(w*x)**2)
print("INTERCEPT: ",intercept)

ycal=slope*x + intercept

df['ycal']=list(ycal)
df['y_cal_var']=df['ycal'].var(axis=0)
df['weight']=list(w)
print(df)

y_var=df['y_sqr_var'].values.tolist()
y_cal_var=df['y_cal_var'].values.tolist()

plt.scatter(x,y)
plt.plot(x,ycal,linestyle='dashdot',color='red')
plt.grid()
plt.show()

plt.scatter(x,y_var)
plt.scatter(x,y_cal_var,color='green')
plt.show()

sigma_m=np.sqrt(np.sum(w)/(np.sum(w)*np.sum(w*(x**2)) - np.sum(w*x)**2))
print("Error in Slope: ",sigma_m)

sigma_c=np.sqrt(np.sum(w*(x**2))/(np.sum(w)*np.sum(w*(x**2)) - np.sum(w*x)**2))
print("Error In Intercept: ",sigma_c)

r=sum(w*(x-np.mean(x))*(y-ymean))/(np.sqrt(sum(w*((x-np.mean(x))**2))*sum(w*((y-np.mean(y))**2))))
print("Correleation Coefficient: ",r)

print("Sum Of Residuals: ")
print(np.sum(y-ycal))
print(np.sum((y-ycal)**2))
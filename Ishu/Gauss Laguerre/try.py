import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import integrate
from sympy import *
from sympy import simplify
import scipy 
from Integration import MySimp
from Integration import MyLaguQuad

#(c)

def new_simp(f,a,R0,R_max,tol):
    j=MySimp(f,a,R0,2,key1=True,N_max=10**8,key2=True,tol=0.1e-5)
    lis=[]
    R_a=[]
    w=0
    a_a=[]
    while R0<=R_max:
        j=MySimp(f,a,R0,2,key1=True,N_max=10**8,key2=True,tol=0.1e-5)
        #j=MySimp(f,a,R0,2,key1=False)
        lis.append(j[0])
        R_a.append(R0)
        a_a.append(a)
        if len(lis)>=2:
            if lis[-1]<=0.1e-5:
                err=abs(lis[-1]-lis[-2])
            else:
                err=abs((lis[-1]-lis[-2])/lis[-1])
            if err<=tol:
                w=1
                break
            else:
                pass
        R0=10*R0
    if w==0:
            s=("R_max reached without achieving required tolerance")
    elif w==1:
             s="Given tolerance achieved with R=",R_a[-1]
    return lis[-1],R_a[-1],s,lis,R_a,a_a     #returning integral,number of intervals and message

'''
#Q3(b)
#(i)
n=2
f_x=["1","x","x**2","x**3","x**4","x**5"]
Calc=[]
Exact=[1,1,2,6,24,120]
f=eval("lambda x:"+input("Enter the value of the FUNCTION 1 F(x): ")) #Defining the function to be used for evaluation
Calc.append(MyLaguQuad(f, n))
f=eval("lambda x:"+input("Enter the value of the FUNCTION 2 F(x): ")) #Defining the function to be used for evaluation
Calc.append(MyLaguQuad(f, n))
f=eval("lambda x:"+input("Enter the value of the FUNCTION 3 F(x): ")) #Defining the function to be used for evaluation
Calc.append(MyLaguQuad(f, n))
f=eval("lambda x:"+input("Enter the value of the FUNCTION 4 F(x): ")) #Defining the function to be used for evaluation
Calc.append(MyLaguQuad(f, n))
f=eval("lambda x:"+input("Enter the value of the FUNCTION 5 F(x): ")) #Defining the function to be used for evaluation
Calc.append(MyLaguQuad(f, n))
f=eval("lambda x:"+input("Enter the value of the FUNCTION 6 F(x): ")) #Defining the function to be used for evaluation
Calc.append(MyLaguQuad(f, n))
data={"f(x)":f_x,"Calculated":Calc,"Exact":Exact}
print()
print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
print()
print("METHOD USED : Gauss Laguerre quadrature (TWO POINT)")
print(pd.DataFrame(data))
print()
print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
print()


n=4
f_x=["1","x","x**2","x**3","x**4","x**5","x**6","x**7","x**8","x**9"]
Calc=[]
Exact=[1,1,2,6,24,120,720,5040,40320,362880]
f=eval("lambda x:"+input("Enter the value of the FUNCTION 1 F(x): ")) #Defining the function to be used for evaluation
Calc.append(MyLaguQuad(f, n))
f=eval("lambda x:"+input("Enter the value of the FUNCTION 2 F(x): ")) #Defining the function to be used for evaluation
Calc.append(MyLaguQuad(f, n))
f=eval("lambda x:"+input("Enter the value of the FUNCTION 3 F(x): ")) #Defining the function to be used for evaluation
Calc.append(MyLaguQuad(f, n))
f=eval("lambda x:"+input("Enter the value of the FUNCTION 4 F(x): ")) #Defining the function to be used for evaluation
Calc.append(MyLaguQuad(f, n))
f=eval("lambda x:"+input("Enter the value of the FUNCTION 5 F(x): ")) #Defining the function to be used for evaluation
Calc.append(MyLaguQuad(f, n))
f=eval("lambda x:"+input("Enter the value of the FUNCTION 6 F(x): ")) #Defining the function to be used for evaluation
Calc.append(MyLaguQuad(f, n))
f=eval("lambda x:"+input("Enter the value of the FUNCTION 7 F(x): ")) #Defining the function to be used for evaluation
Calc.append(MyLaguQuad(f, n))
f=eval("lambda x:"+input("Enter the value of the FUNCTION 8 F(x): ")) #Defining the function to be used for evaluation
Calc.append(MyLaguQuad(f, n))
f=eval("lambda x:"+input("Enter the value of the FUNCTION 9 F(x): ")) #Defining the function to be used for evaluation
Calc.append(MyLaguQuad(f, n))
f=eval("lambda x:"+input("Enter the value of the FUNCTION 10 F(x): ")) #Defining the function to be used for evaluation
Calc.append(MyLaguQuad(f, n))

data={"f(x)":f_x,"Calculated":Calc,"Exact":Exact}
print()
print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
print()
print("METHOD USED : Gauss Laguerre quadrature (FOUR POINT)")
print(pd.DataFrame(data))
print()
print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
print()
'''

#(ii)
I1_a=[]
I2_a=[]
n_a=[2,4,8,16,32,64,128]
f1=lambda x : 1/(1+x**2)
f2=lambda x : np.exp(x)/(1+x**2)
for n in n_a:
    I1_a.append(MyLaguQuad(f1, n))
    I2_a.append(MyLaguQuad(f2, n))

DataOut = np.column_stack((n_a,I1_a,I2_a))
np.savetxt("quad-lag-1092", DataOut,delimiter=',')

print("#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#")
print()
data={"n":n_a,"I1":I1_a,"I2":I2_a}
print(pd.DataFrame(data))


a=0
R0=10
R_max=10**6
tol=0.1e-7
F1=lambda x : np.exp(-1*x)/(1+x**2)
F2=lambda x : 1/(1+x**2)
s1=new_simp(F1,a,R0,R_max,tol)
s2=new_simp(F2,a,R0,R_max,tol)

#d
print()
print("#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#")
print()
print("RESULTS USING SIMPSSON METHOD")
print("Tolerance for MYSimp defined in MyIntegration Module = 0.1e-5")
print("Tolerance for the value of Integral with respect to value of b(upper limit) = 0.1e-7")
print()
data={"a(lower limit)":s1[5],"b(upper limit)":s1[4],"Integral I1":s1[3]}
print(pd.DataFrame(data))
print()
data={"a(lower limit)":s2[5],"b(upper limit)":s2[4],"Integral I2":s2[3]}
print(pd.DataFrame(data))
print()
print("#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#")
print()

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('SIMPSON METHOD (TOLERANCE = 0.1e-7)')
ax1.plot(s1[4],s1[3],marker="*",label="I1 using SIMPSON",linestyle='dashed')
ax2.plot(s2[4],s2[3],marker="*",label="I1 using SIMPSON",linestyle='dashed')
ax1.grid()
ax1.legend()
ax1.set(xlabel="b (upper limit)",ylabel="Integral",title="Integral I1 calculated using SIMPSON")
ax2.set(xlabel="b (upper limit)",ylabel="Integral",title="Integral I2 calculated using SIMPSON")
ax2.grid()
ax2.legend()
plt.show()

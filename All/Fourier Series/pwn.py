from MyIntegration import MySimp
from MyIntegration import MyTrap
from MyIntegration import MyLegQuadrature
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import integrate
from sympy import *
from sympy import simplify

x=symbols('x')

#Function for Fourier Coefficient
def FourierCoeff(f,Prop_Key,a,b,L,N,m_k,tole):
    an=0;an_a=[]
    bn=0;bn_a=[]
    a0=0
    if Prop_Key==1:
        for i in range(1,N+1):
            fun = lambda x: f(x)*np.sin(i*np.pi*x/L)
            if m_k==1:
               bn=(1/L)*MyTrap(fun,a,b,1,key1=True,N_max=1000,key2=True,tol=tole)[0]
            elif m_k==2:
               bn=(1/L)*MyTrap(fun,a,b,1,key1=True,N_max=1000,key2=True,tol=tole)[0]
            elif m_k==3:
                bn=(1/L)*MyLegQuadrature(fun,a,b,10,1,key=True,tol=tole,m_max=1000)[0]
            else:
                return "Wrong Method Key entered"
            bn_a.append(bn)
            an_a.append(0)
        return a0,an_a,bn_a

    elif Prop_Key==0:
        if m_k==1:
            a0=(1/L)*MyTrap(f,a,b,1,key1=True,N_max=1000,key2=True,tol=0.1e-3)[0]
        elif m_k==2:
            a0=(1/L)*MySimp(f,a,b,2,key1=True,N_max=1000,key2=True,tol=tole)[0]
        elif m_k==3:
            a0=(1/L)*MyLegQuadrature(f,a,b,10,1,key=True,tol=tole,m_max=1000)[0]
        else:
            return "Wrong Method Key entered"

        for i in range(1,N+1):
            fun = lambda x: f(x)*np.cos(i*np.pi*x/L)
            if m_k==1:
               an=(1/L)*MyTrap(fun,a,b,1,key1=True,N_max=1000,key2=True,tol=tole)[0]
            elif m_k==2:
               an=(1/L)*MySimp(fun,a,b,1,key1=True,N_max=1000,key2=True,tol=tole)[0]
            elif m_k==3:
                an=(1/L)*MyLegQuadrature(fun,a,b,10,1,key=True,tol=tole,m_max=1000)[0]
            else:
                return "Wrong Method Key entered"
            an_a.append(an)
            bn_a.append(0)
        return a0,an_a,bn_a

    elif Prop_Key==-1:
        if m_k==1:
            a0=(1/L)*MyTrap(f,a,b,1,key1=True,N_max=1000,key2=True,tol=0.1e-3)[0]
        elif m_k==2:
            a0=(1/L)*MySimp(f,a,b,2,key1=True,N_max=1000,key2=True,tol=tole)[0]
        elif m_k==3:
            a0=(1/L)*MyLegQuadrature(f,a,b,10,1,key=True,tol=tole,m_max=1000)[0]
        else:
            return "Wrong Method Key entered"

        for i in range(1,N+1):
            fun1 = lambda x: f(x)*np.cos(i*np.pi*x/L)
            fun2 = lambda x: f(x)*np.sin(i*np.pi*x/L)
            if m_k==1:
               an=(1/L)*MyTrap(fun1,a,b,1,key1=True,N_max=1000,key2=True,tol=tole)[0]
               bn=(1/L)*MyTrap(fun2,a,b,1,key1=True,N_max=1000,key2=True,tol=tole)[0]
            elif m_k==2:
               an=(1/L)*MySimp(fun1,a,b,2,key1=True,N_max=1000,key2=True,tol=tole)[0]
               bn=(1/L)*MySimp(fun2,a,b,2,key1=True,N_max=1000,key2=True,tol=tole)[0]
            elif m_k==3:
                an=(1/L)*MyLegQuadrature(fun1,a,b,10,1,key=True,tol=tole,m_max=1000)[0]
                bn=(1/L)*MyLegQuadrature(fun2,a,b,10,1,key=True,tol=tole,m_max=1000)[0]
            else:
                return "Wrong Method Key entered"
            an_a.append(an)
            bn_a.append(bn)
        return a0,an_a,bn_a
    else :
        return "Wrong key entered"


def Task(L,f1,pf1,Prop_Key,m_k,tit,filename1,filename2,filename3,filename4,filename5):
    N_a=[1,2,5,10, 20]
    x_a=np.linspace(-2.5,2.5,50)
    ex=[]
    f_v=[]
    f_=[]
    a_c=[]
    b_c=[]
    E1=[];E2=[];E3=[];E4=[];E5=[]
    A_X=[-0.5,0,0.5]
    ex2=[]
    for x in x_a:
        ex.append(pf1(x))
    for x in A_X:
        ex2.append(pf1(x))
    for N in N_a:
         s=FourierCoeff(pf1,Prop_Key=Prop_Key,a=-1*L,b=1*L,L=1,N=N,m_k=m_k,tole=0.1e-8)
         a0=s[0]
         #print(a0)
         e=s[1]
         e1=s[2]
         d_a=[]
         d_=[]
         for x in A_X:
             sin=0
             cos=0
        
             for (i,an,bn) in zip(range(1,N+1),e,e1):
                 sin+=bn*np.sin(i*np.pi*x/L)
                 cos+=an*np.cos(i*np.pi*x/L)
             d=a0/2+sin+cos
             d_.append(d)
         f_.append(d_)
         d_=[]
         
         for x in x_a:
             sin=0
             cos=0
        
             for (i,an,bn) in zip(range(1,N+1),e,e1):
                 sin+=bn*np.sin(i*np.pi*x/L)
                 cos+=an*np.cos(i*np.pi*x/L)
             d=a0/2+sin+cos
             d_a.append(d)
         f_v.append(d_a)
         d_a=[]
         a_c.append(e)
         b_c.append(e1)
    
    DataOut1 = np.column_stack((a_c[0],b_c[0]))
    np.savetxt(filename1, DataOut1,delimiter=',')
    DataOut2 = np.column_stack((a_c[1],b_c[1]))
    np.savetxt(filename2, DataOut2,delimiter=',')
    DataOut3 = np.column_stack((a_c[2],b_c[2]))
    np.savetxt(filename3, DataOut3,delimiter=',')
    DataOut4 = np.column_stack((a_c[3],b_c[3]))
    np.savetxt(filename4, DataOut4,delimiter=',')
    DataOut5 = np.column_stack((a_c[4],b_c[4]))
    np.savetxt(filename5, DataOut5,delimiter=',')

    plt.plot(x_a,f_v[0],marker=".",label="i=1",linestyle='dashed')
    plt.plot(x_a,f_v[1],marker=".",label="i=2",linestyle='dashed')
    plt.plot(x_a,f_v[2],marker=".",label="i=5",linestyle='dashed')
    plt.plot(x_a,f_v[3],marker=".",label="i=10",linestyle='dashed')
    plt.plot(x_a,f_v[4],marker=".",label="i=20",linestyle='dashed')
    plt.plot(x_a,ex,linewidth=2,c="black",label="Actual Function")
    plt.grid()
    plt.title(tit)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()
    print()
    print("---------------Value of f(x) calculated using Fourier Expansion------------------")
    data={"x_a":x_a,"f(x)(i=1)":f_v[0],"f(x)(i=2)":f_v[1],"f(x)(i=5)":f_v[2],"f(x)(i=10)":f_v[3],"f(x)(i=20)":f_v[4],"f(x)(exact)":ex}
    print(pd.DataFrame(data))
    
   
    
    for (r,t,y,p,j,l) in zip (f_[0],f_[1],f_[2],f_[3],f_[4],range(3)):
        E1.append(abs(r-ex2[l]))
        E2.append(abs(t-ex2[l]))
        E3.append(abs(y-ex2[l]))
        E4.append(abs(p-ex2[l]))
        E5.append(abs(j-ex2[l]))
   
    print()
    print("----------------Absolute error for x=[-0.5,0,0.5]------------------")   
    data={"x":A_X,"Error(i=1)":E1,"Error(i=2)":E2,"Error(i=5)":E3,"Error(i=10)":E4,"Error(i=20)":E5} 
    print(pd.DataFrame(data))

#Function1          
def f1(x):
    if x>-1 and x<0:
        return 0
    elif x>0 and x<1:
        return 1
    elif x==-1 or x==0 or x==1:
        return 1/2

def pf1(x):
    if x>=-1 and x<=1 :
        return f1(x)
    elif x>1:
        x_new=x-(1-(-1))
        return pf1(x_new)
    elif x<(-1):
        x_new=x+(1-(-1))
        return pf1(x_new)

#Function2         
def f2(x):
    if x>-1 and x<-0.5:
        return 0
    elif x>-0.5 and x<0.5:
        return 1
    elif x>0.5 and x<1:
        return 0
    elif x==-1 or x==0.5 or x==1 or x==-0.5:
        return 1/2

def pf2(x):
    if x>=-1 and x<=1 :
        return f2(x)
    elif x>1:
        x_new=x-(1-(-1))
        return pf2(x_new)
    elif x<(-1):
        x_new=x+(1-(-1))
        return pf2(x_new)

#Function3
def f3(x):
    if x>-1 and x<0:
        return -0.5
    elif x>0 and x<1:
        return 0.5
    elif x==-1 or x==0 or x==1:
        return 0

def pf3(x):
    if x>=-1 and x<=1 :
        return f3(x)
    elif x>1:
        x_new=x-(1-(-1))
        return pf3(x_new)
    elif x<(-1):
        x_new=x+(1-(-1))
        return pf3(x_new)

print("********************************(ALPHA)************************************************")
Task(1,f1,pf1,-1,3,"Fourier Series Approximation for alpha part of Ques(b)(iii)","alpha1.dat","alpha2.dat","alpha3.dat","alpha4.dat","alpha5.dat")
print()
print("********************************(BETA)************************************************")
Task(1,f2,pf2,0,3,"Fourier Series Approximation for beta part of Ques(b)(iii)","beta1.dat","beta2.dat","beta3.dat","beta4.dat","beta5.dat")
print()
print("********************************(GAMMA)************************************************")
Task(1,f3,pf3,1,3,"Fourier Series Approximation for gamma part of Ques(b)(iii)","gamma1.dat","gamma2.dat","gamma3.dat","gamma4.dat","gamma5.dat")

# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:23:20 2020

@author: Guido di Pasquo
"""
#Apologies in advance for any spelling or grammar error, english is not my first language

import matplotlib.pyplot as plt
import numpy as np
import random

###########################################
#known bugs-> 



###########OVERALL CHARACTERISTICS OF THE PROGRAM THAT SHOULD BE TAKEN INTO ACCOUNT IN THE FLIGHT COMPUTER CODE

#Important, all angles are in RADIANS (Standard 1º/57.3 = radian)

#Code already takes the reduction of the TVC into account, so "u_controler" (output to the servo,[u_controler is in RADIANS, if your servo takes degrees you have to convert it])
#should be sent directly to the servo, without multiplying it, that means that the PID gains might seem higher than they
#actually are.

#Parameters related to the servo have the conviniet "s" ending

########################################################################################################################## SET YOUR ROCKET

Thrust=28.8*0.5; #THRUST in N (use peak thrust first, check with minimum after)

m=0.451  #  MASS in kg

Iy=0.0662 #MOMENT OF INERTIA, use the smaller one (motor burnout)

g=9.8  # gravity in m/s^2

U=0.001  # If you use control fins, U is the speed at which the rocket leaves the launch rod
V=0.001

rho=1.225 # air density

q=0.5*rho*V**2 #dynamic pressure                                       
                                       
d=0.05  # DIAMETER of the rocket and cross sectional area (IN METERS)

S=((d**2*3.14159)/4) #cross sectional area

xa=0.17  #DISTANCE from the nose to the center of pressure, cg and tail (either the motor mount for tvc or the mac of the control fin), IN METERS
xcg=0.55 #DISTANCE from the nose to the cg IN METERS
xt=0.85  #DISTANCE from the nose to the tail (either the motor mount for tvc or the 25%mac of the control fin), IN METERS

lt=(xt-xcg)  #lenght to the tail

L=0.85; #LENGHT of the rocket

CD=0.43 #DRAG COEFFICIENT at alpha=0


alpha=1.571

k1=-13.119 #CNalpha coeficients
k2=30.193
k3=0.3948
CNalpha=(k1*alpha**3+k2*alpha**2+k3*alpha)/alpha #Third order aproximation deg, also in 1/radians

CZalpha=-CNalpha

Cw=-(m*g)/(S*q);



fin_area=0.003  #if you use FINS FOR CONTROL, CLde is the lift slope of the fin, be carefull that the bi-dimensional is not accurate

CLde=2; 

CMalpha=(CNalpha*(xcg-xa))/(d);

CMde=(Thrust*lt)/(S*q*d); #Replace (Thrust*lt) by (q*fin_area*CLde*lt) if you use control fins

CZde=(d/(lt))*CMde;

TVC_max=5/57.3 # maximum gimbal angle

TVC_reduction=5 #TVC_reduction of 5 means that 5 degrees in the servo equal 1 degree in the TVC mount, use the higher of you TVC mount.



wind=-2; #Wind speed in m/s (positive right to left)
wind_distribution=0.5*0 # wind*wind_distribution = max gust speed 

########################################################################################################################## IGNORE UP TO PID GAINS

####PLOTS
t_plot=[]
theta_plot=[]
setpoint_plot=[]
servo_plot=[]



alpha_calc=0.
alpha_control=0.
U_vect=np.array([0.1,0])
V_vect=np.array([0.1,0])
wind_vect=np.array([0,wind])
u_eq=0.
u_prev=0.
u_delta=0.
u_controler=0.


a11, a12, a13, a21, a22, a23, a31, a32, a33 = (0.0,)*9
b11, b21, b31=(0.0,)*3
c11, c12, c13, c21, c22, c23, c31, c32, c33 = (0.0,)*9
d11, d21, d31=(0.0,)*3


as11, as12, as21, as22 = (0.0,)*4
bs11, bs21=(0.0,)*2
cs11, cs12, cs21, cs22 = (0.0,)*4
ds11, ds21=(0.0,)*2


#MODELO

A=np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])


B=np.array([[b11],
           [b21],
           [b31]])

C=np.array([[c11, c12, c13],
           [c21, c22, c23],
           [c31, c32, c33]])

D=np.array([[d11],
           [d21],
           [d31]])


x=np.array([[0.],
           [0.],
           [0.]])


xdot=np.array([[0.],
           [0.],
           [0.]])

xdot_prev=np.array([[0.],
           [0.],
           [0.]])

out=np.array([[0.],
           [0.],
           [0.]])

out_prev=np.array([[0.],
           [0.],
           [0.]])
u=0.


#SERVO (SG90)

As=np.array([[as11, as12], [as21, as22]])


Bs=np.array([[bs11],
           [bs21]])

Cs=np.array([[cs11, cs12],
           [cs21, cs22]])

Ds=np.array([[ds11],
           [ds21]])


xs=np.array([[0.],
            [0.]])


xdots=np.array([[0.],
               [0.]])

xdot_prevs=np.array([[0.],
                    [0.]])

outs=np.array([[0.],
              [0.]])

out_prevs=np.array([[0.],
                   [0.]])




#PID
currentTime=0
previousTime=0
elapsedTime=0.
elapsedTimeSeg=0.
errorPID=0.
lastError=0.
output=0.
setPoint=0.
cumError=0.
rateError=0.
out_pid=0.
anti_windup=True



#CONTROL
setpoint=0.
error=0.
dt=0.02;


#TIMERS
timer_run=0
#timer=0
timer_run_sim=0
timer_run_servo=0
t=0.


#FLAGS
flag=False
flag2=False

########################################################################################################################## FUNCTIONS

def control_tita(setpoint):
    global error
    global u_controler,u_prev,u_delta

    
    u_prev=u_controler
    
    error=setpoint-out[0,0] #On your flight computer, replace out[0,0] for you calculated angle
    error=error*k_all
    u_controler=PID(error)
    u_controler=u_controler-out[1,0]*k_damping #On your flight computer, replace out[1,0] for the gyroscope data (angular speed)
    
    if(u_controler>TVC_max*TVC_reduction):  #prevents the TVC from deflecting more that it can
        u_controler=TVC_max*TVC_reduction
    elif(u_controler<-TVC_max*TVC_reduction):
        u_controler=-TVC_max*TVC_reduction
        
        
   
    
    u_delta=abs(u_controler-u_prev) #only for the simualtion, does nothing in the real flight computer
    
    return u_controler



####PID
def PID(inp):
    global t
    global T_Program  #elapsed time between runs
    global lastError,cumError
    global TVC_max,TVC_reduction,anti_windup
    

    errorPID = inp;                                # determine error
    rateError = (errorPID - lastError) / T_Program  # compute derivative

    
    
    if(anti_windup==True):
        out_pid = kp * errorPID + ki * cumError + kd * rateError;          #PID output
        if(out_pid < (TVC_max*TVC_reduction) and out_pid > (-TVC_max*TVC_reduction)):  #Anti windup by clamping
            cumError = ((((lastError) + ((errorPID - lastError) / 2))) * T_Program)+cumError         # compute integral (trapezoidal) only if the TVC is not staurated
            out_pid = kp * errorPID + ki * cumError + kd * rateError #recalculate the output
       
        if(out_pid>TVC_max*TVC_reduction):  #prevents the TVC from deflecting more that it can
            out_pid=TVC_max*TVC_reduction
        elif(out_pid<-TVC_max*TVC_reduction):
            out_pid=-TVC_max*TVC_reduction
    else:
        cumError = ((((lastError) + ((errorPID - lastError) / 2))) * T_Program)+cumError
        out_pid = kp * errorPID + ki * cumError + kd * rateError;          #PID output
        
        
    
    lastError = errorPID;                                #remember current error

    return out_pid;                                        #have function return the PID output



def update_servo(u_controler,u_delta):
    
    if(u_delta<=10/57.3):
        u_delta=10/57.3
    elif(u_delta>=45/57.3):
        u_delta=45/57.3
    
    
    dens=-0.000609663-0.0358604*T+0.0273794*u_delta*T-0.812193*T**2+u_delta*T**2
    
    # A Matrix    
    as11=-(0.000609663 + 0.0358604*T - 0.0273794*u_delta*T - 0.812193*T**2 + u_delta*T**2)/dens
    as12=-(0.000609663*T)/dens
    as21=-(4*(-0.812193+u_delta)*T)/dens
    as22=-((0.000609663-0.0358604*T+0.0273794*u_delta*T-0.812193*T**2+u_delta*T**2))/dens
    
    global As
    As=np.array([[as11, as12], [as21, as22]])
    
    # B Matrix
    bs11=((-0.812193+u_delta)*T**2)/dens
    bs21=(2*(-0.812193+u_delta)*T)/dens
    
    global Bs
    Bs=np.array([[bs11],
                 [bs21]])
    
    # C Matrix
    cs11=(0.0547587*(-0.0222673-1.30976*T*+u_delta*T))/dens
    cs12=-(0.000609663*T)/dens
    cs21=-(4*(-0.812193*T+u_delta*T))/dens
    cs22=((-0.00121933-3.46945*10**-18*u_delta*T))/dens
    
    global Cs
    Cs=np.array([[cs11, cs12],
                 [cs21, cs22]])
    
    # D Matrix
    ds11=((-0.812193*T**2*+u_delta*T**2))/dens
    ds21=(2*(-0.812193*T*+u_delta*T))/dens
    
    global Ds
    Ds=np.array([[ds11],
                 [ds21]])  
    
    return





def u_equivalent(alpha):
          
    if(alpha>0):        
        CNwind2=(k1*alpha**3+k2*alpha**2+k3*alpha)
    elif(alpha<0):
        CNwind2=-(k1*abs(alpha)**3+k2*abs(alpha)**2+k3*abs(alpha))
    else:
        CNwind2=0
        
    u_eq=((CNwind2*(xcg-xa))/(d))/((Thrust*lt)/(S*q*d))
    
    return u_eq


i=1
def update_parameters():
    global U
    global V
    global q
    global alpha,u_eq
    global CNalpha
    global x
    global CZalpha
    global CMalpha,xa,CZde,CMde,Cw
    global i
    global alpha_calc,alpha_control
    global wind
    
    
    U=(Thrust-m*g*np.cos(out[0,0])-S*q*CD)*T+U
    
    V=np.sqrt(U**2+wind**2)
    q=0.5*rho*V**2
    
    wind_rand=(random.uniform(-wind, wind))*wind_distribution
    
    alpha=np.arctan(-(wind+wind_rand)/U) 
    u_eq=u_equivalent(alpha)
    
      
    alpha_control=float(x[2,0])+alpha
#    alpha_calc=alpha+alpha_control
    
    
    if(alpha_control>0):        
        CNalpha=(k1*alpha_control**3+k2*alpha_control**2+k3*alpha_control)/alpha_control
    elif(alpha_control<0):
        CNalpha=(-(k1*abs(alpha_control)**3+k2*abs(alpha_control)**2+k3*abs(alpha_control)))/alpha_control
#        if(out[2,0]>0):
#            CNalpha=CNalpha/abs(alpha_control)
#        else:
#            CNalpha=CNalpha/alpha_control
    else:
        CNalpha=0

    
    CZalpha=-CNalpha
    CMalpha=CNalpha*(xcg-xa)/d
    
  
    q=0.5*rho*V**2
    Cw=-(m*g)/(S*q)    
    CMde=(Thrust*lt)/(S*q*d)
    CZde=(d/(lt))*CMde
    
    
    return
    
    
    
    
 
def update_matrix():
    
    global x
    global xdot
    global out
    global out_prev
    global u,u_delta
    
    
    update_parameters()
    update_servo(u,u_delta)

    
    sin_theta=np.sin(90/57.3-out[0,0])
    den=(-8*m*U*Iy+2*d*m*q*S*T**2*U*CMalpha+d*q**2*S**2*T**3*sin_theta*Cw*CMalpha+4*q*S*T*Iy*CZalpha)

#MATRIX A

    a11=(-8*m*U*Iy+2*d*m*q*S*T**2*U*CMalpha-d*q**2*S**2*T**3*sin_theta*Cw*CMalpha+4*q*S*T*Iy*CZalpha)/den
    a12=(4*T*Iy*(-2*m*U+q*S*T*CZalpha))/den
    a13=-(4*d*m*q*S*T**2*U*CMalpha)/den

    a21=-(4*d*q**2*S**2*T**2*sin_theta*Cw*CMalpha)/den
    a22=-(8*m*U*Iy+2*d*m*q*S*T**2*U*CMalpha+d*q**2*S**2*T**3*sin_theta*Cw*CMalpha-4*q*S*T*Iy*CZalpha)/den
    a23=-(8*d*m*q*S*T*U*CMalpha)/den

    a31=-(8*q*S*T*sin_theta*Cw*Iy)/den
    a32=-(4*T*Iy*(2*m*U+q*S*T*sin_theta*Cw))/den
    a33=-(8*m*U*Iy+2*d*m*q*S*T**2*U*CMalpha+d*q**2*S**2*T**3*sin_theta*Cw*CMalpha+4*q*S*T*Iy*CZalpha)/den

    global A
    A=np.array([[a11, a12, a13],[a21, a22, a23],[a31, a32, a33]])



    #MATRIX B
    
    b11=(d*q*S*T**2*(-2*m*U*CMde+q*S*T*CMde*CZalpha-q*S*T*CMalpha*CZde))/den;
    b21=(2*d*q*S*T*(-2*m*U*CMde+q*S*T*CMde*CZalpha-q*S*T*CMalpha*CZde))/den;
    b31=-(q*S*T*(2*d*m*T*U*CMde+d*q*S*T**2*sin_theta*Cw*CMde+4*Iy*CZde))/den;
    
    global B
    B=np.array([[b11],
                [b21],
                [b31]])
    
    
    
    #MATRIX C
    
    c11=-(4*(4*m*U*Iy-d*m*q*S*T**2*U*CMalpha-2*q*S*T*Iy*CZalpha))/(den);
    c12=(4*T*Iy*(-2*m*U+q*S*T*CZalpha))/(den);
    c13=-(4*d*m*q*S*T**2*U*CMalpha)/(den);
    
    c21=-(4*d*q**2*S**2*T**2*sin_theta*Cw*CMalpha)/(den);
    c22=-(8*Iy*(2*m*U-q*S*T*CZalpha))/(den);
    c23=-(8*d*m*q*S*T*U*CMalpha)/(den);
    
    c31=-(8*q*S*T*sin_theta*Iy*Cw)/(den);
    c32=-(4*T*Iy*(2*m*U+q*S*T*sin_theta*Cw))/(den);
    c33=-(16*m*U*Iy)/(den);
    
    global C
    C=np.array([[c11, c12, c13],
                [c21, c22, c23],
                [c31, c32, c33]])
    
    
    
    #MATRIX D
    
    d11=-(d*q*S*T**2*(2*m*U*CMde-q*S*T*CMde*CZalpha+q*S*T*CMalpha*CZde))/den;
    d21=-(2*d*q*S*T*(2*m*U*CMde-q*S*T*CMde*CZalpha+q*S*T*CMalpha*CZde))/den;
    d31=-(q*S*T*(2*d*m*T*U*CMde+d*q*S*T**2*sin_theta*Cw*CMde+4*Iy*CZde))/den;
    
    global D    
    D=np.array([[d11],
                [d21],
                [d31]])

    
    return


    

def set_setpoint(inp):
    
    if(inp==1):
        setpoint=10/57.3
    elif(inp==2):
        setpoint=(5/57.3)*(t-0.5)
    else:
        setpoint=0
    
    
    return setpoint

      

def simulation():
    
    global x,xs
    global xdot,xdots
    global out,outs
    global out_prev,out_prevs
    global u_controler
    global u,timer_run_servo
    
    
    if(t>timer_run_servo+Ts*0.9999):
        timer_run_servo=t
        u=u_controler        
        u=round(u*57.3,0)/57.3 #definition of the servo, standard 1º
    
    
    xdots=(np.dot(As,xs)+np.dot(Bs,(u)))
    outs=(np.dot(Cs,xs)+np.dot(Ds,u))
    
    outs[0,0]=(outs[0,0]/TVC_reduction) #reduction of the TVC
    
    xdot=(np.dot(A,x)+np.dot(B,(outs[0,0]+u_eq)))
    out=(np.dot(C,x)+np.dot(D,outs[0,0]+u_eq))
    #out=out-out_prev*0.3;  #filter, increasing the number makes it stronger and slower
    #out_prev=out    
    #x=x+xdot*dt;
    #x=x+(xdot_prev*dt+((xdot-xdot_prev)*dt)/2);
    #xdot_prev=xdot;
    
    xs=xdots
    x=xdot
    
    return


def timer():
    global t
    t_plot.append(t)
    t=t+T
    return








########################################################################################################################## PID GAINS
kp=1
ki=1 
kd=0.0 
k_all=10 
k_damping=1.2*0.7/0.7

anti_windup=True #Prevents the integrator for integreting when the TVC is saturated
                 #it also limits the output of the PID so k_damping is more efective
                 
inp=1 #selects the input
      #1-> Step, hard test on stability
      #2-> 5º/s ramp, slow pitch over
      #3-> only stabilization against disturbances


T=0.001   #T=Sample time of the simulation 
Ts=0.02   #Ts=sample time of the servo (0.02 for a SG90)
T_Program=0.01 #T_Program: Sample time of your PID code
Sim_duration=50 #How long will it simulate
########################################################################################################################## HERE THE PROGRAM STARTS


while t<=Sim_duration:

    if(t>(timer_run_sim+T*0.9999)):
        timer_run_sim=t
        update_matrix()
      
        if(t>timer_run+T_Program*0.9999):
            timer_run=t
    
            if(t>0.5):
                setpoint=set_setpoint(inp)

            control_tita(setpoint)
    
      
        simulation()
    
    
    timer()
   # print(out[0,0]*57.3)
    
    setpoint_plot.append(setpoint*57.3)
    theta_plot.append(out[0,0]*57.3)
    servo_plot.append((out[2,0]+alpha)*57.3)
    
    #Plot selectors
    #out[0,0]->Pitch Angle, out[1,0]-> Pitch Rate, out[2,0]->Rection Angle of attack
    #u_controler->controler output, outs[0,0]->Real servo angle, outs[1,0]->Servo speed
    
    
    
    

plt.plot(t_plot,theta_plot,t_plot,setpoint_plot,t_plot,servo_plot)
plt.grid(True,linestyle='--')

plt.xlabel('Time',fontsize=16)
plt.ylabel('Pitch Angle',fontsize=16)

plt.show()



########################################################################################################################## HERE THE PROGRAM ENDS


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
#known bugs-> #Carefull with any optional function, they work but haven't been thoroughly tested

#Mathematical model assumes small angles, so, don't make the wind slower than 1m/s without increasing the initial U (line 100)

#Wind forces and torques are calculated from the xa (center of pressure)
#this is not realistic since the xa at high angles of attack shifts backwards (eg: 90 deg produced by side wind when the rocket is at low speeds)
#the result is that wind effects are stronger in the simulator than in real life

#Changed the discrete matrices for continuous and the discretize it

###########OVERALL CHARACTERISTICS OF THE PROGRAM THAT SHOULD BE TAKEN INTO ACCOUNT IN THE FLIGHT COMPUTER CODE

#Important, all angles are in RADIANS (Standard 1º/57.3 = radian)

#Code simulates the TVC_reduction (gear ratio), it multiplies the output of the controller times the TVC_reduction, and then sends that output to the servo.
#Remember that you have to multiply the output of the controller times the TVC reduction in you flight computer!
#All in all, the overall structure of the function "control_tita" and "PID" should be copied in your flight computer to ensure that the simulator and flight computer are doing the same thing


#Parameters related to the servo have the conviniet "s" ending

#Default settings are -1.5m/s winds and a 2 degree initial offset in the TVC mount, to push hard on the system. Change them accordingly to suit your model

########################################################################################################################## SET UP YOUR ROCKET


## MOTOR PARAMETERS

Thrust=28.8; #THRUST in N (use peak thrust first, check with minimum after)
burnout_time=3.5 #burn time of the motor, stops the simulation

Thrust_curve=True #Thrust=(max_thrust/(burnout_time/7)*t) FOR 0<t<burnout_time/7
                  # Thrust=-(max_thrust-average_thrust)/(burnout_time/7)*(t-burnout_time/7)+max_thrust FOR burnout_time/7<t<2*burnout_time/7
                  # Thrust=average_thrust FOR 2*burnout_time/7<t<burnout_time
max_thrust=28.8 #If you use thrust curve set them to your motor value
average_thrust=10. #




## ROCKET PARAMETERS

m=0.451  #  MASS in kg
Iy=0.0662 #MOMENT OF INERTIA, use the smaller one (motor burnout)
d=0.05  # DIAMETER of the rocket and cross sectional area (IN METERS)
xa=0.17  #DISTANCE from the nose to the center of pressure, cg and tail (either the motor mount for tvc or the mac of the control fin), IN METERS
xcg=0.55 #DISTANCE from the nose to the cg IN METERS
xt=0.85  #DISTANCE from the nose to the tail (either the motor mount for tvc or the 25%mac of the control fin), IN METERS
L=0.85; #LENGHT of the rocket
CD=0.43 #DRAG COEFFICIENT at alpha=0




## AERODYNAMIC PARAMETERS

k1=-13.119 #CNalpha coeficients
k2=30.193
k3=0.3948
#CNalpha=(k1*alpha**3+k2*alpha**2+k3*alpha)/alpha #Third order aproximation deg, also in 1/radians
max_speed=45 #max speed the rocket reaches, in m/s




## TVC PARAMETERS

TVC_max=5/57.3 # maximum gimbal angle
TVC_reduction=5 #TVC_reduction of 5 means that 5 degrees in the servo equal 1 degree in the TVC mount, use the higher on your TVC mount.
u_initial_offset=2/57.3 # Initial angle of the TVC mount due to calibration error

#Torque based controller
reference_thrust=28.8 #Thrust for which you tune the rocket, will be used as reference


## WIND PARAMETERS

wind_beta=True #Uses wind as disturbance, if set to False random noise is introduced in the TVC mount to simulate it.
wind=-1.5; #Wind speed in m/s (positive right to left)
wind_distribution=0.5 # wind*wind_distribution = max gust speed 




## OTHER PARAMETERS OR VARIABLES
CNalpha=0
g=9.8  # gravity in m/s^2
U=0.001  # If you use control fins, U is the speed at which the rocket leaves the launch rod
V=0.001
rho=1.225 # air density
q=0.5*rho*V**2 #dynamic pressure
S=((d**2*3.14159)/4) #cross sectional area
lt=(xt-xcg)  #lenght to the tail
alpha=1.517
CZalpha=-CNalpha
Cw=-(m*g)/(S*q);
CMalpha=(CNalpha*(xcg-xa))/(d);
CMde=(Thrust*lt)/(S*q*d); #Replace (Thrust*lt) by (q*fin_area*CLde*lt) if you use control fins, here and down
fin_area=0.003  #if you use FINS FOR CONTROL, CLde is the lift slope of the fin, be carefull that the bi-dimensional is not accurate
CLde=2;  #fins
CZde=(d/(lt))*CMde;
U_prev=0.
U2=0.


########################################################################################################################## IGNORE UP TO PID GAINS

####PLOTS
t_plot=[]
theta_plot=[]
setpoint_plot=[]
servo_plot=[]

if(Thrust_curve==True):
    Thrust=1.1*m*g

alpha_calc=0.
alpha_control=0.
U_vect=np.array([0.1,0])
V_vect=np.array([0.1,0])
wind_vect=np.array([0,wind])
u_eq=0.
u_prev=0.
u_delta=0.
u_controller=0.
u_servos=0.
TVC_weight_compensation=1.75 # After watching joe's video on servos they are a tad slower that the ones I measured, this simulates that. #1.3 is the servo alone


a11, a12, a13, a21, a22, a23, a31, a32, a33 = (0.0,)*9
b11, b21, b31=(0.0,)*3
c11, c12, c13, c21, c22, c23, c31, c32, c33 = (0.0,)*9
d11, d21, d31=(0.0,)*3


as11, as12, as21, as22 = (0.0,)*4
bs11, bs21=(0.0,)*2
cs11, cs12, cs21, cs22 = (0.0,)*4
ds11, ds21=(0.0,)*2

asc11, asc12, asc21, asc22 = (0.0,)*4
bsc11, bsc21=(0.0,)*2
csc11, csc12, csc21, csc22 = (0.0,)*4

ac11, ac12, ac13, ac21, ac22, ac23, ac31, ac32, ac33 = (0.0,)*9
bc11, bc21, bc31=(0.0,)*3
cc11, cc12, cc13, cc21, cc22, cc23, cc31, cc32, cc33 = (0.0,)*9
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

#continuous time
Ac=np.array([[ac11, ac12, ac13], [ac21, ac22, ac23], [ac31, ac32, ac33]])


Bc=np.array([[bc11],
           [bc21],
           [bc31]])
    
Cc=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

#SERVO (SG90)

As=np.array([[as11, as12], [as21, as22]])


Bs=np.array([[bs11],
           [bs21]])

Cs=np.array([[cs11, cs12],
           [cs21, cs22]])

Ds=np.array([[ds11],
           [ds21]])

#Continous time
    
Asc=np.array([[asc11, asc12], [asc21, asc22]])


Bsc=np.array([[bsc11],
           [bsc21]])

Csc=np.array([[1, 0],
           [0, 1]])
    
    
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

Itwo=np.array([[1, 0],
           [0, 1]])
    
Ithree=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


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
timer_disturbance=0.
timer_U=0.

#FLAGS
flag=False
flag2=False

########################################################################################################################## FUNCTIONS

def control_tita(setpoint):
    global error
    global u_controller,u_prev,u_delta,u_servos

    
    u_prev=u_controller
    
    error=setpoint-out[0,0] #On your flight computer, replace out[0,0] for you calculated angle
    error=error*k_all
    u_controller=PID(error)
    u_controller=u_controller-out[1,0]*k_damping #On your flight computer, replace out[1,0] for the gyroscope data (angular speed)
    
    if(u_controller>TVC_max):  #prevents the TVC from deflecting more that it can
        u_controller=TVC_max
    elif(u_controller<-TVC_max):
        u_controller=-TVC_max
        
    #u_controller=u_controller-u_prev*0.05;  #filter, increasing the number makes it stronger and slower
    
    u_servos=u_controller*TVC_reduction #compensates the TVC reduction (gear ratio)
    
    if(torque_controller==True):

        thrust_controller=Thrust #On the simulation one can access the real Thrust from the thrust curve
                                 #In your flight computer you will have to calculate it with Thrust-Weight-Drag=m*acceleration with Drag being: Drag=0.5*rho*U**2*S*CD and Weight=m*g. If the speed is slow you can neglect the Drag 
        u_servos=(reference_thrust/thrust_controller)*u_servos
        if(u_servos>TVC_max*TVC_reduction):  #prevents the TVC from deflecting more that it can
            u_servos=TVC_max*TVC_reduction
        elif(u_servos<-TVC_max*TVC_reduction):
            u_servos=-TVC_max*TVC_reduction

    
    return u_controller



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
        if(out_pid < (TVC_max ) and out_pid > (-TVC_max )):  #Anti windup by clamping
            cumError = ((((lastError) + ((errorPID - lastError) / 2))) * T_Program)+cumError         # compute integral (trapezoidal) only if the TVC is not staurated
            out_pid = kp * errorPID + ki * cumError + kd * rateError #recalculate the output
       
        if(out_pid>TVC_max ):  #prevents the TVC from deflecting more that it can
            out_pid=TVC_max 
        elif(out_pid<-TVC_max ):
            out_pid=-TVC_max 
    else:
        cumError = ((((lastError) + ((errorPID - lastError) / 2))) * T_Program)+cumError
        out_pid = kp * errorPID + ki * cumError + kd * rateError;          #PID output
        
        
    
    lastError = errorPID;                                #remember current error

    return out_pid;                                        #have function return the PID output



def update_servo():      
    global Asc,Bsc,Csc,As,Bs,Cs,Ds,u_delta, u_servos    
    
    u_delta=TVC_weight_compensation*abs(u_servos-outs[0,0]*TVC_reduction) #only for the simualtion, does nothing in the real flight computer
                                                                           #The tvc compensation slows down the servo to simulate the TVC weight, since the characterization was done without resistance 
                                                                            #outs[0,0]*TVC_reduction follows the current servo angle instead of the TVC angle

    
    if(u_delta<=10/57.3):
        u_delta=10/57.3        
    elif(u_delta>=90/57.3):
        u_delta=90/57.3
    
    
        
    # A Matrix    
    asc11=0.
    asc12=1.
    asc21=-(-2624.5*u_delta**3+9996.2*u_delta**2-13195*u_delta+6616.2)
    asc22=-(39.382*u_delta**2-125.81*u_delta+124.56)
    

    Asc=np.array([[asc11, asc12], [asc21, asc22]])
    
    # B Matrix
    bsc11=0
    bsc21=(-2624.5*u_delta**3+9996.2*u_delta**2-13195*u_delta+6616.2)
    

    Bsc=np.array([[bsc11],
                 [bsc21]])
    
    # C Matrix    
    # D Matrix
    
    As=np.dot(np.linalg.inv((2/T)*Itwo-Asc),((2/T)*Itwo+Asc)) #Now discretization is done in the program, servo still uses the old method (both were verified)
    Bs=np.dot(np.linalg.inv((2/T)*Itwo-Asc),Bsc)
    Cs=(As+Csc)
    Ds=Bs
    
    
    return





def u_equivalent(alpha):
          
    if(alpha>0):        
        CNwind2=(k1*alpha**3+k2*alpha**2+k3*alpha)
    elif(alpha<0):
        CNwind2=-(k1*abs(alpha)**3+k2*abs(alpha)**2+k3*abs(alpha))
    else:
        CNwind2=0
        
    u_eq=((CNwind2*(xcg-(xa)))/(d))/((Thrust*lt)/(S*q*d))
    
    
    
    return u_eq



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
    global Thrust
    global out,timer_disturbance,timer_U,U2
    
    if Thrust_curve==True:
        Thrust_calculator()
        
    U=((Thrust-m*g-0.5*rho*S*U**2*CD)/m)*T+U
    
    if(U>max_speed):
        U=max_speed

    V=np.sqrt(U**2+wind**2)
    q=0.5*rho*V**2
   
    if(wind_beta==True):
        wind_rand=(random.uniform(-wind, wind))*wind_distribution          
        alpha=np.arctan(-(wind+wind_rand)/U)
        if(t>timer_disturbance+2*T_Program*0.9999):
            timer_disturbance=t
            u_eq=u_equivalent(alpha)

    else:
        alpha=0
        if(t>timer_disturbance+2*T_Program*0.9999):
            timer_disturbance=t
            u_eq=random.uniform(-1/57.3, 1/57.3)
    
      
    alpha_control=float(x[2,0])+alpha

    
    if(alpha_control>0 and alpha>=0):        
        CNalpha=((k1*alpha_control**3+k2*alpha_control**2+k3*alpha_control)-(k1*alpha**3+k2*alpha**2+k3*alpha))
    elif(alpha_control<0 and alpha>=0):
        CNalpha=(-(k1*abs(alpha_control)**3+k2*abs(alpha_control)**2+k3*abs(alpha_control))-(k1*alpha**3+k2*alpha**2+k3*alpha))
    elif(alpha_control<0 and alpha<=0):
        CNalpha=-(k1*abs(alpha_control)**3+k2*abs(alpha_control)**2+k3*abs(alpha_control))-(-(k1*abs(alpha)**3+k2*abs(alpha)**2+k3*abs(alpha)))
    elif(alpha_control>0 and alpha<=0):
        CNalpha=(k1*alpha_control**3+k2*alpha_control**2+k3*alpha_control)-(-(k1*abs(alpha)**3+k2*abs(alpha)**2+k3*abs(alpha)))
        
    else:
        CNalpha=0
    
       
    if(abs(float(x[2,0]))>0):
        CNalpha=CNalpha/float(x[2,0])
        
    
    CZalpha=-CNalpha
    CMalpha=CNalpha*(xcg-xa)/d
    

    Cw=-(m*g)/(S*q)    
    CMde=(Thrust*lt)/(S*q*d)
    CZde=(d/(lt))*CMde       
    
    return
    
    



def Thrust_calculator():
    global Thrust, t, burnout_time
    
    if(t<burnout_time/7):            
        Thrust=((max_thrust-m*g)/(burnout_time/7)*t)+m*g
    elif(t>=(burnout_time/7) and t<(2*burnout_time/7)):
        Thrust=-(max_thrust-average_thrust)/(burnout_time/7)*(t-burnout_time/7)+max_thrust
    else:
        Thrust=average_thrust    
 ############################
   
#    if(t<(2*burnout_time/7)):
#        Thrust=-(max_thrust-average_thrust)/(2*burnout_time/7)*(t)+(max_thrust)
#    else:
#        Thrust=average_thrust
 
##############################    
#    if(t<burnout_time/7):            
#        Thrust=-(average_thrust/(burnout_time/7)*t)+average_thrust*2

        
        
        
        
 
def update_matrix():
    
    global x
    global xdot
    global out
    global out_prev
    global u,u_delta
    global A,B,C,D,Ac,Bc,Cc
    
    
    update_parameters()

    update_servo()
    
    sin_theta=np.sin(90/57.3+out[0,0])

#MATRIX A

    ac11=0
    ac12=1
    ac13=0

    ac21=0
    ac22=0
    ac23=(CMalpha*S*q*d)/Iy

    ac31=(Cw*sin_theta*S*0.5*rho*U)/m
    ac32=1
    ac33=(CZalpha*S*0.5*rho*U)/m

    
    Ac=np.array([[ac11, ac12, ac13],[ac21, ac22, ac23],[ac31, ac32, ac33]])



    #MATRIX B
    
    bc11=0
    bc21=(CMde*S*q*d)/Iy
    bc31=(CZde*S*0.5*rho*U)/m
    
    
    Bc=np.array([[bc11],
                [bc21],
                [bc31]])
    
    
    #DISCRETIZATION
    
    A=np.dot(np.linalg.inv((2/T)*Ithree-Ac),((2/T)*Ithree+Ac)) #Now discretization is done in the program, servo still uses the old method (both were verified)
    B=np.dot(np.linalg.inv((2/T)*Ithree-Ac),Bc)
    C=(A+Cc)
    D=B



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
    global u_controller
    global u,timer_run_servo,u_servos
      
    
    if(t>timer_run_servo+Ts*0.9999+0.001):  #+0.001 desincronizes the servo and the program, thing that would likely happen in a real flight computer
        timer_run_servo=t
        u=u_servos
        u=round(u*57.3,0)/57.3 #definition of the servo, standard 1º
    
    
    xdots=(np.dot(As,xs)+np.dot(Bs,(u)))
    outs=(np.dot(Cs,xs)+np.dot(Ds,u))
    
    outs[0,0]=(outs[0,0]/TVC_reduction) #reduction of the TVC
    
    xdot=(np.dot(A,x)+np.dot(B,(outs[0,0]+u_eq+u_initial_offset)))
    out=(np.dot(C,x)+np.dot(D,outs[0,0]+u_eq+u_initial_offset))
    

    xs=xdots
    x=xdot

  
    return


def timer():
    global t
    t=t+T
    return








########################################################################################################################## PID GAINS
kp=0.8 #1  #Gains might be quite high to compensate for aerodynamic forces
ki=0.4  #1
kd=0.08 #0
k_all=1 #10
k_damping=0 #1.2 #when possible use gyro feedback instead of derivative so as to not amplify noise

anti_windup=True #Prevents the integrator for integreting when the TVC is saturated
                 #it also limits the output of the PID so k_damping is more efective

torque_controller=True #Activates the torque controller

inp=3 #selects the input
      #1-> Step, hard test on stability
      #2-> 5º/s ramp, slow pitch over
      #3-> only stabilization against disturbances, simulates the real, straight-up flight


T=0.001   #T=Sample time of the simulation 
Ts=0.02   #Ts=sample time of the servo (0.02 for a SG90)
T_Program=0.01 #T_Program: Sample time of your PID code
Sim_duration=10 #How long will it simulate
########################################################################################################################## HERE THE PROGRAM STARTS

while t<=Sim_duration:
    if(t>burnout_time):
        break
    
    if(abs(out[0,0])>180/57.3):
        print("Pitch angle greater than 180º, the rocket is flying pointy end down.")
        break
    
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

    
    setpoint_plot.append(setpoint*57.3)
    theta_plot.append(out[0,0]*57.3)
    servo_plot.append(outs[0,0]*57.3)
    
    #Plot selectors
    #out[0,0]->Pitch Angle, out[1,0]-> Pitch Rate, out[2,0]->Rection Angle of attack
    #u_controller->controler output, outs[0,0]->Real servo angle, outs[1,0]->Servo speed
    
    t_plot.append(t)
    
    
    #t_plot,theta_plot,

plt.plot(t_plot,theta_plot,t_plot,setpoint_plot,t_plot,servo_plot)
plt.grid(True,linestyle='--')

plt.xlabel('Time',fontsize=16)
plt.ylabel('Pitch Angle',fontsize=16)

plt.show()


########################################################################################################################## HERE THE PROGRAM ENDS

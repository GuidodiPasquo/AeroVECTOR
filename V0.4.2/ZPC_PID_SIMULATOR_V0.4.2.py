# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:23:20 2020

@author: Guido di Pasquo
"""
#Apologies in advance for any spelling or grammar error, english is not my first language

## IGNORE
import matplotlib.pyplot as plt
import numpy as np
import random



class Integrable_Variable:
    f_dot_dot=0.
    f_dot=0.
    f=0.
    
    f_dot_dot_1=0. #previous samples
    f_dot_1=0.
    f_1=0.
    
    f_dot_dot_2=0.
    f_dot_2=0.
    f_2=0.
    
    f_dot_dot_3=0.
    f_dot_3=0.
    f_3=0.
    
    def new_f_dot_dot(self,a):
        self.f_dot_dot_3=self.f_dot_dot_2
        self.f_dot_dot_2=self.f_dot_dot_1
        self.f_dot_dot_1=self.f_dot_dot
        self.f_dot_dot=a
        
    def new_f_dot(self,a):
        self.f_dot_3=self.f_dot_2
        self.f_dot_2=self.f_dot_1
        self.f_dot_1=self.f_dot
        self.f_dot=a
        
    def new_f(self,a):
        self.f_3=self.f_2
        self.f_2=self.f_1
        self.f_1=self.f
        self.f=a
    
    def integrate_f_dot_dot(self):
        self.f_dot_3 = self.f_dot_2
        self.f_dot_2 = self.f_dot_1
        self.f_dot_1 = self.f_dot
        
        #self.delta_f_dot = T * self.f_dot_dot # Euler
        
        self.delta_f_dot = 0.5 * T * (self.f_dot_dot_1 + self.f_dot_dot) # Trapezoidal
        
        #Because the accelerations rotates I'm not a fan of using previous measurements to integrate, so I went for the safer trapezoidal
        #self.delta_f_dot= (T/6) * (self.f_dot_dot_2 + 4 * (self.f_dot_dot_1) + self.f_dot_dot) # Simpson's (Runs each timestep -> (b-a)=h=T)        
        #self.delta_f_dot= (T/8) * (self.f_dot_dot_3 + 3 * (self.f_dot_dot_2) + 3 * self.f_dot_dot_1 + self.f_dot_dot) # Simpson's 3/8       
    
        self.f_dot += self.delta_f_dot   
         
        return self.f_dot
        
        
    def integrate_f_dot(self):
        self.f_3 = self.f_2
        self.f_2 = self.f_1
        self.f_1 = self.f   
        
        self.delta_f = 0.5 * T * (self.f_dot_1 + self.f_dot) # Trapezoidal        
        
        self.f += self.delta_f        
         
        return self.f
    
###########################################
#known bugs-> Arrows are hit or miss, sometimes they aim in the right direction, sometimes they don't




########### OVERALL CHARACTERISTICS OF THE PROGRAM THAT SHOULD BE TAKEN INTO ACCOUNT IN THE FLIGHT COMPUTER CODE

# Non-linear model integrates local accelerations into global velocities. An alternate method of vector derivates is still in the program, results are better with the first method
# Angular velocity is not damped, so when tumbling the rocket will turn way more than in real life.
# If your angular velocities are small, the lack of damping doesn't affect too much the flight, if the velocities aren't small, then the damping in real life would make the real model more stable than in the simulation (most times)

#Important, all angles are in RADIANS (Standard 1º*np.pi/180 = radian)
deg2rad=np.pi/180
rad2deg=1/deg2rad

#Code simulates the TVC_reduction (gear ratio), it multiplies the output of the controller times the TVC_reduction, and then sends that output to the servo.
#Remember that you have to multiply the output of the controller times the TVC reduction in you flight computer!
#All in all, the overall structure of the function "control_theta" and "PID" should be copied in your code to ensure that the simulator and flight computer are doing the same thing

#Parameters related to the servo have the conviniet "s" ending

#Default settings are 2m/s winds and a 2 degree initial offset in the TVC mount. Change them accordingly to suit your model.



########################################################################################################################## SET UP YOUR ROCKET

## MOTOR PARAMETERS

Thrust=28.8 # Leave as it is if you use the Thrust curve, otherwhise set it to you avg thrust
burnout_time=3.5 # Burn time of the motor, stops the simulation

Thrust_curve=True         #Thrust=(max_thrust/(burnout_time/7)*t) DURING 0<t<burnout_time/7
                          # Thrust=-(max_thrust-average_thrust)/(burnout_time/7)*(t-burnout_time/7)+max_thrust DURING burnout_time/7<t<2*burnout_time/7
                          # Thrust=average_thrust DURING 2*burnout_time/7<t<burnout_time
max_thrust=28.8           #If you use thrust curve set them to your motor value
average_thrust=10


## ROCKET PARAMETERS

m=0.451  #  MASS in kg
Iy=0.0662 #MOMENT OF INERTIA, use the smaller one (motor burnout)
d=0.05  # DIAMETER of the rocket and cross sectional area (IN METERS)
xa=0.17  #DISTANCE from the nose to the center of pressure, cg and tail (either the motor mount for tvc or the mac of the control fin), IN METERS
xcg=0.55 #DISTANCE from the nose to the cg IN METERS
xt=0.85  #DISTANCE from the nose to the tail (either the motor mount for tvc or the 25%mac of the control fin), IN METERS
L=0.85 #LENGHT of the rocket
nosecone_length = 0.2 # Self explanatory
CD=0.43 #DRAG COEFFICIENT at alpha=0




## AERODYNAMIC PARAMETERS

k1=-13.119 #CNalpha coeficients
k2=30.193
k3=0.3948
#CNalpha=(k1*alpha**3+k2*alpha**2+k3*alpha)/alpha #Third order aproximation deg, also in 1/radians




## TVC PARAMETERS

TVC_max=10*deg2rad # Maximum gimbal angle
TVC_reduction=5 # A TVC_reduction of 5 means that 5 degrees in the servo equal 1 degree in the TVC mount, use the higher on your TVC mount.
u_initial_offset=2*deg2rad # Initial angle of the TVC mount due to calibration error

#Torque based controller
reference_thrust=28.8 #Thrust for which you tune the rocket, will be used as reference


## WIND PARAMETERS

wind=2; #Wind speed in m/s (positive right to left) 
wind_distribution=0.1  # wind*wind_distribution = max gust speed 






## OTHER PARAMETERS OR VARIABLES
Nalpha=0
g=9.8  # gravity in m/s^2
U=0.001  #Initial velocity
w=0
rho=1.225 # air density
q=0 #dynamic pressure
S=((d**2*np.pi)/4) #cross sectional area
lt=(xt-xcg)  #lenght to the tail
U_prev=0.
U2=0.
wind_rand=0
i_turns=0

############# NEW SIMULATION PARAMETERS
theta = 0
AoA = 0
U = 0
W = 0
Q = 0
U_dot = Integrable_Variable()
W_dot = Integrable_Variable()
Q_dot = Integrable_Variable()
X_dot = Integrable_Variable()
Z_dot = Integrable_Variable()
V_loc=[0.00001,0.00001]
V_loc_tot=[0.00001,0.00001]
V_glob=[0.00001,0.00001]
g_loc=[0.0000,0.0000]
F_loc=[0.0000,0.0000]
F_glob=[0.0000,0.0000]
Position_global=[0,0]

########################################################################################################################## IGNORE UP TO PID GAINS

####PLOTS
t_plot=[]
theta_plot=[]
setpoint_plot=[]
servo_plot=[]


#3D plots
t_3D=[]
theta_3D=[]
setpoint_3D=[]
servo_3D=[]
V_loc_3D=[]
V_glob_3D=[]
Airspeed_3D=[]
Position_3D=[]
X_3D=[]
Z_3D=[]
Nalpha_3D=[]
Thrust_3D=[]
xa_3D=[]
AoA_3D=[]


alpha_calc=0.
AoA=0.
U_vect=np.array([0.1,0])
V_vect=np.array([0.1,0])
wind_vect=np.array([0,wind])
u_eq=0.
u_prev=0.
u_delta=0.
u_controller=0.


#SERVO (SG90)

u_servos=0.
TVC_weight_compensation=1.75 # After watching joe's video on servos they are a tad slower that the ones I measured, this simulates that. #1.3 is the servo alone

as11, as12, as21, as22 = (0.0,)*4
bs11, bs21=(0.0,)*2
cs11, cs12, cs21, cs22 = (0.0,)*4
ds11, ds21=(0.0,)*2

asc11, asc12, asc21, asc22 = (0.0,)*4
bsc11, bsc21=(0.0,)*2
csc11, csc12, csc21, csc22 = (0.0,)*4

u=0.

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



#TIMERS
timer_run=0
t_timer_3D=0
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

# Transforms from local coordinates to global coordinates
# (body to world)
def loc2glob(u0,v0,theta):
    A=np.array([[np.cos(theta), np.sin(theta)],             #Rotational matrix 2x2
                [-np.sin(theta), np.cos(theta)]])           # Axes are rotated, there is more info in the Technical documentation 
    
    u=np.array([[u0],[v0]])
    
    x=np.dot(A,u)
    
    a=[x[0,0],x[1,0]]
    
    return a

def glob2loc(u0,v0,theta):
    A=np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
    
    u=np.array([[u0],[v0]]) 
    
    x=np.dot(A,u)    

    a=[x[0,0],x[1,0]]
    
    return a
    


# CONTROL FUNCTION - FLIGHT COMPUTER
def control_theta(setpoint):
    global error
    global u_controller,u_prev,u_delta,u_servos

    
    u_prev=u_controller
    
    error=setpoint-theta # On your flight computer, replace theta for you calculated angle
    error=error*k_all
    u_controller=PID(error)
    u_controller=u_controller - Q*k_damping # On your flight computer, replace Q for the gyroscope data (angular speed)
    
    if(u_controller>TVC_max):  # prevents the TVC from deflecting more that it can
        u_controller=TVC_max
    elif(u_controller<-TVC_max):
        u_controller=-TVC_max
        
    #u_controller=u_controller-u_prev*0.05;  #filter, increasing the number makes it stronger and slower
    
    u_servos=u_controller*TVC_reduction #compensates the TVC reduction (gear ratio)
    
    if(torque_controller==True):

        thrust_controller=Thrust #On the simulation one can access the real Thrust from the thrust curve
                                 #In your flight computer you will have to calculate it
        u_servos=(reference_thrust/thrust_controller)*u_servos
        if(u_servos>TVC_max*TVC_reduction):  #prevents the TVC from deflecting more that it can
            u_servos=TVC_max*TVC_reduction
        elif(u_servos<-TVC_max*TVC_reduction):
            u_servos=-TVC_max*TVC_reduction

    
    return u_controller



# PID
def PID(inp):
    global t
    global T_Program  #elapsed time between runs
    global lastError,cumError
    global TVC_max,TVC_reduction,anti_windup
    

    errorPID = inp;                                 # determine error
    rateError = (errorPID - lastError) / T_Program  # compute derivative

    
    
    if(anti_windup==True):
        out_pid = kp * errorPID + ki * cumError + kd * rateError;          #PID output
        
         #Anti windup by clamping
        if(out_pid < (TVC_max ) and out_pid > (-TVC_max )):
            cumError = ((((lastError) + ((errorPID - lastError) / 2))) * T_Program)+cumError    # compute integral (trapezoidal) only if the TVC is not staurated
            out_pid = kp * errorPID + ki * cumError + kd * rateError                            # recalculate the output
    
        # Saturation, prevents the TVC from deflecting more that it can
        if(out_pid>TVC_max ):   
            out_pid=TVC_max 
        elif(out_pid<-TVC_max ):
            out_pid=-TVC_max 
    else:
        cumError = ((((lastError) + ((errorPID - lastError) / 2))) * T_Program)+cumError
        out_pid = kp * errorPID + ki * cumError + kd * rateError;                            #PID output
        
        
    
    lastError = errorPID    #remember current error

    return out_pid          #have function return the PID output



def update_servo():      
    global Asc,Bsc,Csc,As,Bs,Cs,Ds,u_delta, u_servos    
    
    u_delta=TVC_weight_compensation*abs(u_servos-outs[0,0]*TVC_reduction)   #only for the simualtion, does nothing in the real flight computer
                                                                            #The tvc compensation slows down the servo to simulate the TVC weight, since the characterization was done without resistance 
                                                                            #outs[0,0]*TVC_reduction follows the current servo angle instead of the TVC angle

    
    if(u_delta<=10/57.3):
        u_delta=10/57.3        
    elif(u_delta>=90/57.3):
        u_delta=90/57.3
    
    
        
    # A Matrix    
    asc11=0.
    asc12=1.
    asc21=-(-2624.5 * u_delta**3 + 9996.2 * u_delta**2 - 13195 * u_delta + 6616.2)  # Straights that modify the model of the sevo based on its current position and setpoint
    asc22=-(39.382 * u_delta**2 - 125.81 * u_delta + 124.56)
    

    Asc=np.array([[asc11, asc12], [asc21, asc22]])
    
    # B Matrix
    bsc11=0
    bsc21=(-2624.5 * u_delta**3 + 9996.2 * u_delta**2 - 13195 * u_delta + 6616.2)
    

    Bsc=np.array([[bsc11],
                 [bsc21]])
    
    # C Matrix    
    # D Matrix
    
    As=np.dot(np.linalg.inv((2/T)*Itwo-Asc),((2/T)*Itwo+Asc)) #Now discretization is done in the program
    Bs=np.dot(np.linalg.inv((2/T)*Itwo-Asc),Bsc)
    Cs=(As+Csc)
    Ds=Bs
    
    
    return








def update_parameters():
    global wind_rand
    global q
    global Nalpha
    global x
    global xa
    global i
    global AoA
    global wind
    global Thrust
    global out,timer_disturbance,timer_U,U2,q_wind
    
    
    # NEW SIMULATION
    global V_loc , V_loc_tot , V_glob
    global U_dot , U , X
    global W_dot , W , Z
    global Q_dot , Q , Q_1
    global theta, AoA, g , g_loc
    
    if Thrust_curve==True:
        Thrust=Thrust_calculator(t,burnout_time)
    
    # Times the disturbances so they don't change that often
    if(t>timer_disturbance+2*T_Program*0.9999):
        wind_rand=(random.uniform(-wind, wind))*wind_distribution 
        timer_disturbance=t
   
    
    # NEW SIMULATION
    
    wind_loc = glob2loc(0, wind+wind_rand, theta) # Computes the velocity of the wind in local coordinates    
    
    V_loc_tot = [ V_loc[0]-wind_loc[0] , V_loc[1]-wind_loc[1] ] # Computes the total airspeed in local coordinates
    
    if V_loc_tot[0] != 0:    
        AoA = np.arctan2(V_loc_tot[1] , V_loc_tot[0]) # Computes the angle of attack
    else:
        AoA = np.pi/2
    
    # Movable CP , Uses linear functions but it could use anything
    xa=CP_calculator(AoA)
    
    # Non-linear Normal Force coeficient, uses third order polinomials but it could use anything
    Nalpha = Nalpha_calculator(AoA) 
        
    Nalpha=-Nalpha # Normal force and Z axes are opposite    
        
    q = 0.5 * rho * (np.sqrt( V_loc_tot[0]**2 + V_loc_tot[1]**2 ))**2 # Computes the dynamic pressure
    
    g_loc = glob2loc(-g,0,theta) # Gravity in local coordinates, theta=0 equals to rocket up
           
      
    
    return
    
    


def Thrust_calculator(t,burnout_time):
    
    if(t<burnout_time/7):            
        Thrust=((max_thrust-m*g)/(burnout_time/7)*t)+m*g
    elif(t>=(burnout_time/7) and t<(2*burnout_time/7)):
        Thrust=-(max_thrust-average_thrust)/(burnout_time/7)*(t-burnout_time/7)+max_thrust
    elif(t>=(2*burnout_time/7) and t<burnout_time):
        Thrust=average_thrust
    else:
        Thrust=0.000000001    
    

    return Thrust

        
        
def CP_calculator(AoA):
    x=abs(AoA)   
    x_cp=0.15
    
    if x<0.19296:
        x_cp=1.2902*x + 0.1527
        
    elif x>=0.19296 and x<0.5:
        x_cp=0.2226*x + 0.3587
        
    elif x>=0.5 and x<np.pi/2:
        x_cp=0.0374*x + 0.4513
        
    elif x>=np.pi/2 and x<np.pi-0.5:
        x_cp=0.0187*x + 0.4807
    
    elif x>=np.pi-0.5 and x<np.pi-0.19296:
        x_cp=0.2226*x - 0.0581
        
    elif x>=np.pi-0.19296:
        x_cp=1.2902*x - 3.2061
   
    # This CP movement was based in a 1m long filess rocket 
    x_cp = x_cp * (L / 1)
    
    return x_cp 


def Nalpha_calculator(AoA):    
    
    # sin()+sin**2() aproximated with polinomials, they are valid from 0 to 90º so there is a little bit of meddleing arround to make it work
    # lookup tables could have been a good alternative
    
    if(AoA>=0 and AoA<np.pi/2):        
        Nalpha = k1 * AoA**3 + k2 * AoA**2 + k3 * AoA
        
    elif(AoA<0 and AoA>=-np.pi/2):
        Nalpha = -( k1 * abs(AoA)**3 + k2 * abs(AoA)**2 + k3 * abs(AoA))
        
    elif(AoA<-np.pi/2):        
        AoA_loc=-AoA-np.pi # Eg, -91º = -89º
        Nalpha = -( k1 * abs(AoA_loc)**3 + k2 * abs(AoA_loc)**2 + k3 * abs(AoA_loc))
        
    elif(AoA>=np.pi/2):
        AoA_loc=np.pi-(AoA)  # Eg, 91º = 89º
        Nalpha = k1 * AoA_loc**3 + k2 * AoA_loc**2 + k3 * AoA_loc 
    else:
        Nalpha = 0
        
    return Nalpha



      
k=0
def simulation():
    
    global x,xs,k
    global xdot,xdots
    global out,outs
    global out_prev,out_prevs
    
    global u_controller
    global u,timer_run_servo,u_servos
    
    global V_loc , V_loc_tot , V_glob
    global U_dot , U , X
    global W_dot , W , Z
    global Q_dot , Q , Q_1
    global theta_1 , theta, AoA, g , g_loc
    global F_loc , F_glob
    global Nalpha , Thrust
    global t_timer_3D
    global Position_global
    global i_turns
    
    
    
    
    # UPDATES
    update_parameters()     
    update_servo()
    
    
    
    # SERVO SIMULATION
    if(t>timer_run_servo+Ts*0.999+0.001):  #+0.001 desincronizes the servo and the program, thing that would likely happen in a real flight computer
        timer_run_servo=t
        u=u_servos
        u=round(u*rad2deg,0)*deg2rad #definition of the servo, standard 1º
        
    xdots=(np.dot(As,xs)+np.dot(Bs,(u)))
    outs=(np.dot(Cs,xs)+np.dot(Ds,u))
    xs=xdots
    
    outs[0,0]=(outs[0,0]/TVC_reduction) #reduction of the TVC    

   
    
    
    """
    NEW METHOD, DIRECTLY INTEGRATES THE DIFFERENTIAL EQUATIONS
    U=local speed in X
    W=local speed in Z
    Q=pitch rate
    AoA=angle of attack
    
    V_glob = global velocity
    X_dot = global X speed (Y in Vpython)
    Z_dot = global Z speed (-X in Vpython)
    """
     
    
    v_d=0 # 0 uses Local and Global Velocities, 1 uses vector derivatives.
    
    if np.sign(V_loc[0]) > 0:
        Drag = q*S*CD        
    else:
        Drag=-5*q*S*CD  # Increased drag due to the rocket flying backwards
    
    Accx = ( Thrust*np.cos(outs[0,0]+u_initial_offset) + m*g_loc[0] - Drag) / m - W*Q*v_d                # Longitudinal Acceleration (local)
    Accz = ( Thrust*np.sin(outs[0,0]+u_initial_offset) + m*g_loc[1] + q*S*Nalpha) / m + U*Q*v_d          # Transversal Acceleration (local)
    AccQ = ( Thrust*np.sin(outs[0,0]+u_initial_offset) * (xt-xcg) + q*S*Nalpha * (xa-xcg)) / Iy            # Angular acceleration
    
    # Updates the variables
    U_dot.new_f_dot_dot(Accx)
    W_dot.new_f_dot_dot(Accz)
    Q_dot.new_f_dot_dot(AccQ)
    
    # Integrates the angular acceleration and velocity    
    Q = Q_dot.integrate_f_dot_dot()
    theta = Q_dot.integrate_f_dot()
 
    
    # In case theta is greater than 180º, to keep it between -180 and 180
    if theta > np.pi:
        theta -= 2*np.pi
        Q_dot.new_f(theta)
        i_turns+=1
        
    if theta < -np.pi:
        theta += 2*np.pi
        Q_dot.new_f(theta)
        i_turns+=1
    # It's alright to do this as long as theta is not integrated

    if v_d==1:
        # Just integrates, the transfer of velocities was already done in the vector derivative          
        V_loc[0] = U_dot.integrate_f_dot_dot()
        V_loc[1] = W_dot.integrate_f_dot_dot()
    else:
        # Takes the global velocity, transforms it into local coordinates, adds the accelerations and transforms the velocity back into global coordinates
        V_loc = glob2loc(V_glob[0], V_glob[1], theta)
        U_dot.integrate_f_dot_dot()
        W_dot.integrate_f_dot_dot()
        V_loc[0] += U_dot.delta_f_dot
        V_loc[1] += W_dot.delta_f_dot 
        
        
         
    V_glob=loc2glob(V_loc[0], V_loc[1], theta)      # New velocity in global coordinates
    
    # Updates the global velocity in the X_dot class
    X_dot.new_f_dot(V_glob[0])
    Z_dot.new_f_dot(V_glob[1])
    
    # Integrates the velocities to get the position, be it local?¿ or global
    Position_local = [U_dot.integrate_f_dot() , W_dot.integrate_f_dot()]
    
    Position_global = [X_dot.integrate_f_dot() , Z_dot.integrate_f_dot()]
    

        
    
    """
    Adding -W*Q to U_dot and +U*Q to W_dot but eliminating the global to local transfer of the velocity accounts for the vector totation
    Using the vector derivative (U_dot = .... - W*Q and W_dot = .... + U*Q) is the same as transforming the global vector in local coordinates, adding the local accelerations and transforming it back to global
    So:
        Vector Derivative -> No need to transform the velocity from global to local, you work only with the local
        No Vector Derivative -> Equations are simpler, but you have to transform the global vector to local and then to global again
        
        Still have to see how it scales with more DOF
    """
 
    # Only saves the points used in the animation
    # (1000) is the rate of the animation, when you use slow_mo it drops, to ensure fluidity at least a rate of 100 ish is recommended, so a rate of 1000 allows for 10 times slower animations
    if t>=t_timer_3D+((1/1000))*0.999:      
   
        #### 3D
        t_3D.append(t)
        theta_3D.append(theta)
        servo_3D.append(outs[0,0]+u_initial_offset)
        V_loc_3D.append(V_loc)
        V_glob_3D.append(V_glob)
        Position_3D.append(Position_global)
        xa_3D.append(xa)
        Thrust_3D.append(Thrust)
        Nalpha_3D.append(Nalpha*S*q)
        AoA_3D.append(AoA)
        setpoint_3D.append(setpoint)
        t_timer_3D=t

  
    return


def timer():
    global t
    t=round(t+T,12) #Trying to avoid error, not sure it works
    return




def set_setpoint(inp):
    
    if(inp==1):
        setpoint=23*deg2rad
    elif(inp==2):
        setpoint=(10*deg2rad)*(t-0.5)
    else:
        setpoint=0
        
    return setpoint




########################################################################################################################## 3D PARAMETERS

toggle_3D=True
camera_shake_toggle=False #Camera Shake
slow_mo=1 # How much slower the 3D animation goes -> 1 = Real time, 5 = five times slower
force_reduction=5 #smaller or bigger arrows 1= one meter per newton, 5=0.2m/N
hide_forces=False # Truns off forces 
Camera_tipe="Fixed" # "Follow" -> Follows the rocket , "Fixed" -> Watches from the ground , "Follow Far" -> Almost 2D


########################################################################################################################## PID GAINS
kp=0.4 #0.4  #Gains might be quite high to compensate for aerodynamic forces
ki=0 #0
kd=0.136 #0.136
k_all=1 
k_damping=0 #when possible use gyro feedback instead of derivative so as to not amplify noise

anti_windup=True #Prevents the integrator for integreting when the TVC is saturated
                 #it also limits the output of the PID so k_damping is more efective

torque_controller=True #Activates the torque controller

inp=1 #selects the input
      #1-> Step, hard test on stability
      #2-> 10º/s ramp, slow pitch over
      #3-> only stabilization against disturbances, simulates the real, straight-up flight


T=0.001 #0.001   #T=Sample time of the simulation 
Ts=0.02   #Ts=sample time of the servo (0.02 for a SG90)
T_Program=0.01 #0.01 #T_Program: Sample time of your PID code
Sim_duration=30 #How long will it simulate
########################################################################################################################## HERE THE PROGRAM STARTS

while t<=Sim_duration:
    if(t>burnout_time*10):
        break
    
    if(i_turns>=5):
        print("Pitch angle greater than 180\xb0, the rocket is flying pointy end down.")
        break
    
    if Position_global[0]<-1:
        print("CRASH")
        break
    
    # *.999 corrects the error in t produced by adding t=t+T for sample times smaller than 0.001
    # If it was exact, it starts acumulating error, and the smaller the sample time, the more error it accumulates
    # So, be carefull with the sample time, 0.001, 0.0005 and 0.0001 give all similar results, so, if it takes to long to run you can increase T to 0.001
    if t>=timer_run_sim+T*0.999:
      
        if(t>=timer_run+T_Program*0.999):
            timer_run=t   
            
            if(t>=0.5):
                setpoint=set_setpoint(inp)                  
            control_theta(setpoint)
            
        timer_run_sim=t
        simulation()
    
    timer()
       
    setpoint_plot.append(setpoint)
    theta_plot.append(theta)
    servo_plot.append(outs[0,0])
    
    #Plot selectors
    #theta->Pitch Angle, Q-> Pitch Rate, AoA->Angle of attack
    #u_controller->controler output, outs[0,0]->Real servo angle, outs[1,0]->Servo speed
    
    t_plot.append(t)
    
 

plt.plot(t_plot,theta_plot,t_plot,setpoint_plot,t_plot,servo_plot)
plt.grid(True,linestyle='--')

plt.xlabel('Time',fontsize=16)
plt.ylabel('Pitch Angle',fontsize=16)

plt.show()


########################################################################################################################## HERE THE PROGRAM ENDS
########################################################################################################################## HERE THE PROGRAM ENDS
########################################################################################################################## HERE THE PROGRAM ENDS
########################################################################################################################## HERE THE PROGRAM ENDS


if toggle_3D==True:
    import vpython as vp
   
    #Creates the window
    scene=vp.canvas(width=1280,height=720,center=vp.vector(0,0,0),background=vp.color.white)
    scene.lights = []
    vp.distant_light(direction=vp.vector(1 , 1 , -1), color=vp.color.gray(0.9))
    
    t=0
    i=0
    #floor
    dim_x_floor=1000
    dim_z_floor=1000  
  
    vp.box(pos=vp.vector(dim_x_floor/2 , -0.5 , dim_z_floor/2),size=vp.vector(dim_x_floor,1,dim_z_floor) , texture={'file':'grass_texture.jpg'})
    
    

    # Sky    
    vp.box(pos=vp.vector(dim_x_floor/2,dim_z_floor/2,dim_z_floor/2+20*2) , size = vp.vector(dim_x_floor,dim_z_floor,1) , texture={'file':'sky_texture.jpg'})
    
    rod=vp.cylinder(pos=vp.vector(dim_x_floor/2,0,dim_z_floor/2),axis=vp.vector(0,1,0),radius=d/2,color=vp.color.black,length=L-nosecone_length)
    nosecone=vp.cone(pos=vp.vector(dim_x_floor/2,(L-nosecone_length),dim_z_floor/2),axis=vp.vector(0,1,0),radius=d/2,color=vp.color.black,length=nosecone_length)
    
    rocket=vp.compound([rod, nosecone])
    motor=vp.cone(pos=vp.vector(dim_x_floor/2,0,dim_z_floor/2),axis=vp.vector(0,-1,0),radius=(d-0.03)/2,color=vp.color.red,length=0.15,make_trail=True)
    
    motor.trail_color=vp.color.red
    motor.rotate(u_initial_offset,axis=vp.vector(0,0,-1))
    
    Tmotor_pos=vp.arrow(pos=vp.vector(motor.pos.x-(d/2+0.015),motor.pos.y,motor.pos.z),axis=vp.vector(0.0001,0,0),shaftwidth=0,color=vp.color.red) #Torque motor
    Tmotor_neg=vp.arrow(pos=vp.vector(motor.pos.x+(d/2+0.015),motor.pos.y,motor.pos.z),axis=vp.vector(-0.0001,0,0),shaftwidth=0,color=vp.color.red) #Torque motor
    Nforce_pos=vp.arrow(pos=vp.vector(motor.pos.x+(d/2+0.015),motor.pos.y+(L-xa_3D[0]),motor.pos.z),axis=vp.vector(0.0001,0,0),shaftwidth=0,color=vp.color.green) 
    Nforce_neg=vp.arrow(pos=vp.vector(motor.pos.x-(d/2+0.015),motor.pos.y+(L-xa_3D[0]),motor.pos.z),axis=vp.vector(0.0001,0,0),shaftwidth=0,color=vp.color.green) 
    
    Nforce_neg.visible=False
    Nforce_pos.visible=False
    Tmotor_pos.visible=False
    Tmotor_neg.visible=False
        
    #Labels
    labels=vp.canvas(width=1280,height=200,center=vp.vector(0,0,0),background=vp.color.white)
    
    Setpoint_label = vp.label(canvas=labels, pos=vp.vector(-60,7,0), text="Setpoint = %.2f"+str(setpoint_plot[i]*rad2deg), xoffset=0, zoffset=0, yoffset=0, space=30, height=30, border=0,font='sans',box=False,line=False , align ="left")
    theta_label = vp.label(canvas=labels, pos=vp.vector(-60,4,0), text="Setpoint = %.2f"+str(setpoint_plot[i]*rad2deg), xoffset=0, zoffset=0, yoffset=0, space=30, height=30, border=0,font='sans',box=False,line=False , align ="left")
    V = vp.label(canvas=labels, pos=vp.vector(-60,1,0), text="Velocity"+str(setpoint_plot[i]*rad2deg), xoffset=0, zoffset=0, yoffset=0, space=30, height=30, border=0,font='sans',box=False,line=False,align ="left")
    AoA_plot = vp.label(canvas=labels, pos=vp.vector(-60,-2,0), text="Velocity"+str(setpoint_plot[i]*rad2deg), xoffset=0, zoffset=0, yoffset=0, space=30, height=30, border=0,font='sans',box=False,line=False,align ="left")
    Altitude = vp.label(canvas=labels, pos=vp.vector(-60,-5,0), text="Velocity"+str(setpoint_plot[i]*rad2deg), xoffset=0, zoffset=0, yoffset=0, space=30, height=30, border=0,font='sans',box=False,line=False,align ="left")
    
    
    i=0    
    for i in range(len(theta_3D)-2):
        vp.rate(1000/slow_mo) #60 FPS
        
        # How much to move each time-step , X and Z are the rocket's axes, not the world's
        delta_pos_X=(Position_3D[i+1][0]-Position_3D[i][0]) 
        delta_pos_Z=(Position_3D[i+1][1]-Position_3D[i][1])
        

        # Moving the rocket
        rocket.pos.y+=delta_pos_X 
        rocket.pos.x+=delta_pos_Z     
        
        
        # Creates a cg and cp vector with reference to the origin of the 3D rocket (not the math model rocket)
        xcg_radius = loc2glob((L-xcg),0,theta_3D[i])
        xa_radius = loc2glob(xa_3D[i+1]-xa_3D[i],0,theta_3D[i])  #Delta xa_radius, this is then integrated when you move the Aerodynamic Force arrow
        
        #CP and CG global vectors
        vect_cg=vp.vector(rocket.pos.x+xcg_radius[0] , rocket.pos.y + xcg_radius[1] , 0)
        vect_cp=vp.vector(rocket.pos.x+xa_radius[0],rocket.pos.y+xa_radius[1],0)
        
        # Rotate rocket from the CG
        rocket.rotate((theta_3D[i+1]-theta_3D[i]),axis=vp.vector(0,0,1),origin=vect_cg)        
        
        # Move the motor together with the rocket
        motor.pos.y+=delta_pos_X 
        motor.pos.x+=delta_pos_Z 
        motor.rotate((theta_3D[i+1]-theta_3D[i]),axis=vp.vector(0,0,1),origin=vect_cg)  # Rigid rotation with the rocket 
        motor.rotate((servo_3D[i+1]-servo_3D[i]),axis=vp.vector(0,0,-1))                # TVC mount rotation 
        
          
        # Motor Burnout, stops the red trail of the rocket
        if(t_3D[i]>=burnout_time):
            motor.visible=False
            motor.make_trail=False
            Tmotor_pos.visible=False
            Tmotor_neg.visible=False
               
        else:
            aux=np.sin(servo_3D[i])*Thrust_3D[i]/force_reduction # Arrows are hit or miss, tried this to avoid them going in the wrong direction, didn't work
            
            # Makes visible one arrow or the other, be it left or right
            if aux>0:
                Tmotor_pos.visible=False
                Tmotor_neg.visible=True
            else:
                Tmotor_pos.visible=True 
                Tmotor_neg.visible=False            
            
            # Displacements and rotations of the arrows

            Tmotor_pos.pos.y+=delta_pos_X
            Tmotor_pos.pos.x+=delta_pos_Z     
            Tmotor_pos.rotate((theta_3D[i+1]-theta_3D[i]),axis=vp.vector(0,0,1),origin=vect_cg)
            Tmotor_pos.axis=vp.vector(aux,0,0)     
            Tmotor_pos.rotate(theta_3D[i],axis=vp.vector(0,0,1))
            

            Tmotor_neg.pos.y+=delta_pos_X
            Tmotor_neg.pos.x+=delta_pos_Z     
            Tmotor_neg.rotate((theta_3D[i+1]-theta_3D[i]),axis=vp.vector(0,0,1),origin=vect_cg)
            Tmotor_neg.axis=vp.vector(aux,0,0)     
            Tmotor_neg.rotate(theta_3D[i],axis=vp.vector(0,0,1))
            
         
            
            
            
      
        #Normal force arrow
        
        # Same as before, makes the one active visible
        if Nalpha_3D[i]<=0:
            Nforce_pos.visible=False
            Nforce_neg.visible=True
        else:
            Nforce_pos.visible=True 
            Nforce_neg.visible=False      
        
        # Displacements and rotations
         
        Nforce_pos.pos.y+=delta_pos_X - xa_radius[0]
        Nforce_pos.pos.x+=delta_pos_Z - xa_radius[1]
        Nforce_pos.axis=vp.vector(Nalpha_3D[i]/force_reduction,0,0) 
        Nforce_pos.rotate(theta_3D[i],axis=vp.vector(0,0,1), origin=Nforce_pos.pos)    
        Nforce_pos.rotate((theta_3D[i+1]-theta_3D[i]),axis=vp.vector(0,0,1),origin=vect_cg)
        
            
        Nforce_neg.pos.y+=delta_pos_X - xa_radius[0] 
        Nforce_neg.pos.x+=delta_pos_Z - xa_radius[1]
        Nforce_neg.axis=vp.vector(Nalpha_3D[i]/force_reduction,0,0) 
        Nforce_neg.rotate(theta_3D[i],axis=vp.vector(0,0,1), origin=Nforce_neg.pos)
        Nforce_neg.rotate((theta_3D[i+1]-theta_3D[i]),axis=vp.vector(0,0,1),origin=vect_cg)
    
    
        if hide_forces==True:
            Nforce_pos.visible = False
            Nforce_neg.visible = False
            Tmotor_pos.visible = False
            Tmotor_neg.visible = False
    
        #To avoid the ugly arrows before the rocket starts going
        if V_glob_3D[i][0]<0.0001:
            Nforce_neg.visible = False
            Nforce_pos.visible = False
            Tmotor_pos.visible = False
            Tmotor_neg.visible = False
    
        #Camera
        if camera_shake_toggle==True:
            camera_shake=loc2glob(V_glob_3D[i][0],V_glob_3D[i][1],theta_3D[i])
        else:
            camera_shake=[0,0]
        
        #Follows almost 45 deg up
        if Camera_tipe=="Follow":
            #scene.fov = 4*deg2rad
            # scene.camera.pos = vp.vector(rocket.pos.x+camera_shake[1]/50,rocket.pos.y+15-camera_shake[0]/500,rocket.pos.z-10)
            # scene.camera.axis=vp.vector(rocket.pos.x-scene.camera.pos.x , rocket.pos.y-scene.camera.pos.y , rocket.pos.z-scene.camera.pos.z)
            scene.camera.pos = vp.vector(rocket.pos.x+camera_shake[1]/50,rocket.pos.y+1.2-camera_shake[0]/500,rocket.pos.z-1)
            scene.camera.axis=vp.vector(rocket.pos.x-scene.camera.pos.x , rocket.pos.y-scene.camera.pos.y , rocket.pos.z-scene.camera.pos.z)
            
        # Simulates someone in the ground
        elif Camera_tipe=="Fixed":
            scene.fov=4*deg2rad
            scene.camera.pos = vp.vector(dim_x_floor/2,1,dim_z_floor/2-70)
            scene.camera.axis = vp.vector(rocket.pos.x-scene.camera.pos.x , rocket.pos.y-scene.camera.pos.y , rocket.pos.z-scene.camera.pos.z)
          
        # Lateral camera, like if it was 2D
        elif Camera_tipe=="Follow Far":
            scene.fov=60*deg2rad
            scene.camera.pos = vp.vector(rocket.pos.x+camera_shake[1]/50,rocket.pos.y+0.0-camera_shake[0]/200,rocket.pos.z-5)
            scene.camera.axis=vp.vector(rocket.pos.x-scene.camera.pos.x , rocket.pos.y-scene.camera.pos.y , rocket.pos.z-scene.camera.pos.z)
            
        
        #Labels       
        Setpoint_label.text = "Setpoint = %.0f" % round(setpoint_3D[i]*rad2deg,1) + u'\xb0'
        theta_label.text = "Pitch Angle = " + str(round(theta_3D[i]*rad2deg,2)) + u'\xb0'
        V.text = "Local Velocity = " + str(V_loc_3D[i]) + " m/s"
        AoA_plot.text = "AoA = " + str(round(AoA_3D[i]*rad2deg,2)) + u'\xb0'
        Altitude.text = "Altitude = " + str(round(Position_3D[i][0],2)) + "m"
        
        if Position_3D[i][0]<-0:
            break
    

    

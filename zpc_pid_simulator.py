# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:23:20 2020

@author: Guido di Pasquo
"""

"""
Thanks to:
     LukeDeWaal for the Standard Atmosphere Calculator
     https://github.com/LukeDeWaal/ISA_Calculator

###########################################
Apologies in advance for any spelling or grammar error, english is not my first language
known bugs-> Arrows are hit or miss, sometimes they aim in the right
direction, sometimes they don't.

########### OVERALL CHARACTERISTICS OF THE PROGRAM THAT SHOULD BE
TAKEN INTO ACCOUNT IN THE FLIGHT COMPUTER CODE.

Non-linear model integrates local accelerations into global
velocities. An alternate method of vector derivates is still in the
program, results are better with the first method.

Important, all angles are in RADIANS (Standard 1º*np.pi/180 = radian)

DEG2RAD=np.pi/180
RAD2DEG=1/DEG2RAD

Code simulates the Actuator_reduction (gear ratio), it multiplies the
output of the controller times the Actuator_reduction, and then sends
that output to the servo.
Remember that you have to multiply the output of the controller times
the Actuator reduction in you flight computer!
All in all, the overall structure of the function "control_theta" and
"PID" should be copied in your code to ensure that the simulator and
flight computer are doing the same thing

Parameters related to the servo have the convenient "s" ending.
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import random
import vpython as vp
import time
import gui_setup as gui
import rocket_functions as rkt
import ISA_calculator as atm
import control
import servo_lib

rocket = rkt.Rocket()
atmosphere = atm.get_atmosphere()
controller = control.Controller()
servo = servo_lib.Servo()

DEG2RAD = np.pi / 180
RAD2DEG = 1 / DEG2RAD

## WIND PARAMETERS
wind = 2 #Wind speed in m/s (positive right to left)
wind_distribution = 0.1  # wind*wind_distribution = max gust speed

## OTHER PARAMETERS OR VARIABLES
cn = 0
ca = 0
cm_xcg = 0
xa = 0
fin_force = 0
g = 9.8  # gravity in m/s^2
U = 0.001  #Initial velocity
w = 0
rho = 1.225 # air density
q = 0 #dynamic pressure
U_prev = 0.
U2 = 0.
wind_rand = 0
i_turns = 0
actuator_angle = 0
CA0 = 0
wind_total = 0

############# NEW SIMULATION PARAMETERS
class IntegrableVariable:
    def __init__(self):
        self.f_dd = 0.
        self.f_d = 0.
        self.f = 0.
        self.f_dd_1 = 0. #previous samples
        self.f_d_1 = 0.
        self.f_1 = 0.
        self.f_dd_2 = 0.
        self.f_d_2 = 0.
        self.f_2 = 0.
        self.f_dd_3 = 0.
        self.f_d_3 = 0.
        self.f_3 = 0.

    def new_f_dd(self, a):
        self.f_dd_3 = self.f_dd_2
        self.f_dd_2 = self.f_dd_1
        self.f_dd_1 = self.f_dd
        self.f_dd = a

    def new_f_d(self, a):
        self.f_d_3 = self.f_d_2
        self.f_d_2 = self.f_d_1
        self.f_d_1 = self.f_d
        self.f_d = a

    def new_f(self, a):
        self.f_3 = self.f_2
        self.f_2 = self.f_1
        self.f_1 = self.f
        self.f = a

    def integrate_f_dd(self):
        self.f_d_3 = self.f_d_2
        self.f_d_2 = self.f_d_1
        self.f_d_1 = self.f_d
        #self.delta_f_d = T * self.f_dd # Euler
        self.delta_f_d = 0.5 * T * (self.f_dd_1+self.f_dd) # Trapezoidal
        # Because the accelerations rotates I'm not a fan of using previous
        # measurements to integrate, so I went for the safer trapezoidal

        # self.delta_f_d= (T/6) * (self.f_dd_2 + 4 * (self.f_dd_1) + self.f_dd)
        # Simpson's (Runs each timestep -> (b-a)=h=T)

        # self.delta_f_d= (T/8) * (self.f_dd_3 + 3 * (self.f_dd_2) + 3 * self.f_dd_1 + self.f_dd)
        # Simpson's 3/8
        self.f_d += self.delta_f_d
        return self.f_d

    def integrate_f_d(self):
        self.f_3 = self.f_2
        self.f_2 = self.f_1
        self.f_1 = self.f
        self.delta_f = 0.5 * T * (self.f_d_1 + self.f_d) # Trapezoidal
        self.f += self.delta_f
        return self.f

theta = 0
aoa = 0
U = 0
W = 0
Q = 0
accx = 0
accy = 0
U_d = IntegrableVariable()
W_d = IntegrableVariable()
Q_d = IntegrableVariable()
X_d = IntegrableVariable()
Z_d = IntegrableVariable()
V_loc = [0.00001,0.00001]
V_loc_tot = [0.00001,0.00001]
V_glob = [0.00001,0.00001]
g_loc = [0.0000,0.0000]
acc_glob = [0.0001,0.0001]
F_loc = [0.0000,0.0000]
F_glob = [0.0000,0.0000]
position_global = [0,0]

####################################################### IGNORE UP TO PID GAINS

####PLOTS
t_plot = []
first_plot = []
second_plot = []
third_plot = []
fourth_plot = []
fifth_plot = []

#3D plots
t_3d = []
theta_3d = []
setpoint_3d = []
servo_3d = []
V_loc_3d = []
V_glob_3d = []
Airspeed_3d = []
position_3d = []
X_3d = []
Z_3d = []
cn_3d = []
fin_force_3d = []
thrust_3d = []
xa_3d = []
aoa_3d = []

# Other Variables
alpha_calc = 0.
aoa = 0.
U_vect = np.array([0.1,0])
V_vect = np.array([0.1,0])
wind_vect = np.array([0,wind])

#PID
anti_windup = True

#CONTROL
setpoint = 0.
error = 0.
okp, oki, okd, totError = (0.0,)*4

#TIMERS
timer_run = 0
t_timer_3d = 0
#timer = 0
timer_run_sim = 0
timer_run_servo = 0
t = 0.
timer_disturbance = 0.
timer_U = 0.

#FLAGS
flag = False
flag2 = False

# SITL
Activate_SITL = False
port = "COM3"
baudrate = 115200
send_gyro = 0
send_accx = 0
send_accz = 0
send_alt = 0

################################################################ FUNCTIONS

def get_data_savefile():
    param = gui.param_file_tab.get_configuration_destringed()
    rocket_dim = gui.draw_rocket_tab.get_configuration_destringed()
    conf_3d = gui.conf_3d_tab.get_configuration_destringed()
    conf_sitl = gui.conf_sitl_tab.get_configuration_destringed()
    conf_controller = gui.sim_setup_tab.get_configuration_destringed()
    return param, conf_3d, conf_controller, conf_sitl, rocket_dim

def update_all_parameters(parameters,conf_3d,conf_controller,conf_sitl, rocket_dim):
    global thrust, burnout_time, thrust_curve, max_thrust, average_thrust
    global m, Iy, d, xcg, xt, L, nosecone_length, CA0
    global k1, k2, k3, Actuator_max, Actuator_reduction, u_initial_offset
    global wind, wind_distribution

    m = parameters[1]
    Iy = parameters[2]
    xcg = parameters[3]
    xt = parameters[4]
    servo_definition = parameters[5]
    Actuator_max = parameters[6] * DEG2RAD
    Actuator_reduction = parameters[7]
    u_initial_offset = parameters[8] * DEG2RAD
    Actuator_weight_compensation = parameters[9]
    wind = parameters[10]
    wind_distribution = parameters[11]

    ##
    global toggle_3d, camera_shake_toggle, slow_mo, force_scale, hide_forces
    global Camera_type, variable_fov, fov
    toggle_3d = conf_3d[0]
    camera_shake_toggle = conf_3d[1]
    hide_forces = conf_3d[2]
    variable_fov = conf_3d[3]
    Camera_type = conf_3d[4]
    slow_mo = conf_3d[5]
    force_scale = conf_3d[6]
    fov = conf_3d[7]

    ## rocket Class
    global S, Q_damp_body, d
    rocket.set_motor(gui.savefile.get_motor_data())
    burnout_time = rocket.burnout_time()
    rocket.update_rocket(gui.draw_rocket_tab.get_configuration_destringed(), xcg)
    Q_damp_body = rocket.get_q_damp_body()
    S = rocket.reference_area
    d = rocket.max_diam

    ## controller
    global kp, ki, kd, k_all, k_damping, anti_windup, torque_controller, inp, inp_time, t_launch
    global T, Ts, T_Program, sim_duration,input_type,reference_thrust
    input_type = conf_controller[2]
    controller.setup_controller(conf_controller[0:9], Actuator_reduction, Actuator_max)
    inp = conf_controller[9]
    inp_time = conf_controller[10]
    t_launch = conf_controller[11]
    Ts = conf_controller[12]
    T_Program = conf_controller[13]
    sim_duration = conf_controller[14]
    T = conf_controller[15]

    ## SITL
    global Activate_SITL, use_noise, port, baudrate, gyro_sd, acc_sd, alt_sd
    Activate_SITL = conf_sitl[0]
    use_noise = conf_sitl[1]
    port = conf_sitl[2]
    baudrate = conf_sitl[3]
    gyro_sd = conf_sitl[4]
    acc_sd = conf_sitl[5]
    alt_sd = conf_sitl[6]

    ## Servo Class
    servo.setup(Actuator_weight_compensation,servo_definition,Ts)


def reset_variables():
    ## Ugly ugly piece of code
    ##
    global cn, w, q, U_prev, U2, wind_rand, i_turns, fin_force, wind_total
    cn = 0
    fin_force = 0
    w = 0
    q = 0
    U_prev = 0.
    U2 = 0.
    wind_rand = 0
    i_turns = 0
    wind_total = 0

    ##
    global theta, ca, aoa, U, W, Q, U_d, W_d, Q_d, X_d, Z_d, V_loc, V_loc_tot
    global V_glob, g_loc, F_loc, F_glob, position_global, acc_glob
    theta = 0
    ca = 0
    aoa = 0
    U = 0
    W = 0
    Q = 0
    U_d = IntegrableVariable()
    W_d = IntegrableVariable()
    Q_d = IntegrableVariable()
    X_d = IntegrableVariable()
    Z_d = IntegrableVariable()
    V_loc = [0.00001,0.00001]
    V_loc_tot = [0.00001,0.00001]
    V_glob = [0.00001,0.00001]
    g_loc = [0.0000,0.0000]
    acc_glob = [0.0001,0.0001]
    F_loc = [0.0000,0.0000]
    F_glob = [0.0000,0.0000]
    position_global = [0,0]

    ##
    global t_3d, theta_3d, servo_3d, V_loc_3d, V_glob_3d, position_3d, xa_3d
    global thrust_3d, cn_3d, fin_force_3d, aoa_3d, setpoint_3d, Airspeed_3d
    global X_3d, Z_3d
    t_3d = []
    theta_3d = []
    setpoint_3d = []
    servo_3d = []
    V_loc_3d = []
    V_glob_3d = []
    Airspeed_3d = []
    position_3d = []
    X_3d = []
    Z_3d = []
    cn_3d = []
    fin_force_3d = []
    thrust_3d = []
    xa_3d = []
    aoa_3d = []

    ##
    global second_plot, first_plot, third_plot, fourth_plot, fifth_plot, t_plot
    first_plot = []
    second_plot = []
    third_plot = []
    fourth_plot = []
    fifth_plot = []
    t_plot = []

    ##
    global alpha_calc, U_vect, V_vect, wind_vect, u_eq, u_prev, u_delta, u_controller
    alpha_calc = 0.
    U_vect = np.array([0.1,0])
    V_vect = np.array([0.1,0])
    wind_vect = np.array([0,wind])

    ##
    global u_servos, u
    u_servos = 0.
    u = 0.

    #CONTROL
    global setpoint, error
    setpoint = 0.
    error = 0.

    #TIMERS
    global timer_run, t_timer_3d, timer_run_sim, timer_run_servo, t, timer_disturbance, timer_U
    timer_run = 0
    t_timer_3d = 0
    timer_run_sim = 0
    timer_run_servo = 0
    t = 0.
    timer_disturbance = 0.
    timer_U = 0.

    #FLAGS
    global flag, flag2
    flag = False
    flag2 = False

    ## SITL
    global arduino_ready_flag0, timer_flag_t0, t0_timer, parachute
    arduino_ready_flag0 = ""
    timer_flag_t0 = False
    t0_timer = 0.
    parachute = 0
    rocket.reset_variables()

# Transforms from local coordinates to global coordinates
# (body to world)
def loc2glob(u0,v0,theta):
    # Rotational matrix 2x2
    # Axes are rotated, there is more info in the Technical documentation.
    A = np.array([[np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)]])
    u = np.array([[u0],[v0]])
    x = np.dot(A,u)
    a = [x[0,0],x[1,0]]
    return a

def glob2loc(u0,v0,theta):
    A = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
    u = np.array([[u0],[v0]])
    x = np.dot(A,u)
    a = [x[0,0],x[1,0]]
    return a

def update_parameters():
    global wind_rand, wind_total
    global q
    global cn, Q_damp_body, fin_force
    global x
    global xa
    global i
    global aoa
    global wind
    global thrust,t_launch
    global out,timer_disturbance,timer_U,U2,q_wind
    global cm_xcg, ca, S
    global actuator_angle

    # NEW SIMULATION
    global V_loc , V_loc_tot , V_glob
    global U_d , U , X
    global W_d , W , Z
    global Q_d , Q
    global theta, aoa, g , g_loc

    # Times the disturbances so they don't change that often
    if t>timer_disturbance + 0.1:
        wind_rand = random.gauss(0, wind_distribution)
        wind_total = wind_rand + wind
        timer_disturbance = t

    # NEW SIMULATION
    # Computes the velocity of the wind in local coordinates
    wind_loc = glob2loc(0, wind_total, theta)
    # Computes the total airspeed in local coordinates
    V_loc_tot = [ V_loc[0]-wind_loc[0] , V_loc[1]-wind_loc[1] ]
    aoa = calculate_aoa(V_loc_tot)
    T, P, rho, spd_sound, mu = atm.calculate_at_h(position_global[0], atmosphere)
    thrust = rocket.get_thrust(t, t_launch)
    S = rocket.reference_area
    V_modulus = np.sqrt(V_loc_tot[0]**2 + V_loc_tot[1]**2)
    M = V_modulus/spd_sound
    if rocket.use_fins_control is True:
        # Detailed explanation in rocket_functions
        cn, cm_xcg, ca, xa = rocket.calculate_aero_coef(V_loc_tot, Q, rho, mu, M, actuator_angle)
    else:
        cn, cm_xcg, ca, xa = rocket.calculate_aero_coef(V_loc_tot, Q, rho, mu, M)
    if abs(xa) > 1.5 * rocket.length:
        xa = 1.5 * rocket.length * np.sign(xa)
    # Computes the dynamic pressure
    q = 0.5 * rho * V_modulus**2
    # Gravity in local coordinates, theta=0 equals to rocket up
    g_loc = glob2loc(-g,0,theta)

def calculate_aoa(V_loc_tot):
    if V_loc_tot[0] != 0:
        aoa = np.arctan2(V_loc_tot[1], V_loc_tot[0]) # Computes the angle of attack
    else:
        aoa = np.pi/2
    return aoa
k=0
def simulation():
    global x,xs,k
    global xdot,xdots
    global out,outs
    global out_prev,out_prevs

    global u_controller
    global u,timer_run_servo,u_servos,actuator_angle

    global V_loc, V_loc_tot , V_glob
    global U_d, U , X
    global W_d, W , Z
    global Q_d, Q
    global theta, aoa, g
    global F_loc , F_glob
    global cn, thrust, rho, Q_damp_body, fin_force
    global cm_xcg, ca
    global t_timer_3d
    global position_global
    global i_turns

    global accx, accz, accQ, g_loc
    global t_launch

    # SERVO SIMULATION
    servo_current_angle = servo.simulate(u_servos,t)
    # Reduction of the TVC
    actuator_angle = (servo_current_angle/Actuator_reduction)
    update_parameters()
    """
    NEW METHOD, DIRECTLY INTEGRATES THE DIFFERENTIAL EQUATIONS
    U = local speed in X
    W = local speed in Z
    Q = pitch rate
    aoa = angle of attack

    V_glob = global velocity
    X_d = global X speed (Y in Vpython)
    Z_d = global Z speed (-X in Vpython)
    """
    v_d = 0 # 0 uses Local and Global Velocities, 1 uses vector derivatives.
    if rocket.is_in_the_pad(position_global[0]) and thrust < m*g or t < t_launch:
        accx = 0
        accz = 0
        accQ = 0
    else:
        if rocket.is_in_the_pad(position_global[0]):
            launchrod_lock = 1
            # launchrod_lock = 0 in case one day I add a launchrod
        else:
            launchrod_lock = 1
        if rocket.use_fins_control is False:
            # Longitudinal acceleration (local)
            accx = ((thrust * np.cos(actuator_angle+u_initial_offset) + m*g_loc[0] - q*S*ca)
                    / m - W*Q*v_d)
            # Transversal acceleration (local)
            accz = ((thrust * np.sin(actuator_angle+u_initial_offset) + m*g_loc[1] + q*S*cn)
                    / m + U*Q*v_d) * launchrod_lock
            accQ = ((thrust * np.sin(actuator_angle+u_initial_offset) * (xt-xcg)
                     + S * q * d * cm_xcg
                     - rho * Q * (Q_damp_body * abs(Q)*0.2))
                     / Iy) * launchrod_lock
        else:
            # Longitudinal acceleration (local)
            accx = ((thrust * np.cos(u_initial_offset) + m*g_loc[0] - q*S*ca)
                    / m - W*Q*v_d)
            # Transversal acceleration (local)
            accz = ((thrust * np.sin(u_initial_offset) + m*g_loc[1] + q*S*cn)
                    / m + U*Q*v_d) * launchrod_lock
            accQ = ((thrust * np.sin(u_initial_offset) * (rocket.length-xcg)
                     + S * q * d * cm_xcg
                     - rho * Q * (Q_damp_body * abs(Q)*0.2))
                     / Iy) * launchrod_lock
            fin_force = q * S * rocket.cn_alpha_ctrl_fin_3d_arrow * actuator_angle

    # Updates the variables
    U_d.new_f_dd(accx)
    W_d.new_f_dd(accz)
    Q_d.new_f_dd(accQ)

    # Integrates the angular acceleration and velocity
    Q = Q_d.integrate_f_dd()
    theta = Q_d.integrate_f_d()

    # In case theta is greater than 180º, to keep it between -180 and 180
    # It's alright to do this as long as theta is not integrated
    if theta > np.pi:
        theta -= 2*np.pi
        Q_d.new_f(theta)
        i_turns += 1
    if theta < -np.pi:
        theta += 2*np.pi
        Q_d.new_f(theta)
        i_turns += 1


    # New acceleration in global coordinates
    global acc_glob
    acc_glob = loc2glob(accx, accz, theta)
    if v_d == 1:
        # Just integrates, the transfer of velocities was already done in the vector derivative
        V_loc[0] = U_d.integrate_f_dd()
        V_loc[1] = W_d.integrate_f_dd()
    else:
        # Takes the global velocity, transforms it into local coordinates, adds the accelerations
        # and transforms the velocity back into global coordinates
        V_loc = glob2loc(V_glob[0], V_glob[1], theta)
        U_d.integrate_f_dd()
        W_d.integrate_f_dd()
        V_loc[0] += U_d.delta_f_d
        V_loc[1] += W_d.delta_f_d

    # New velocity in global coordinates
    V_glob = loc2glob(V_loc[0], V_loc[1], theta)

    # Updates the global velocity in the X_d class
    X_d.new_f_d(V_glob[0])
    Z_d.new_f_d(V_glob[1])

    # Integrates the velocities to get the position, be it local?¿ or global
    position_local = [U_d.integrate_f_d() , W_d.integrate_f_d()]

    position_global = [X_d.integrate_f_d() , Z_d.integrate_f_d()]

    """
    Adding -W*Q to U_d and +U*Q to W_d but eliminating the global to
    local transfer of velocity accounts for the vector rotation.
    Using the vector derivative (U_d = .... - W*Q and W_d = .... +
    U*Q) is the same as transforming the global vector in local
    coordinates, adding the local accelerations and transforming it
    back to global, in theory (didn't work for me, gotta se why)
    So:
    Vector Derivative -> No need to transform the velocity from
        global to local, you work only with the local
    No Vector Derivative -> Equations are simpler, but you have
        to transform the global vector to local and then to global
        again
    Still have to see how it scales with more DOF
    """
    """
    Only saves the points used in the animation.
    (1000) is the rate of the animation, when you use slow_mo it drops.
    To ensure fluidity at least a rate of 100 ish is recommended, so a
    rate of 1000 allows for 10 times slower animations.
    """
    if t >= t_timer_3d + (1/1000)*0.999:
        #### 3d
        t_3d.append(t)
        theta_3d.append(theta)
        servo_3d.append(actuator_angle+u_initial_offset)
        V_loc_3d.append(V_loc)
        V_glob_3d.append(V_glob)
        position_3d.append(position_global)
        xa_3d.append(rocket.cp_w_o_ctrl_fin)
        thrust_3d.append(thrust)
        cn_3d.append(cn*S*q)
        fin_force_3d.append(fin_force)
        aoa_3d.append(aoa)
        setpoint_3d.append(setpoint)
        t_timer_3d = t


def timer():
    global t
    t = round(t + T,12) #Trying to avoid error, not sure it works

def timer_SITL():
    global t, timer_flag_t0, t0_timer, T
    if timer_flag_t0 is False:
        t0_timer = time.perf_counter()/clock_dif
        timer_flag_t0 = True
    t_prev = t
    t = time.perf_counter()/clock_dif - t0_timer
    # Sample time in SITL = Time elapsed between runs
    T = t-t_prev

def set_setpoint(inp):
    global input_type
    if input_type == "Step [º]":
        setpoint = inp*DEG2RAD
    elif input_type == "Ramp [º/s]":
        setpoint = (inp*DEG2RAD)*(t-inp_time)
    else:
        setpoint = 0
    return setpoint

def check_which_plot(s):
    global okp, oki, okd, totError
    global send_gyro, send_accx, send_accz, send_alt
    global actuator_angle
    if s == "Setpoint":
        return setpoint * RAD2DEG
    if s == "Pitch Angle":
        return theta * RAD2DEG
    if s == "Actuator deflection":
        return actuator_angle * RAD2DEG
    if s == "Pitch Rate":
        return Q*RAD2DEG
    if s == "Local Velocity X":
        return V_loc[0]
    if s == "Local Velocity Z":
        return V_loc[1]
    if s == "Global Velocity X":
        return V_glob[0]
    if s == "Global Velocity Z":
        return V_glob[1]
    if s == "Total Velocity":
        return np.sqrt(V_loc_tot[0]**2 + V_loc_tot[1]**2)
    if s == "Local Acc X":
        return accx
    if s == "Local Acc Z":
        return accz
    if s == "Global Acc X":
        return acc_glob[0]
    if s == "Global Acc Z":
        return acc_glob[1]
    if s == "Angle of Atack":
        return aoa * RAD2DEG
    if s == "Cp Position":
        return xa
    if s == "Normal Force Coefficient":
        return cn
    if s == "Axial Force Coefficient":
        return ca
    if s == "Moment Coefficient":
        return cm_xcg
    if s == "Altitude":
        return position_global[0]
    if s == "Distance Downrange":
        return position_global[1]
    if s == "Proportional Contribution":
        return okp * RAD2DEG
    if s == "Integral Contribution":
        return oki * RAD2DEG
    if s == "Derivative Contribution":
        return okd * RAD2DEG
    if s == "Total Error":
        return totError * RAD2DEG
    if s == "Simulated Gyro":
        return send_gyro
    if s == "Simulated Acc X":
        return send_accx
    if s == "Simulated Acc Z":
        return send_accz
    if s == "Simulated Altimeter":
        return send_alt
    if s == "Off":
        return None

def plot_data():
    s = gui.run_sim_tab.get_configuration_destringed()
    a_plt = check_which_plot(s[0])
    b_plt = check_which_plot(s[1])
    c_plt = check_which_plot(s[2])
    d_plt = check_which_plot(s[3])
    e_plt = check_which_plot(s[4])
    first_plot.append(a_plt)
    second_plot.append(b_plt)
    third_plot.append(c_plt)
    fourth_plot.append(d_plt)
    fifth_plot.append(e_plt)
    t_plot.append(t)

def plot_plots():
    figure = Figure(figsize=(5, 4), dpi=100)
    plot = figure.add_subplot(1, 1, 1)
    s = gui.run_sim_tab.get_configuration_destringed()
    if s[0] != "Off":
        plt.plot(t_plot,first_plot, label=gui.run_sim_tab.get_configuration_destringed()[0])
    if s[1] != "Off":
        plt.plot(t_plot,second_plot, label=gui.run_sim_tab.get_configuration_destringed()[1])
    if s[2] != "Off":
        plt.plot(t_plot,third_plot, label=gui.run_sim_tab.get_configuration_destringed()[2])
    if s[3] != "Off":
        plt.plot(t_plot,fourth_plot, label=gui.run_sim_tab.get_configuration_destringed()[3])
    if s[4] != "Off":
        plt.plot(t_plot,fifth_plot, label=gui.run_sim_tab.get_configuration_destringed()[4])
    plt.grid(True,linestyle='--')
    plt.xlabel('Time',fontsize=16)
    plt.ylabel('',fontsize=16)
    legend = plt.legend(loc='upper right', shadow=True, fontsize='x-small')
    plt.axvline(x=burnout_time+t_launch, color="black", linewidth=1)

def run_sim_local():
    global parameters, conf_3d, conf_controller
    global timer_run_sim,timer_run,setpoint, t_launch,inp_time, u_servos
    global okp, oki, okd, totError

    while t <= sim_duration:
        simulation()
        if t > burnout_time * 10:
            print("Nice Coast")
            break
        if i_turns >= 5:
            print("Pitch angle greater than 180\xb0, the rocket is flying pointy end down.")
            break
        if position_global[0] < -0.1:
            print("CRASH")
            break
        if t == sim_duration * 0.999:
            print("Simulation Ended")
            break
        """
        *.999 corrects the error in t produced by adding t=t+T for sample times smaller than 0.001
        If it was exact, it starts accumulating  error, and the smaller the sample time, the more error it accumulates
        So, be careful with the sample time, 0.001, 0.0005 and 0.0001 give all similar results, so,
        if it takes to long to run you can increase T:
        """
        if t >= timer_run_sim+T*0.999:
            if t >= timer_run+T_Program*0.999:
                timer_run = t
                if t >= inp_time:
                    setpoint = set_setpoint(inp)
                u_servos, okp, oki, okd, totError = controller.control_theta(setpoint,
                                                                             theta, Q,
                                                                             thrust, t)
            timer_run_sim = t
        timer()
        plot_data()

def run_sim_sitl():
    global parameters, conf_3d, conf_controller, setpoint
    global timer_run_sim,timer_run,setpoint, parachute, t_launch, u_servos
    global send_gyro, send_accx, send_accz, send_alt
    global timer_flag_t0, clock_dif, T_glob, parachute
    import serial

    timer_seconds = 0
    timer_flag_t0 = False
    t0_timer = 0.
    # Clock_dif because the Arduino clock is slower than the PC one (0.94 for mine)
    clock_dif = 1
    #0.001 sample time of the simulation loop, the sample time of the simulation
    # is the real time between runs
    T_glob = 0.001
    parachute = 0

    serialArduino = serial.Serial(port, baudrate, writeTimeout=0)
    arduino_ready_flag = "not_ready"
    while arduino_ready_flag != "A":
        arduino_ready_flag0 = serialArduino.read()
        arduino_ready_flag = arduino_ready_flag0.decode("ASCII").strip()
        serialArduino.flushInput()
    t0 = time.perf_counter() / clock_dif
    i = 0
    while t <= sim_duration:
        if time.perf_counter()/clock_dif > t0+T_glob:
            t0 = time.perf_counter()/clock_dif
            # Timer runs at the begining, so it calculates the actual
            # T between runs and integrates more accurately
            simulation()
            timer_SITL()
            if t >= timer_seconds + 1:
                timer_seconds = t
                print("Time is ",round(t,0)," seconds")
            if t > burnout_time * 10:
                break
            if i_turns >= 5:
                print("Pitch angle greater than 180\xb0, the rocket is flying pointy end down.")
                break
            if position_global[0] < -0.1:
                print("CRASH")
                break
            if parachute == 1:
                print("Parachute Deployed")
                break
            if t == sim_duration*0.999:
                print("Simulation Ended")
                break
            if t >= timer_run_sim + T_glob*0.999:
                timer_run_sim = t
                if t >= inp_time:
                    setpoint = set_setpoint(inp)
            plot_data()
            i += 1
        ##
        if t >= 0.003:
            if serialArduino.inWaiting()>1:
                read0 = serialArduino.readline()
                read = read0.decode("ASCII")
                read = read.strip()
                if use_noise is True:
                    send_gyro = random.gauss(Q*RAD2DEG, gyro_sd)
                    send_accx = random.gauss((accx-g_loc[0])/9.81, acc_sd)
                    send_accz = random.gauss((accz-g_loc[1])/9.81, acc_sd)
                    send_alt = random.gauss(position_global[0], alt_sd)
                else:
                    send_gyro = Q*RAD2DEG
                    send_accx = (accx-g_loc[0])/9.81
                    send_accz = (accz-g_loc[1])/9.81
                    send_alt = position_global[0]
                if read == "R":
                    # Arduino ready to Read
                    # last comma because the Arduino library separates the string at commas
                    send = (str(round(send_gyro,6))+","+str(round(send_accx,6))
                            +","+str(round(send_accz,6))+","
                            +str(round(send_alt,2))+",")
                    serialArduino.write(str(send).encode("ASCII"))
                    serialArduino.write('\n'.encode("ASCII"))
                else:
                    # Arduino sent data for the program to read
                    # Arduino is not ready to read data
                    read_split = read.split(",")
                    u_servos = float(read_split[0])*DEG2RAD
                    parachute = int(read_split[1])

def run_simulation():
    global parameters, conf_3d, conf_controller
    global timer_run_sim,timer_run,setpoint
    reset_variables()
    parameters, conf_3d, conf_controller, conf_sitl, rocket_dim = get_data_savefile()
    update_all_parameters(parameters, conf_3d, conf_controller, conf_sitl, rocket_dim)
    print("Simulation Started")
    if Activate_SITL is False:
        run_sim_local()
    else:
        run_sim_sitl()
    plot_plots()
    return


############################################ 3D 3D 3D 3D 3D
############################################ 3D 3D 3D 3D 3D
############################################ 3D 3D 3D 3D 3D
############################################ 3D 3D 3D 3D 3D

def run_3d():
    if toggle_3d is False:
        plt.draw()
        plt.show()
    if toggle_3d is True:

        rocket_dim = gui.draw_rocket_tab.get_points_float(0)

        #Creates the window
        scene = vp.canvas(width=1280, height=720, center=vp.vector(0,0,0),
                          background=vp.color.white)
        scene.lights = []
        vp.distant_light(direction=vp.vector(1, 1, -1), color=vp.color.gray(0.9))
        i = 0
        #floor
        dim_x_floor = 3000
        dim_z_floor = 4000
        n = 21
        # Sky (many panels)
        for i in range(n):
            for j in range(n):
                vp.box(pos=vp.vector(dim_x_floor * (i - n/2 + 1) - 100,
                                     dim_z_floor * (j + 0.5),
                                     dim_x_floor),
                       size=vp.vector(dim_x_floor,dim_z_floor,1),
                       texture={'file':'sky_texture.jpg'})
        n = 3
        #Floor (many panels)
        for i in range(n):
            for j in range(n):
                vp.box(pos=vp.vector(dim_x_floor/2*(i-n/2+1), -0.5, dim_z_floor*(j+0.5)),
                       size=vp.vector(dim_x_floor, 1, dim_z_floor),
                       texture={'file':'grass_texture.jpg'})


        ## rocket
        n_c = 20 #how many pieces have each non standard component
        L = rocket_dim[-1][0]
        nosecone_length = rocket_dim[1][0]
        d = rocket_dim[1][1]
        R_ogive = rocket_dim[1][1]
        L_ogive = rocket_dim[1][0]
        rho_radius = (R_ogive**2 + L_ogive**2)/(2 * R_ogive)
        compound_list = []
        for i in range(len(rocket_dim)-1):
            if i == 0 and rocket.ogive is True:
                # Ogive goes brrrrr
                l_partial2 = 0
                l_partial = L_ogive / n_c
                for j in range(n_c):
                    # diameter can never be 0
                    d = ((np.sqrt(rho_radius**2 - (L_ogive-l_partial2)**2) + R_ogive - rho_radius)
                        + 0.000001)
                    pos = l_partial * (j+1)
                    rod = vp.cylinder(pos=vp.vector(dim_x_floor/2, L-pos, dim_z_floor/2),
                                    axis=vp.vector(0, 1, 0), radius=d/2,
                                    color=vp.color.black, length=l_partial)
                    compound_list.append(rod)
                    l_partial2 += l_partial
                continue
            if i == 0 and rocket.ogive is False:
                nosecone = vp.cone(pos=vp.vector(dim_x_floor/2, (L-nosecone_length), dim_z_floor/2),
                                 axis=vp.vector(0,1,0), radius=d/2,
                                 color=vp.color.black, length=nosecone_length)
                compound_list.append(nosecone)
                continue
            if rocket_dim[i][1] == rocket_dim[i+1][1]:
                l = rocket_dim[i+1][0] - rocket_dim[i][0]
                d = rocket_dim[i][1]
                rod = vp.cylinder(pos=vp.vector(dim_x_floor/2, L-rocket_dim[i+1][0], dim_z_floor/2),
                                axis=vp.vector(0,1,0), radius=d/2,
                                color=vp.color.black, length=l)
                compound_list.append(rod)
            if rocket_dim[i][1] != rocket_dim[i+1][1]:
                l = rocket_dim[i+1][0] - rocket_dim[i][0]
                d0 = rocket_dim[i][1]
                d1 = rocket_dim[i+1][1]
                x0 = rocket_dim[i][0]
                x1 = rocket_dim[i+1][0]
                # Truncated cone goes brrrrr
                for i in range(n_c):
                    d = d0 + i*((d1-d0)/n_c)
                    l_partial = l/n_c
                    pos = x0 + l_partial * (i+1)
                    rod=vp.cylinder(pos=vp.vector(dim_x_floor/2,L-pos,dim_z_floor/2),
                                    axis=vp.vector(0,1,0),radius=d/2,
                                    color=vp.color.black, length=l_partial)
                    compound_list.append(rod)
        # Fins
        throwaway_box = vp.box(pos=vp.vector(dim_x_floor/2,0.1,dim_z_floor/2),
                               axis=vp.vector(1,0,0),
                               size = vp.vector(0.001,0.001,0.001),
                               color=vp.color.black)
        fin_compound = [0]*4
        for i in range(4):
            fin_compound[i] = throwaway_box
        fin_compound_control = [0]*4
        for i in range(4):
            fin_compound_control[i] = throwaway_box
        if rocket.use_fins is True:
            compound_fins = []
            fins = rkt.fin[0].dim
            if fins[1][0] > 0.001:
                for i in range(len(fins)-1):
                    if fins[i][1] == fins[i+1][1]:
                        l = abs(fins[i+1][0] - fins[i][0])+0.001
                        if l==0:
                            continue
                        b0 = fins[i][1]
                        b1 = fins[i+1][1]
                        x0 = abs(fins[i][0] - rocket_dim[-1][0])
                        x1 = abs(fins[i+1][0] - rocket_dim[-1][0])
                        c = l
                        b = fins[i][1]
                        posy = c/2 + x0 - c * (i)
                        posx = b/2 + rocket_dim[-1][1]/2 * 0
                        fin = vp.box(pos=vp.vector(dim_x_floor/2+posx,posy,dim_z_floor/2),
                                     axis=vp.vector(1,0,0),
                                     size = vp.vector(b,c,0.007),
                                     color=vp.color.black)
                        compound_fins.append(fin)
                    if fins[i][1] != fins[i+1][1]:
                        l = abs(fins[i+1][0] - fins[i][0])
                        if l==0:
                            continue
                        b0 = fins[i][1]
                        b1 = fins[i+1][1]
                        x0 = abs(fins[i][0] - rocket_dim[-1][0])
                        x1 = abs(fins[i+1][0] - rocket_dim[-1][0])
                        for i in range(n_c):
                            b = b0 + i*((b1-b0)/n_c)+0.0001
                            c = l/n_c
                            posy = -c/2 + x0 - c * (i)
                            posx = b/2 + rocket_dim[-1][1]/2 * 0
                            fin = vp.box(pos=vp.vector(dim_x_floor/2+posx,posy,dim_z_floor/2),
                                         axis=vp.vector(1,0,0),
                                         size = vp.vector(b,c,0.007),
                                         color=vp.color.black)
                            compound_fins.append(fin)
                fin_compound = [0]*4
                fin_compound[0] = vp.compound(compound_fins)
                for i in range(3):
                    fin_compound[i+1] = fin_compound[0].clone(pos=fin_compound[0].pos)
                    fin_compound[i+1].rotate(np.pi/2*(i+1),axis=vp.vector(0,1,0),
                                             origin=vp.vector(dim_x_floor/2,0,dim_z_floor/2))

            if rocket.use_fins_control is True:
                fins = rkt.fin[1].dim
                compound_fins_control = []
                for i in range(len(fins)-1):
                    if fins[i][1] == fins[i+1][1]:
                        l = abs(fins[i+1][0] - fins[i][0])
                        if l==0:
                            continue
                        b0 = fins[i][1]
                        b1 = fins[i+1][1]
                        x0 = abs(fins[i][0] - rocket_dim[-1][0])
                        x1 = abs(fins[i+1][0] - rocket_dim[-1][0])
                        c = l
                        b = fins[i][1]
                        posy = c/2 + x0 - c * (i)
                        posx = b/2 + rocket_dim[-1][1]/2 * 0
                        fin = vp.box(pos=vp.vector(dim_x_floor/2+posx,posy,dim_z_floor/2),
                                     axis=vp.vector(1,0,0),
                                     size = vp.vector(b,c,0.007),
                                     color=vp.color.red)
                        compound_fins_control.append(fin)
                    if fins[i][1] != fins[i+1][1]:
                        l = abs(fins[i+1][0] - fins[i][0])
                        if l==0:
                            continue
                        l = fins[i+1][0] - fins[i][0]
                        b0 = fins[i][1]
                        b1 = fins[i+1][1]
                        x0 = abs(fins[i][0] - rocket_dim[-1][0])
                        x1 = abs(fins[i+1][0] - rocket_dim[-1][0])
                        for i in range(n_c):
                            b = b0 + i*((b1-b0)/n_c)+0.0001
                            c = l/n_c
                            posy = -c/2 + x0 - c * (i)
                            posx = b/2 + rocket_dim[-1][1]/2 * 0
                            fin = vp.box(pos=vp.vector(dim_x_floor/2+posx,posy,dim_z_floor/2),
                                         axis=vp.vector(1,0,0),
                                         size = vp.vector(b,c,0.007),
                                         color=vp.color.red)
                            compound_fins_control.append(fin)
                fin_compound_control = [0]*4
                fin_compound_control[0] = vp.compound(compound_fins_control)
                for i in range(3):
                    fin_compound_control[i+1] = fin_compound_control[0].clone(pos=fin_compound_control[0].pos)
                    fin_compound_control[i+1].rotate(np.pi/2*(i+1),
                                                     axis=vp.vector(0,1,0),
                                                     origin=vp.vector(dim_x_floor/2,
                                                                      0,
                                                                      dim_z_floor/2))

        motor_radius = 0.015
        def maximum_diameter(rocket_dim):
            d = 0
            for i in range(len(rocket_dim)):
                if rocket_dim[i][1] > d:
                    d = rocket_dim[i][1]
            return d

        # Create final components
        d = maximum_diameter(rocket_dim)
        compound_rocket = (compound_list + fin_compound
                           + [fin_compound_control[0],fin_compound_control[2]])
        rocket_3d = vp.compound(compound_rocket)
        motor=vp.cone(pos=vp.vector(dim_x_floor/2,0,dim_z_floor/2),
                      axis=vp.vector(0,-1,0),radius=motor_radius,
                      color=vp.color.red,length=0.15,make_trail=True)
        control_fins = vp.compound([fin_compound_control[1],fin_compound_control[3]])

        motor.trail_color=vp.color.red
        motor.rotate(u_initial_offset,axis=vp.vector(0,0,-1))

        if rocket.use_fins is True:
            sep = rkt.fin[0].dim[1][1] - rocket.rocket_dim[-1][1]/2
        else:
            sep = 0
        #Torque motor
        Tmotor_pos=vp.arrow(pos=vp.vector(motor.pos.x-(d/2+0.015+sep),
                                          motor.pos.y,motor.pos.z),
                            axis=vp.vector(0.0001,0,0),
                            shaftwidth=0,color=vp.color.red)
        Tmotor_neg=vp.arrow(pos=vp.vector(motor.pos.x+(d/2+0.015+sep),
                                          motor.pos.y,motor.pos.z),
                            axis=vp.vector(-0.0001,0,0),
                            shaftwidth=0,color=vp.color.red) #Torque motor
        Nforce_pos=vp.arrow(pos=vp.vector(motor.pos.x+(d/2+0.015),
                                          motor.pos.y+(L-xa_3d[0]),
                                          motor.pos.z),
                            axis=vp.vector(0.0001,0,0),
                            shaftwidth=0,color=vp.color.green)
        Nforce_neg=vp.arrow(pos=vp.vector(motor.pos.x-(d/2+0.015),
                                          motor.pos.y+(L-xa_3d[0]),
                                          motor.pos.z),
                            axis=vp.vector(0.0001,0,0),
                            shaftwidth=0,color=vp.color.green)

        if rocket.use_fins_control is True:
            #Torque control fin
            T_fin_pos = vp.arrow(pos=vp.vector(control_fins.pos.x-(d/2+rkt.fin[1].wingspan+0.015)
                                               ,control_fins.pos.y,
                                               control_fins.pos.z),
                                 axis=vp.vector(0.0001,0,0),shaftwidth=0,
                                 color=vp.color.red)
            T_fin_neg = vp.arrow(pos=vp.vector(control_fins.pos.x+(d/2+rkt.fin[1].wingspan+0.015)
                                               ,control_fins.pos.y,
                                               control_fins.pos.z),
                                 axis=vp.vector(-0.0001,0,0),shaftwidth=0,
                                 color=vp.color.red) #Torque control fin

        Nforce_neg.visible=False
        Nforce_pos.visible=False
        Tmotor_pos.visible=False
        Tmotor_neg.visible=False
        if rocket.use_fins_control is True:
            T_fin_pos.visible=False
            T_fin_neg.visible=False

        #Labels
        labels=vp.canvas(width=1280,height=200,center=vp.vector(0,0,0),background=vp.color.white)

        Setpoint_label = vp.label(canvas=labels, pos=vp.vector(-60,7,0),
                                  text="Setpoint = %.2f"+str(""),
                                  xoffset=0, zoffset=0, yoffset=0,
                                  space=30, height=30, border=0,
                                  font='sans', box=False, line=False, align ="left")
        theta_label = vp.label(canvas=labels, pos=vp.vector(-60,4,0),
                               text="Setpoint = %.2f"+str(""),
                               xoffset=0, zoffset=0, yoffset=0,
                               space=30, height=30, border=0,
                               font='sans',box=False,line=False, align ="left")
        servo_label = vp.label(canvas=labels, pos=vp.vector(-60,1,0),
                               text="Setpoint = %.2f"+str(""),
                               xoffset=0, zoffset=0, yoffset=0,
                               space=30, height=30, border=0,
                               font='sans',box=False,line=False, align ="left")
        V = vp.label(canvas=labels, pos=vp.vector(-60,-2,0),
                     text="Velocity"+str(""),
                     xoffset=0, zoffset=0, yoffset=0,
                     space=30, height=30, border=0,
                     font='sans',box=False,line=False,align ="left")
        aoa_plot = vp.label(canvas=labels, pos=vp.vector(-60,-5,0),
                            text="Velocity"+str(""),
                            xoffset=0, zoffset=0, yoffset=0,
                            space=30, height=30, border=0,
                            font='sans',box=False,line=False,align ="left")
        Altitude = vp.label(canvas=labels, pos=vp.vector(-60,-8,0),
                            text="Velocity"+str(""),
                            xoffset=0, zoffset=0, yoffset=0,
                            space=30, height=30, border=0,
                            font='sans',box=False,line=False,align ="left")
        Time_label = vp.label(canvas=labels, pos=vp.vector(40,7,0),
                              text="Time"+str(""),
                              xoffset=0, zoffset=0, yoffset=0,
                              space=30, height=30, border=0,
                              font='sans',box=False,line=False,align ="left")
        i=0
        for i in range(len(theta_3d)-2):
            vp.rate(1000/slow_mo) # For a smooth animation

            # How much to move each time-step , X and Z are the rocket's axes, not the world's
            delta_pos_X=(position_3d[i+1][0]-position_3d[i][0])
            delta_pos_Z=(position_3d[i+1][1]-position_3d[i][1])

            # Moving the rocket
            rocket_3d.pos.y+=delta_pos_X
            rocket_3d.pos.x+=delta_pos_Z

            # Creates a cg and cp vector with reference to the origin of the
            # 3d rocket (not the math model rocket)
            # Delta xa_radius, this is then integrated when you move the Aerodynamic Force arrow
            xcg_radius = loc2glob((L-xcg),0,theta_3d[i])
            xa_radius = loc2glob(xa_3d[i+1]-xa_3d[i],0,theta_3d[i])
            if rocket.use_fins_control is True:
                T_control_fin_radius = loc2glob((rkt.fin[1].cp-xcg),0,theta_3d[i])

            #CP and CG global vectors
            vect_cg = vp.vector(rocket_3d.pos.x + xcg_radius[0],
                              rocket_3d.pos.y + xcg_radius[1],
                              0)
            vect_cp = vp.vector(rocket_3d.pos.x + xa_radius[0],
                              rocket_3d.pos.y + xa_radius[1],
                              0)

            # Rotate rocket from the CG
            rocket_3d.rotate((theta_3d[i+1]-theta_3d[i]),axis=vp.vector(0,0,1),
                             origin=vect_cg)

            # Move the motor together with the rocket
            motor.pos.y+=delta_pos_X
            motor.pos.x+=delta_pos_Z
            motor.rotate((theta_3d[i+1]-theta_3d[i]),axis=vp.vector(0,0,1),
                         origin=vect_cg)  # Rigid rotation with the rocket
            if rocket.use_fins_control is False:
                # TVC mount rotation
                motor.rotate((servo_3d[i+1]-servo_3d[i]),axis=vp.vector(0,0,-1))

            if rocket.use_fins_control is True:
                control_fins.pos.y+=delta_pos_X
                control_fins.pos.x+=delta_pos_Z
                control_fins.rotate((theta_3d[i+1] - theta_3d[i]),
                                    axis=vp.vector(0,0,1),origin=vect_cg)
                control_fins.rotate(servo_3d[i+1] - servo_3d[i],
                                    axis=vp.vector(0,0,-1))

            # Motor Burnout, stops the red trail of the rocket
            if t_3d[i] > burnout_time + t_launch:
                motor.visible=False
                motor.make_trail=False
                Tmotor_pos.visible=False
                Tmotor_neg.visible=False
            else:
                # Arrows are hit or miss, tried this to avoid them going in the
                # wrong direction, didn't work
                aux=np.sin(servo_3d[i])*thrust_3d[i]*force_scale
                if rocket.use_fins_control is False:
                    # Makes visible one arrow or the other, be it left or right
                    if aux > 0:
                        Tmotor_pos.visible = False
                        Tmotor_neg.visible = True
                    else:
                        Tmotor_pos.visible = True
                        Tmotor_neg.visible = False
                    # Displacements and rotations of the arrows
                    Tmotor_pos.pos.y+=delta_pos_X
                    Tmotor_pos.pos.x+=delta_pos_Z
                    Tmotor_pos.rotate((theta_3d[i+1]-theta_3d[i]),
                                      axis=vp.vector(0,0,1),
                                      origin=vect_cg)
                    Tmotor_pos.axis=vp.vector(aux,0,0)
                    Tmotor_pos.rotate(theta_3d[i],axis=vp.vector(0,0,1))
                    Tmotor_neg.pos.y+=delta_pos_X
                    Tmotor_neg.pos.x+=delta_pos_Z
                    Tmotor_neg.rotate((theta_3d[i+1]-theta_3d[i]),
                                      axis=vp.vector(0,0,1),
                                      origin=vect_cg)
                    Tmotor_neg.axis=vp.vector(aux,0,0)
                    Tmotor_neg.rotate(theta_3d[i],axis=vp.vector(0,0,1))


            #Normal force arrow
            # Same as before, makes the one active visible
            if cn_3d[i] <= 0:
                Nforce_pos.visible = False
                Nforce_neg.visible = True
            else:
                Nforce_pos.visible = True
                Nforce_neg.visible = False
            # Displacements and rotations
            Nforce_pos.pos.y+=delta_pos_X - xa_radius[0]
            Nforce_pos.pos.x+=delta_pos_Z - xa_radius[1]
            Nforce_pos.axis=vp.vector(cn_3d[i]*force_scale,0,0)
            Nforce_pos.rotate(theta_3d[i],axis=vp.vector(0,0,1),
                              origin=Nforce_pos.pos)
            Nforce_pos.rotate((theta_3d[i+1]-theta_3d[i]),axis=vp.vector(0,0,1),
                              origin=vect_cg)
            Nforce_neg.pos.y+=delta_pos_X - xa_radius[0]
            Nforce_neg.pos.x+=delta_pos_Z - xa_radius[1]
            Nforce_neg.axis=vp.vector(cn_3d[i]*force_scale,0,0)
            Nforce_neg.rotate(theta_3d[i],axis=vp.vector(0,0,1),
                              origin=Nforce_neg.pos)
            Nforce_neg.rotate((theta_3d[i+1]-theta_3d[i]),axis=vp.vector(0,0,1),
                              origin=vect_cg)

            if rocket.use_fins_control is True:
                if fin_force_3d[i] >= 0:
                    T_fin_pos.visible = False
                    T_fin_neg.visible = True
                else:
                    T_fin_pos.visible = True
                    T_fin_neg.visible = False
                # Displacements and rotations
                T_fin_pos.pos.y+=delta_pos_X - xa_radius[0]*0
                T_fin_pos.pos.x+=delta_pos_Z - xa_radius[1]*0
                T_fin_pos.axis=vp.vector(fin_force_3d[i]*force_scale,0,0)
                T_fin_pos.rotate(theta_3d[i],axis=vp.vector(0,0,1),
                                 origin=T_fin_pos.pos)
                T_fin_pos.rotate((theta_3d[i+1]-theta_3d[i]),axis=vp.vector(0,0,1),
                                 origin=vect_cg)
                T_fin_neg.pos.y+=delta_pos_X - xa_radius[0]*0
                T_fin_neg.pos.x+=delta_pos_Z - xa_radius[1]*0
                T_fin_neg.axis=vp.vector(fin_force_3d[i]*force_scale,0,0)
                T_fin_neg.rotate(theta_3d[i],axis=vp.vector(0,0,1),
                                 origin=T_fin_neg.pos)
                T_fin_neg.rotate((theta_3d[i+1]-theta_3d[i]),axis=vp.vector(0,0,1),
                                 origin=vect_cg)

            if hide_forces is True:
                Nforce_pos.visible = False
                Nforce_neg.visible = False
                Tmotor_pos.visible = False
                Tmotor_neg.visible = False
                if rocket.use_fins_control is True:
                    T_fin_pos.visible=False
                    T_fin_neg.visible=False

            #To avoid the ugly arrows before the rocket starts going
            if V_glob_3d[i][0] < 0.1:
                Nforce_neg.visible = False
                Nforce_pos.visible = False
                Tmotor_pos.visible = False
                Tmotor_neg.visible = False

            #Camera
            if camera_shake_toggle is True:
                camera_shake=loc2glob(V_glob_3d[i][0],V_glob_3d[i][1],theta_3d[i])
            else:
                camera_shake = [0,0]

            #Follows almost 45 deg up
            if Camera_type=="Follow":
                scene.camera.pos = vp.vector(rocket_3d.pos.x+camera_shake[1]/50,
                                             rocket_3d.pos.y+1.2-camera_shake[0]/500,
                                             rocket_3d.pos.z-1)
                scene.camera.axis=vp.vector(rocket_3d.pos.x-scene.camera.pos.x,
                                            rocket_3d.pos.y-scene.camera.pos.y,
                                            rocket_3d.pos.z-scene.camera.pos.z)

            # Simulates someone in the ground
            elif Camera_type=="Fixed":
                if variable_fov == True:
                    scene.fov=fov*DEG2RAD/(0.01/10 * np.sqrt(position_3d[i][0]**2
                                                             + position_3d[i][1]**2)
                                           + 1)
                else:
                    scene.fov=fov*DEG2RAD
                scene.camera.pos = vp.vector(dim_x_floor/2,1,dim_z_floor/2-70*5)
                scene.camera.axis = vp.vector(rocket_3d.pos.x-scene.camera.pos.x,
                                              rocket_3d.pos.y-scene.camera.pos.y,
                                              rocket_3d.pos.z-scene.camera.pos.z)

            # Lateral camera, like if it was 2D
            elif Camera_type=="Follow Far":
                scene.fov=fov*DEG2RAD
                scene.camera.pos = vp.vector(rocket_3d.pos.x+camera_shake[1]/50,
                                             rocket_3d.pos.y+0.0-camera_shake[0]/200,
                                             rocket_3d.pos.z-70*5)
                scene.camera.axis=vp.vector(rocket_3d.pos.x-scene.camera.pos.x,
                                            rocket_3d.pos.y-scene.camera.pos.y,
                                            rocket_3d.pos.z-scene.camera.pos.z)

            #Labels
            Setpoint_label.text = "Setpoint = %.0f" % round(setpoint_3d[i]*RAD2DEG,1) + u'\xb0'
            theta_label.text = "Pitch Angle = " + str(round(theta_3d[i]*RAD2DEG,2)) + u'\xb0'
            servo_label.text = ("Actuator deflection = " +
                                str(round((servo_3d[i]-u_initial_offset)*RAD2DEG,2))
                                + u'\xb0')
            V.text = ("Local Velocity => " + " X = "+
                      str(round(V_loc_3d[i][0],2)) + " m/s , Z = "
                      + str(round(V_loc_3d[i][1],2)) + " m/s")
            aoa_plot.text = "aoa = " + str(round(aoa_3d[i]*RAD2DEG,2)) + u'\xb0'
            Altitude.text = "Altitude = " + str(round(position_3d[i][0],2)) + "m"
            Time_label.text = "Time = " + str(round(t_3d[i],3))
        plt.draw()
        plt.show()

# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:23:20 2020

@author: Guido di Pasquo
"""


import sys
import matplotlib
import matplotlib.pyplot as plt
# from matplotlib.figure import Figure
import numpy as np
import random
import vpython as vp
import time
import importlib
from pathlib import Path
from scipy.interpolate import interp1d
from src.gui import gui_setup as gui
from src.aerodynamics import rocket_functions as rkt
from src import control
from src.simulation import servo_lib
from src import files

matplotlib.use('TkAgg')
textures_path = Path("src/textures")


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



rocket = rkt.Rocket()
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
        # self.delta_f_d = T * self.f_dd # Euler
        self.delta_f_d = 0.5 * T * (self.f_dd_1+self.f_dd)  # Trapezoidal
        # Because the accelerations rotates I'm not a fan of using previous
        # measurements to integrate, so I went for the safer trapezoidal

        # self.delta_f_d= (T/6) * (self.f_dd_2 + 4 * (self.f_dd_1) + self.f_dd)
        # Simpson's (Runs each timestep -> (b-a)=h=T)

        # self.delta_f_d= ((T/8) * (self.f_dd_3 + 3 * (self.f_dd_2) +
        # 3 * self.f_dd_1 + self.f_dd))
        # Simpson's 3/8
        self.f_d += self.delta_f_d
        return self.f_d

    def integrate_f_d(self):
        self.f_3 = self.f_2
        self.f_2 = self.f_1
        self.f_1 = self.f
        self.delta_f = 0.5 * T * (self.f_d_1 + self.f_d)  # Trapezoidal
        self.f += self.delta_f
        return self.f


class ProgressBar:

    def __init__(self):
        self.delta_t = 0.5
        self.prev_t = 0
        self.bar_length = 20

    def update(self, t, end_val, i=1):
        if t >= (self.delta_t + self.prev_t)*i:
            self.re_write_progress_bar(t, end_val)
            self.prev_t = t

    def re_write_progress_bar(self, t, end_val):
        percent = float(t) / end_val
        hashes = '#' * int(round(percent * self.bar_length))
        spaces = ' ' * (self.bar_length - len(hashes))
        sys.stdout.write("\rProgress: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
        sys.stdout.flush()



theta = 0
aoa = 0
U = 0
W = 0
Q = 0
accx = 0
accy = 0
accQ = 0
U_d = IntegrableVariable()
W_d = IntegrableVariable()
Q_d = IntegrableVariable()
x_d = IntegrableVariable()
z_d = IntegrableVariable()
v_loc = [0.00001, 0.00001]
v_loc_tot = [0.00001, 0.00001]
v_glob = [0.00001, 0.00001]
g_loc = [0., 0.]
acc_glob = [0.0001, 0.0001]
F_loc = [0., 0.]
F_glob = [0., 0.]
position_global = [0, 0]
position_local = [0, 0]
force_app_point = 0
normal_force = 0

# LOTS
t_plot = []
first_plot = []
second_plot = []
third_plot = []
fourth_plot = []
fifth_plot = []

# 3D plots
t_3d = [0]
theta_3d = [0]
setpoint_3d = [0]
servo_3d = [0]
v_loc_3d = [[0, 0]]
v_glob_3d = [[0, 0]]
Airspeed_3d = [[0, 0]]
position_3d = [[0, 0]]
X_3d = [0]
Z_3d = [0]
cn_3d = [0]
fin_force_3d = [0]
thrust_3d = [0]
xa_3d = [0]
xcg_3d = [0]
aoa_3d = [0]

# Other Variables
alpha_calc = 0.
aoa = 0.
U_vect = np.array([0.1, 0])
V_vect = np.array([0.1, 0])
wind_vect = np.array([0, wind])

# PID
anti_windup = True

# CONTROL
setpoint = 0.
error = 0.
okp, oki, okd, totError = (0.0,)*4

# TIMERS
timer_run = 0
t_timer_3d = 0
timer_run_sim = 0
timer_run_servo = 0
t = 0.
timer_disturbance = 0.
timer_U = 0.

# FLAGS
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
send_gnss_pos = 0
send_gnss_vel = 0
gyro_sd = 0
acc_sd = 0
alt_sd = 0
gnss_pos_sd = 0
gnss_vel_sd = 0
gyro_st = 0
acc_st = 0
alt_st = 0
gnss_st = 0
var_sitl_plot = [0]*10

# FUNCTIONS

def get_data_savefile():
    param = gui.param_file_tab.get_configuration_destringed()
    rocket_dim = gui.draw_rocket_tab.get_configuration_destringed()
    conf_3d = gui.conf_3d_tab.get_configuration_destringed()
    conf_sitl = gui.conf_sitl_tab.get_configuration_destringed()
    conf_controller = gui.sim_setup_tab.get_configuration_destringed()
    return param, conf_3d, conf_controller, conf_sitl, rocket_dim

def update_all_parameters(parameters,conf_3d,conf_controller,conf_sitl, rocket_dim):
    global thrust, burnout_time, thrust_curve, max_thrust, average_thrust
    global m, m_liftoff, m_burnout, Iy, Iy_liftoff, Iy_burnout, d, xcg
    global xcg_liftoff, xcg_burnout, xt
    global k1, k2, k3, Actuator_max, Actuator_reduction
    global u_initial_offset, motor_offset
    global wind, wind_distribution, launchrod_lenght, theta, wind_total, launchrod_angle

    m_liftoff = parameters[1]
    m_burnout = parameters[2]
    Iy_burnout = parameters[3]
    Iy_liftoff = parameters[4]
    xcg_liftoff = parameters[5]
    xcg_burnout = parameters[6]
    xt = parameters[7]
    servo_definition = parameters[8]
    Actuator_max = parameters[9] * DEG2RAD
    Actuator_reduction = parameters[10]
    u_initial_offset = parameters[11] * DEG2RAD
    Actuator_weight_compensation = parameters[12]
    wind = parameters[13]
    wind_distribution = parameters[14]
    launchrod_lenght = parameters[15]
    launchrod_angle = parameters[16] * DEG2RAD
    Q_d.f = launchrod_angle
    motor_offset = parameters[17] * DEG2RAD
    theta = Q_d.f
    wind_total = wind
    m = m_liftoff
    Iy = Iy_liftoff
    xcg = xcg_liftoff
    rocket_mass_parameters = [m_liftoff, m_burnout, Iy_burnout, Iy_liftoff,
                              xcg_liftoff, xcg_burnout]
    roughness = [0, 0, 0]  # Rocket, stabilization fin, control fin.
    roughness[0] = parameters[18]*1e-6 + 1e-9
    roughness[1] = parameters[19]*1e-6 + 1e-9
    roughness[2] = parameters[20]*1e-6 + 1e-9

    ##
    global toggle_3d, camera_shake_toggle, slow_mo, force_scale, hide_forces
    global hide_cg, camera_type, variable_fov, fov
    toggle_3d = conf_3d[0]
    camera_shake_toggle = conf_3d[1]
    hide_forces = conf_3d[2]
    variable_fov = conf_3d[3]
    hide_cg = conf_3d[4]
    camera_type = conf_3d[5]
    slow_mo = conf_3d[6]
    force_scale = conf_3d[7]
    fov = conf_3d[8]

    # rocket Class
    global S, d
    gui.savefile.read_motor_data(gui.param_file_tab.combobox[0].get())
    rocket.set_motor(gui.savefile.get_motor_data())
    burnout_time = rocket.burnout_time()
    rocket.update_rocket(gui.draw_rocket_tab.get_configuration_destringed(),
                         rocket_mass_parameters, roughness)
    S = rocket.area_ref
    d = rocket.max_diam

    # controller
    global kp, ki, kd, k_all, k_damping, anti_windup
    global torque_controller, inp, inp_time, t_launch
    global T, Ts, T_Program, sim_duration
    global input_type, reference_thrust
    global average_T, launch_altitude
    global position_global, position_local, v_glob, Q
    global export_T
    input_type = conf_controller[2]
    controller.setup_controller(conf_controller[0:9],
                                Actuator_reduction,
                                Actuator_max)
    inp = conf_controller[9]
    inp_time = conf_controller[10]
    t_launch = conf_controller[11]
    Ts = conf_controller[12]
    T_Program = conf_controller[13]
    sim_duration = conf_controller[14]
    T = conf_controller[15]
    export_T = conf_controller[16]
    launch_altitude = conf_controller[17]
    position_global[0] = conf_controller[18]
    x_d.f = position_global[0]
    v_glob = [conf_controller[19], conf_controller[20]]
    x_d.f_d, z_d.f_d = v_glob[0], v_glob[1]
    Q_d.f += conf_controller[21] * DEG2RAD
    Q_d.f_d = conf_controller[22] * DEG2RAD
    Q = Q_d.f_d
    theta = Q_d.f
    [U_d.f, W_d.f] = glob2loc(position_global[0], 0, theta)
    average_T = T

    # SITL
    global Activate_SITL, use_noise, enable_python_sitl, module, port, baudrate
    global gyro_sd, acc_sd, alt_sd, gnss_pos_sd, gnss_vel_sd, gyro_st, acc_st
    global alt_st, gnss_st
    Activate_SITL = conf_sitl[0]
    use_noise = conf_sitl[1]
    enable_python_sitl = conf_sitl[2]
    module = conf_sitl[3]
    port = conf_sitl[4]
    baudrate = conf_sitl[5]
    gyro_sd = conf_sitl[6]
    acc_sd = conf_sitl[7]
    alt_sd = conf_sitl[8]
    gnss_pos_sd = conf_sitl[9]
    gnss_vel_sd = conf_sitl[10]
    gyro_st = conf_sitl[11]
    acc_st = conf_sitl[12]
    alt_st = conf_sitl[13]
    gnss_st = conf_sitl[14]

    global send_gyro, send_alt, send_gnss_vel
    send_gyro = Q
    send_alt = position_global[0]
    send_gnss_vel = v_glob[0]

    global data_plot
    data_plot = gui.run_sim_tab.get_configuration_destringed()

    # Servo Class
    servo.setup(Actuator_weight_compensation, servo_definition, Ts)


def reset_variables():
    # Ugly ugly piece of code
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
    global theta, ca, aoa, U, W, Q, U_d, W_d, Q_d, x_d, z_d, v_loc, v_loc_tot
    global v_glob, g_loc, F_loc, F_glob, position_global, acc_glob, xa
    global position_local
    theta = 0
    ca = 0
    aoa = 0
    xa = 0
    U = 0
    W = 0
    Q = 0
    U_d = IntegrableVariable()
    W_d = IntegrableVariable()
    Q_d = IntegrableVariable()
    x_d = IntegrableVariable()
    z_d = IntegrableVariable()
    v_loc = [0.0001, 0.00001]
    v_loc_tot = [0.0001, 0.00001]
    v_glob = [0.00001, 0.00001]
    g_loc = [0.0000, 0.0000]
    acc_glob = [0.0001, 0.0001]
    F_loc = [0.0000, 0.0000]
    F_glob = [0.0000, 0.0000]
    position_global = [0, 0]
    position_local = [0, 0]

    ##
    global t_3d, theta_3d, servo_3d, v_loc_3d, v_glob_3d, position_3d, xa_3d
    global thrust_3d, cn_3d, fin_force_3d, aoa_3d, setpoint_3d, Airspeed_3d
    global X_3d, Z_3d, xcg_3d
    t_3d = [0]
    theta_3d = [0]
    setpoint_3d = [0]
    servo_3d = [0]
    v_loc_3d = [[0, 0]]
    v_glob_3d = [[0, 0]]
    Airspeed_3d = [[0, 0]]
    position_3d = [[0, 0]]
    X_3d = [0]
    Z_3d = [0]
    cn_3d = [0]
    fin_force_3d = [0]
    thrust_3d = [0]
    xa_3d = [0]
    xcg_3d = [0]
    aoa_3d = [0]

    ##
    global second_plot, first_plot, third_plot, fourth_plot, fifth_plot, t_plot
    global sixth_plot, seventh_plot, eighth_plot, ninth_plot, tenth_plot
    first_plot = []
    second_plot = []
    third_plot = []
    fourth_plot = []
    fifth_plot = []
    sixth_plot = []
    seventh_plot = []
    eighth_plot = []
    ninth_plot = []
    tenth_plot = []
    t_plot = []

    ##
    global alpha_calc, U_vect, V_vect, wind_vect, u_eq, u_prev
    global u_delta, u_controller
    alpha_calc = 0.
    U_vect = np.array([0.1, 0])
    V_vect = np.array([0.1, 0])
    wind_vect = np.array([0, wind])

    ##
    global u_servos, u
    u_servos = 0.
    u = 0.

    # CONTROL
    global setpoint, error
    setpoint = 0.
    error = 0.

    # TIMERS
    global timer_run, t_timer_3d, timer_run_sim, timer_run_servo
    global t, timer_disturbance, timer_U
    timer_run = 0
    t_timer_3d = 0
    timer_run_sim = 0
    timer_run_servo = 0
    t = 0.
    timer_disturbance = 0.
    timer_U = 0.

    # FLAGS
    global flag, flag2
    flag = False
    flag2 = False

    # SITL
    global arduino_ready_flag0, timer_flag_t0, t0_timer, parachute
    global average_T_counter
    global send_gyro, send_accx, send_accz, send_alt
    global send_gnss_pos, send_gnss_vel
    arduino_ready_flag0 = ""
    timer_flag_t0 = False
    t0_timer = 0.
    parachute = 0
    average_T_counter = 0
    send_gyro = 0
    send_accx = 0
    send_accz = 0
    send_alt = 0
    send_gnss_pos = 0
    send_gnss_vel = 0
    rocket.reset_variables()


# Transforms from local coordinates to global coordinates
# (body to world)
def loc2glob(u0, v0, theta):
    # Rotational matrix 2x2
    # Axes are rotated, there is more info in the Technical documentation.
    A = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    u = np.array([[u0], [v0]])
    x = np.dot(A, u)
    a = [x[0, 0], x[1, 0]]
    return a


def glob2loc(u0, v0, theta):
    A = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    u = np.array([[u0], [v0]])
    x = np.dot(A, u)
    a = [x[0, 0], x[1, 0]]
    return a


def update_parameters():
    global wind_rand, wind_total
    global q
    global cn, fin_force
    global x
    global xa
    global i
    global aoa
    global wind
    global thrust, t_launch, t, xcg, m, Iy
    global out, timer_disturbance, timer_U, U2, q_wind
    global cm_xcg, ca, S
    global actuator_angle, launch_altitude

    # NEW SIMULATION
    global v_loc, v_loc_tot, v_glob
    global U_d, U, X
    global W_d, W, Z
    global Q_d, Q
    global theta, aoa, g, g_loc

    # Times the disturbances so they don't change that often
    if t>timer_disturbance + 0.1:
        wind_rand = random.gauss(0, wind_distribution)
        wind_total = wind_rand + wind
        timer_disturbance = t

    # NEW SIMULATION
    # Computes the velocity of the wind in local coordinates
    wind_loc = glob2loc(0, wind_total, theta)
    # Computes the total airspeed in local coordinates
    v_loc_tot = [v_loc[0]-wind_loc[0], v_loc[1]-wind_loc[1]]
    aoa = calculate_aoa(v_loc_tot)
    thrust = rocket.get_thrust(t, t_launch)
    m, Iy, xcg = rocket.get_mass_parameters(t, t_launch)
    S = rocket.area_ref
    v_modulus = np.sqrt(v_loc_tot[0]**2 + v_loc_tot[1]**2)
    if rocket.use_fins_control is True:
        # Detailed explanation in rocket_functions
        cn, cm_xcg, ca, xa = rocket.calculate_aero_coef(v_loc_tot, Q,
                                                        position_global[0] + launch_altitude,
                                                        actuator_angle)
    else:
        cn, cm_xcg, ca, xa = rocket.calculate_aero_coef(v_loc_tot, Q,
                                                        position_global[0] + launch_altitude)
    # Computes the dynamic pressure
    rho = rocket.rho
    q = 0.5 * rho * v_modulus**2
    # Gravity in local coordinates, theta=0 equals to rocket up
    g_loc = glob2loc(-g, 0, theta)


def calculate_aoa(v_loc_tot):
    if v_loc_tot[0] != 0:
        aoa = np.arctan2(v_loc_tot[1], v_loc_tot[0])
    else:
        aoa = np.pi/2
    return aoa


def simulation():
    global x, xs
    global xdot, xdots
    global out, outs
    global out_prev, out_prevs
    global u_controller
    global u, timer_run_servo, u_servos, actuator_angle
    global v_loc, v_loc_tot, v_glob
    global U_d, U, X
    global W_d, W, Z
    global Q_d, Q
    global theta, aoa, g
    global F_loc, F_glob
    global cn, thrust, rho, fin_force
    global cm_xcg, ca, xcg, m, Iy
    global t_timer_3d
    global position_global, position_local
    global i_turns
    global accx, accz, accQ, g_loc
    global t_launch, launchrod_angle

    global force_app_point, normal_force

    # SERVO SIMULATION
    servo_current_angle = servo.simulate(u_servos, t)
    # Reduction of the TVC
    actuator_angle = (servo_current_angle/Actuator_reduction) + u_initial_offset
    update_parameters()
    """
    NEW METHOD, DIRECTLY INTEGRATES THE DIFFERENTIAL EQUATIONS
    U = local speed in X
    W = local speed in Z
    Q = pitch rate
    aoa = angle of attack

    v_glob = global velocity
    x_d = global X speed (Y in Vpython)
    z_d = global Z speed (-X in Vpython)
    """
    v_d = 0  # 0 uses Local and Global Velocities, 1 uses vector derivatives.

    if rocket.is_in_the_pad(position_global[0]) and thrust < m*g:
        accx = 0
        accz = 0
        accQ = 0
        force_app_point = 0
        normal_force = 0
    else:
        launchrod_global_coor = loc2glob(launchrod_lenght, 0, launchrod_angle)
        if position_global[0] <= launchrod_global_coor[0]:
            launchrod_lock = 0
        else:
            launchrod_lock = 1
        if rocket.use_fins_control is False:
            motor_angle = actuator_angle + motor_offset
        else:
            motor_angle = motor_offset
            fin_force = q * S * rocket.fin_cn[1]
        x_force = thrust * np.cos(motor_angle) - q*S*ca + m*g_loc[0]
        z_force = thrust * np.sin(motor_angle) + m*g_loc[1] + q*S*cn
        Q_moment = (thrust * np.sin(motor_angle) * (xt-xcg) + S*q*d*cm_xcg)
        accx = x_force/m - W*Q*v_d
        accz = (z_force/m + U*Q*v_d) * launchrod_lock
        accQ = (Q_moment/Iy) * launchrod_lock
        normal_force = z_force - m*g_loc[1]
        force_app_point = Q_moment / normal_force + xcg
        force_app_point = saturate_plot_xa_force_app(force_app_point)

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
    if theta < -np.pi:
        theta += 2*np.pi
        Q_d.new_f(theta)

    # New acceleration in global coordinates
    global acc_glob
    acc_glob = loc2glob(accx, accz, theta)
    if v_d == 1:
        # Just integrates, the transfer of velocities was already
        # done in the vector derivative
        v_loc[0] = U_d.integrate_f_dd()
        v_loc[1] = W_d.integrate_f_dd()
    else:
        # Takes the global velocity, transforms it into local coordinates,
        # adds the accelerations
        # and transforms the velocity back into global coordinates
        v_loc = glob2loc(v_glob[0], v_glob[1], theta)
        U_d.integrate_f_dd()
        W_d.integrate_f_dd()
        v_loc[0] += U_d.delta_f_d
        v_loc[1] += W_d.delta_f_d

    # New velocity in global coordinates
    v_glob = loc2glob(v_loc[0], v_loc[1], theta)

    # Updates the global velocity in the x_d class
    x_d.new_f_d(v_glob[0])
    z_d.new_f_d(v_glob[1])

    # Integrates the velocities to get the position, be it local or global
    position_local = [U_d.integrate_f_d(), W_d.integrate_f_d()]
    position_global = [x_d.integrate_f_d(), z_d.integrate_f_d()]

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
    (500) is the rate of the animation, when you use slow_mo it drops.
    To ensure fluidity at least a rate of 100 ish is recommended, so a
    rate of 1000 allows for 10 times slower animations.
    """
    if t >= t_timer_3d + 0.00499:
        # 3d
        t_3d.append(t)
        theta_3d.append(theta)
        servo_3d.append(actuator_angle)
        v_loc_3d.append(v_loc)
        v_glob_3d.append(v_glob)
        position_3d.append(position_global)
        xa_3d.append(rocket.cp_w_o_ctrl_fin)
        xcg_3d.append(rocket.xcg)
        thrust_3d.append(thrust)
        cn_3d.append(rocket.passive_cn*S*q)
        fin_force_3d.append(fin_force)
        aoa_3d.append(aoa)
        setpoint_3d.append(setpoint)
        t_timer_3d = t


def timer():
    global t
    t = round(t + T, 12)  # Trying to avoid error, not sure it works

def timer_SITL():
    global t, timer_flag_t0, t0_timer, T, average_T, average_T_counter
    if timer_flag_t0 is False:
        t0_timer = time.perf_counter()/clock_dif
        timer_flag_t0 = True
    t_prev = t
    t = time.perf_counter()/clock_dif - t0_timer
    # Sample time in SITL = Time elapsed between runs
    T = t-t_prev
    average_T_counter += 1
    average_T = t / average_T_counter

def set_setpoint(inp):
    global input_type
    if input_type == "Step [º]":
        setpoint = inp*DEG2RAD
    elif input_type == "Ramp [º/s]":
        setpoint = (inp*DEG2RAD)*(t-inp_time)
    else:
        setpoint = 0
    return setpoint

def saturate_plot_xa_force_app(v):
    if abs(v) > 3 * rocket.length:
        v_plot = 3 * rocket.length * np.sign(v)
    else:
        v_plot = v
    return v_plot

def check_which_plot(s):
    global okp, oki, okd, totError
    global send_gyro, send_accx, send_accz, send_alt
    global send_gnss_pos, send_gnss_vel
    global actuator_angle, normal_force
    global var_sitl_plot
    if s == "Setpoint [º]":
        return setpoint * RAD2DEG
    elif s == "Pitch Angle [º]":
        return theta * RAD2DEG
    elif s == "Actuator deflection [º]":
        return actuator_angle * RAD2DEG
    elif s == "Pitch Rate [º/s]":
        return Q * RAD2DEG
    elif s == "Local Velocity X [m/s]":
        return v_loc[0]
    elif s == "Local Velocity Z [m/s]":
        return v_loc[1]
    elif s == "Global Velocity X [m/s]":
        return v_glob[0]
    elif s == "Global Velocity Z [m/s]":
        return v_glob[1]
    elif s == "Total Velocity [m/s]":
        return np.sqrt(v_loc_tot[0]**2 + v_loc_tot[1]**2)
    elif s == "Local Acc X [m^2/s]":
        return accx
    elif s == "Local Acc Z [m^2/s]":
        return accz
    elif s == "Global Acc X [m^2/s]":
        return acc_glob[0]
    elif s == "Global Acc Z [m^2/s]":
        return acc_glob[1]
    elif s == "Angle of Atack [º]":
        return aoa * RAD2DEG
    elif s == "CP Position [m]":
        xa_plot = saturate_plot_xa_force_app(xa)
        return xa_plot
    elif s == "Mass [kg]":
        return m
    elif s == "Iy [kg*m^2]":
        return Iy
    elif s == "CG Position [m]":
        return xcg
    elif s == "Thrust [N]":
        return thrust
    elif s == "Normal Force Coefficient":
        return cn
    elif s == "Axial Force Coefficient":
        return ca
    elif s == "Moment Coefficient":
        return cm_xcg
    elif s == "Force Application Point [m]":
        return force_app_point
    elif s == "Normal Force [N]":
        return normal_force
    elif s == "Altitude [m]":
        return position_global[0]
    elif s == "Distance Downrange [m]":
        return position_global[1]
    elif s == "Proportional Contribution":
        return okp * RAD2DEG
    elif s == "Integral Contribution":
        return oki * RAD2DEG
    elif s == "Derivative Contribution":
        return okd * RAD2DEG
    elif s == "Total Error":
        return totError * RAD2DEG
    elif s == "Simulated Gyro [º/s]":
        return send_gyro
    elif s == "Simulated Acc X [m^2/s]":
        return send_accx
    elif s == "Simulated Acc Z [m^2/s]":
        return send_accz
    elif s == "Simulated Altimeter":
        return send_alt
    elif s == "Simulated GNSS Position [m]":
        return send_gnss_pos
    elif s == "Simulated GNSS Velocity [m/s]":
        return send_gnss_vel
    elif s == "Variable SITL 1":
        return var_sitl_plot[0]
    elif s == "Variable SITL 2":
        return var_sitl_plot[1]
    elif s == "Variable SITL 3":
        return var_sitl_plot[2]
    elif s == "Variable SITL 4":
        return var_sitl_plot[3]
    elif s == "Variable SITL 5":
        return var_sitl_plot[4]
    elif s == "Variable SITL 6":
        return var_sitl_plot[5]
    elif s == "Variable SITL 7":
        return var_sitl_plot[6]
    elif s == "Variable SITL 8":
        return var_sitl_plot[7]
    elif s == "Variable SITL 9":
        return var_sitl_plot[8]
    elif s == "Variable SITL 10":
        return var_sitl_plot[9]
    elif s == "Off":
        return None

def plot_data():
    global data_plot
    a_plt = check_which_plot(data_plot[0])
    b_plt = check_which_plot(data_plot[1])
    c_plt = check_which_plot(data_plot[2])
    d_plt = check_which_plot(data_plot[3])
    e_plt = check_which_plot(data_plot[4])
    f_plt = check_which_plot(data_plot[5])
    g_plt = check_which_plot(data_plot[6])
    h_plt = check_which_plot(data_plot[7])
    i_plt = check_which_plot(data_plot[8])
    j_plt = check_which_plot(data_plot[9])
    first_plot.append(a_plt)
    second_plot.append(b_plt)
    third_plot.append(c_plt)
    fourth_plot.append(d_plt)
    fifth_plot.append(e_plt)
    sixth_plot.append(f_plt)
    seventh_plot.append(g_plt)
    eighth_plot.append(h_plt)
    ninth_plot.append(i_plt)
    tenth_plot.append(j_plt)
    t_plot.append(t)

def plot_plots():
    plt.figure(1, figsize=(12, 7), dpi=100)  # First Plot
    s = gui.run_sim_tab.get_configuration_destringed()
    if s[0] != "Off":
        plt.plot(t_plot,
                 first_plot,
                 label=gui.run_sim_tab.get_configuration_destringed()[0])
    if s[1] != "Off":
        plt.plot(t_plot,
                 second_plot,
                 label=gui.run_sim_tab.get_configuration_destringed()[1])
    if s[2] != "Off":
        plt.plot(t_plot,
                 third_plot,
                 label=gui.run_sim_tab.get_configuration_destringed()[2])
    if s[3] != "Off":
        plt.plot(t_plot,
                 fourth_plot,
                 label=gui.run_sim_tab.get_configuration_destringed()[3])
    if s[4] != "Off":
        plt.plot(t_plot,
                 fifth_plot,
                 label=gui.run_sim_tab.get_configuration_destringed()[4])
    plt.grid(True, linestyle='--')
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('', fontsize=16)
    plt.legend(shadow=True, fontsize='small')
    plt.axvline(x=burnout_time+t_launch, color="black", linewidth=1)

    # Second Plot
    if s[5] != "Off" or s[6] != "Off" or s[7] != "Off" or s[8] != "Off" or s[9] != "Off":
        plt.figure(2, figsize=(12, 7), dpi=100)
        s = gui.run_sim_tab.get_configuration_destringed()
        if s[5] != "Off":
            plt.plot(t_plot,
                     sixth_plot,
                     label=gui.run_sim_tab.get_configuration_destringed()[5])
        if s[6] != "Off":
            plt.plot(t_plot,
                     seventh_plot,
                     label=gui.run_sim_tab.get_configuration_destringed()[6])
        if s[7] != "Off":
            plt.plot(t_plot,
                     eighth_plot,
                     label=gui.run_sim_tab.get_configuration_destringed()[7])
        if s[8] != "Off":
            plt.plot(t_plot,
                     ninth_plot,
                     label=gui.run_sim_tab.get_configuration_destringed()[8])
        if s[9] != "Off":
            plt.plot(t_plot,
                     tenth_plot,
                     label=gui.run_sim_tab.get_configuration_destringed()[9])
        plt.grid(True, linestyle='--')
        plt.xlabel('Time', fontsize=16)
        plt.ylabel('', fontsize=16)
        plt.legend(shadow=True, fontsize='small')
        plt.axvline(x=burnout_time+t_launch, color="black", linewidth=1)

def export_plots(file_name):
    names = gui.run_sim_tab.get_configuration_destringed()
    names_to_csv = ["Time"]
    for name in names:
        if name != "Off":
            names_to_csv.append(name)
    plots_to_csv = [t_plot]
    if first_plot[0] != None:
        plots_to_csv.append(first_plot)
    if second_plot[0] != None:
        plots_to_csv.append(second_plot)
    if third_plot[0] != None:
        plots_to_csv.append(third_plot)
    if fourth_plot[0] != None:
        plots_to_csv.append(fourth_plot)
    if fifth_plot[0] != None:
        plots_to_csv.append(fifth_plot)
    if sixth_plot[0] != None:
        plots_to_csv.append(sixth_plot)
    if seventh_plot[0] != None:
        plots_to_csv.append(seventh_plot)
    if eighth_plot[0] != None:
        plots_to_csv.append(eighth_plot)
    if ninth_plot[0] != None:
        plots_to_csv.append(ninth_plot)
    if tenth_plot[0] != None:
        plots_to_csv.append(tenth_plot)
    export_T = gui.sim_setup_tab.get_configuration_destringed()[16]
    files.export_plots(file_name, names_to_csv, plots_to_csv, export_T)


def run_sim_local():
    global parameters, conf_3d, conf_controller
    global timer_run_sim, timer_run, setpoint, t_launch,inp_time, u_servos
    global okp, oki, okd, totError
    progress_bar = ProgressBar()

    while t <= sim_duration:
        simulation()
        """
        *.999 corrects the error in t produced by adding t=t+T for sample times
        smaller than 0.001. If it was exact, it starts accumulating error,
        and the smaller the sample time, the more error it accumulates
        So, be careful with the sample time, 0.001, 0.0005 and 0.0001
        give all similar results, so, if it takes to long to run you can
        increase T:
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
        progress_bar.update(t, sim_duration)
        plot_data()
        if position_global[0] < -0.55:
            progress_bar.update(t, t, 0)
            if abs(v_glob[0]) < 2:
                print("\nLanding!")
            else:
                print("\nCRASH")
            break
        if t >= sim_duration:
            progress_bar.update(t, t, 0)
            print("\nSimulation Ended")
            break
        if rocket.is_supersonic:
            progress_bar.update(t, t, 0)
            print("\nTransonic and supersonic flow, abort!")
            break
        timer()

def run_sim_sitl():
    global parameters, conf_3d, conf_controller, setpoint
    global timer_run_sim, timer_run, setpoint, parachute, t_launch, u_servos
    global send_gyro, send_accx, send_accz, send_alt
    global timer_flag_t0, clock_dif, T_glob, parachute, t0_timer
    import serial

    timer_seconds = 0
    timer_flag_t0 = False
    t0_timer = 0.
    # Clock_dif because the Arduino clock is slower than the PC one
    # (0.94 for mine)
    clock_dif = 1
    # 0.001 sample time of the simulation loop,
    # the sample time of the simulation
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
            if t >= timer_run_sim + T_glob*0.999:
                timer_run_sim = t
                if t >= inp_time:
                    setpoint = set_setpoint(inp)
            plot_data()
            if t >= timer_seconds + 1:
                timer_seconds = t
                print("Time is ", round(t, 0), " seconds")
            if t > burnout_time * 10:
                break
            if position_global[0] < -0.55:
                if abs(v_glob[0]) < 2:
                    print("Landing!")
                else:
                    print("CRASH")
                break
            if parachute == 1:
                print("Parachute Deployed")
                break
            if t >= sim_duration:
                print("Simulation Ended")
                break
            if rocket.is_supersonic:
                print("Transonic and supersonic flow, abort!")
                break
            i += 1
        ##
        if t >= 0.003:
            if serialArduino.inWaiting() > 1:
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
                    # last comma because the Arduino library separates the
                    # string at commas
                    send = (str(round(send_gyro, 6))+","
                            + str(round(send_accx, 6)) + ","
                            + str(round(send_accz, 6)) + ","
                            + str(round(send_alt, 2)) + ",")
                    serialArduino.write(str(send).encode("ASCII"))
                    serialArduino.write('\n'.encode("ASCII"))
                else:
                    # Arduino sent data for the program to read
                    # Arduino is not ready to read data
                    read_split = read.split(",")
                    u_servos = float(read_split[0])*DEG2RAD
                    parachute = int(read_split[1])


def run_sim_python_sitl():
    global parameters, conf_3d, conf_controller, setpoint
    global timer_run_sim, timer_run, setpoint, parachute, t_launch, u_servos
    global send_gyro, send_accx, send_accz, send_alt
    global send_gnss_pos, send_gnss_vel
    global parachute, python_sitl
    progress_bar = ProgressBar()

    timer_gyro = 0
    timer_acc = 0
    timer_alt = 0
    timer_gnss = 0
    parachute = 0
    python_sitl = importlib.import_module("SITL Modules."+module, package="SITL Modules")
    python_sitl_program = python_sitl.SITLProgram()
    python_sitl_program.everything_that_is_outside_functions()
    python_sitl_program.void_setup()

    while t <= sim_duration:
        simulation()

        python_sitl_program.void_loop()

        if use_noise is True:
            if t >= timer_gyro + gyro_st*0.999:
                send_gyro = round(random.gauss(Q*RAD2DEG, gyro_sd), 6)
                timer_gyro = t
            if t >= timer_acc + acc_st*0.999:
                send_accx = round(random.gauss((accx-g_loc[0])/9.81, acc_sd), 6)
                send_accz = round(random.gauss((accz-g_loc[1])/9.81, acc_sd), 6)
                timer_acc = t
            if t >= timer_alt + alt_st*0.999:
                send_alt = round(random.gauss(position_global[0], alt_sd), 6)
                timer_alt = t
            if t >= timer_gnss + gnss_st*0.999:
                send_gnss_pos = round(random.gauss(position_global[1], gnss_pos_sd), 6)
                send_gnss_vel = round(random.gauss(v_glob[1], gnss_vel_sd), 6)
                timer_gnss = t
        else:
            if t >= timer_gyro + gyro_st*0.999:
                send_gyro = round(Q*RAD2DEG, 6)
                timer_gyro = t
            if t >= timer_acc + acc_st*0.999:
                send_accx = round((accx-g_loc[0])/9.81, 6)
                send_accz = round((accz-g_loc[1])/9.81, 6)
                timer_acc = t
            if t >= timer_alt + alt_st*0.999:
                send_alt = round(position_global[0], 6)
                timer_alt = t
            if t >= timer_gnss + gnss_st*0.999:
                send_gnss_pos = round(position_global[1], 6)
                send_gnss_vel = round(v_glob[1], 6)
                timer_gnss = t
        progress_bar.update(t, sim_duration)
        plot_data()
        if position_global[0] < -0.55:
            progress_bar.update(t, t, 0)
            if abs(v_glob[0]) < 2:
                print("\nLanding!")
            else:
                print("\nCRASH")
            break
        if parachute == 1:
            progress_bar.update(t, t, 0)
            print("\nParachute Deployed")
            break
        if t >= sim_duration:
            progress_bar.update(t, t, 0)
            print("\nSimulation Ended")
            break
        if rocket.is_supersonic:
            progress_bar.update(t, t, 0)
            print("\nTransonic and supersonic flow, abort!")
            break
        timer()
    del python_sitl_program


def run_simulation():
    global parameters, conf_3d, conf_controller
    global timer_run_sim,timer_run, setpoint
    reset_variables()
    parameters, conf_3d, conf_controller, conf_sitl, rocket_dim = get_data_savefile()
    update_all_parameters(parameters,
                          conf_3d,
                          conf_controller,
                          conf_sitl,
                          rocket_dim)
    print("Simulation Started")
    if Activate_SITL is False:
        run_sim_local()
    elif enable_python_sitl is False:
        run_sim_sitl()
    else:
        run_sim_python_sitl()
    plot_plots()
    return

"""
3D 3D 3D 3D 3D
3D 3D 3D 3D 3D
3D 3D 3D 3D 3D
3D 3D 3D 3D 3D
"""
break_flag_button = False
pause_resume_flag = False
skip_flag = False
skip_ahead_flag = False
skip_backwards_flag = False
skip_steps = 10
widgets = []
widgets_text = []
def run_3d():
    global break_flag_button, pause_resume_flag, skip_flag, skip_ahead_flag
    global skip_backwards_flag, skip_steps, hide_cg, widgets
    global labels, scene, widgets_text

    if toggle_3d is False:
        plt.draw()
        plt.show()
        print("\n")
    if toggle_3d is True:
        try:
            labels.delete()
            scene.delete()
            for i in range(len(widgets)):
                widgets[i].delete()
            for i in range(len(widgets_text)):
                widgets_text[i].text = ""
        except:
            pass

        max_hight = 0
        for i in range(len(position_3d)):
            if position_3d[i][0] > max_hight:
                max_hight = position_3d[i][0]

        rocket_dim = rocket.rocket_dim
        L_body = rocket_dim[-1][0]
        L_total = gui.draw_rocket_tab.max_length
        nosecone_length = rocket_dim[1][0]
        d = rocket_dim[1][1]

        scene = vp.canvas(width=1280, height=720, center=vp.vector(0, 0, 0),
                          background=vp.color.white)
        scene.lights = []
        vp.distant_light(direction=vp.vector(1, 1, -1), color=vp.color.gray(0.9))
        i = 0

        """Background ######################################################"""
        dim_x_sky = 3000
        dim_z_sky = 3000
        dim_x_floor = 1500
        dim_z_floor = 800

        n = 7
        # Sky (many panels)
        for i in range(n):
            for j in range(n):
                vp.box(pos=vp.vector(dim_x_sky * (i - n/2 + 1) - 1000,
                                     dim_z_sky * (j + 0.5) - 30,
                                     dim_x_sky*0 + 1600*0+dim_x_floor*0.7),
                       size=vp.vector(dim_x_sky, dim_z_sky, 1),
                       color=vp.color.white,
                       texture={'file': str(textures_path / 'sky_texture.jpg')})

        floor1 = vp.box(pos=vp.vector(dim_x_floor*0.8,
                                      -0.5,
                                      dim_z_floor/2+dim_x_floor/4*0.7),
                        size=vp.vector(dim_x_floor, 1, dim_z_floor),
                        texture={'file': str(textures_path / '1500x800.jpg')})
        floor1.rotate(180*DEG2RAD, axis=vp.vector(0,1,0))

        floor2 = vp.box(pos=vp.vector(dim_x_floor + dim_x_floor*0.8,
                                      -0.5,
                                      dim_z_floor/2+dim_x_floor/4*0.7),
                        size=vp.vector(dim_x_floor, 1, dim_z_floor),
                        texture={'file': str(textures_path / '1500x800.jpg')})
        floor2.rotate(180*DEG2RAD, axis=vp.vector(0,1,0))

        floor3 = vp.box(pos=vp.vector(dim_x_floor*0.8 - dim_x_floor,
                                      -0.5,
                                      dim_z_floor/2+dim_x_floor/4*0.7),
                        size=vp.vector(dim_x_floor, 1, dim_z_floor),
                        texture={'file': str(textures_path / '1500x800.jpg')})
        floor3.rotate(180*DEG2RAD, axis=vp.vector(0,1,0))

        floor4 = vp.box(pos=vp.vector(dim_x_floor*0.8 - 2*dim_x_floor,
                                      -0.5,
                                      dim_z_floor/2+dim_x_floor/4*0.7),
                        size=vp.vector(dim_x_floor, 1, dim_z_floor),
                        texture={'file': str(textures_path / '1500x800.jpg')})
        floor4.rotate(180*DEG2RAD, axis=vp.vector(0,1,0))


        """Rocket #########################################################"""
        n_c = 50  # how many pieces have each non standard component
        R_ogive = rocket_dim[1][1] / 2
        L_ogive = rocket_dim[1][0]
        rho_radius = (R_ogive**2 + L_ogive**2)/(2 * R_ogive)
        compound_list = []
        for i in range(len(rocket_dim)-1):
            if i == 0 and rocket.ogive_flag is True:
                # Ogive goes brrrrr
                l_partial2 = 0
                l_partial = L_ogive / n_c
                for j in range(n_c):
                    # diameter must never be 0
                    r = ((np.sqrt(rho_radius**2 - (L_ogive-l_partial2)**2)
                          + R_ogive - rho_radius)
                         + 0.000001)
                    pos = l_partial * (j+1)
                    rod = vp.cylinder(pos=vp.vector(dim_x_floor/2,
                                                    L_total-pos,
                                                    dim_z_floor/2),
                                      axis=vp.vector(0, 1, 0), radius=r,
                                      color=vp.color.black, length=l_partial)
                    compound_list.append(rod)
                    l_partial2 += l_partial
                continue
            if i == 0 and rocket.ogive_flag is False:
                nosecone = vp.cone(pos=vp.vector(dim_x_floor/2,
                                                 L_total-nosecone_length,
                                                 dim_z_floor/2),
                                   axis=vp.vector(0, 1, 0), radius=d/2,
                                   color=vp.color.black,
                                   length=nosecone_length)
                compound_list.append(nosecone)
                continue
            if rocket_dim[i][1] == rocket_dim[i+1][1]:
                l = rocket_dim[i+1][0] - rocket_dim[i][0]
                d = rocket_dim[i][1]
                rod = vp.cylinder(pos=vp.vector(dim_x_floor/2,
                                                L_total-rocket_dim[i+1][0],
                                                dim_z_floor/2),
                                  axis=vp.vector(0, 1, 0),
                                  radius=d/2,
                                  color=vp.color.black,
                                  length=l)
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
                    rod = vp.cylinder(pos=vp.vector(dim_x_floor/2,
                                                    L_total-pos,
                                                    dim_z_floor/2),
                                      axis=vp.vector(0, 1, 0),
                                      radius=d/2,
                                      color=vp.color.black,
                                      length=l_partial)
                    compound_list.append(rod)

        """Fins ###########################################################"""
        throwaway_box = vp.box(pos=vp.vector(dim_x_floor/2,
                                             0.1,
                                             dim_z_floor/2),
                               axis=vp.vector(1, 0, 0),
                               size=vp.vector(0.001, 0.001, 0.001),
                               color=vp.color.black)
        fin_compound = [0]*4
        fin_outline_interp = interp1d([0, 1], [0, 0], fill_value=0, bounds_error=False)
        for i in range(4):
            fin_compound[i] = throwaway_box
        fin_compound_control = [0]*4
        for i in range(4):
            fin_compound_control[i] = throwaway_box
        if rocket.use_fins is True:
            compound_fins = []
            chord_list = [0]*2
            fins = gui.draw_rocket_tab.get_points_float(1)
            fin_parameters = gui.draw_rocket_tab.get_param_fin_float(0)
            wingspan = fin_parameters[2]
            le_list = [[fins[0][0], fins[1][0]],
                       [fins[0][1], fins[1][1]]]
            te_list = [[fins[3][0], fins[2][0]],
                       [fins[3][1], fins[2][1]]]

            if fins[3][0] >= fins[2][0] and fins[0][0] <= fins[1][0]:  # Delta
                fin_outline = [[fins[0][0], fins[1][0], fins[2][0], fins[3][0]],
                               [fins[0][1], fins[1][1], fins[2][1], fins[3][1]]]
            elif fins[2][0] > fins[3][0]:  # swept back
                fin_outline = [[fins[0][0], fins[1][0], fins[2][0]],
                               [fins[0][1], fins[1][1], fins[2][1]]]
            else:  # swept forward
                fin_outline = [[fins[1][0], fins[2][0], fins[3][0]],
                               [fins[1][1], fins[2][1], fins[3][1]]]
            fin_outline_interp = interp1d(fin_outline[0], fin_outline[1],
                                          fill_value=0, bounds_error=False)
            for i in range(len(le_list)):
                chord_list[i] = te_list[0][i] - le_list[0][i]
            thickness = rkt.fin[0].pp.thickness
            if thickness < 0.002:
                thickness = 0.002
            if fins[1][0] > 0.001:
                n = 50
                b = wingspan/n
                for i in range(n):
                    wing_station = b * (i+1) + fins[0][1] - b/2
                    le = np.interp(wing_station, le_list[1], le_list[0])
                    chord = (np.interp(wing_station, te_list[1], te_list[0])
                             - np.interp(wing_station, le_list[1], le_list[0]))
                    posy = L_total - le - chord/2
                    fin = vp.box(pos=vp.vector(dim_x_floor/2+wing_station,
                                               posy,
                                               dim_z_floor/2),
                                 axis=vp.vector(1, 0, 0),
                                 size=vp.vector(b*1.5, chord, thickness),
                                 color=vp.color.black)
                    compound_fins.append(fin)

                fin_compound = [0]*4
                fin_compound[0] = vp.compound(compound_fins)
                for i in range(3):
                    fin_compound[i+1] = fin_compound[0].clone(pos=fin_compound[0].pos)
                    fin_compound[i+1].rotate(np.pi/2*(i+1),
                                             axis=vp.vector(0, 1, 0),
                                             origin=vp.vector(dim_x_floor/2,
                                                              0,
                                                              dim_z_floor/2))

            if rocket.use_fins_control is True:
                fins = gui.draw_rocket_tab.get_points_float(2)
                fin_parameters = gui.draw_rocket_tab.get_param_fin_float(1)
                le_list = [[fins[0][0], fins[1][0]],
                           [fins[0][1], fins[1][1]]]
                te_list = [[fins[3][0], fins[2][0]],
                           [fins[3][1], fins[2][1]]]
                for i in range(len(le_list)):
                    chord_list[i] = te_list[0][i] - le_list[0][i]
                thickness = rkt.fin[1].pp.thickness
                if thickness < 0.002:
                    thickness = 0.002
                compound_fins_control = []
                if fins[1][0] > 0.001:
                    n = 50
                    b = fin_parameters[2]/n
                    for i in range(n):
                        wing_station = b * (i+1) + fins[0][1] - b/2
                        le = np.interp(wing_station, le_list[1], le_list[0])
                        chord = (np.interp(wing_station, te_list[1], te_list[0])
                                 - np.interp(wing_station, le_list[1], le_list[0]))
                        posy = L_total - le - chord/2
                        fin = vp.box(pos=vp.vector(dim_x_floor/2+wing_station,
                                                   posy,
                                                   dim_z_floor/2),
                                     axis=vp.vector(1, 0, 0),
                                     size=vp.vector(b*1.5, chord, thickness),
                                     color=vp.color.red)
                        compound_fins.append(fin)
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

        d = rocket.max_diam
        d_at_motor = rocket.diam_interp(L_body)
        motor_radius = d_at_motor / 4.5
        motor_lenght = motor_radius * 10

        compound_rocket = (compound_list + fin_compound
                           + [fin_compound_control[0], fin_compound_control[2]])
        rocket_3d = vp.compound(compound_rocket)
        motor = vp.cone(pos=vp.vector(dim_x_floor/2,
                                      L_total - L_body,
                                      dim_z_floor/2),
                        axis=vp.vector(0,-1,0),
                        radius=motor_radius,
                        color=vp.color.red,
                        length=motor_lenght,
                        make_trail=True)
        control_fins = vp.compound([fin_compound_control[1],
                                    fin_compound_control[3]])

        launchrod_3d_pos_z = dim_z_floor/2 + d
        launchrod_3d = vp.cylinder(pos=vp.vector(dim_x_floor/2,
                                                 0,
                                                 launchrod_3d_pos_z),
                                   axis=vp.vector(0, 1, 0),
                                   radius=d/10,
                                   color=vp.color.gray(0.5),
                                   length=launchrod_lenght)

        motor.trail_color = vp.color.red
        motor.rotate(motor_offset, axis=vp.vector(0,0,-1))

        if rocket.use_fins is True:
            sep = fin_outline_interp(L_body-0.001)
        else:
            sep = 0

        # Torque motor
        Tmotor_pos = vp.arrow(pos=vp.vector(motor.pos.x-(d_at_motor/2 + d_at_motor/3 + sep),
                                            motor.pos.y,
                                            motor.pos.z),
                              axis=vp.vector(-0.001, 0.001, 0.001),
                              shaftwidth=0,
                              color=vp.color.red)
        Tmotor_neg = vp.arrow(pos=vp.vector(motor.pos.x+(d_at_motor/2 + d_at_motor/3 + sep),
                                            motor.pos.y,
                                            motor.pos.z),
                              axis=vp.vector(-0.001, 0.001, 0.001),
                              shaftwidth=0,
                              color=vp.color.red)
        Nforce_pos = vp.arrow(pos=vp.vector(rocket_3d.pos.x + d/3,
                                            L_total,
                                            rocket_3d.pos.z),
                              axis=vp.vector(-0.001, 0.001, 0.001),
                              shaftwidth=0,
                              color=vp.color.green)
        Nforce_neg = vp.arrow(pos=vp.vector(rocket_3d.pos.x - d/3,
                                            L_total,
                                            rocket_3d.pos.z),
                              axis=vp.vector(-0.001, 0.001, 0.001),
                              shaftwidth=0,
                              color=vp.color.green)

        if rocket.use_fins_control is True:
            # Torque control fin
            T_fin_pos = vp.arrow(pos=vp.vector((control_fins.pos.x -
                                                (d/2 + rkt.fin[1].pp.wingspan + d/3)),
                                               control_fins.pos.y,
                                               control_fins.pos.z),
                                 axis=vp.vector(0.0001, 0, 0),
                                 shaftwidth=0,
                                 color=vp.color.red)
            T_fin_neg = vp.arrow(pos=vp.vector((control_fins.pos.x +
                                                (d/2 + rkt.fin[1].pp.wingspan + d/3)),
                                               control_fins.pos.y,
                                               control_fins.pos.z),
                                 axis=vp.vector(-0.0001, 0, 0),
                                 shaftwidth=0,
                                 color=vp.color.red)

        """Activate for the CG to vary with time"""
        variable_radius_cg = True
        if variable_radius_cg is True:
            max_rad_cg_ball = d/2*1.6
            min_rad_cg_ball = d/2*1.3
            rad_cg_ball_time = [[t_launch, rocket.t_burnout+t_launch],
                                [max_rad_cg_ball, min_rad_cg_ball]]
        cg_ball = vp.sphere(pos=vp.vector(dim_x_floor/2,
                                          0,
                                          dim_z_floor/2),
                            axis=vp.vector(1,0,0),
                            radius=d/2*1.5,
                            color=vp.color.white,
                            texture={'file': str(textures_path / 'center_of_gravity.jpg')})
        cg_ball.rotate(np.pi, axis=vp.vector(0,1,0))
        cg_ball.visilbe = not hide_cg

        """Activate for the velocity arrow to vary with time"""
        variable_length_velocity_arrow = False
        if variable_length_velocity_arrow is True:
            max_vel = 0
            for i in range(len(v_loc_3d)):
               v = np.sqrt(v_loc_3d[i][0]**2 + v_loc_3d[i][1]**2)
               if v > max_vel:
                   max_vel = v
            vel_arrow_length_interp = [[0, max_vel],
                                       [L_total*0.5, L_total*0.8]]
        velocity_arrow = vp.arrow(pos=(rocket_3d.pos + vp.vector(0, L_total/2, 0)),
                                  axis=vp.vector(0, 1, 0), shaftwidth=d/4,
                                  length=L_total*0.7, color=vp.color.blue,
                                  headwidth=2*d/4, headlength=3*d/4)

        """buttons & Sliders ##############################################"""
        break_flag_button = False
        pause_resume_flag = False

        widgets = []
        widgets_text = []
        widgets_text.append(vp.wtext(text="\n"))
        widgets_text.append(vp.wtext(text="          "
                                     + "                                     "
                                     + "         "))

        def skip_back_super_coarse(b):
            global skip_flag, skip_backwards_flag, skip_steps
            skip_flag = True
            skip_backwards_flag = True
            skip_steps = int(super_coarse_step)
        skip_back_super_coarse_button = vp.button(canvas=scene,
                                                  bind=skip_back_super_coarse,
                                                  text=' <<<<<<<<< ',
                                                  color=vp.vec(1, 1, 1),
                                                  background=vp.vec(0.333,
                                                                    0.465,
                                                                    0.561))
        widgets.append(skip_back_super_coarse_button)

        def skip_back_coarse(b):
            global skip_flag, skip_backwards_flag, skip_steps
            skip_flag = True
            skip_backwards_flag = True
            skip_steps = int(coarse_step)
        skip_back_coarse_button = vp.button(canvas=scene,
                                            bind=skip_back_coarse,
                                            text='   <<<<<   ',
                                            color=vp.vec(1, 1, 1),
                                            background=vp.vec(0.333,
                                                              0.465,
                                                              0.561))
        widgets.append(skip_back_coarse_button)

        def skip_back_fine(b):
            global skip_flag, skip_backwards_flag, skip_steps
            skip_flag = True
            skip_backwards_flag = True
            skip_steps = int(fine_step)
        skip_back_fine_button = vp.button(canvas=scene,
                                          bind=skip_back_fine,
                                          text='    <<<    ',
                                          color=vp.vec(1, 1, 1),
                                          background=vp.vec(0.333,
                                                            0.465,
                                                            0.561))
        widgets.append(skip_back_fine_button)

        def pause_resume(b):
            global pause_resume_flag
            pause_resume_flag = not pause_resume_flag
            if pause_resume_flag is True:
                pause_resume_button.background = vp.vec(0.35, 0.35, 0.35)
                pause_resume_button.color = vp.vec(1, 1, 1)
            else:
                pause_resume_button.color = vp.vec(0, 0, 0)
                pause_resume_button.background = vp.vec(1, 1, 1)
        pause_resume_button = vp.button(canvas=scene,
                                        bind=pause_resume,
                                        text='  >||  ',
                                        color=vp.vec(0, 0, 0),
                                        background=vp.vec(1, 1, 1))
        widgets.append(pause_resume_button)

        def skip_ahead_fine(b):
            global skip_flag, skip_ahead_flag, skip_steps
            skip_flag = True
            skip_ahead_flag = True
            skip_steps = int(fine_step)
        skip_ahead_fine_button = vp.button(canvas=scene,
                                           bind=skip_ahead_fine,
                                           text='    >>>    ',
                                           color=vp.vec(1, 1, 1),
                                           background=vp.vec(0.273,
                                                             0.588,
                                                             0))
        widgets.append(skip_ahead_fine_button)

        def skip_ahead_coarse(b):
            global skip_flag, skip_ahead_flag, skip_steps
            skip_flag = True
            skip_ahead_flag = True
            skip_steps = int(coarse_step)
        skip_ahead_coarse_button = vp.button(canvas=scene,
                                             bind=skip_ahead_coarse,
                                             text='   >>>>>   ',
                                             color=vp.vec(1,1,1),
                                             background=vp.vec(0.273,0.588,0))
        widgets.append(skip_ahead_coarse_button)

        def skip_ahead_super_coarse(b):
            global skip_flag, skip_ahead_flag, skip_steps
            skip_flag = True
            skip_ahead_flag = True
            skip_steps = int(super_coarse_step)
        skip_ahead_coarse_button = vp.button(canvas=scene,
                                             bind=skip_ahead_super_coarse,
                                             text=' >>>>>>>>> ',
                                             color=vp.vec(1,1,1),
                                             background=vp.vec(0.273,0.588,0))
        widgets.append(skip_ahead_coarse_button)
        widgets_text.append(vp.wtext(text="                       " +
                                     "                     "))

        def exit_3d(b):
            global break_flag_button
            break_flag_button = True
            print("3D Forced Stop \n")
        exit_button = vp.button(canvas=scene,
                                bind=exit_3d,
                                text='   Finish    ',
                                color=vp.vec(1,1,1),
                                background=vp.vec(1,0,0))
        widgets.append(exit_button)

        widgets_text.append(vp.wtext(text="\n\n"))

        def slider_slow_mo_3d(s):
            global slow_mo
            if break_flag_button is False:
                slow_mo = s.value
                slider_slow_mo_caption.text = ' Slow Mo = '+'{:1.2f}'.format(slider_slow_mo.value)
        slider_slow_mo = vp.slider(canvas=scene,
                                   bind=slider_slow_mo_3d,
                                   text='Slow Motion',
                                   min=1, max=10,
                                   value=slow_mo,
                                   left=440,
                                   right=12)
        slider_slow_mo_caption = vp.wtext(text=' Slow Mo = '+'{:1.2f}'.format(slider_slow_mo.value))
        widgets.append(slider_slow_mo)
        widgets_text.append(slider_slow_mo_caption)
        widgets_text.append(vp.wtext(text="                     "
                                     + "                          "))

        def hide_forces_3d(b):
            global hide_forces
            hide_forces = not hide_forces
            if hide_forces is True:
                hide_forces_button.color = vp.vec(0,0,0)
                hide_forces_button.background = vp.vec(1,1,1)
                hide_forces_button.text = "Show Forces "
            else:
                hide_forces_button.background = vp.vec(0.35,0.35,0.35)
                hide_forces_button.color = vp.vec(1,1,1)
                hide_forces_button.text = " Hide Forces "
        hide_forces_button = vp.button(canvas=scene,
                                       bind=hide_forces_3d,
                                       text='Hide/Show Forces',
                                       color=vp.color.white,
                                       background=vp.vector(0,0.557,0.7))
        hide_forces_3d(hide_forces_button)
        hide_forces_3d(hide_forces_button)
        widgets.append(hide_forces_button)
        widgets_text.append(vp.wtext(text="\n\n"))

        def slider_fov_3d(s):
            global fov
            if break_flag_button is False:
                fov = s.value
                slider_fov_text.text = ' Fov = '+'{:1.2f}'.format(slider_fov.value)+"    "
                run_camera_3d(i,j)
        slider_fov = vp.slider(canvas=scene,
                               bind=slider_fov_3d,
                               text='Fov',
                               min=0.001,
                               max=4*fov,
                               value=fov,
                               left=440,
                               right=12)
        slider_fov_text = vp.wtext(text=' Fov = '+'{:1.2f}'.format(slider_fov.value)+"    ")
        widgets.append(slider_fov)
        widgets_text.append(slider_fov_text)

        def change_camera(m):
            global camera_type
            camera_type = camera_options[m.index]
            run_camera_3d(i, j)

        camera_options = ["Follow", "Fixed", "Follow Far", "Drone"]
        menu_camera = vp.menu(bind=change_camera, choices=camera_options,
                              selected=camera_type)
        widgets.append(menu_camera)
        widgets_text.append(vp.wtext(text="                          "))

        def hide_cg_3d(b):
            global hide_cg
            hide_cg = not hide_cg
            if hide_cg is False:
                hide_cg_button.background = vp.vec(0.35,0.35,0.35)
                hide_cg_button.color = vp.vec(1,1,1)
                hide_cg_button.text = " Hide CG & AoA"
            else:
                hide_cg_button.color = vp.vec(0,0,0)
                hide_cg_button.background = vp.vec(1,1,1)
                hide_cg_button.text = "Show CG & AoA"

        hide_cg_button = vp.button(canvas=scene,
                                   bind=hide_cg_3d,
                                   text='Hide/Show CG',
                                   color=vp.color.white,
                                   background=vp.vector(0,0.557,0.7))
        hide_cg_3d(hide_cg_button)
        hide_cg_3d(hide_cg_button)
        widgets.append(hide_cg_button)
        widgets_text.append(vp.wtext(text="\n\n"))


        """Labels #########################################################"""
        labels = vp.canvas(width=1280, height=200,
                           center=vp.vector(0,0,0),
                           background=vp.color.white)

        Setpoint_label = vp.label(canvas=labels, pos=vp.vector(-60, 7, 0),
                                  text="Setpoint = %.2f"+str(""),
                                  xoffset=0, zoffset=0, yoffset=0,
                                  space=30, height=30, border=0,
                                  font='sans', box=False, line=False,
                                  align="left")
        theta_label = vp.label(canvas=labels, pos=vp.vector(-60, 4, 0),
                               text="Setpoint = %.2f"+str(""),
                               xoffset=0, zoffset=0, yoffset=0,
                               space=30, height=30, border=0,
                               font='sans', box=False, line=False,
                               align="left")
        servo_label = vp.label(canvas=labels, pos=vp.vector(-60, 1, 0),
                               text="Setpoint = %.2f"+str(""),
                               xoffset=0, zoffset=0, yoffset=0,
                               space=30, height=30, border=0,
                               font='sans', box=False, line=False,
                               align="left")
        V = vp.label(canvas=labels, pos=vp.vector(-60, -2, 0),
                     text="Velocity"+str(""),
                     xoffset=0, zoffset=0, yoffset=0,
                     space=30, height=30, border=0,
                     font='sans', box=False, line=False,
                     align="left")
        aoa_plot = vp.label(canvas=labels, pos=vp.vector(-60, -5, 0),
                            text="Velocity"+str(""),
                            xoffset=0, zoffset=0, yoffset=0,
                            space=30, height=30, border=0,
                            font='sans', box=False, line=False,
                            align="left")
        Position_label = vp.label(canvas=labels, pos=vp.vector(-60, -8, 0),
                                  text="Velocity"+str(""),
                                  xoffset=0, zoffset=0, yoffset=0,
                                  space=30, height=30, border=0,
                                  font='sans', box=False, line=False,
                                  align="left")
        Time_label = vp.label(canvas=labels, pos=vp.vector(40, 7, 0),
                              text="Time"+str(""),
                              xoffset=0, zoffset=0, yoffset=0,
                              space=30, height=30, border=0,
                              font='sans', box=False, line=False,
                              align="left")

        def slider_time_3d(s):
            if break_flag_button is False:
                s.value = t_3d[i]
        slider_time = vp.slider(canvas=labels, bind=slider_time_3d,
                                text='t', min=t_3d[0], max=t_3d[-1], value=0,
                                pos=labels.title_anchor, length=1280)
        widgets.append(slider_time)


        """Rotations and displacements #####################################"""
        def run_3d_graphics(i, j):
            global vect_cg
            # How much to move each time-step , X and Z are the rocket's axes, not the world's
            delta_pos_X = position_3d[j][0] - position_3d[i][0]
            delta_pos_Z = position_3d[j][1] - position_3d[i][1]
            delta_theta = theta_3d[j] - theta_3d[i]
            delta_servo = servo_3d[j] - servo_3d[i]
            delta_aoa = aoa_3d[j] - aoa_3d[i]
            delta_xa = xa_3d[j] - xa_3d[i]
            delta_xa_radius = ((rocket.diam_interp(xa_3d[j])
                                - rocket.diam_interp(xa_3d[i])) / 2
                               + (fin_outline_interp(xa_3d[j])
                                  - fin_outline_interp(xa_3d[i])))

            # Moves the rocket
            rocket_3d.pos.y += delta_pos_X
            rocket_3d.pos.x += delta_pos_Z

            # Creates a cg and cp vector with reference to the origin of the
            # 3d rocket (its centroid)
            # Delta xa_radius, this is then integrated when you move
            # the Aerodynamic Force arrow
            xcg_radius = loc2glob((L_total/2-xcg_3d[j]), 0, theta_3d[i])
            xa_radius_pos = loc2glob(delta_xa, -delta_xa_radius, theta_3d[i])
            xa_radius_neg = loc2glob(delta_xa, delta_xa_radius, theta_3d[i])

            # CP and CG global vectors
            if i == 0 and position_3d[j][0] < 1:
                vect_cg = rocket_3d.pos - vp.vector(0, L_total/2, 0)
                launchrod_3d.rotate(theta_3d[1], axis=vp.vector(0,0,1), origin=vect_cg)
            else:
                vect_cg = vp.vector(rocket_3d.pos.x + xcg_radius[1],
                                    rocket_3d.pos.y + xcg_radius[0],
                                    dim_z_floor / 2)

            # Put the ball in the cg
            cg_ball.visible = not hide_cg
            velocity_arrow.visible = not hide_cg
            cg_ball.pos = vect_cg
            velocity_arrow.pos = vect_cg

            if variable_length_velocity_arrow is True:
                velocity = np.sqrt(v_loc_3d[i][0]**2 + v_loc_3d[i][1]**2)
                vel_arrow_length = np.interp(velocity,
                                             vel_arrow_length_interp[0],
                                             vel_arrow_length_interp[1])
                velocity_arrow.axis = vp.vector(0, vel_arrow_length, 0)
                velocity_arrow.rotate(theta_3d[i],
                                      axis=vp.vector(0,0,1),
                                      origin=velocity_arrow.pos)
                velocity_arrow.rotate(-aoa_3d[i],
                                      axis=vp.vector(0,0,1),
                                      origin=velocity_arrow.pos)
            velocity_arrow.rotate(delta_theta, axis=vp.vector(0,0,1),
                                  origin=vect_cg)
            velocity_arrow.rotate(-delta_aoa, axis=vp.vector(0,0,1),
                                  origin=vect_cg)

            if variable_radius_cg is True:
                cg_ball.radius = np.interp(t_3d[i],
                                           rad_cg_ball_time[0],
                                           rad_cg_ball_time[1])
            cg_ball.rotate(delta_theta, axis=vp.vector(0,0,1))

            # Rotate rocket from the CG
            rocket_3d.rotate(delta_theta, axis=vp.vector(0,0,1),
                             origin=vect_cg)

            # Move the motor together with the rocket
            motor.pos.y += delta_pos_X
            motor.pos.x += delta_pos_Z
            motor.rotate(delta_theta, axis=vp.vector(0,0,1),
                         origin=vect_cg)  # Rigid rotation with the rocket

            if rocket.use_fins_control is False:
                # TVC mount rotation
                motor.rotate(delta_servo, axis=vp.vector(0,0,-1))

            if rocket.use_fins_control is True:
                control_fins.pos.y += delta_pos_X
                control_fins.pos.x += delta_pos_Z
                control_fins.rotate(delta_theta,
                                    axis=vp.vector(0,0,1),
                                    origin=vect_cg)
                control_fins.rotate(delta_servo,
                                    axis=vp.vector(0,0,-1))

            # Arrows are hit or miss, tried this to avoid them going in the
            # wrong direction, didn't work
            if rocket.use_fins_control is False:
                aux = np.sin(servo_3d[i] + motor_offset)*thrust_3d[i] * force_scale
            else:
                aux = np.sin(motor_offset) * thrust_3d[i] * force_scale
            axis_motor_arrow = vp.vector(aux, 0, 0)
            # Makes visible one arrow or the other, be it left or right
            if aux > 0:
                Tmotor_pos.visible = False
                Tmotor_neg.visible = True
            else:
                Tmotor_pos.visible = True
                Tmotor_neg.visible = False
            # Displacements and rotations of the arrows
            Tmotor_pos.axis = axis_motor_arrow
            Tmotor_pos.pos.y += delta_pos_X
            Tmotor_pos.pos.x += delta_pos_Z
            Tmotor_pos.rotate(theta_3d[i],
                              axis=vp.vector(0,0,1),
                              origin=Tmotor_pos.pos)
            Tmotor_pos.rotate(delta_theta,
                              axis=vp.vector(0,0,1),
                              origin=vect_cg)
            Tmotor_neg.axis = axis_motor_arrow
            Tmotor_neg.pos.y += delta_pos_X
            Tmotor_neg.pos.x += delta_pos_Z
            Tmotor_neg.rotate(theta_3d[i],
                              axis=vp.vector(0,0,1),
                              origin=Tmotor_neg.pos)
            Tmotor_neg.rotate(delta_theta,
                              axis=vp.vector(0,0,1),
                              origin=vect_cg)

            # Motor Burnout, stops the red trail of the rocket
            if t_3d[i] > burnout_time + t_launch or t_3d[i] < t_launch:
                motor.visible = False
                motor.make_trail = False
            else:
                motor.visible = True
                motor.make_trail = True

            # Normal force arrow
            # Same as before, makes the one active visible
            if cn_3d[i] <= 0:
                Nforce_pos.visible = False
                Nforce_neg.visible = True
            else:
                Nforce_pos.visible = True
                Nforce_neg.visible = False
            # Displacements and rotations
            Nforce_axis = vp.vector(cn_3d[i]*force_scale, 0, 0)
            Nforce_pos.pos.y += delta_pos_X - xa_radius_pos[0]
            Nforce_pos.pos.x += delta_pos_Z - xa_radius_pos[1]
            Nforce_pos.axis = Nforce_axis
            Nforce_pos.rotate(theta_3d[i],
                              axis=vp.vector(0,0,1),
                              origin=Nforce_pos.pos)
            Nforce_pos.rotate(delta_theta,
                              axis=vp.vector(0,0,1),
                              origin=vect_cg)
            Nforce_neg.pos.y += delta_pos_X - xa_radius_neg[0]
            Nforce_neg.pos.x += delta_pos_Z - xa_radius_neg[1]
            Nforce_neg.axis = Nforce_axis
            Nforce_neg.rotate(theta_3d[i],
                              axis=vp.vector(0,0,1),
                              origin=Nforce_neg.pos)
            Nforce_neg.rotate(delta_theta,
                              axis=vp.vector(0,0,1),
                              origin=vect_cg)

            if rocket.use_fins_control is True:
                if fin_force_3d[i] >= 0:
                    T_fin_pos.visible = False
                    T_fin_neg.visible = True
                else:
                    T_fin_pos.visible = True
                    T_fin_neg.visible = False
                # Displacements and rotations
                T_fin_pos.pos.y += delta_pos_X
                T_fin_pos.pos.x += delta_pos_Z
                T_fin_pos.axis=vp.vector(fin_force_3d[i]*force_scale, 0, 0)
                T_fin_pos.rotate(theta_3d[i],
                                 axis=vp.vector(0,0,1),
                                 origin=T_fin_pos.pos)
                T_fin_pos.rotate(delta_theta,
                                 axis=vp.vector(0,0,1),
                                 origin=vect_cg)
                T_fin_neg.pos.y += delta_pos_X
                T_fin_neg.pos.x += delta_pos_Z
                T_fin_neg.axis=vp.vector(fin_force_3d[i]*force_scale, 0, 0)
                T_fin_neg.rotate(theta_3d[i],
                                 axis=vp.vector(0,0,1),
                                 origin=T_fin_neg.pos)
                T_fin_neg.rotate(delta_theta,
                                 axis=vp.vector(0,0,1),
                                 origin=vect_cg)
            if hide_forces is True:
                Nforce_pos.visible = False
                Nforce_neg.visible = False
                Tmotor_pos.visible = False
                Tmotor_neg.visible = False
                if rocket.use_fins_control is True:
                    T_fin_pos.visible = False
                    T_fin_neg.visible = False


        def run_camera_3d(i,j):
            #Camera
            global vect_cg
            if camera_shake_toggle is True:
                camera_shake=loc2glob(v_glob_3d[i][0],
                                      v_glob_3d[i][1],
                                      theta_3d[i])
            else:
                camera_shake = [0,0]

            # Follows almost 45 deg up
            if camera_type == "Follow":
                scene.fov = fov*DEG2RAD
                scene.camera.pos = vp.vector(rocket_3d.pos.x-L_total*50+camera_shake[1]/50,
                                             rocket_3d.pos.y+L_total*70-camera_shake[0]/500,
                                             rocket_3d.pos.z-L_total*33)
                scene.camera.axis = vp.vector(rocket_3d.pos.x-scene.camera.pos.x,
                                              rocket_3d.pos.y-scene.camera.pos.y,
                                              rocket_3d.pos.z-scene.camera.pos.z)

            # Simulates someone in the ground
            elif camera_type == "Fixed":
                if variable_fov is True:
                    scene.fov = fov*DEG2RAD/(0.035/10 * np.sqrt(position_3d[i][0]**2
                                                                + position_3d[i][1]**2)
                                             + 1)
                else:
                    scene.fov = fov*DEG2RAD
                scene.camera.pos = vp.vector(dim_x_floor/2, 1, dim_z_floor/2-70*2)
                scene.camera.axis = vp.vector(rocket_3d.pos.x-scene.camera.pos.x,
                                              rocket_3d.pos.y-scene.camera.pos.y,
                                              rocket_3d.pos.z-scene.camera.pos.z)

            # Lateral camera, like if it was 2D
            elif camera_type == "Follow Far":
                scene.fov = fov*DEG2RAD
                scene.camera.pos = vp.vector(rocket_3d.pos.x+camera_shake[1]/50,
                                             rocket_3d.pos.y+0.0-camera_shake[0]/200,
                                             rocket_3d.pos.z-70*2)
                scene.camera.axis = vp.vector(rocket_3d.pos.x-scene.camera.pos.x,
                                              rocket_3d.pos.y-scene.camera.pos.y,
                                              rocket_3d.pos.z-scene.camera.pos.z)
            elif camera_type == "Drone":
                scene.fov = fov*DEG2RAD
                scene.camera.pos = vp.vector(dim_x_floor/2-70*2,
                                             max_hight*0.75,
                                             dim_z_floor/2-70*2)
                scene.camera.axis = vp.vector(rocket_3d.pos.x-scene.camera.pos.x,
                                              rocket_3d.pos.y-scene.camera.pos.y,
                                              rocket_3d.pos.z-scene.camera.pos.z)


            # Labels
            Setpoint_label.text = ("Setpoint = %.0f" % round(setpoint_3d[i]*RAD2DEG, 1)
                                   + u'\xb0')
            theta_label.text = ("Pitch Angle = " +
                                str(round(theta_3d[i]*RAD2DEG, 2))
                                + u'\xb0')
            servo_label.text = ("Actuator deflection = " +
                                str(round((servo_3d[i])*RAD2DEG, 2))
                                + u'\xb0')
            V.text = ("Global Velocity => " + " Up = " +
                      str(round(v_glob_3d[i][0], 2))
                      + " m/s , Left = "
                      + str(round(v_glob_3d[i][1], 2))
                      + " m/s; Total => " + str(round(np.sqrt(v_glob_3d[i][1]**2 + v_glob_3d[i][0]**2), 2)))
            aoa_plot.text = "AoA = " + str(round(aoa_3d[i]*RAD2DEG, 2)) + u'\xb0'
            Position_label.text = ("Position => " + "Altitude = "
                                   + str(round(position_3d[i][0], 2))
                                   + "m , Distance Downrange = "
                                   + str(round(position_3d[i][1], 2))
                                   + "m")
            Time_label.text = "Time = %.3f" % round(t_3d[i], 3)


        """ Simualtion Control #############################################"""
        t_total = t_3d[-3] - t_3d[0]
        t_step = (t_3d[-1]-t_3d[0]) / len(t_3d)
        super_coarse_step = t_total/5 / t_step
        coarse_step = 1 / t_step
        fine_step = 0.015 / t_step
        i = 0
        j = 1
        list_length = len(theta_3d)-2
        while True:
            vp.rate(len(t_3d)/t_total/slow_mo)
            play_video = skip_flag is False and pause_resume_flag is False and i < list_length
            slider_time_3d(slider_time)
            if play_video is True:
                run_3d_graphics(i, j)
                run_camera_3d(i, j)
                i += 1
                j += 1
            if skip_flag is True:
                if skip_ahead_flag is True:
                    for t in range(skip_steps):
                        if i >= list_length-1:
                            break
                        run_3d_graphics(i, j)
                        run_camera_3d(i, j)
                        i += 1
                        j += 1
                    skip_flag = False
                    skip_ahead_flag = False
                if skip_backwards_flag is True:
                    for t in range(skip_steps):
                        if i <= 1:
                            break
                        i -= 1
                        j -= 1
                        run_3d_graphics(j, i)
                        run_camera_3d(j, i)
                        motor.clear_trail()
                    skip_flag = False
                    skip_backwards_flag = False
            if break_flag_button is True:
                break
        plt.draw()
        plt.show()

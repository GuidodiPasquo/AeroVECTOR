
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 23:48:04 2021

@author: guido
"""

import matplotlib.pyplot as plt
import rocket_functions
import random
import ISA_calculator as atm
import numpy as np

deg2rad=np.pi/180
rad2deg=1/deg2rad
atmosphere = atm.get_atmosphere()

Rocket = rocket_functions.rocket_class()


# l = [[0,0],[0.2,0.08],[0.23,0.066],[0.8+0.23,0.066],[0.8+0.23+0.075,0.05],[0.8+0.23+0.075+0.1,0.10]]
# l = [[0,0],[0.2,0.08],[0.23,0.066],[0.33,0.1],[0.405,0.05],[1.205,0.05]]
# l = [[0,0],[0.2,0.1],[0.23,0.066],[0.8+0.23,0.066]]
#l = [[0,0],[0.2,0.1],[1,0.1]]
# l = [[0,0],[0.2,0.05],[1,0.07]]
# l = [[0,0],[0.2,0.05],[1,0.07]]
#l = [[0,0.05],[0.2,0.1],[1,0.1],[1.2,0.05]]
l = [[0,0],[0.2,0.066],[1,0.066]]
#l = [[0,0],[0.2,0.066],[0.25,0.05],[1,0.066]]
#l = [[0,0],[0.2,0.1],[0.4,0.1],[0.5,0.05],[1.5,0.05]]
# l = [[0,0],[0.3,0.066],[0.4,0.04],[1.2,0.04]]

fins = [[0.8,0.033],[0.9,0.079],[0.95,0.079],[1,0.033]]
# fins = [[1.1,0.02],[1.15,0.08],[1.16,0.08],[1.2,0.02]]
# fins = [[0.0000,0.0000],[0.00001,0.00001],[0.00002,0.00001],[0.00002,0.0]]

# fins_control = [[0.0,0.02],[0.1,0.08],[0.1,0.08],[0.1,0.02]]
fins_control = [[0.2,0.033],[0.24,0.07],[0.25,0.07],[0.255,0.033]]

pep = [True, True, True, True, True]
l=pep + [l]+ [fins] + [fins_control]
print(l)
Rocket.Update_Rocket(l,0.5)
Q_damp = Rocket.get_Q_damp()
# print("b ", rocket_functions.fin.b)
# print(Rocket.Cp90)
# print(Rocket.station_cross_area)

list_plot = []
list_plot2 = []
list_plot3 = []
list_plot4 = []
x_plot = []
AoA = -3.14159
M = 0.0
T, P, rho, spd_sound, mu = atm.calculate_at_h(0, atmosphere)
Actuator_angle= -60*deg2rad
# for i in range(1500):
#     V0 = 10
#     cn, cp, ca = Rocket.Calculate_Aero_coef(AoA,V0, rho, mu, M, Actuator_angle)
#     list_plot.append(cp)
#     a = -Rocket.CN_Alpha_fin[1] * AoA
#     b = Rocket.CN_Alpha_fin[1] * Actuator_angle
#     list_plot2.append(a+b)
#     list_plot3.append(ca)
#     x_plot.append(AoA*57.295)
#     AoA += 2*3.14159/1000
#     # M+=0.8/10
#     # x_plot.append(M)


# # plt.plot(x_plot,list_plot)
# plt.plot(x_plot,list_plot2)
# # plt.plot(x_plot,list_plot3)
# plt.grid(True,linestyle='--')    
# plt.xlabel('Angle of Atack [º]',fontsize=16)
# plt.ylabel('x Cp [m]',fontsize=16)
# # plt.ylabel('Cn',fontsize=16)   
# # legend = plt.legend(loc='upper right', shadow=True, fontsize='x-small') 
# plt.title('Modified Extended Barrowman')
# plt.show()

Actuator_angle= 0*deg2rad
AoA = -360*deg2rad
for i in range(1000):
    V0 = 10
    cn, cp, ca = Rocket.Calculate_Aero_coef(AoA,V0, rho, mu, M, Actuator_angle)
    """
    slope as a function of Actuator angle for a determined AoA
    list_plot.append(cp)
    a = -Rocket.CN_Alpha_fin[1] * AoA
    b = Rocket.CN_Alpha_fin[1] * Actuator_angle
    list_plot2.append(Rocket.CN_Alpha_fin[1])
    list_plot3.append(ca)
    x_plot.append(Actuator_angle*57.295)
    Actuator_angle += 2*90*deg2rad/1000    
    """
    list_plot.append(rocket_functions.fin[1].flat_plate.CN_Alpha)
    list_plot2.append(rocket_functions.fin[1].flat_plate.CN)
    x_plot.append(AoA*rad2deg)
    AoA += 4*np.pi / 1000
    
    # M+=0.8/10
    # x_plot.append(M)


plt.plot(x_plot,list_plot)
plt.plot(x_plot,list_plot2)
# plt.plot(x_plot,list_plot3)
plt.grid(True,linestyle='--')    
plt.xlabel('Angle of Atack [º]',fontsize=16)
plt.ylabel('x Cp [m]',fontsize=16)
# plt.ylabel('Cn',fontsize=16)   
# legend = plt.legend(loc='upper right', shadow=True, fontsize='x-small') 
plt.title('Modified Extended Barrowman')
plt.show()


for i in range(1):
    T, P, rho, spd_sound, mu = atm.calculate_at_h(150, atmosphere)
    nu = mu/rho
    # print(T, P, rho, spd_sound, mu, nu)




# Rocket._Calculate_total_Ca(AoA = 3)
# print(Rocket.Calculate_Aero_coef(0,V0 = 10, M = 0.5))
# print(Rocket.CD0_friction)
# print(Rocket.pressure_CD)
# print(Rocket.base_drag_CD)

# print(Rocket.CA)

# print(Rocket.A_ref)

# R = 0.05/2
# L = 0.2
# rho_radius = (R**2 + L**2)/(2 * R)

# x = 0
# y = np.sqrt(rho_radius**2 - (L-x)**2)+R-rho_radius

# x_plot = []
# for i in range(100):    
#     list_plot4.append(np.sqrt(rho_radius**2 - (L-x)**2)+R-rho_radius)
#     x_plot.append(x)
#     x += L/100
    
# plt.plot(x_plot,list_plot4)
# plt.show()

import servo_lib

servo = servo_lib.servo_class()

T = 0.001
# TVC_weight_compensation = 1.45 for servo alone
# TVC_wight_compensation = 2.1 for TVC mount

servo.setup(2.1, 1,0.02, T)
# servo.test()

# u=60*np.pi/180
# x_plot = []
# list_plot4 = []
# t = 0
# for i in range(1000):
#     servo.update()
#     list_plot4.append(servo.simulate(u)/(np.pi/180))
#     x_plot.append(t)
#     t+=T
    
# plt.plot(x_plot,list_plot4)
# plt.grid(True,linestyle='--')  
# plt.show()


# flat_plate = rocket_functions.airfoil()
# x_plot = []
# list_plot = []
# list_plot2 = []
# list_plot3 = []
# list_plot4 = []
# t = 0
# x = -np.pi
# for i in range(1000):
#     CN, CA, CN_A = flat_plate.get_Aero_coef(x,1)
#     list_plot.append(CN)
#     list_plot2.append(CA)
#     list_plot3.append(CN_A)
#     # list_plot4.append(CD)
#     x_plot.append(x)
#     x+=2*np.pi/1000
    
# plt.plot(x_plot,list_plot)
# plt.plot(x_plot,list_plot2)
# plt.plot(x_plot,list_plot3)
# # plt.plot(x_plot,list_plot4)
# plt.grid(True,linestyle='--')  
# plt.show()

# AoA = 5*deg2rad
# Actuator_angle = 30*deg2rad
# tot_AoA = AoA + Actuator_angle
# # print(Rocket.Calculate_Aero_coef(AoA))
# # print(rocket_functions.fin.sb_angle)
# # print(Rocket.CN_Alpha_0)
# print( Rocket.Calculate_Aero_coef(AoA,10, rho, mu, M=0, Actuator_angle=Actuator_angle))
# print(Rocket.CN_Alpha_og)
# print("CN 1 = ",Rocket.CN_Alpha_og[1]*tot_AoA)
# print("CN 2 = ", Rocket.CN_Alpha_og[1]*AoA + Rocket.CN_Alpha_og[1]*Actuator_angle)
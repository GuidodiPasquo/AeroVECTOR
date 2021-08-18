# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 23:40:58 2021

@author: Guido di Pasquo
"""
import importlib
import sys
sys.path.append("..")
import os
os.chdir(os.path.dirname(sys.argv[0]))
import numpy as np
import matplotlib.pyplot as plt
from src.aerodynamics import rocket_functions
from src.aerodynamics import flight_conditions


DEG2RAD = np.pi/180
RAD2DEG = 1/DEG2RAD


def transform_AoA_2_v(aoa):
    if aoa <= -np.pi/2:
        v = [-1, -np.tan(aoa)]/np.sqrt(1+np.tan(aoa)**2)
    elif aoa <= 0:
        v = [1, np.tan(aoa)]/np.sqrt(1+np.tan(aoa)**2)
    elif aoa <= np.pi/2:
        v = [1, np.tan(aoa)]/np.sqrt(1+np.tan(aoa)**2)
    elif aoa <= np.pi:
        v = [-1, -np.tan(aoa)]/np.sqrt(1+np.tan(aoa)**2)
    else:
        v = [-1, -np.tan(aoa)]/np.sqrt(1+np.tan(aoa)**2)
    return v




"""Raw wind tunnel data #==================================================#"""

wind_tunnel_data = importlib.import_module("Wind Tunnel Data.wind_tunnel_data_complete")
wtd_planform = [0]*4
planform_list = ["Rectangular", "Zimmerman", "Inverse Zimmerman", "Elliptical"]
for i in range(4):
    wtd_planform[i] = wind_tunnel_data.FinWindTunnelData(planform_list[i])

AR_list = ["0.5", "0.75", "1", "1.25"]
legend_list = [""]*7
for i in range(4):
    legend_list[i] = "AR=" + AR_list[i]
marker = ["v", "s", "v", "^"]

for j in range(4):
    plt.figure()
    plt.grid(True)
    plt.axis([-10, 50, -0.5, 1.75])
    plt.xlabel('α (degrees)', fontsize=20)
    plt.ylabel(r'${C_{L}}$      ', fontsize=20, rotation=0)
    plt.title(planform_list[j], fontsize=20)

    for i in range(4):
        plt.plot(wtd_planform[j].cl_vs_alpha[AR_list[i]][0],
                 wtd_planform[j].cl_vs_alpha[AR_list[i]][1],
                 marker[i],
                 markerfacecolor='none')
        plt.legend(legend_list, loc=4, fontsize=13)




"""CL CN and CD CA Examples #==============================================#"""

rocket_points = [[0, 0], [0.2, 0.066], [1.2, 0.066], [1.5, 0.066]]
fins = [[0, 0.1], [0, 0.1], 0.25/10, 0.001]  # AR = 0.5
fins_control = [[0, 0.1], [0, 0.1], 0.5, 0.001]
flags = [True, True, True, True, True]
random_mass_parameters = [0.7, 2, 1, 1, 2, 1]
to_rocket_list = flags + [rocket_points] + [fins] + [fins_control]
rocket = rocket_functions.Rocket()
rocket.update_rocket(to_rocket_list, random_mass_parameters)
ac = flight_conditions.FinFlightCondition()
ac.mach = 0.01
ac.Re = 1e5

AR_list = [0.5, 0.75, 1, 1.25]

list_plot = []
x_plot = []
AoA = 0


label_font_size = 17
legend_font_size = 15
tick_font_size = 14

n = 200
for i in range(n):
    v = transform_AoA_2_v(AoA)
    cn, cm_xcg, ca, xa = rocket.calculate_aero_coef(v, 0, 0, 0)
    x_plot.append(AoA * RAD2DEG)
    ac.aoa = AoA
    acoeff = rocket_functions.fin[0].aero_properties.get_aero_coeff(ac,
                                                                    100,
                                                                    100,
                                                                    use_rocket_re=True)
    list_plot.append(acoeff.cl)
    AoA += 90*DEG2RAD / n
plt.figure()
plt.plot(x_plot, list_plot)
plt.axis([0, 90, 0, 1.5])
plt.grid(True)
plt.xlabel('α (degrees)', fontsize=label_font_size)
plt.ylabel(r'${C_{L}}$      ', fontsize=label_font_size, rotation=0)
plt.legend(["AR = 0.5"], fontsize=legend_font_size)
plt.xticks(fontsize=tick_font_size)
plt.yticks(fontsize=tick_font_size)
plt.show()

list_plot = []
x_plot = []
AoA = 0
for i in range(n):
    v = transform_AoA_2_v(AoA)
    cn, cm_xcg, ca, xa = rocket.calculate_aero_coef(v, 0, 0, 0)
    x_plot.append(AoA * RAD2DEG)
    ac.aoa = AoA
    acoeff = rocket_functions.fin[0].aero_properties.get_aero_coeff(ac,
                                                                    100,
                                                                    100,
                                                                    use_rocket_re=True)
    list_plot.append(acoeff.cd)
    AoA += 90*DEG2RAD / n

plt.figure()
plt.plot(x_plot, list_plot)
plt.axis([0, 90, 0, 1.75])
plt.grid(True)
plt.xlabel('α (degrees)', fontsize=label_font_size)
plt.ylabel(r'${C_{D}}$      ', fontsize=label_font_size, rotation=0)
plt.legend(["AR = 0.5"], fontsize=legend_font_size)
plt.xticks(fontsize=tick_font_size)
plt.yticks(fontsize=tick_font_size)
plt.show()

list_plot = [[], []]
x_plot = [[], []]
label_font_size = 17
legend_font_size = 15
tick_font_size = 14
AoA = 0
n = 200
for i in range(n):
    v = transform_AoA_2_v(AoA)
    cn, cm_xcg, ca, xa = rocket.calculate_aero_coef(v, 0, 0, 0)
    x_plot[0].append(AoA * RAD2DEG)
    x_plot[1].append(AoA * RAD2DEG)
    ac.aoa = AoA
    acoeff = rocket_functions.fin[0].aero_properties.get_aero_coeff(ac,
                                                                    100,
                                                                    100,
                                                                    use_rocket_re=True)
    list_plot[0].append(acoeff.cn)
    list_plot[1].append(acoeff.ca)
    AoA += 180 * DEG2RAD / (n-1)
plt.figure()
for i in range(2):
    plt.plot(x_plot[i], list_plot[i])
plt.axis([0, 180, -0.10, 2])
plt.grid(True)
plt.xlabel('α (degrees)', fontsize=label_font_size)
# plt.ylabel(r'${C_{N}}$      ', fontsize=label_font_size, rotation=0)
plt.legend([r'${C_{N}}$', r'${C_{A}}$'], fontsize=legend_font_size)
plt.xticks(fontsize=tick_font_size)
plt.yticks(fontsize=tick_font_size)
plt.show()


"""Comparison with experimental Data #====================================#"""
wind_tunnel_data = importlib.import_module("Wind Tunnel Data.wind_tunnel_data_complete")
wtd_planform = [0]*4
planform_list = ["Rectangular", "Zimmerman", "Inverse Zimmerman", "Elliptical"]
for i in range(4):
    wtd_planform[i] = wind_tunnel_data.FinWindTunnelData(planform_list[i])

rocket_points = [[0, 0], [0.2, 0.066], [1.2, 0.066], [1.5, 0.066]]
fins = [[0, 0.1], [0, 0.1], 0.25/10, 0.001]  # AR = 0.5
fins_control = [[0, 0.1], [0, 0.1], 0.5, 0.001]
flags = [True, True, True, True, True]
random_mass_parameters = [0.7, 2, 1, 1, 2, 1]
to_rocket_list = flags + [rocket_points] + [fins] + [fins_control]
rocket = rocket_functions.Rocket()
rocket.update_rocket(to_rocket_list, random_mass_parameters, roughness=[10e-6]*3)  # Aluminum roughness
ac = flight_conditions.FinFlightCondition()
ac.mach = 0.01
ac.Re = 1e5

AR_list = [0.5, 0.75, 1]
AR_list_str = ["0.5", "0.75", "1"]
legend_list = [""]*7
for i in range(3):
    legend_list[i] = "AR=" + AR_list_str[i]
for i in range(3):
    legend_list[i+3] = "AR=" + AR_list_str[i]
marker = ["v", "s", "v"]
colors = ["C0", "C1", "C2"]

label_font_size = 17
legend_font_size = 15
tick_font_size = 14

list_plot = [[], [], [], []]
x_plot = [[], [], [], []]
AoA = 0

n = 200
for j in range(len(AR_list)):
    rocket_points = [[0, 0], [0.2, 0.066], [1.2, 0.066], [1.5, 0.066]]
    fins = [[0, 0.1], [0, 0.1], (AR_list[j])/10/2, 0.001]
    fins_control = [[0, 0.1], [0, 0.1], 0.5, 0.001]
    pep = [True, True, True, True, True]
    to_rocket_list = pep + [rocket_points] + [fins] + [fins_control]
    rocket.update_rocket(to_rocket_list, random_mass_parameters, roughness=[10e-6]*3)
    AoA = 0
    for i in range(n):
        v = transform_AoA_2_v(AoA)
        x_plot[j].append(AoA * RAD2DEG)
        ac.aoa = AoA
        acoeff = rocket_functions.fin[0].aero_properties.get_aero_coeff(ac,
                                                                        100,
                                                                        100,
                                                                        use_rocket_re=True)
        list_plot[j].append(acoeff.cl)
        AoA += 90*DEG2RAD / n
plt.figure()
for i in range(3):
    plt.plot(x_plot[i], list_plot[i], colors[i])
# plt.legend(legend_list + legend_list, loc=4, fontsize=legend_font_size)
plt.axis([0, 90, 0, 1.5])
plt.grid(True)
plt.xlabel('α (degrees)', fontsize=label_font_size)
plt.ylabel(r'${C_{L}}$      ', fontsize=label_font_size, rotation=0)
plt.xticks(fontsize=tick_font_size)
plt.yticks(fontsize=tick_font_size)
for i in range(3):
    plt.plot(wtd_planform[0].cl_vs_alpha[AR_list_str[i]][0],
             wtd_planform[0].cl_vs_alpha[AR_list_str[i]][1],
             color=colors[i],
             marker=marker[i],
             markerfacecolor='none',
             linestyle="None")
plt.legend(legend_list + legend_list, loc=4, fontsize=legend_font_size)
plt.show()

# =======================================================================#
wtd_planform = [0]*4
planform_list = ["Rectangular CD", "Zimmerman", "Inverse Zimmerman CD", "Elliptical CD"]
for i in range(4):
    wtd_planform[i] = wind_tunnel_data.FinWindTunnelData(planform_list[i])

list_plot = [[], [], [], []]
x_plot = [[], [], [], []]
AoA = 0

n = 200
for j in range(len(AR_list)):
    rocket_points = [[0, 0], [0.2, 0.066], [1.2, 0.066], [1.5, 0.066]]
    fins = [[0, 0.1], [0, 0.1], (AR_list[j])/10/2, 0.001]
    fins_control = [[0, 0.1], [0, 0.1], 0.5, 0.001]
    pep = [True, True, True, True, True]
    to_rocket_list = pep + [rocket_points] + [fins] + [fins_control]
    rocket.update_rocket(to_rocket_list, random_mass_parameters, roughness=[10e-6]*3)
    AoA = 0
    for i in range(n):
        v = transform_AoA_2_v(AoA)
        x_plot[j].append(AoA * RAD2DEG)
        ac.aoa = AoA
        acoeff = rocket_functions.fin[0].aero_properties.get_aero_coeff(ac,
                                                                        100,
                                                                        100,
                                                                        use_rocket_re=False)
        list_plot[j].append(acoeff.cd)
        AoA += 90*DEG2RAD / n
plt.figure()
for i in range(len(AR_list)):
    plt.plot(x_plot[i], list_plot[i], colors[i])
# plt.legend(legend_list + legend_list, loc=4, fontsize=legend_font_size)
plt.axis([0, 90, 0, 1.75])
plt.grid(True)
plt.xlabel('α (degrees)', fontsize=label_font_size)
plt.ylabel(r'${C_{D}}$      ', fontsize=label_font_size, rotation=0)
plt.xticks(fontsize=tick_font_size)
plt.yticks(fontsize=tick_font_size)
for i in range(len(AR_list)):
    plt.plot(wtd_planform[0].cl_vs_alpha[AR_list_str[i]][0],
             wtd_planform[0].cl_vs_alpha[AR_list_str[i]][1],
             color=colors[i],
             marker=marker[i],
             markerfacecolor='none',
             linestyle="None")
plt.legend(legend_list + legend_list, loc=4, fontsize=legend_font_size)
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 23:40:58 2021

@author: Guido di Pasquo
"""
import importlib
import sys
sys.path.append("..")
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




"""Comparison with experimental Data #====================================#"""

wind_tunnel_data = importlib.import_module("Wind Tunnel Data.wind_tunnel_data_complete")
wtd_planform = [0]*4
planform_list = ["Rectangular", "Zimmerman", "Inverse Zimmerman", "Elliptical"]
for i in range(4):
    wtd_planform[i] = wind_tunnel_data.FinWindTunnelData(planform_list[i])

rocket_points = [[0, 0], [0.2, 0.066], [1.2, 0.066], [1.5, 0.066]]
fins = [[0, 0.1], [0, 0.1], 1/10, 0.00196]  # AR = 2
fins_control = [[0, 0.1], [0, 0.1], 0.5, 0.001]
flags = [True, True, True, True, True]
random_mass_parameters = [0.7, 2, 1, 1, 2, 1]
to_rocket_list = flags + [rocket_points] + [fins] + [fins_control]
rocket = rocket_functions.Rocket()
rocket.update_rocket(to_rocket_list, random_mass_parameters, roughness=[10e-6]*3)
ac = flight_conditions.FinFlightCondition()
ac.mach = 0.01
ac.Re = 1e5

cte_to_fit_AR = [1/10/2,  # Rectangular
                 1/10*0.3125,  # Zimmerman
                 1/10*0.3125,  # Inverse Zimmerman
                 1/10*0.375  # Elliptical
                 ]
tip_sweep_chord = [
    [0, 0.1],
    [0.01, 0.025],
    [0.065, 0.025],
    [0.025, 0.05]
    ]


AR_list = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
AR_list_str = ["0.5", "0.75", "1", "1.25", "1.5", "1.75", "2"]

legend_list = [""]*(7*2)
for i in range(len(AR_list)):
    legend_list[i] = "AR=" + AR_list_str[i]
for i in range(len(AR_list)):
    legend_list[i+7] = "AR=" + AR_list_str[i]

marker = ["v", "s", "v", "^", "o", "D", "h"]
colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

label_font_size = 17
legend_font_size = 15
tick_font_size = 14


for k in range(4):
    list_plot = [[], [], [], [], [], [], []]
    x_plot = [[], [], [], [], [], [], []]
    AoA = 0
    n = 200
    for j in range(len(AR_list)):
        fins = [[0, 0.1], tip_sweep_chord[k], AR_list[j] * cte_to_fit_AR[k], 0.001]
        pep = [True, True, True, True, True]
        to_rocket_list = pep + [rocket_points] + [fins] + [fins_control]
        rocket.update_rocket(to_rocket_list, [0.7, 2, 1, 1, 2, 1], roughness=[10e-6]*3)
        AoA = 0
        for i in range(n):
            v = transform_AoA_2_v(AoA)
            x_plot[j].append(AoA*RAD2DEG)
            ac.aoa = AoA
            acoeff = rocket_functions.fin[0].aero_properties.get_aero_coeff(ac,
                                                                            100,
                                                                            100,
                                                                            use_rocket_re=True)
            list_plot[j].append(acoeff.cl)
            AoA += 90*DEG2RAD / n
        print(planform_list[k], AR_list[j], fins)

    plt.figure()
    for i in range(len(AR_list)):
        plt.plot(x_plot[i], list_plot[i], colors[i])

    plt.axis([0, 90, 0, 1.5])
    plt.grid(True)
    plt.xlabel('α (degrees)', fontsize=label_font_size)
    plt.ylabel(r'${C_{L}}$      ', fontsize=label_font_size, rotation=0)
    plt.xticks(fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)
    for i in range(len(AR_list)):
        plt.plot(wtd_planform[k].cl_vs_alpha[AR_list_str[i]][0],
                 wtd_planform[k].cl_vs_alpha[AR_list_str[i]][1],
                 color=colors[i],
                 marker=marker[i],
                 markerfacecolor='none',
                 linestyle="None")

    plt.legend(legend_list + legend_list, loc=4, fontsize=legend_font_size)
    plt.title(planform_list[k], fontsize=20)
    plt.show()

# CD ================================================================== #
wind_tunnel_data = importlib.import_module("Wind Tunnel Data.wind_tunnel_data_complete")
wtd_planform = [0]*4
planform_list = ["Rectangular CD", "Zimmerman CD", "Inverse Zimmerman CD", "Elliptical CD"]
for i in range(4):
    wtd_planform[i] = wind_tunnel_data.FinWindTunnelData(planform_list[i])

rocket_points = [[0, 0], [0.2, 0.066], [1.2, 0.066], [1.5, 0.066]]
fins = [[0, 0.1], [0, 0.1], 1/10, 0.00196]  # AR = 2
fins_control = [[0, 0.1], [0, 0.1], 0.5, 0.001]
flags = [True, True, True, True, True]
random_mass_parameters = [0.7, 2, 1, 1, 2, 1]
to_rocket_list = flags + [rocket_points] + [fins] + [fins_control]
rocket = rocket_functions.Rocket()
rocket.update_rocket(to_rocket_list, random_mass_parameters, roughness=[10e-6]*3)
ac = flight_conditions.FinFlightCondition()
ac.mach = 0.01
ac.Re = 1e5

cte_to_fit_AR = [1/10/2,  # Rectangular
                 1/10*0.3125,  # Zimmerman
                 1/10*0.3125,  # Inverse Zimmerman
                 1/10*0.375  # Elliptical
                 ]
tip_sweep_chord = [
    [0, 0.1],
    [0.01, 0.025],
    [0.065, 0.025],
    [0.025, 0.05]
    ]


AR_list = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
AR_list_str = ["0.5", "0.75", "1", "1.25", "1.5", "1.75", "2"]

legend_list = [""]*(7*2)
for i in range(len(AR_list)):
    legend_list[i] = "AR=" + AR_list_str[i]
for i in range(len(AR_list)):
    legend_list[i+7] = "AR=" + AR_list_str[i]

marker = ["v", "s", "v", "^", "o", "D", "h"]
colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

label_font_size = 17
legend_font_size = 15
tick_font_size = 14


for k in range(4):
    list_plot = [[], [], [], [], [], [], []]
    x_plot = [[], [], [], [], [], [], []]
    AoA = 0
    n = 200
    for j in range(len(AR_list)):
        fins = [[0, 0.1], tip_sweep_chord[k], AR_list[j] * cte_to_fit_AR[k], 0.00196]
        pep = [True, True, True, True, True]
        to_rocket_list = pep + [rocket_points] + [fins] + [fins_control]
        rocket.update_rocket(to_rocket_list, [0.7, 2, 1, 1, 2, 1], roughness=[10e-6]*3)
        AoA = 0
        for i in range(n):
            v = transform_AoA_2_v(AoA)
            x_plot[j].append(AoA*RAD2DEG)
            ac.aoa = AoA
            acoeff = rocket_functions.fin[0].aero_properties.get_aero_coeff(ac,
                                                                            100,
                                                                            100,
                                                                            use_rocket_re=True)
            list_plot[j].append(acoeff.cd)
            AoA += 90*DEG2RAD / n

    plt.figure()
    for i in range(len(AR_list)):
        plt.plot(x_plot[i], list_plot[i], colors[i])

    plt.axis([0, 90, 0, 1.75])
    plt.grid(True)
    plt.xlabel('α (degrees)', fontsize=label_font_size)
    plt.ylabel(r'${C_{D}}$      ', fontsize=label_font_size, rotation=0)
    plt.xticks(fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)
    for i in range(len(AR_list)):
        plt.plot(wtd_planform[k].cl_vs_alpha[AR_list_str[i]][0],
                 wtd_planform[k].cl_vs_alpha[AR_list_str[i]][1],
                 color=colors[i],
                 marker=marker[i],
                 markerfacecolor='none',
                 linestyle="None")

    plt.legend(legend_list + legend_list, loc=4, fontsize=legend_font_size)
    plt.title(planform_list[k][:-3], fontsize=20)
    plt.show()

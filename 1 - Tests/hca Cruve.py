# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 12:04:47 2021

@author: guido
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
os.chdir(os.path.dirname(sys.argv[0]))
sys.path.append("..")
from src.aerodynamics import rocket_functions
from src.aerodynamics import flight_conditions

DEG2RAD = np.pi / 180
RAD2DEG = 1 / DEG2RAD


def transform_AoA_2_v(aoa):
    if aoa <= -np.pi/2:
        v = [-1, -np.tan(aoa)]/np.sqrt(1+np.tan(aoa)**2)
    elif aoa <= 0:
        v = [1, np.tan(aoa)]/np.sqrt(1+np.tan(aoa)**2)
    elif aoa <= np.pi/2:
        v = [1, np.tan(aoa)]/np.sqrt(1+np.tan(aoa)**2)
    elif aoa <= np.pi:
        v = [-1, -np.tan(aoa)]/np.sqrt(1+np.tan(aoa)**2)
    return v


rocket_points = [[0, 0], [0.2, 0.066], [1.2, 0.066], [1.5, 0.066]]
fins = [[0, 0.1], [0, 0.1], 0.025, 0.001]  # AR = 0.5
fins_control = [[0, 0.1], [0, 0.1], 0.5, 0.001]
flags = [True, True, True, True, True]
random_mass_parameters = [0.7, 2, 1, 1, 2, 1]
to_rocket_list = flags + [rocket_points] + [fins] + [fins_control]
rocket = rocket_functions.Rocket()
rocket.update_rocket(to_rocket_list, random_mass_parameters)
ac = flight_conditions.FinFlightCondition()
ac.mach = 0.01
ac.Re = 1e5

AR_list = [4, 2, 1]
legend_list = ["MAR", "LAR", "ULAR"]
colors = ["C0", "C1", "C2", "C3"]
label_font_size = 17
legend_font_size = 15
tick_font_size = 14

list_plot = [[], [], [], []]
x_plot = [[], [], [], []]
AoA = 0

n = 200
for j in range(len(AR_list)):
    to_rocket_list = [[0, 0], [0.2, 0.066], [1.2, 0.066], [1.5, 0.066]]
    fins = [[0, 0.1], [0, 0.1], (AR_list[j])/10/2, 0.001]
    fins_control = [[0, 0.1], [0, 0.1], 0.5, 0.001]
    pep = [True, True, True, True, True]
    to_rocket_list = pep + [to_rocket_list] + [fins] + [fins_control]
    rocket.update_rocket(to_rocket_list, random_mass_parameters)
    AoA = 0
    for i in range(n):
        v = transform_AoA_2_v(AoA)
        x_plot[j].append(AoA * RAD2DEG)
        ac.aoa = AoA
        acoeff = rocket_functions.fin[0].aero_properties.get_aero_coeff(ac,
                                                                        100,
                                                                        100,
                                                                        use_rocket_re=True)
        list_plot[j].append(acoeff.hac)
        AoA += 90 * DEG2RAD / n
plt.figure()
for i in range(len(AR_list)):
    plt.plot(x_plot[i], list_plot[i], colors[i])
plt.axis([0, 90, 0, 0.5])
plt.grid(True)
plt.xlabel('α (degrees)', fontsize=label_font_size)
plt.ylabel(r"${h_{ac}}$         ", fontsize=label_font_size, rotation=0)
plt.xticks(fontsize=tick_font_size)
plt.yticks(fontsize=tick_font_size)
plt.legend(legend_list + legend_list, loc=4, fontsize=legend_font_size)
plt.show()

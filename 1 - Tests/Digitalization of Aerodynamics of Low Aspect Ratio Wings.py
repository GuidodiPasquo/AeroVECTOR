# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 17:16:35 2021

@author: Guido di Pasquo
"""
import importlib
import matplotlib.pyplot as plt

wind_tunnel_data = importlib.import_module("Wind Tunnel Data.wind_tunnel_data_complete")
wtd_planform = [0]*4
planform_list = ["Rectangular", "Zimmerman", "Inverse Zimmerman", "Elliptical"]
for i in range(4):
    wtd_planform[i] = wind_tunnel_data.FinWindTunnelData(planform_list[i])

AR_list = ["0.5", "0.75", "1", "1.25", "1.5", "1.75", "2"]
legend_list = [""]*7
for i in range(7):
    legend_list[i] = "AR=" + AR_list[i]
marker = ["v", "s", "v", "^", "o", "D", "h"]
plt.figure()
for j in range(4):
    plt.figure()
    plt.grid(True)
    plt.axis([-10, 50, -0.5, 1.75])
    plt.xlabel('α (degrees)', fontsize=20)
    plt.ylabel(r'${C_{L}}$      ', fontsize=20, rotation=0)
    plt.title(planform_list[j], fontsize=20)

    for i in range(7):
        plt.plot(wtd_planform[j].cl_vs_alpha[AR_list[i]][0],
                 wtd_planform[j].cl_vs_alpha[AR_list[i]][1],
                 marker[i],
                 markerfacecolor='none')
        plt.legend(legend_list, loc=4, fontsize=13)

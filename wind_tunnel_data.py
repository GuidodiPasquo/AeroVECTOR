# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 17:49:12 2021

@author: Guido di Pasquo
"""

from scipy.interpolate import interp1d
from numpy import pi

DEG2RAD = pi/180


class FinWindTunnelData:

    def __init__(self):
        self.alpha = [[], [], [], []]
        self.cl = [[], [], [], []]
        self._fill_alpha_and_cl()
        self.cl_f_of_alpha_for_AR_125 = interp1d(self.alpha[3], self.cl[3])
        self.cl_f_of_alpha_for_AR_1 = interp1d(self.alpha[2], self.cl[2])
        self.cl_f_of_alpha_for_AR_075 = interp1d(self.alpha[1], self.cl[1])
        self.cl_f_of_alpha_for_AR_05 = interp1d(self.alpha[0], self.cl[0])
        self.alpha_f_of_cl_for_AR_125 = interp1d(self.cl[3], self.alpha[3])
        self.alpha_f_of_cl_for_AR_1 = interp1d(self.cl[2], self.alpha[2])
        self.alpha_f_of_cl_for_AR_075 = interp1d(self.cl[1], self.alpha[1])
        self.alpha_f_of_cl_for_AR_05 = interp1d(self.cl[0], self.alpha[0])

    def _fill_alpha_and_cl(self):
        for i in range(4):
            with open(".\\Wind Tunnel Data\\InvZimmerman.csv", "r") as file:
                for line in file:
                    try:
                        a = float(line.split(",")[0 + i*2])
                        b = float(line.split(",")[1 + i*2])
                        if self.cl[i] != []:
                            if b < self.cl[i][-1]:
                                b = self.cl[i][-1]
                        self.alpha[i].append(a * DEG2RAD)
                        self.cl[i].append(b)
                    except ValueError:
                        pass

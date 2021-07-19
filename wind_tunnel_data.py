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
        self.cl_f_of_alpha = {
            "1.25": interp1d(self.alpha[3], self.cl[3]),
            "1": interp1d(self.alpha[2], self.cl[2]),
            "0.75": interp1d(self.alpha[1], self.cl[1]),
            "0.5": interp1d(self.alpha[0], self.cl[0])
            }
        self.alpha_f_of_cl = {
            "1.25": interp1d(self.cl[3], self.alpha[3]),
            "1": interp1d(self.cl[2], self.alpha[2]),
            "0.75": interp1d(self.cl[1], self.alpha[1]),
            "0.5": interp1d(self.cl[0], self.alpha[0])
            }

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

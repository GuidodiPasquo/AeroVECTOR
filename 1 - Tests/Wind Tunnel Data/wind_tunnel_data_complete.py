# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 17:49:12 2021

@author: Guido di Pasquo
"""

from numpy import pi

DEG2RAD = pi/180


class FinWindTunnelData:

    def __init__(self, planform):
        self.planform = planform
        self.alpha = [[], [], [], [], [], [], []]
        self.cl = [[], [], [], [], [], [], []]
        self._fill_alpha_and_cl()
        self.cl_vs_alpha = {
            "0.5": [self.alpha[0], self.cl[0]],
            "0.75": [self.alpha[1], self.cl[1]],
            "1": [self.alpha[2], self.cl[2]],
            "1.25": [self.alpha[3], self.cl[3]],
            "1.5": [self.alpha[4], self.cl[4]],
            "1.75": [self.alpha[5], self.cl[5]],
            "2": [self.alpha[6], self.cl[6]],
            }

    def _fill_alpha_and_cl(self):
        for i in range(7):
            with open(".\\Wind Tunnel Data\\" + self.planform + ".csv", "r") as file:
                for line in file:
                    try:
                        a = float(line.split(",")[0 + i*2])
                        b = float(line.split(",")[1 + i*2])
                        self.alpha[i].append(a)
                        self.cl[i].append(b)
                    except ValueError:
                        pass

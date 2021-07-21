# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 19:07:07 2021

@author: Guido di Pasquo
"""


import ISA_calculator as atm
import numpy as np


class AtmosphericConditions:

    def __init__(self):
        self.atmosphere = atm.get_atmosphere()

    def update(self, h):
        self.T, self.P, self.rho, self.spd_sound, self.mu = atm.calculate_at_h(h, self.atmosphere)


class FlightConditions:

    def __init__(self):
        self.atm = AtmosphericConditions()

    def update(self, h=0, v=1):
        self.h = h
        self.atm.update(h)
        self.v = v
        self.mach = self._calculate_mach()
        self.beta = np.sqrt(1 - self.mach**2)

    def _calculate_mach(self):
        mach = self.v / self.atm.spd_sound
        if mach < 0.001:
            mach = 0.001
        elif mach >= 0.9:
            self.is_supersonic = True
        else:
            mach = mach
        return mach

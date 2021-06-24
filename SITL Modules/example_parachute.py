# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 15:17:35 2021

@author: Guido di Pasquo
"""
import python_sitl_functions as Sim
import numpy as np
import importlib
class SITLProgram:
    def __init__(self):
        pass

    """ Available funtions, called with Sim.
    Sim.millis(), Sim.micros(),
    gyro, accx, accz, alt, pos_gnss, vel_gnss = Sim.getSimData()
    Sim.sendCommand(servo, parachute)
    Sim.plot_variable(variable, number) (from 1 to 5 for diferent plots)
    -->
    -->
    -->
    -->
    -->
    -->
    -->
    -->
    -->
    -->
    """
    """!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""

    def everything_that_is_outside_functions(self):
        self.alt_prev = 0
        self.timer_all = 0
        self.sample_time_program = 0.1






    def void_setup(self):
        pass







    def void_loop(self):
        self.t = Sim.micros()/1000000
        if self.t >= self.timer_all + self.sample_time_program*0.999:
            self.gyro, self.accx, self.accz, self.alt, self.pos_gnss, self.vel_gnss = Sim.getSimData()
            parachute = self.parachute_deployment()
            Sim.sendCommand(0, parachute)






    """########"""

    def parachute_deployment(self):
        if self.alt < self.alt_prev and self.alt > 10:
            return 1
        else:
            self.alt_prev = self.alt
            return 0

























































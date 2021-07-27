# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 15:14:18 2021

@author: Guido di Pasquo
"""

import numpy as np
from src.simulation import main_simulation as sim

DEG2RAD = np.pi/180
RAD2DEG = 1/DEG2RAD


def millis():
    return int(sim.t * 1000)


def micros():
    return int(sim.t * 1000000)


def getSimData():
    data = [sim.send_gyro, sim.send_accx, sim.send_accz, sim.send_alt,
            sim.send_gnss_pos, sim.send_gnss_vel]
    return data


def sendCommand(servo, parachute, ignition=0):
    sim.u_servos = servo * DEG2RAD
    sim.parachute = int(parachute)
    if int(ignition) == 1:
        if sim.t_launch > sim.t:
            sim.t_launch = sim.t


def plot_variable(var, i):
    sim.var_sitl_plot[i-1] = var

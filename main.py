# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:23:20 2020
@author: Guido di Pasquo

Thanks to:
     LukeDeWaal for the Standard Atmosphere Calculator
     https://github.com/LukeDeWaal/ISA_Calculator

Apologies in advance for any spelling or grammar error, english is not my first language

###########################################
known bugs-> Arrows are hit or miss, sometimes they aim in the right direction, sometimes they don't




########### OVERALL CHARACTERISTICS OF THE PROGRAM

Non-linear model integrates local accelerations into global velocities. An alternate method of vector derivates 
is still in the program, results are better with the first method.

No supersonic rockets!
Subsonic/Compressible Aerodynamics are very similar to Open Rocket's, there are some differences, especially 
in drag calculations (not more than 20%)
Fins aren't exactly accurate above 90º AoA Total, so be wary if you pretend to do some nice acrobatics. It
should be fine even for hard manouvers tho, but the descent might not be realistic/accurate.

Important, all angles are in RADIANS (Standard 1º*np.pi/180 = radian)
deg2rad=np.pi/180
rad2deg=1/deg2rad

Code simulates the Actuator reduction (gear/lever ratio), it multiplies the output of the controller times the
that reduction, and then sends that output to the servo.
Remember that you must multiply the output of the controller times the TVC reduction in you flight computer!
All in all, the overall structure of the function "control_theta" and "PID" should be copied in your code to 
ensure that the simulator and flight computer are doing the same thing.

"""

import tkinter as tk
from tkinter import ttk
import GUI_setup as GUI

#### Main
root = tk.Tk()
root.title("ZPC Actively Stabilized Model Rocket Tuner / Simulator")
root.geometry("600x600")

notebook = ttk.Notebook(root)
GUI.create_file_tab(notebook)
GUI.create_parameters_tab(notebook)
GUI.create_draw_rocket_tab(notebook)
GUI.create_conf_3D_tab(notebook)
GUI.create_SITL_tab(notebook)
GUI.create_simulation_setup_tab(notebook)
GUI.create_run_sim_tab(notebook)
GUI.configure_root(root,notebook)
root.mainloop()
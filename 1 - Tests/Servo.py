import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from src.simulation import servo_lib


DEG2RAD = np.pi / 180
RAD2DEG = 1 / DEG2RAD


servo = servo_lib.Servo()
servo.setup(actuator_weight_compensation=1.45, definition=1, servo_s_t=0.02)
servo.test(u_deg=20)


""" #======================================================================#"""
label_font_size = 17
legend_font_size = 15
tick_font_size = 14

list_plot = [[], []]
x_plot = []
AoA = 0

n = 1000
inp = 0
for i in range(n):
    K = np.interp(inp,
              [10*DEG2RAD, 20*DEG2RAD, 45*DEG2RAD, 90*DEG2RAD],
              [3761, 2159*1.1, 691.9, 256.9])  # *1.1 because it gives much better results
    J = np.interp(inp,
              [10*DEG2RAD, 20*DEG2RAD, 45*DEG2RAD, 90*DEG2RAD],
              [91.31, 59.97, 30.42, 16])
    x_plot.append(inp*RAD2DEG)
    list_plot[0].append(K)
    list_plot[1].append(J)
    inp += (np.pi/2) / n
plt.figure()
plt.plot(x_plot, list_plot[0], "C0")
plt.grid(True)
plt.xlabel('Input (degrees)', fontsize=label_font_size)
plt.ylabel("K    ", fontsize=label_font_size, rotation=0)
plt.xticks(fontsize=tick_font_size)
plt.yticks(fontsize=tick_font_size)
plt.show()

plt.figure()
plt.plot(x_plot, list_plot[1], "C1")
plt.grid(True)
plt.xlabel('Input (degrees)', fontsize=label_font_size)
plt.ylabel("J    ", fontsize=label_font_size, rotation=0)
plt.xticks(fontsize=tick_font_size)
plt.yticks(fontsize=tick_font_size)
plt.show()

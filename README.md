# PID-Simulator
This is a Model Rocket Simulator oriented towards active stabilization. It integrates the 3DOF Equations of Motion, allowing to tune controllers used in Model Rockets. There is pre-coded controller in the file control.py, one can modify it or run the real Flight Computer's software through Software in the Loop.

#### Simulation Procedure
The program integrates the 3DOF Equations of Motion of the Rocket and integrates their result. The Aerodynamics are calculated using the same Extended Barrowman Equations thet Open Rocket uses. The fins use modified wind tunnel data to better model their behavior. More information can be found inside ZPC_PID_SIMULATOR_V05.py and rocket_functions.py.

## Dependencies
Numpy and matplotlib are mandatory.
VPython and pyserial are recomended.

## How To
The setup of the rocket is fairly simple. However, the program is not ment to desing the rocket. Open Rocket is a more confortable option.
###

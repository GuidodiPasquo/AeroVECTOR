



# AeroVECTOR - The Model Rocket Simulator & Tuner
This is a Model Rocket Simulator oriented towards active stabilization. It integrates the Three Degree of Freedom Equations of Motion, allowing to tune controllers used in Model Rockets. There is pre-coded controller in the file *src/control.py*, one can use it or run the Flight Computer's software through Software in the Loop.



https://user-images.githubusercontent.com/65097298/128649259-8a93ab7a-ccb8-4f38-8864-71a1805a3afd.mp4



#### Simulation Procedure
The program computes the 3DoF Equations of Motion of the Rocket and integrates their result with the trapezoidal rule. The Aerodynamic Coefficients are calculated using the same Extended Barrowman Equations that Open Rocket uses (plus some modifications). The fins' forces are computed with interpolated wind tunnel data and Diederich's Semi-Empirical Method to accurately model the behaviour for every aspect ratio and angle of attack. The program also allows for fins separated from the body. More information can be found in the technical documentation or inside *src/simulation/main_simulation.py*, *src/aerodynamics/rocket_functions.py* or *src/aerodynamics/fin_aerodynamics.py*

## Dependencies
### Mandatory
1. tkinter
2. numpy 
3. scipy
4. matplotlib
5. pandas
6. vpython
### Optional
7. pyserial  
  
**Without the optional dependencies, the Software in the Loop module will not work.**  
If someone can make an .exe that works with VPython, please let me know.

## How To
The setup of the rocket is fairly simple. However, the program is not meant for designing the rocket. Open Rocket is a more comfortable option.
### Creating and Opening files
To run the program, run *AeroVECTOR.py*.
The program will open in *the file tab*.
![](/3 - README Images/Readme/Screenshot_1.png)

One can create a new file or open an existing one. Once a file is open, a copy can be created with the *Save As* button.

### Setting up the Rocket
![](/3 - README Images/Readme/Screenshot_2.png)

One must fill the required parameters of the rocket. New motors can be added in the *motors* folder. 
- *Iy Liftoff/Burnout* are the wet and dry pitching moment of inertia.
- *Xcg Liftoff/Burnout* are the position of the wet and dry Center of Gravity.
- *Xt* is the position of the TVC mount. If one is using fins, the program automatically calculates the force application point.
  - All distances are measured from the tip of the nose cone.
- The *Servo Resolution* is the minimum angle it can rotate.
- The *Max Actuator Angle* is the maximum angle the actuator can move (either the TVC mount or the fin).
- The *Actuator Reduction* is the gear ratio between the servo and the actuator.
- The *Initial Misalignment* only modifies the initial angle of the TVC mount or Active Fin.
- The *Servo Velocity Compensation* slows down the servo according  to the load, its value is 1.45 for an SG90 without load, and 2.1 with a TVC mount. The *servo* class found in *servo_lib.py* has a test method to modify this value to fit one's servo.
- The wind is positive from right to left, and the gusts follow a Gaussian distribution.
- The effective launch rod length is the length at which the rocket can pitch somewhat freely.
- Angle of the launch rod.
- The *Motor Misalignment* sets the motor at an angle (mainly for active fin control).  
- The roughness applies to all the components of the body and fins; therefore, a little tuning might be required to get the performances right.


**Please do not leave blank entries**

**THE SAVE BUTTON SAVES ONLY THE CURRENT TAB, BE SURE TO CLICK IT ON ALL OF THEM**

### Setting up the Rocket 2 - The Empire Strikes Back
![](/3 - README Images/Readme/Screenshot_3.png)

#### Body
To draw the rocket, one must insert the point as **coordinate from the nose cone tip, diameter in that point**.
With the *Add Point* button, one adds the point written in the entry. The *Delete Point* button deletes the point currently selected in the combo box. To modify a point, one has to select the desired point in the combo box, click the *Select Point* button, write the new coordinates in the entry, and at last, click the *Modify Point* button.  

#### Fins
To draw the fins, one must insert the position and chord of the root and tip, separated by comma. The wingspan is measured from the root to the tip.
**All dimensions are in meters.**  
The order is the following:  
![](/3 - README Images/Readme/Screenshot_8.png)

**Only trapezoidal fins are modelled.**

After the points are written in the entries, one can either update the stabilization or control fin. Clicking the *Load "" Fins* button will fill the entries with the current data of the fin. The button *Reset Fin* sets the fin entries to a zero-area fin.

**WARNING: THE USE OF CONTROL FINS DISABLES THE TVC STABILIZATION.**  

**Examples of detached fins:**
![](/3 - README Images/Readme/Screenshot_9.png)
*Sprint - BPS.space, and our own Roll Control System Testbed*  
  
The space between the body and the fin must be considerable, if one is unsure about a fin being attached or detached, the most conservative option is the right option.  
The Angle of Attack slider allows to change the AoA at which the CP (red point) is calculated. The blue point represents the CG of the rocket. One can enable and disable the fins to quickly redraw the rocket and ensure that the CG is in the correct position.
  
### 3D Graphics
![](/3 - README Images/Readme/Screenshot_4.png)  
  
One can activate the 3D Graphics by clicking the checkbox. **IT REQUIRES VPYTHON**  

- *Camera Shake* moves the camera based on the accelerations of the rocket.
- *Hide Forces* hides the force arrows.
- *Variable fov* decreases the fov variably, maintaining the rocket of approximately the same size during the flight.
- *Hide CG* hides a ball that represents the CG.
- *Camera type*
  - *"Follow"* -> Follows the rocket.
  - *"Fixed"* -> Looks from the ground.
  - *"Follow Far"* -> 2D.
- *Slow Motion* slows the animation -> 1 = Real time, 5 = five times slower.
- *Force Scale Factor* scales the forces accordingly, 1 = 1 meter/Newton.
- *Fov* is the field of view of the camera.
  
### Software in the Loop
![](/3 - README Images/Readme/Screenshot_5.png)

#### Python SITL
In case of using the Python SITL module, refer to the example. One can create functions, classes, modules, etc., or modify the code at will. The objective was to make it as similar as possible to an Arduino, but there are some differences, especially with the global variables having the prefix self. Only the Python SITL module is compatible with GNSS and sample times.
  
#### On what boards can I use this software?
It was only tested on an Arduino Nano clone, so compatibility is not ensured.  
Even on the Arduino, the program did not work properly with program runtimes times smaller than 5 milliseconds.

#### How to set up the serial communication in Python.
To use the Software in the Loop function, one must set the *Port* to the one in which the board is connected, and the *Baudrate* to the one set in the Arduino program.  
One can simulate sensor noise by filling the entries with the Noise Standard Deviation of the specified sensor.
  
#### How to set up the simulation in your Arduino.
Download and include the library and create the instance with the name you want.  

![](/3 - README Images/Readme/Screenshot_10.png)  
  
At the end of void setup, start the simulation.  

![](/3 - README Images/Readme/Screenshot_11.png)


Replace your sensor readings with *Sim.getSimData()* and the name of your variables.   
![](/3 - README Images/Readme/Screenshot_13.png)


- **The Gyroscope data is in º/s.**  
- **The Accelerometer measures the reaction force applied to it (like the real ones), the data is in g's.**   
- **The Altimeter measures the data in meters.**  
- **The GNSS measures the distance in the horizontal axis from the launch pad, in m.**
- **The GNSS measures the horizontal velocity, in m/s.**   
  
**Positive values are positive in the direction of the axes!**    
**(Refer to the Technical Documentation)**  

![](/3 - README Images/Readme/Screenshot_14.png)    
  
   
Replace your *servo.write()* for:    
![](/3 - README Images/Readme/Screenshot_12.png)  
Replace *servo_command* for your servo variable (in º).    
The parachute variable is an int, it's normally 0 and one must make it 1 when the parachute would deploy. 

**REMEMBER THAT THE DATA IS IN DEGREES, G'S AND M, AND YOU HAVE TO SEND THE SERVO COMMAND IN DEGREES AND THE PARACHUTE DEPLOYMENT SIGNAL AS 0 OR 1.**  

### Tuning the Internal Controller   
![](/3 - README Images/Readme/Screenshot_6.png)

**IF ONE IS USING SOFTWARE IN THE LOOP, THE CONTROLLER SETTINGS DO NOTHING**  

- *Kp, Ki, and Kd* are self-explanatory.
- *K All* scales the error before sending it to the PID.
- *K Damping* feeds the gyro back at the output of the PID and acts as damping.
  - To disable the controller, one must set *K All and K Damping* to zero.
  - In case the Control Fins are ahead of the CG, the controller multiplies *K All* and *K Damping* by -1.
- *Reference Thrust* is the Reference Thrust of the Torque Controller, more info in the *control.py* file.
- *Input* is the input to the rocket, be it a Step (deg), or a Ramp (deg/s)
  - If the selected input is *Up*, then this entry is bypassed
- *Input Time* is the instant at which the input is changed from 0 to the selected one.
- *Launch Time* is the instant at which the motor is ignited.
- *Servo Sample Time and Program Sample Time* are self-explanatory, note they are in seconds and not Hz.
- *Maximum Sim Duration* specifies the maximum duration of the simulation. 
  - The simulation will stop if the Rocket tumbles more than 2 times, hits the ground, or coasts for more than 10 burnout times.
- *Sim Delta T* is the sample time of the simulation, one can increase it to hasten it.
  - Since the Software in the Loop simulation runs in real time, this setting does not affect it.
- Export T is the sample time of the data exports, i.e., the time between exported data points.
- Launch altitude is the height above sea level of the launchpad.
- The next entries deal with the initial state of the rocket, useful for second stages or landings (landing aerodynamics are not well modelled, please do not rely exclusively on them.)
  
  
### Plotting the results
![](/3 - README Images/Readme/Screenshot_7.png)
  
There are ten plots in total, one can choose between a variety of variables to be plotted in two different figures.
The Export Plots button creates a *.csv* file containing all the plotted variables. The name of the *.csv* is the save file's name with a subscript and is created in the *Exports* folder.
The exports are useful for feeding simulated sensor data from the Python SITL modules to a real flight computer and debug the latter by comparing the outputs. In consequence, there will be faster code prototyping and more complex controllers with less failures.

### 3D Representation
![](/3 - README Images/Readme/Screenshot_15.png)
If the 3D Graphics were enabled, once the simulation finishes a new tab in your default web browser should open.
- One can seek backward/forward 1/5 of the simulation time, 1 or 0.015 seconds using the first row of buttons, the center one pauses and resumes the animation.
- The first slider controls the *Slow Motion* and the second one the *FOV* (zoom).
- The menu enables the change of camera angle on the fly.
- One can enable and disable the *Forces* and the *CG & AoA* (relative wind) with the buttons on the right.
- To see the plots, one must click the *Finish* button. This disables the animation.

NOTE: If the animation is paused, some of the features might require you to seek forward/backward to update the frame.

### Experimental Features
In *src/aerodynamics/rocket_functions.py* are some experimental features turned off by default, you can enable them by searching **"experimental ="** and setting them to True. They include dynamic pressure scaling for, theoretically, better damping, and drag calculations based on the component or fin Reynolds instead of the rocket's Re.

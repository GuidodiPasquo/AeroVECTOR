# PID-Simulator
This is a Model Rocket Simulator oriented towards active stabilization. It integrates the 3DOF Equations of Motion, allowing to tune controllers used in Model Rockets. There is pre-coded controller in the file control.py, one can modify it or run the real Flight Computer's software through Software in the Loop.

![](/Images/Readme/GIF-TVC-only.gif)

#### Simulation Procedure
The program integrates the 3DOF Equations of Motion of the Rocket and integrates their result. The Aerodynamics are calculated using the same Extended Barrowman Equations that Open Rocket uses. The fins use modified wind tunnel data to better model their behavior. More information can be found inside *ZPC_PID_SIMULATOR.py* and *rocket_functions.py*

## Dependencies
### Mandatory
1. numpy 
2. matplotlib
### Optional
3. VPython
4. pyserial


**Without the optional modules, the 3D graphics and Software in the Loop will not work.**

## How To
The setup of the rocket is fairly simple. However, the program is not meant for designing the rocket. Open Rocket is a more comfortable option.
### First Step
To run the program, run *main.py*.
The program will open in *the file tab*.
![](/Images/Readme/Screenshot_1.png)

One can create a new file or open an existing one. Once a file is open, a copy can be created with the *Save as* button.

### Second Step
![](/Images/Readme/Screenshot_2.png)

One must fill the required parameters of the rocket. New motors can be added in the *motors* folder. 
- *Iy* is the pitching moment of inertia.
- All distances are measured from the tip of the nosecone.
- The *Servo Definition* is the minimum angle it can rotate.
- The *Max Actuator Angle* is the maximum angle the actuator can move (either the TVC mount or the fin).
- The *Actuator Reduction* is the gear ratio between the servo and the actuator.
- The *Initial Misalignment* only modifies the initial angle of the TVC mount (in case of using TVC stabilization).
- The *Servo Velocity Compensation* slows down the servo according  to the load, its value is 1.45 for an SG90 without load, and 2.1 with a TVC mount. The *servo* class found in *servo_lib.py* has a test method to modify this value to fit one's servo.
- The wind is positive from right to left, and the gusts follow a Gaussian distribution.
**Please do not leave blank entries**

### Third Step
![](/Images/Readme/Screenshot_3.png)

#### Body
To draw the rocket, one must insert the point as **coordinate from the nosecone tip, diameter in that point**.
With the *Add Point* button, one adds the point in the entry. The *Delete Point* button deletes the point currently selected in the combobox. To modify a point, one has to select the desired point in the combobox, click the *Select Point* button, write the new coordinates in the entry, and at last, click the *Modify Point* button.  

#### Fins
To draw the fins, the procedure is similar. One must fill the entries with their respective points as:  
**coordinate from the nosecone tip, *radius* to that point**.   
The order is the following:  
![](/Images/Readme/Screenshot_8.png)

**Only trapezoidal fins are modeled**, so please make sure that the root and tip chords are parallel.

After the points are written in the entries, one can either update the stabilization or control fin. Clicking the *Load "" Fins* button will fill the entries with the current data of the fin. The button *Reset Fin* sets the fin entries to a zero-area fin.

**The use of control fins disables the TVC stabilization**  

Examples of detached fins:




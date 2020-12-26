Download and include the library
#include <SinL.h>

Create the instance with the name you want.
SinL Sim; 

At the end of void setup, start the simulation.
Sim.StartSinL();

Replace your sensor readings with .getSimData() and the name of your variables. 
Sim.getSimData(GyroY, AccX, AccZ, Alt);

**All the data is in º/s, g's, and m!!**

Insert image of the rocket


Replace your servo.write() for 
Sim.sendCommand(servo, parachute);

Replace servo for your servo variable (in º)
The parachute variable is an int, it's normally 0 and you have to make it 1 when the parachute would be deployed.

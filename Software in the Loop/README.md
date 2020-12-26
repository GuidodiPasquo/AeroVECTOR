Download and include the library
#include <SinL.h>
![](/images/Screenshot_1.png)

Create the instance with the name you want.
SinL Sim; 
![](/images/Screenshot_2.png)

At the end of void setup, start the simulation.
Sim.StartSinL();
![](/images/Screenshot_3.png)

Replace your sensor readings with .getSimData() and the name of your variables. 
Sim.getSimData(GyroY, AccX, AccZ, Alt);
![](/images/Screenshot_4.png)

**All the data is in º/s, g's, and m!**

![](/images/Screenshot_5.png)


Replace your servo.write() for 
Sim.sendCommand(servo, parachute);

![](/images/Screenshot_6.png)

Replace servo for your servo variable (in º)
The parachute variable is an int, it's normally 0 and you have to make it 1 when the parachute would be deployed.

**_REMEMBER THAT THE DATA IS IN DEGREES, G'S AND M, AND YOU HAVE TO SEND THE SERVO COMMAND IN DEGREES AND THE PARACHUTE DEPLOYMENT SIGNAL AS 0 OR 1!_**

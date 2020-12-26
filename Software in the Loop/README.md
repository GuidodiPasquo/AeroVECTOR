# On what boards can I use this software?
It was only tested in an Arduino Nano clone, so compatibility is not ensured.



# How to set up the simulation in your Arduino.

Download and include the library.

_#include <SinL.h>_

![](/Software%20in%20the%20Loop/images/Screenshot_1.png)



Create the instance with the name you want.

_SinL Sim;_

![](/Software%20in%20the%20Loop/images/Screenshot_2.png)



At the end of void setup, start the simulation.

_Sim.StartSinL();_


![](/Software%20in%20the%20Loop/images/Screenshot_3.png)


Replace your sensor readings with _Sim.getSimData()_ and the name of your variables. 


![](/Software%20in%20the%20Loop/images/Screenshot_4.png)


**All the data is in º/s, g's, and m!**
**Positive values are positive in the direction of the axes!**

![Axes of the Rocket](/Software%20in%20the%20Loop/images/Screenshot_6.png)


Replace your servo.write() for:

![Sim.sendCommand(servo, parachute);](/Software%20in%20the%20Loop/images/Screenshot_5.png)

Replace _servo_ for your servo variable (in º)

The parachute variable is an int, it's normally 0 and you have to make it 1 when the parachute would be deployed.


-


**_REMEMBER THAT THE DATA IS IN DEGREES, G'S AND M, AND YOU HAVE TO SEND THE SERVO COMMAND IN DEGREES AND THE PARACHUTE DEPLOYMENT SIGNAL AS 0 OR 1_**

# How to set up the serial communication in Python.

You only have to replace the baudrate and the port of your Arduino in the lines 840. A baudrate of 1000000 is recommended.

![PythonSetup](/Software%20in%20the%20Loop/images/Screenshot_7.png)


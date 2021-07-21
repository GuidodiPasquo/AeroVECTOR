#include "SITL.h"


const unsigned int MAX_INPUT = 100;

SITL::SITL(bool hello){

}
// Starts the Simulation
void SITL::StartSITL(){
  delay(100);
  Serial.println("A");
}



// Functions that read the data sended by Python
void SITL::process_data (const char * data) // here to process incoming serial data after a terminator received
  {
  _rxString = "";
  int stringStart = 0;
  int arrayIndex = 0;
  _rxString = (String) data;

  for (int i = 0; i  < _rxString.length(); i++) {
      //Get character and check if it's our "special" character.
      if (_rxString.charAt(i) == ',') {
          //Clear previous values from array.
          _strArr[arrayIndex] = "";
          //Save substring into array.
          _strArr[arrayIndex] = _rxString.substring(stringStart, i);
          //Set new string starting point.
          stringStart = (i + 1);
          arrayIndex++;
      }
    }

  }  // end of process_data



void SITL::processIncomingByte (const byte inByte)
  {
  static char input_line [MAX_INPUT];
  static unsigned int input_pos = 0;

  switch (inByte)
    {

    case '\n':   // end of text
      input_line [input_pos] = 0;  // terminating null byte

      // terminator reached! process input_line here ...
      process_data (input_line);

      // reset buffer for next time
      input_pos = 0;
      break;

    case '\r':   // discard carriage return
      break;

    default:
      // keep adding if not full ... allow for terminating null byte
      if (input_pos < (MAX_INPUT - 1))
        input_line [input_pos++] = inByte;
      break;

    }  // end of switch

  } // end of processIncomingByte


void SITL::getSimData(float & SimGiroY, float & SimAccX, float & SimAccZ, float & SimAlt)
{

  Serial.println("R");

  unsigned long timer_send = micros();

  while (micros() < timer_send+3UL*1000UL)
  {
    while (Serial.available()>0)
    {
       processIncomingByte (Serial.read ());
       delayMicroseconds(20);
    }
  }


  SimGiroY = _strArr[0].toFloat();
  SimAccX = _strArr[1].toFloat();
  SimAccZ = _strArr[2].toFloat();
  SimAlt = _strArr[3].toFloat();
}



// Send the Servo and Parachute commands
void SITL::sendCommand (float servo, int parachute)
{
  Serial.print(servo,6);
  Serial.print(",");
  Serial.print(parachute);
  Serial.print('\n');
}

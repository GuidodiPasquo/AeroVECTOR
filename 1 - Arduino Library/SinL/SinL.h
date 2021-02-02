#ifndef SIL
#define SIL

#if (ARDUINO >= 100)
#include "Arduino.h"
#else
#include "WProgram.h"
#endif

class SinL {
  public:
    // Constructor
    SinL(bool hello = true);

    // Methods
    void sendCommand (float servo, int parachute);
    void getSimData (float & SimGiroY, float & SimAccX, float & SimAccZ, float & SimAlt);
    void StartSinL();

  private:
    String _rxString;
    String _strArr[4];


    void process_data (const char * data);
    void processIncomingByte (const byte inByte);
};


#endif

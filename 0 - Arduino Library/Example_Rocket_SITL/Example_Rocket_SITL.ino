#include <Servo.h>
#include <SITL.h>

SITL Sim;
Servo servo;

double DEG2RAD = 3.14159265/180;
double RAD2DEG = 180/3.14159265;

// Sample Time
double T_Program = 0.01;
unsigned long T_Program_micros = T_Program * 1000000;
unsigned long timer_run;

// PID
unsigned long currentTime, previousTime;
double elapsedTime,elapsedTimeSeg;
double errorPID;
double lastError;
double input, output, setPoint;
double cumError, rateError;
double okp, oki, okd;
float kp = 0.4;
float ki = 0.0;
float kd = 0.136;
double out;

// Saturation and Real Actuator
double Actuator_reduction = 5;
double Max_Actuator_Angle = 10 * DEG2RAD;
double Max_servo_Angle = Max_Actuator_Angle * Actuator_reduction;

// SinL
float theta, servo_command, Alt_prev;
int parachute = 0;
float GyroY, AccX, AccZ, Alt;

void setup() {
  Serial.begin(1000000);   
  servo.attach(10);
  servo.write(90);
  Sim.StartSITL();
}

void loop() {
// Sample time
if (micros() >= timer_run + T_Program_micros){  
  double dt = double(micros() - timer_run);
  // micros to seconds
  dt /= 1000000;
  timer_run = micros();  

  // SinL Simulation
  Sim.getSimData(GyroY, AccX, AccZ, Alt);

  // Integrate the Gyros to find the angle.
  theta += (GyroY * DEG2RAD) * dt;

  double setpoint = 10 * DEG2RAD;
  out = PID(setpoint, theta);
  servo_command = PID2Servo(out);
  
  // Parachute
  parachute = Deploy_Parachute(Alt);

  // Real servo
  double servo_center = 90;
  servo.write(-servo_command + servo_center); 
   
  // Simulated servo and parachute
  Sim.sendCommand(servo_command, parachute); 
  }
}


double PID(double set_point, double inp){
  currentTime = micros();  //get current time
  elapsedTime = (double)(currentTime - previousTime);  // compute time elapsed from previous computation
  elapsedTimeSeg = elapsedTime / 1000000;
    
  errorPID = set_point - inp;  // determine error
  rateError = (errorPID - lastError) / (elapsedTimeSeg);  // compute derivative
  cumError += ((((lastError) + ((errorPID - lastError) / 2))) * elapsedTimeSeg);  // compute integral
  out = kp * errorPID + ki * cumError + kd * rateError;
  
  lastError = errorPID;  //remember current error
  previousTime = currentTime;  //remember current time
  return out;
  }

double PID2Servo(double out){
  // Actuator  
  double servo_command = out * Actuator_reduction;
  
  // Saturation
  if(servo_command > Max_servo_Angle){servo_command = Max_servo_Angle;}
  if(servo_command < -Max_servo_Angle){servo_command = -Max_servo_Angle;}
  servo_command *= RAD2DEG;
  return servo_command;  
}

int Deploy_Parachute (double Alt){
  // No noise pls
  if (Alt > 10 and Alt_prev > Alt)
  {
    parachute = 1;
  }
  Alt_prev = Alt;
  return parachute; 
}
 

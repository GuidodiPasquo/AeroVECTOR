# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 14:05:35 2021

@author: guido
"""
import ZPC_PID_SIMULATOR as sim

class controller_class:

    def __init__(self):
        self.torque_controller = True
        self.anti_windup = True
        self.input_type = "Step [º]"
        self.kp = 0.4
        self.ki = 0
        self.kd = 0.136
        self.k_all = 1
        self.k_damping = 0
        self.reference_thrust = 28
        self.u_controller = 0
        self.t_prev = 0
        self.lastError = 0
        self.cumError = 0
        return

    def setup_controller(self, conf_controller, Actuator_reduction, TVC_max):
        self.u_controller = 0
        self.t_prev = 0
        self.lastError = 0
        self.cumError = 0
        self.torque_controller = conf_controller[0]
        self.anti_windup = conf_controller[1]
        self.input_type = conf_controller[2]
        self.kp = conf_controller[3]
        self.ki = conf_controller[4]
        self.kd = conf_controller[5]
        self.k_all = conf_controller[6]
        self.k_damping = conf_controller[7]
        self.reference_thrust = conf_controller[8]
        self.Actuator_reduction = Actuator_reduction
        self.TVC_max = TVC_max
        if sim.rocket.use_fins_control == True and sim.rkt.fin[1].CP < sim.xcg:
            """
            Must invert the gains if the fins are ahead of the CG, or else
            the controller has positive feedback due to the torque being
            opposite to the one expected.
            """
            self.k_all *= -1
            self.k_damping *= -1

    def control_theta(self, setpoint, theta, Q, Thrust, t):
        self.u_prev = self.u_controller
        error = setpoint - theta
        error = error * self.k_all
        self.u_controller = self.PID(error, t)
        self.u_controller = self.u_controller - Q*self.k_damping
        #Saturation
        if(self.u_controller > self.TVC_max):
            self.u_controller = self.TVC_max
        elif(self.u_controller < -self.TVC_max):
            self.u_controller = -self.TVC_max
        #u_controller=u_controller-u_prev*0.05;  #filter, increasing the number makes it stronger and slower
        self.u_servos = self.u_controller*self.Actuator_reduction
        if(self.torque_controller==True):
            """
            On the simulation one can access the real Thrust from the thrust curve
            In your flight computer you will have to calculate it.
            """
            thrust_controller = Thrust
            self.u_servos=(self.reference_thrust/thrust_controller) * self.u_servos
            # Prevents the TVC from deflecting more that it can after being corrected
            # for the Thrust.
            if(self.u_servos > self.TVC_max * self.Actuator_reduction):
                self.u_servos = self.TVC_max * self.Actuator_reduction
            elif(self.u_servos < -self.TVC_max * self.Actuator_reduction):
                self.u_servos = -self.TVC_max * self.Actuator_reduction
        return self.u_servos, self.okp, self.oki, self.okd, self.totError

    # PID
    def PID(self, inp, t):
        T_Program = t - self.t_prev
        # Determine the error
        errorPID = inp;
        # Compute its derivative
        rateError = (errorPID - self.lastError) / T_Program
        if(self.anti_windup==True):
            # PID output
            out_pid = self.kp * errorPID + self.ki * self.cumError + self.kd * rateError;
            # Anti windup by clamping
            if out_pid < self.TVC_max and out_pid > -self.TVC_max:
                # Compute integral (trapezoidal) only if the TVC is not staurated
                self.cumError = ((((self.lastError) + ((errorPID - self.lastError) / 2))) * T_Program) + self.cumError
                # Recalculate the output
                out_pid = self.kp * errorPID + self.ki * self.cumError + self.kd * rateError
            # Saturation, prevents the TVC from deflecting more that it can
            if(out_pid > self.TVC_max ):
                out_pid = self.TVC_max
            elif(out_pid < -self.TVC_max ):
                out_pid = -self.TVC_max
        else:
            self.cumError = ((((self.lastError) + ((errorPID - self.lastError) / 2))) * T_Program)+self.cumError
            out_pid = self.kp * errorPID + self.ki * self.cumError + self.kd * rateError;
        self.okp = self.kp * errorPID
        self.oki = self.ki * self.cumError
        self.okd = self.kd * rateError
        self.totError = errorPID
        # Remember current error
        self.lastError = errorPID
        self.t_prev = t
        # Have function return the PID output
        return out_pid
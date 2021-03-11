# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 14:05:35 2021

@author: guido
"""
import zpc_pid_simulator as sim


class Controller:
    """
    Controller class, the default is a PID controller
    """

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
        self.actuator_reduction = 0.
        self.tvc_max = 0.
        self.u_controller = 0
        self.u_prev = 0
        self.u_servos = 0
        self.t_prev = 0
        self.last_error = 0
        self.cum_error = 0
        self.okp = 0
        self.oki = 0
        self.okd = 0
        self.tot_error = 0

    def setup_controller(self, conf_controller, actuator_reduction, tvc_max):
        """
        Set up the controller

        Parameters
        ----------
        conf_controller : list
            List with the controller configuration.
        actuator_reduction : float
            TVC reduction rate.
        tvc_max : float
            Maximum deflection angle (rad).

        Returns
        -------
        None.

        """
        self.u_controller = 0
        self.t_prev = 0
        self.last_error = 0
        self.cum_error = 0
        self.torque_controller = conf_controller[0]
        self.anti_windup = conf_controller[1]
        self.input_type = conf_controller[2]
        self.kp = conf_controller[3]
        self.ki = conf_controller[4]
        self.kd = conf_controller[5]
        self.k_all = conf_controller[6]
        self.k_damping = conf_controller[7]
        self.reference_thrust = conf_controller[8]
        self.actuator_reduction = actuator_reduction
        self.tvc_max = tvc_max
        if sim.rocket.use_fins_control is True and sim.rkt.fin[1].cp < sim.xcg:
            # Must invert the gains if the fins are ahead of the CG, or else
            # the controller has positive feedback due to the torque being
            # opposite to the one expected.
            self.k_all *= -1
            self.k_damping *= -1

    def control_theta(self, setpoint, theta, Q, thrust, t):
        """
        Input the setpoint and the rest of the parameters to obtain
        the servo command.

        Parameters
        ----------
        setpoint : float
            Input to the controller.
        theta : float
            Current angle.
        Q : float
            Current pitching speed.
        thrust : Float
            Current thrust.
        t : float
            Current time.

        Returns
        -------
        u_servos : float
            Output of the controller.
        okp : float
            Proportional contribution.
        oki : float
            Integral contribution.
        okd : float
            Derivative contribution.
        tot_error : float
            Total error.

        """
        self.u_prev = self.u_controller
        error = setpoint - theta
        error = error * self.k_all
        self.u_controller = self._pid(error, t)
        self.u_controller = self.u_controller - Q*self.k_damping
        # Saturation
        if self.u_controller > self.tvc_max:
            self.u_controller = self.tvc_max
        elif self.u_controller < -self.tvc_max:
            self.u_controller = -self.tvc_max
        # u_controller=u_controller-u_prev*0.05;  #filter, increasing the
        # number makes it stronger and slower
        self.u_servos = self.u_controller*self.actuator_reduction
        if self.torque_controller is True:
            # On the simulation one can access the real thrust from the thrust curve
            # In your flight computer you will have to calculate it.
            thrust_controller = thrust
            self.u_servos=(self.reference_thrust/thrust_controller) * self.u_servos
            # Prevents the TVC from deflecting more that it can after being corrected
            # for the thrust.
            if self.u_servos > self.tvc_max * self.actuator_reduction:
                self.u_servos = self.tvc_max * self.actuator_reduction
            elif self.u_servos < -self.tvc_max * self.actuator_reduction:
                self.u_servos = -self.tvc_max * self.actuator_reduction
        return self.u_servos, self.okp, self.oki, self.okd, self.tot_error

    # PID
    def _pid(self, inp, t):
        T_program = t - self.t_prev
        # Determine the error
        error_pid = inp
        # Compute its derivative
        rate_error = (error_pid - self.last_error) / T_program
        if self.anti_windup is True:
            # PID output
            out_pid = self.kp * error_pid + self.ki * self.cum_error + self.kd * rate_error
            # Anti windup by clamping
            if -self.tvc_max < out_pid < self.tvc_max:
                # Compute integral (trapezoidal) only if the TVC is not staurated
                self.cum_error = (((((self.last_error) + ((error_pid - self.last_error) / 2)))
                                  * T_program) + self.cum_error)
                # Recalculate the output
                out_pid = self.kp * error_pid + self.ki * self.cum_error + self.kd * rate_error
            # Saturation, prevents the TVC from deflecting more that it can
            if out_pid > self.tvc_max:
                out_pid = self.tvc_max
            elif out_pid < -self.tvc_max:
                out_pid = -self.tvc_max
        else:
            self.cum_error = ((((self.last_error + ((error_pid - self.last_error) / 2)))
                             * T_program) + self.cum_error)
            out_pid = self.kp * error_pid + self.ki * self.cum_error + self.kd * rate_error
        self.okp = self.kp * error_pid
        self.oki = self.ki * self.cum_error
        self.okd = self.kd * rate_error
        self.tot_error = error_pid
        # Remember current error
        self.last_error = error_pid
        self.t_prev = t
        # Have function return the PID output
        return out_pid

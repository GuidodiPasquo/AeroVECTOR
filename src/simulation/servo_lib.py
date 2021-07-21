# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 15:05:28 2021

@author: Guido di Pasquo
"""


import numpy as np
import matplotlib.pyplot as plt


"""
Handles the actuator dynamics.

Classes:
    Servo - Simulates an SG90.
"""

"""
The Actuator compensation slows down the servo to simulate the TVC weight,
since the characterization was done without resistance, with the test
method you can adjust fairly simply the compensation to match your servo
"""


DEG2RAD = np.pi / 180
RAD2DEG = 1 / DEG2RAD


class Servo:
    """
    Handles the servo simulation and internal variables.

    Methods:
        setup -- Set the servo characteristics
        simulate -- Simulate the servo and obtain the current position
        test -- Test the servo to ensure its speed is correct
    """

    def __init__(self):
        # After watching joe's video on servos, they are a tad slower that the
        # ones I measured, this simulates that. 1.45 is the servo alone
        self._actuator_weight_compensation = 2.1
        self._sample_time = 0.001
        self._servo_sample_time = 0.02
        self._definition = 1
        self._u = 0.
        # Matrices left in uppercase because it's how they apear in the books
        self._A_s = np.array([
                            [0., 0.],
                            [0., 0.]
                            ])
        self._B_s = np.array([
                            [0.],
                            [0.]
                            ])
        self._C_s = np.array([
                            [0., 0.],
                            [0., 0.]
                            ])
        self._D_s = np.array([
                            [0.],
                            [0.]
                            ])
        # Continous time
        self._A_s_c = np.array([
                            [0., 0.],
                            [0., 0.]
                            ])
        self._B_s_c = np.array([
                            [0.],
                            [0.]
                            ])
        self._C_s_c = np.array([
                            [1, 0],
                            [0, 1]
                            ])
        self._x_s = np.array([
                            [0.],
                            [0.]
                            ])
        self._x_dot_s = np.array([
                            [0.],
                            [0.]
                            ])

        self._xdot_prev_s = np.array([
                            [0.],
                            [0.]
                            ])
        self._out_s = np.array([
                            [0.],
                            [0.]
                            ])
        self._out_prev_s = np.array([
                            [0.],
                            [0.]
                            ])
        self._identity_2x2 = np.array([
                            [1, 0],
                            [0, 1]
                            ])
        self._u_delta = 0
        self._t_prev = -0.0001
        self._sample_time = 0.001
        self._timer_run = 0

    def __reset_variables(self):
        # sets the variables and matrices to zero
        self._u = 0.
        self._A_s = np.array([
                            [0., 0.],
                            [0., 0.]
                            ])
        self._B_s = np.array([
                            [0.],
                            [0.]
                            ])
        self._C_s = np.array([
                            [0., 0.],
                            [0., 0.]
                            ])
        self._D_s = np.array([
                            [0.],
                            [0.]
                            ])
        # Continous time
        self._A_s_c = np.array([
                            [0., 0.],
                            [0., 0.]
                            ])
        self._B_s_c = np.array([
                            [0.],
                            [0.]
                            ])
        self._C_s_c = np.array([
                            [1, 0],
                            [0, 1]
                            ])
        self._x_s = np.array([
                            [0.],
                            [0.]
                            ])
        self._x_dot_s = np.array([
                            [0.],
                            [0.]
                            ])

        self._xdot_prev_s = np.array([
                            [0.],
                            [0.]
                            ])
        self._out_s = np.array([
                            [0.],
                            [0.]
                            ])
        self._out_prev_s = np.array([
                            [0.],
                            [0.]
                            ])
        self._identity_2x2 = np.array([
                            [1, 0],
                            [0, 1]
                            ])
        self._u_delta = 0
        self._t_prev = -0.0001
        self._sample_time = 0.001
        self._timer_run = 0

    def setup(self, actuator_weight_compensation, definition, servo_s_t):
        """
        Sets the servo speed compensation, definition [º], and sample time

        Parameters
        ----------
        actuator_weight_compensation : float -- Speed compensation.
        definition : float -- servo definition [deg].
        servo_s_t : float -- servo sample time [seg].

        Returns
        -------
        None.
        """
        self._actuator_weight_compensation = actuator_weight_compensation
        self._definition = definition
        self._servo_sample_time = servo_s_t
        self.__reset_variables()

    def _update(self):
        self._u_delta = (self._actuator_weight_compensation
                         * abs(self._u-self._out_s[0,0]))
        if self._u_delta <= 10 * DEG2RAD:
            self._u_delta = 10 * DEG2RAD
        elif self._u_delta >= 90 * DEG2RAD:
            self._u_delta = 90 * DEG2RAD
        # A Matrix
        asc11 = 0.
        asc12 = 1.
        asc21 = -(-2624.5*self._u_delta**3 + 9996.2*self._u_delta**2
                  - 13195*self._u_delta + 6616.2)
        asc22 = -(39.382*self._u_delta**2 - 125.81*self._u_delta + 124.56)
        # Third order polinomial that modifies the model of the sevo based
        # on its current position and setpoint
        self._A_s_c = np.array([
            [asc11, asc12],
            [asc21, asc22]
            ])
        # B Matrix
        bsc11 = 0
        bsc21 = (-2624.5*self._u_delta**3 + 9996.2*self._u_delta**2
                 - 13195*self._u_delta + 6616.2)
        self._B_s_c = np.array([
            [bsc11],
            [bsc21]
            ])
        # C Matrix
        self._C_s_c = np.array([
            [1, 0],
            [0, 1]
            ])
        # D Matrix
        # Tustin integration with variable paramenters (u_delta)
        self._tustin_discretization()

    def _tustin_discretization(self):
        self._A_s = (np.dot(
            np.linalg.inv((2/self._sample_time)*self._identity_2x2 - self._A_s_c),
            ((2/self._sample_time)*self._identity_2x2 + self._A_s_c)
            ))
        self._B_s = (np.dot(
            np.linalg.inv((2/self._sample_time)*self._identity_2x2 - self._A_s_c),
            self._B_s_c
            ))
        self._C_s = self._A_s + self._C_s_c
        self._D_s = self._B_s

    def simulate(self, u_servo, t_current):
        """
        Introduce the servo command and the current time to simulate the servo
        movement.

        Parameters
        ----------
        u_servo : float -- Servo command.
        t_current : float -- Current time.

        Returns
        -------
        numpy float -- Current servo position.
        """
        self._sample_time = t_current - self._t_prev
        self._t_prev = t_current
        self._update()
        u_2_round = self._u
        # +0.001 desynchronizes the servo and the program, thing
        # that would likely happen in a real flight computer
        if t_current > (self._timer_run + self._servo_sample_time*0.999 + 0.001):
            self._timer_run = t_current
            u_2_round = u_servo
        self._u = self._round_input(u_2_round)
        self._x_dot_s = np.dot(self._A_s, self._x_s) + np.dot(self._B_s, self._u)
        self._out_s = np.dot(self._C_s, self._x_s) + np.dot(self._D_s, self._u)
        self._x_s = self._x_dot_s
        return self._out_s[0,0]

    def _round_input(self, u_inp):
        u_inp *= RAD2DEG
        u_inp /= self._definition
        u_inp = round(u_inp, 0)
        u_inp *= self._definition
        u_inp *= DEG2RAD
        return u_inp

    def test(self, u_deg):
        """
        Call this method to test the servo speed.

        Use as argument the input you used to caracterize your ACTUATOR
        (TVC mount/fin). The output of this method is a plot that should
        look similar to your actuator movement, or at least its rise time.

        Parameters
        ----------
        u_deg : float -- Command to the servo.

        Returns
        -------
        A plot?
        """
        u_test = u_deg * DEG2RAD
        x_plot = []
        list_plot = []
        t_current = 0
        for _ in range(1000):
            list_plot.append(self.simulate(u_test, t_current) * RAD2DEG)
            x_plot.append(t_current)
            t_current += 0.001
        plt.plot(x_plot, list_plot)
        plt.grid(True, linestyle='--')
        plt.show()

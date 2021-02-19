# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 15:05:28 2021

@author: guido
"""
import numpy as np
deg2rad=np.pi/180
rad2deg=1/deg2rad
import matplotlib.pyplot as plt

"""
The Actuator compensation slows down the servo to simulate the TVC weight, 
since the characterization was done without resistance, with the test
method you can adjust fairly simply the compensation to match your 
servo        
"""

class servo_class:

    def __init__(self):
        self.u_servos=0.
        # After watching joe's video on servos, they are a tad slower that the ones I measured, 
        # this simulates that. #1.45 is the servo alone
        self.Actuator_weight_compensation=2.1 
        self.T = 0.001        
        self.Ts = 0.02
        self.definition  = 1
        self.__initializate()
        
    def __initializate(self):
        # Initializes the variables and matrices
        self.u = 0.        
        self.As=np.array([[0., 0.], [0., 0.]]) 
        self.Bs=np.array([[0.],[0.]])
        self.Cs=np.array([[0., 0.], [0., 0.]])
        self.Ds=np.array([[0.],[0.]])        
        #Continous time
        self.Asc=np.array([[0., 0.], [0., 0.]])            
        self.Bsc=np.array([[0.],[0.]]) 
        self.Csc=np.array([[1, 0],[0, 1]])                
        self.xs=np.array([[0.],[0.]])            
        self.xdots=np.array([[0.],[0.]])
        self.xdot_prevs=np.array([[0.],[0.]])
        self.outs=np.array([[0.],[0.]])
        self.out_prevs=np.array([[0.],[0.]])
        self.Itwo=np.array([[1, 0],[0, 1]])
        self.u_delta = 0
        self.t_prev = -0.00000001
        self.T = 0.001
        self.timer_run = 0
        
    def setup(self, Actuator_weight_compensation, definition,Ts):        
        self.Actuator_weight_compensation = Actuator_weight_compensation
        self.definition = definition
        self.Ts = Ts                
        self.__initializate()
    
    def update(self):   
        self.u_delta = self.Actuator_weight_compensation*abs(self.u-self.outs[0,0])         
        if(self.u_delta <= 10*deg2rad):
            self.u_delta = 10*deg2rad        
        elif(self.u_delta >= 90*deg2rad):
            self.u_delta = 90*deg2rad            
        # A Matrix    
        asc11=0.
        asc12=1.
        asc21=-(-2624.5 * self.u_delta**3 + 9996.2 * self.u_delta**2 - 13195 * self.u_delta + 6616.2)
        asc22=-(39.382 * self.u_delta**2 - 125.81 * self.u_delta + 124.56)
        # Third order polinomial that modifies the model of the sevo based on its current position and setpoint
    
        self.Asc=np.array([[asc11, asc12], [asc21, asc22]])
        
        # B Matrix
        bsc11=0
        bsc21=(-2624.5 * self.u_delta**3 + 9996.2 * self.u_delta**2 - 13195 * self.u_delta + 6616.2)
            
        self.Bsc=np.array([[bsc11],[bsc21]])
        
        # C Matrix  
        self.Csc=np.array([[1, 0],[0, 1]])  
        # D Matrix
        ## Tustin integration with variable paramenters (u_delta)
        self.Tustin_discretization()
        return
    
    def Tustin_discretization(self):
        self.As = np.dot(np.linalg.inv( (2 / self.T) * self.Itwo -self.Asc), ( (2/self.T) * self.Itwo + self.Asc)) 
        self.Bs = np.dot(np.linalg.inv( (2/self.T) * self.Itwo - self.Asc) , self.Bsc )
        self.Cs = (self.As + self.Csc)
        self.Ds = self.Bs
    
    def simulate(self, u_servo, t):
        self.T = t - self.t_prev
        self.t_prev = t
        self.update()
        u = self.u
        # +0.001 desynchronizes the servo and the program, thing 
        # that would likely happen in a real flight computer
        if(t > (self.timer_run + self.Ts * 0.999 + 0.001)):  
            self.timer_run=t
            u=u_servo
        self.u = self.round_input(u)
        self.xdots = np.dot(self.As , self.xs) + np.dot(self.Bs , self.u)
        self.outs = np.dot(self.Cs , self.xs) + np.dot(self.Ds , self.u)
        self.xs = self.xdots        
        return self.outs[0,0]
    
    def round_input(self,u):
        u = u * rad2deg
        u = u / self.definition        
        u = round(u,0)
        u = u * self.definition
        u = u * deg2rad
        return u
    
    def test(self,u_deg):
        u=u_deg*np.pi/180
        x_plot = []
        list_plot = []
        t = 0
        for i in range(1000):                    
            list_plot.append(self.simulate(u,t)/(np.pi/180))
            x_plot.append(t)
            t += 0.001            
        plt.plot(x_plot,list_plot)
        plt.grid(True,linestyle='--')  
        plt.show()
        
    
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 18:28:36 2021

@author: guido
"""

import os
import copy

def get_save_names():
    filenames = os.listdir(".\\saves")    
    for i in range(len(filenames)):
        # Removes the .txt
        filenames[i] = filenames[i][:-4]       
    return copy.deepcopy(filenames)

def get_motor_names():
    return os.listdir(".\\motors") 

class savefile:
    def __init__(self):
        self.parameter_names = ["Motor[N] = ", "Mass [kg] = ","Iy [kg*m] = ", "Xcg [m] = ","Xt [m] = ", "Servo definition [º] = ",\
             "Max Actuator Angle [º] = ", "Actuator Reduction = ","Initial Misalignment [º] = ","Servo Compensation = ",
             "Wind [m/s] = ","Wind Gust = "]    
        self.conf_3D_names = ["###=#","toggle_3D=","camera_shake_toggle=","hide_forces=","variable_fov=","Camera_type=","slow_mo=","force_scale=","fov="]        
        self.conf_controller_names = ["###=#","Torque Controller = ","Anti Windup = ","Input Type = ","Kp = ", "Ki = ", "Kd = ", "K All = ",
                                      "K Damping = ", "Reference Thrust [N] = ", "Input = ","Input time = ","Launch Time = ",
                                      "Servo Sample Time [s] = ","Controller Sample Time [s] = "
                                      , "Maximum Sim Duration [s] = ", "Sim Delta T [s] = "]
        self.conf_SITL_names = ["###=#","Activate_SITL=", "Use Sensor Noise=", "Port=", "Baudrate=",
                                "Gyroscope SD=", "Accelerometer SD=","Altimeter SD="]
        self.conf_plot_names = ["###=#","Frist_plot=", "Second_plot=", "Third_plot=", "Fourth_plot=", "Fifth_plot="]
        self.rocket_dim_names = ["###=#"]
        
        self.overwrite_flag = False
    
    def update_name(self,n):
        self.name = n
        
    def _save_parameters(self, tofile):
        for i in range(len(self.parameter_names)):
            tofile+=self.parameter_names[i]+self.parameters[i]+"\n"
        return tofile
            
    def _save_conf_3D(self, tofile):
        for i in range(len(self.conf_3D_names)):
            if i == 0:
                tofile+="###=#\n"
            else:
                tofile+=self.conf_3D_names[i]+self.conf_3D[i-1]+"\n"
        return tofile
                
    def _save_conf_SITL(self, tofile):
        for i in range(len(self.conf_SITL_names)):
            if i == 0:
                tofile+="###=#\n"
            else:
                tofile+=self.conf_SITL_names[i]+self.conf_SITL[i-1]+"\n"
        return tofile
                
    def _save_conf_controller(self, tofile):
        for i in range(len(self.conf_controller_names)):
            if i == 0:
                tofile+="###=#\n"
            else:
                tofile+=self.conf_controller_names[i]+self.conf_controller[i-1]+"\n"
        return tofile
    
    def _save_conf_plots(self, tofile):
        for i in range(len(self.conf_plot_names)):
            if i == 0:
                tofile+="###=#\n"
            else:
                tofile+=self.conf_plot_names[i]+self.conf_plots[i-1]+"\n"
        return tofile
    
    def _save_rocket_dim(self, tofile):
        tofile+="###=#\n"
        for i in range(len(self.rocket_dim)):
            tofile+=self.rocket_dim[i]+"\n"
        return tofile
                
    def _save_all(self, tofile):
        tofile = self._save_parameters(tofile)
        tofile = self._save_conf_3D(tofile)
        tofile = self._save_conf_controller(tofile)
        tofile = self._save_conf_SITL(tofile)
        tofile = self._save_conf_plots(tofile)
        tofile = self._save_rocket_dim(tofile)
        return tofile    
    
    def check_if_file_exists2overwrite(self, n):
        self.overwrite_flag = False
        names = get_save_names()
        flag_found_name = False
        for i in range(len(names)):
            if names[i] == n: 
                x = ""
                flag_found_name = True
                while x != "y" or x != "n":
                    x = str(input("File already exists, overwrite? [y/n]"))                
                    if x == "y":
                        self.overwrite_flag = True 
                        return self.overwrite_flag                       
                    elif x == "n":
                        self.overwrite_flag = False 
                        return self.overwrite_flag                          
                    else:
                        print("Insert valid option")
        if flag_found_name == False:
            self.overwrite_flag = True
            return self.overwrite_flag
    
    def create_file(self,n):        
        self.update_name(n)        
        self.parameters = ["Estes_D12.csv", '0.451', '0.0662', '0.55',"0.85", '1', '10', '5', '2',"2.1", '2', '0.1']
        self.conf_3D = ['True', 'False',"False","False","Fixed", '3', '0.2', "0.75"]        
        self.conf_controller = ['False', 'True', 'Step [º]', '0.4', '0', '0.136', '1', '0', '30','20',"0.5","1", '0.02', '0.01', "30", "0.001"]
        self.conf_SITL = ["False","False","COM3","115200","0","0","0"]
        self.conf_plots = ["Pitch Angle", "Setpoint", "Actuator deflection", "Off", "Off"]
        self.rocket_dim = ["True","False","True","False","False","0,0","0.2,0.066","1,0.066",
                           "Fins_s","0.0000,0.0000","0.0001,0.0001","0.0002,0.0001","0.0002,0.0000",
                           "Fins_c","0.0000,0.0000","0.0001,0.0001","0.0002,0.0001","0.0002,0.0000"]
        try:
            with open(".\\saves\\" + self.name + ".txt","w", encoding="utf-8") as file:
                self.tofile = ""                
                self.tofile = self._save_all(self.tofile)
                file.write(self.tofile)
            print("File Created Successfully")
        except EnvironmentError:
            print("Error")
            
    def create_file_as(self,n):
        self.update_name(n)
        try:
            with open(".\\saves\\" + self.name + ".txt","w", encoding="utf-8") as file:
                self.tofile = ""                
                self.tofile = self._save_all(self.tofile)
                file.write(self.tofile)
            print("File Created Successfully")
        except EnvironmentError:
            print("Error")
    # Parameters are set (from the GUI_Setup file) before saving the whole file        
    def set_parameters(self,a):
        self.parameters = copy.deepcopy(a)    
    def set_conf_3D(self,a):        
        self.conf_3D = copy.deepcopy(a)         
    def set_conf_controller(self,a):        
        self.conf_controller = copy.deepcopy(a)
    def set_conf_SITL(self,a):        
        self.conf_SITL = copy.deepcopy(a)
    def set_conf_plots(self,a):        
        self.conf_plots = copy.deepcopy(a)
    def set_rocket_dim(self,a):        
        self.rocket_dim = copy.deepcopy(a)
    
    def save_all_configurations(self):
        try:
            with open(".\\saves\\" + self.name + ".txt","w", encoding="utf-8") as file:        
                self.tofile = ""                
                self.tofile = self._save_all(self.tofile)
                file.write(self.tofile)
                print("Configuration Saved Successfully")
        except EnvironmentError:
            print("Error")  

    def split_list(self,content,split_index):
        # splits the list in the selected indexes        
        res = [content[i : j] for i, j in zip([0] + split_index, split_index + [None])]
        # Deletes the # that's left from the ###=# separator
        del res[1][0]
        del res[2][0]
        del res[3][0]
        del res[4][0]
        del res[5][0]
        return res    
    def read_file(self):
        try:
            with open(".\\saves\\" + self.name + ".txt","r") as file:     
                content = []
                split_index=[]                
                for line in file:
                    try:
                        content.append(line.split("=")[1].strip())
                    except IndexError:
                        # For the rocket Dimensions
                        content.append(line.split("=")[0].strip())
                for i in range(len(content)):
                    if content[i]=="#":
                        # where to cut the list to send to each tab
                        split_index.append(i)    
                res = self.split_list(content,split_index)                
                self.parameters = res[0]
                self.conf_3D = res[1]
                self.conf_controller = res[2]
                self.conf_SITL = res[3]
                self.conf_plots = res[4]
                self.rocket_dim = res[5]
            print("File Opened Successfully")
        except EnvironmentError:
            print("Error")
            
    def get_parameters(self):        
        return copy.deepcopy(self.parameters)    
    def get_conf_3D(self):        
        return copy.deepcopy(self.conf_3D)        
    def get_conf_controller(self):        
        return copy.deepcopy(self.conf_controller)    
    def get_conf_SITL(self):        
        return copy.deepcopy(self.conf_SITL)
    def get_conf_plots(self):        
        return copy.deepcopy(self.conf_plots)
    def get_rocket_dim(self):        
        return copy.deepcopy(self.rocket_dim)
    
    def read_motor_data(self,name):
        # To make sure it starts form t=0 and Thrust = 0
        self.t_mot = [0]
        self.y_mot = [0]
        try:
            with open(".\\motors\\" + name,"r", encoding="utf-8") as file:                        
                for line in file:                
                    try:
                        a = float(line.split(",")[0])
                        b = float(line.split(",")[1])
                        self.t_mot.append(a)
                        self.y_mot.append(b)
                    except ValueError:
                        pass                
            print("Motor Opened Successfully")
        except EnvironmentError:
            print("Error Reading Motor")
            
    def get_motor_data(self):
        return [copy.deepcopy(self.t_mot), copy.deepcopy(self.y_mot)]
        
    
    

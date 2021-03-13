# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 18:28:36 2021

@author: Guido di Pasquo
"""

"""
Handles the savefile and motor file.

Methods:
    get_save_names -- Returns save names.
    get_motor_names -- Returns available motor names.

Classes:
    SaveFile -- Handles all the rocket's data.
"""

import os
import copy


def get_save_names():
    """
    Returns a list with the names of the files in the folder .\\saves.

    Returns
    -------
    List of strings
        Save files names.
    """
    filenames = os.listdir(".\\saves")
    for i, filename in enumerate(filenames):
        # Removes the .txt
        filenames[i] = filename[:-4]
    return copy.deepcopy(filenames)


def get_motor_names():
    """
    Returns a list with the names of the motor files in the folder .\\motors.

    Returns
    -------
    List of strings
        Motor names.
    """
    return os.listdir(".\\motors")


class SaveFile:
    """
    Save file class. Handles the opening, reading and writing of the .txt's
    used to store the rocket and motor data.

    Methods:
        update_name -- Update the name of the savefile instance.
        check_if_file_exists2overwrite -- Check if the file "n" exists.
        create_file -- Creates a file with default configurations.
        create_file_as -- Creates a file with the GUI's configurations
        save_all_configurations - Saves the GUI's config into the file.
        read_file -- Reads the file.
        read_motor_data -- Reads the motor file.
        get_motor_data -- Returns the motor data.
    """

    def __init__(self):
        self.parameter_names = ["Motor[N] = ", "Mass [kg] = ", "Iy [kg*m] = ",
                                "Xcg [m] = ", "Xt [m] = ", "Servo definition [º] = ",
                                "Max Actuator Angle [º] = ", "Actuator Reduction = ",
                                "Initial Misalignment [º] = ", "Servo Compensation = ",
                                "Wind [m/s] = ", "Wind Gust = "]
        self.conf_3d_names = ["###=#", "toggle_3D=", "camera_shake_toggle=",
                              "hide_forces=", "variable_fov=", "Camera_type=",
                              "slow_mo=", "force_scale=", "fov="]
        self.conf_controller_names = ["###=#", "Torque Controller = ", "Anti Windup = ",
                                      "Input Type = ", "Kp = ", "Ki = ", "Kd = ",
                                      "K All = ", "K Damping = ", "Reference Thrust [N] = ",
                                      "Input = ", "Input time = ", "Launch Time = ",
                                      "Servo Sample Time [s] = ", "Controller Sample Time [s] = ",
                                      "Maximum Sim Duration [s] = ", "Sim Delta T [s] = "]
        self.conf_sitl_names = ["###=#", "Activate_SITL=", "Use Sensor Noise=",
                                "Port=", "Baudrate=", "Gyroscope SD=",
                                "Accelerometer SD=", "Altimeter SD="]
        self.conf_plot_names = ["###=#", "Frist_plot=", "Second_plot=", "Third_plot=",
                                "Fourth_plot=", "Fifth_plot="]
        self.rocket_dim_names = ["###=#"]
        self.name = ""
        self.parameters = []
        self.conf_3d = []
        self.conf_controller = []
        self.conf_sitl = []
        self.conf_plots = []
        self.rocket_dim = []
        self.tofile = ""
        self.t_mot = []
        self.thrust_mot = []
        self.overwrite_flag = False

    def update_name(self,n):
        """Update the name of the savefile instance (not the actual file) to n."""
        self.name = n

    def _save_parameters(self, tofile):
        for i in range(len(self.parameter_names)):
            tofile += self.parameter_names[i] + self.parameters[i] + "\n"
        return tofile

    def _save_conf_3d(self, tofile):
        for i in range(len(self.conf_3d_names)):
            if i == 0:
                tofile += "###=#\n"
            else:
                tofile += self.conf_3d_names[i] + self.conf_3d[i-1] + "\n"
        return tofile

    def _save_conf_sitl(self, tofile):
        for i in range(len(self.conf_sitl_names)):
            if i == 0:
                tofile += "###=#\n"
            else:
                tofile += self.conf_sitl_names[i] + self.conf_sitl[i-1] + "\n"
        return tofile

    def _save_conf_controller(self, tofile):
        for i in range(len(self.conf_controller_names)):
            if i == 0:
                tofile += "###=#\n"
            else:
                tofile += self.conf_controller_names[i] + self.conf_controller[i-1] + "\n"
        return tofile

    def _save_conf_plots(self, tofile):
        for i in range(len(self.conf_plot_names)):
            if i == 0:
                tofile += "###=#\n"
            else:
                tofile += self.conf_plot_names[i] + self.conf_plots[i-1] + "\n"
        return tofile

    def _save_rocket_dim(self, tofile):
        tofile += "###=#\n"
        for i in range(len(self.rocket_dim)):
            tofile += self.rocket_dim[i] + "\n"
        return tofile

    def _save_all(self, tofile):
        tofile = self._save_parameters(tofile)
        tofile = self._save_conf_3d(tofile)
        tofile = self._save_conf_controller(tofile)
        tofile = self._save_conf_sitl(tofile)
        tofile = self._save_conf_plots(tofile)
        tofile = self._save_rocket_dim(tofile)
        return tofile

    def check_if_file_exists2overwrite(self, n):
        """
        Check if the file "n" exists and decide if it will be overwritten
        or not.

        Parameters
        ----------
        n : string
            File name.

        Returns
        -------
        Bool
            Flag to overwrite the file or not.
        """
        self.overwrite_flag = False
        names = get_save_names()
        flag_found_name = False
        for name in names:
            if name == n:
                x = ""
                flag_found_name = True
                while x != "y" or x != "n":
                    x = str(input("File already exists, overwrite? [y/n]"))
                    if x == "y":
                        self.overwrite_flag = True
                        return self.overwrite_flag
                    if x == "n":
                        self.overwrite_flag = False
                        return self.overwrite_flag
                    print("Insert valid option")
        if flag_found_name is False:
            self.overwrite_flag = True
            return self.overwrite_flag
        return None

    def create_file(self, n):
        """
        Create a file named "n" with default parameters.

        Parameters
        ----------
        n : string
            File name.

        Returns
        -------
        None.
        """
        self.update_name(n)
        self.parameters = ["Estes_D12.csv", '0.451', '0.0662', '0.55',"0.85",
                           '1', '10', '5', '2',"2.1", '2', '0.1']
        self.conf_3d = ['True', 'False',"False","False","Fixed", '3', '0.2', "0.75"]
        self.conf_controller = ['False', 'True', 'Step [º]', '0.4', '0',
                                '0.136', '1', '0', '30','20',"0.5","1", '0.02',
                                '0.01', "30", "0.001"]
        self.conf_sitl = ["False","False","COM3","115200","0","0","0"]
        self.conf_plots = ["Pitch Angle", "Setpoint", "Actuator deflection",
                           "Off", "Off"]
        self.rocket_dim = ["True","False","True","False","False",
                           "0,0","0.2,0.066","1,0.066",
                           "Fins_s",
                           "0.0000,0.0000","0.0001,0.0001","0.0002,0.0001","0.0002,0.0000",
                           "Fins_c",
                           "0.0000,0.0000","0.0001,0.0001","0.0002,0.0001","0.0002,0.0000"]
        try:
            with open(".\\saves\\"+self.name+".txt", "w", encoding="utf-8") as file:
                self.tofile = ""
                self.tofile = self._save_all(self.tofile)
                file.write(self.tofile)
            print("File Created Successfully")
        except EnvironmentError:
            print("Error")

    def create_file_as(self, n):
        """
        Create a file named "n" with the data saved in the GUI.

        Parameters
        ----------
        n : string
            File name.

        Returns
        -------
        None.
        """
        self.update_name(n)
        try:
            with open(".\\saves\\"+self.name+".txt", "w", encoding="utf-8") as file:
                self.tofile = ""
                self.tofile = self._save_all(self.tofile)
                file.write(self.tofile)
            print("File Created Successfully")
        except EnvironmentError:
            print("Error")
    # Parameters are set (from the GUI_Setup file) before saving the whole file
    def set_parameters(self, data):
        self.parameters = copy.deepcopy(data)
    def set_conf_3d(self, data):
        self.conf_3d = copy.deepcopy(data)
    def set_conf_controller(self, data):
        self.conf_controller = copy.deepcopy(data)
    def set_conf_sitl(self, data):
        self.conf_sitl = copy.deepcopy(data)
    def set_conf_plots(self, data):
        self.conf_plots = copy.deepcopy(data)
    def set_rocket_dim(self, data):
        self.rocket_dim = copy.deepcopy(data)

    def save_all_configurations(self):
        """
        Save the configuration in the GUI to the active file.

        Returns
        -------
        None.
        """
        try:
            with open(".\\saves\\"+self.name+".txt", "w", encoding="utf-8") as file:
                self.tofile = ""
                self.tofile = self._save_all(self.tofile)
                file.write(self.tofile)
                print("Configuration Saved Successfully")
        except EnvironmentError:
            print("Error")

    def _split_list(self, content, split_index):
        # splits the list in the selected indexes
        res = [content[i : j] for i, j in zip([0]+split_index, split_index+[None])]
        # Deletes the # that's left from the ###=# separator
        del res[1][0]
        del res[2][0]
        del res[3][0]
        del res[4][0]
        del res[5][0]
        return res
    def read_file(self):
        """
        Set the Savefile instance parameters to the ones on the save file.

        Returns
        -------
        None.
        """
        try:
            with open(".\\saves\\"+self.name+".txt", "r", encoding="utf-8") as file:
                content = []
                split_index=[]
                for line in file:
                    try:
                        content.append(line.split("=")[1].strip())
                    except IndexError:
                        # For the rocket Dimensions
                        content.append(line.split("=")[0].strip())
                for i, element in enumerate(content):
                    if element == "#":
                        # where to cut the list to send to each tab
                        split_index.append(i)
                res = self._split_list(content,split_index)
                self.parameters = res[0]
                self.conf_3d = res[1]
                self.conf_controller = res[2]
                self.conf_sitl = res[3]
                self.conf_plots = res[4]
                self.rocket_dim = res[5]
            print("File Opened Successfully")
        except EnvironmentError:
            print("Error")

    def get_parameters(self):
        return copy.deepcopy(self.parameters)
    def get_conf_3d(self):
        return copy.deepcopy(self.conf_3d)
    def get_conf_controller(self):
        return copy.deepcopy(self.conf_controller)
    def get_conf_sitl(self):
        return copy.deepcopy(self.conf_sitl)
    def get_conf_plots(self):
        return copy.deepcopy(self.conf_plots)
    def get_rocket_dim(self):
        return copy.deepcopy(self.rocket_dim)

    def read_motor_data(self, name):
        """
        Load motor data into the Savefile instance.

        Parameters
        ----------
        name : string.
            Save file name.

        Returns
        -------
        None.
        """
        # To make sure it starts form t=0 and Thrust = 0
        self.t_mot = [0]
        self.thrust_mot = [0]
        try:
            with open(".\\motors\\"+name, "r", encoding="utf-8") as file:
                for line in file:
                    try:
                        a = float(line.split(",")[0])
                        b = float(line.split(",")[1])
                        self.t_mot.append(a)
                        self.thrust_mot.append(b)
                    except ValueError:
                        pass
            print("Motor Opened Successfully")
        except EnvironmentError:
            print("Error Reading Motor")

    def get_motor_data(self):
        """
        Get motor data as a list of [time, thrust]

        Returns
        -------
        list
            thrust data.
        """
        return [copy.deepcopy(self.t_mot), copy.deepcopy(self.thrust_mot)]

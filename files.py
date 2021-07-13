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
import gui_functions


def get_save_names():
    r"""
    Return a list with the names of the files in the folder .\\saves.

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
    r"""
    Return a list with the names of the motor files in the folder .\\motors.

    Returns
    -------
    List of strings
        Motor names.
    """
    return os.listdir(".\\motors")


def get_sitl_modules_names():
    r"""
    Return a list with the names of the motor files in the folder .\\motors.

    Returns
    -------
    List of strings
        SITL modules names.
    """
    filenames = os.listdir(".\\SITL Modules")
    names = []
    for i, filename in enumerate(filenames):
        if filename[-1] == "y":
            names.append(filename[:-3])
    return copy.deepcopy(names)


class SaveFile:
    """
    Save file class. Handles the opening, reading and writing of the .txt's
    used to store the rocket and motor data.

    Methods
    -------
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
        self.parameter_names = ["Motor = ",
                                "Mass Liftoff [kg] = ",
                                "Mass Burnout [kg] = ",
                                "Iy Liftoff [kg*m] = ",
                                "Iy Burnout [kg*m] = ",
                                "Xcg Liftoff [m] = ",
                                "Xcg Burnout [m] = ",
                                "Xt [m] = ",
                                "Servo definition [º] = ",
                                "Max Actuator Angle [º] = ",
                                "Actuator Reduction = ",
                                "Initial Misalignment [º] = ",
                                "Servo Compensation = ",
                                "Wind [m/s] = ",
                                "Wind Gust = ",
                                "Launch Rod Length = ",
                                "Launch Rod Angle [º] = ",
                                "Motor Misalignment [º] = "]
        self.conf_3d_names = ["###=#",
                              "toggle_3D=",
                              "camera_shake_toggle=",
                              "hide_forces=",
                              "variable_fov=",
                              "hide_cg=",
                              "Camera_type=",
                              "slow_mo=",
                              "force_scale=",
                              "fov="]
        self.conf_controller_names = ["###=#",
                                      "Torque Controller = ",
                                      "Anti Windup = ",
                                      "Input Type = ",
                                      "Kp = ",
                                      "Ki = ",
                                      "Kd = ",
                                      "K All = ",
                                      "K Damping = ",
                                      "Reference Thrust [N] = ",
                                      "Input = ",
                                      "Input time = ",
                                      "Launch Time = ",
                                      "Servo Sample Time [s] = ",
                                      "Controller Sample Time [s] = ",
                                      "Maximum Sim Duration [s] = ",
                                      "Sim Delta T [s] = "]
        self.conf_sitl_names = ["###=#",
                                "Activate_SITL=",
                                "Use Sensor Noise=",
                                "Python SITL=",
                                "File=",
                                "Port=",
                                "Baudrate=",
                                "Gyroscope SD=",
                                "Accelerometer SD=",
                                "Altimeter SD=",
                                "GNSS Pos SD=",
                                "GNSS Vel SD=",
                                "Gyroscope ST=",
                                "Accelerometer ST=",
                                "Altimeter ST=",
                                "GNSS ST="]
        self.conf_plot_names = ["###=#",
                                "Frist_plot=",
                                "Second_plot=",
                                "Third_plot=",
                                "Fourth_plot=",
                                "Fifth_plot="]
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

    def update_name(self, n):
        """Update the name of the savefile instance (not the actual file) to n."""
        self.name = n

    def _save_all(self, tofile):
        tofile = self._save_parameters(tofile)
        tofile = self._save_conf_3d(tofile)
        tofile = self._save_conf_controller(tofile)
        tofile = self._save_conf_sitl(tofile)
        tofile = self._save_conf_plots(tofile)
        tofile = self._save_rocket_dim(tofile)
        return tofile

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

    def _save_conf_controller(self, tofile):
        for i in range(len(self.conf_controller_names)):
            if i == 0:
                tofile += "###=#\n"
            else:
                tofile += self.conf_controller_names[i] + self.conf_controller[i-1] + "\n"
        return tofile

    def _save_conf_sitl(self, tofile):
        for i in range(len(self.conf_sitl_names)):
            if i == 0:
                tofile += "###=#\n"
            else:
                tofile += self.conf_sitl_names[i] + self.conf_sitl[i-1] + "\n"
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
        self.parameters = ["Estes_D12.csv",
                           "0.451",
                           "0.351",
                           "0.0662",
                           "0.0601",
                           "0.55",
                           "0.51",
                           "0.85",
                           "1",
                           "10",
                           "5",
                           "2",
                           "2.1",
                           "2",
                           "0.1",
                           "0",
                           "0",
                           "0"]
        self.conf_3d = ["True",
                        "False",
                        "False",
                        "False",
                        "True",
                        "Fixed",
                        "3",
                        "0.2",
                        "0.75"]
        self.conf_controller = ["False",
                                "True",
                                "Step [º]",
                                "0.4",
                                "0",
                                "0.136",
                                "1",
                                "0",
                                "30",
                                "20",
                                "0.5",
                                "0",
                                "0.02",
                                "0.01",
                                "30",
                                "0.001"]
        self.conf_sitl = ["False",
                          "False",
                          "False",
                          "example_python_sitl",
                          "COM3",
                          "115200",
                          "0",
                          "0",
                          "0",
                          "0",
                          "0",
                          "0.0025",
                          "0.0025",
                          "0.005",
                          "1"]
        self.conf_plots = ["Pitch Angle [º]",
                           "Setpoint [º]",
                           "Actuator deflection [º]",
                           "Off",
                           "Off"]
        self.rocket_dim = ["True",
                           "False",
                           "True",
                           "False",
                           "False",
                           "0,0",
                           "0.2,0.066",
                           "1,0.066",
                           "Fins_s",
                           "0, 0",
                           "0, 0",
                           "0",
                           "0",
                           "Fins_c",
                           "0, 0",
                           "0, 0",
                           "0",
                           "0"]

        try:
            with open(".\\saves\\"+self.name+".txt", "w", encoding="utf-8") as file:
                self.tofile = ""
                self.tofile = self._save_all(self.tofile)
                file.write(self.tofile)
            print("File Created Successfully")
        except EnvironmentError:
            print("Error Creating File")

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
        res = [content[i: j] for i, j in zip([0]+split_index, split_index+[None])]
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
        self.open_and_split_file()
        self.check_and_correct_v11_save()
        self.open_and_split_file()
        self.check_and_correct_v20_save()
        self.open_and_split_file()
        if self.error_opening_file_flag is False:
            print("File Opened Successfully")

    def open_and_split_file(self):
        try:
            with open(".\\saves\\"+self.name+".txt", "r", encoding="utf-8") as file:
                content = []
                split_index = []
                self.raw_data = []
                for line in file:
                    self.raw_data.append(line)
                    try:
                        content.append(line.split("=")[1].strip())
                    except IndexError:
                        # For the rocket Dimensions
                        content.append(line.split("=")[0].strip())
                for i, element in enumerate(content):
                    if element == "#":
                        # where to cut the list to send to each tab
                        split_index.append(i)
                res = self._split_list(content, split_index)
                self.parameters = res[0]
                self.conf_3d = res[1]
                self.conf_controller = res[2]
                self.conf_sitl = res[3]
                self.conf_plots = res[4]
                self.rocket_dim = res[5]
                self.error_opening_file_flag = False
        except EnvironmentError:
            print("EnvironmentError Opening File")
            self.error_opening_file_flag = True

    def check_and_correct_v11_save(self):
        if self.check_file("Wind Gust =", "###"):
            try:
                with open(".\\saves\\"+self.name+".txt", "w", encoding="utf-8") as file:
                    end = "\n"
                    for i, line in enumerate(self.raw_data):
                        file.write(line)
                        if line.startswith("Wind Gust"):
                            file.write("Launch Rod Length [m] = 0" + end)
                            file.write("Launch Rod Angle [º] = 0" + end)
                        elif line.startswith("Use Sensor Noise"):
                            file.write("Python SITL=True" + end)
                        elif line.startswith("Altimeter SD="):
                            file.write("GNSS Pos SD=0" + end)
                            file.write("GNSS Vel SD=0" + end)
                            file.write("Gyroscope ST=0.0025" + end)
                            file.write("Accelerometer ST=0.0025" + end)
                            file.write("Altimeter ST=0.005" + end)
                            file.write("GNSS ST=1" + end)
                print("Save updated to v2.0")
            except EnvironmentError:
                print("Error updating to v2.0")

    def check_and_correct_v20_save(self):
        if self.check_file("Mass [kg]", "Iy"):
            try:
                with open(".\\saves\\"+self.name+".txt", "w", encoding="utf-8") as file:
                    counter = 4
                    end = "\n"
                    for i, line in enumerate(self.raw_data):
                        if line.startswith("Mass"):
                            mass = self.parameters[1]
                            file.write("Mass Liftoff [kg] = " + mass + end)
                            file.write("Mass Burnout [kg] = " + mass + end)
                        elif line.startswith("Iy"):
                            Iy = self.parameters[2]
                            file.write("Iy Liftoff [kg*m^2] = " + Iy + end)
                            file.write("Iy Burnout [kg*m^2] = " + Iy + end)
                        elif line.startswith("Xcg"):
                            xcg = self.parameters[3]
                            file.write("Xcg Liftoff [m] = " + xcg + end)
                            file.write("Xcg Burnout [m] = " + xcg + end)
                        elif line.startswith("Launch Rod Angle"):
                            file.write(line)
                            file.write("Motor Misalignment = 0" + end)
                        elif line.startswith("variable_fov"):
                            file.write(line)
                            file.write("hide_cg=True" + end)
                        elif line.startswith("Python SITL="):
                            file.write(line)
                            file.write("File=example_python_sitl" + end)
                        elif line.startswith("Fins_s") or line.startswith("Fins_c"):
                            file.write(line)
                            counter = 0
                            if line.startswith("Fins_s"):
                                n = 9
                            else:
                                n = 4
                            s = [0]*4
                            for i in range(4):
                                s[i] = self.rocket_dim[-(n-i)]
                            l2 = gui_functions.points_2_param_fins(s)
                            for i in range(4):
                                file.write(l2[i] + end)
                        elif counter < 4:
                            counter += 1
                            continue
                        elif (line.startswith("Frist_plot") or
                              line.startswith("Second_plot") or
                              line.startswith("Third_plot") or
                              line.startswith("Fourth_plot") or
                              line.startswith("Fifth_plot")):
                            line_split = line.split("=")[1].strip()
                            if (line_split.startswith("Pitch Angle") or
                                line_split.startswith("Actuator deflection") or
                                line_split.startswith("Angle of Atack") or
                                    line_split.startswith("Setpoint")):
                                s = line.strip() + " [º]" + end
                                file.write(s)
                            elif (line_split.startswith("Pitch Rate") or
                                  line_split.startswith("Simulated Gyro")):
                                s = line.strip() + " [º/s]" + end
                                file.write(s)
                            elif (line_split.startswith("Total Velocity") or
                                  line_split.startswith("Local Velocity") or
                                  line_split.startswith("Global Velocity") or
                                  line_split.startswith("Simulated GNSS Velocity")):
                                s = line.strip() + " [m/s]" + end
                                file.write(s)
                            elif (line_split.startswith("Altitude") or
                                  line_split.startswith("Distance") or
                                  line_split.startswith("CP") or
                                  line_split.startswith("CG") or
                                  line_split.startswith("Force") or
                                  line_split.startswith("Simulated Altimeter") or
                                  line_split.startswith("Simulated GNSS Position")):
                                s = line.strip() + " [m]" + end
                                file.write(s)
                            elif line_split.startswith("Mass"):
                                s = line.strip() + " [kg]" + end
                                file.write(s)
                            elif line_split.startswith("Iy"):
                                s = line.strip() + " [kg*m^2]" + end
                                file.write(s)
                            elif line == "Normal Force":
                                s = line.strip() + " [N]" + end
                                file.write(s)
                            elif (line_split.startswith("Local Acc") or
                                  line_split.startswith("Global Acc") or
                                  line_split.startswith("Simulated Acc")):
                                s = line.strip() + " [m^2/s]" + end
                                file.write(s)
                            else:
                                file.write(line)
                        else:
                            file.write(line)
                    print("Save updated to v2.1")
            except EnvironmentError:
                print("Error updating to v2.1")

    def check_file(self, parameter1, parameter2):
        flag = False
        found_param = False
        try:
            with open(".\\saves\\"+self.name+".txt", "r", encoding="utf-8") as file:
                for line in file:
                    try:
                        if found_param is True:
                            if line.startswith(parameter2):
                                flag = True
                                break
                            else:
                                break
                        if line.startswith(parameter1):
                            found_param = True
                    except IndexError:
                        pass
        except EnvironmentError:
            print("EnvironmentError Opening File")
        return flag

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
        except EnvironmentError:
            print("Error Reading Motor")

    def get_motor_data(self):
        """
        Get motor data as a list of [time, thrust].

        Returns
        -------
        list
            thrust data.
        """
        return [copy.deepcopy(self.t_mot), copy.deepcopy(self.thrust_mot)]

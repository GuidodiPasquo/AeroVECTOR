# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 18:28:36 2021

@author: Guido di Pasquo
"""

import os
import copy
from src.gui import gui_functions
from pathlib import Path
import re
from tkinter import filedialog
import shutil


exports_path = ""
motors_path = Path("Motors/")


"""
Handles the savefile and motor file.

Methods:
    get_save_names -- Returns save names.
    get_motor_names -- Returns available motor names.

Classes:
    SaveFile -- Handles all the rocket's data.
"""

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_export_names(filepath, exports_path):
    r"""
    Return a list with the names of the files in the folder /Exports.

    Returns
    -------
    List of strings
        Save files names.
    """
    if exports_path == "":
        exports_path = filepath
    filenames = natural_sort(os.listdir(exports_path))
    for i, filename in enumerate(filenames):
        # Removes the .csv
        filenames[i] = filename[:-4]
    return copy.deepcopy(filenames)


def get_motor_names():
    r"""
    Return a list with the names of the motor files in the folder /Motors.

    Returns
    -------
    List of strings
        Motor names.
    """
    return natural_sort(os.listdir(motors_path))


def get_sitl_modules_names(filepath):
    r"""
    Return a list with the names of the motor files in the folder /\Motors.

    Returns
    -------
    List of strings
        SITL modules names.
    """
    filenames = []
    path_without_name = [e+"/" for e in filepath.split("/") if e != ""][:-1]
    path_without_name = "".join(path_without_name)
    if path_without_name == "":
        return [""]
    path_without_name = Path(path_without_name + "/SITL Modules")
    try:
        filenames = os.listdir(path_without_name)
    except FileNotFoundError:
        print("WARNING: Could not find the folders 'SITL Modules' and 'SITL Modules/Complementary Modules'.")
    filenames = natural_sort(filenames)
    names = []
    for i, filename in enumerate(filenames):
        if filename[-3:] == ".py":
            names.append(filename[:-3])
    return copy.deepcopy(names)


def export_plots(file_name, filepath, names, data, T):
    global exports_path
    file_name += "_0"
    export_names = get_export_names(filepath, exports_path)

    number_found = False
    counter = 0
    while number_found is False:  # Check for the file and add _1, _2, etc if they exist
        for name in export_names:
            if name == file_name:
                counter += 1
                if counter <= 10:
                    n = 1
                else:
                    n = 2
                file_name = file_name[:-n] + str(counter)
                continue
        number_found = True

    if exports_path == "":
        exports_path = filepath
    exports_path_total = filedialog.asksaveasfilename(initialdir=exports_path,
                                                      initialfile=file_name,
                                                      defaultextension=".csv",
                                                      title="Export Data",
                                                      filetypes=[("Export File", ".csv"),
                                                                 ("All Files", ".*")])

    if exports_path_total == "":
        return
    path_without_name = [e+"/" for e in exports_path_total.split("/") if e != ""][:-1]
    exports_path = "".join(path_without_name)
    if exports_path_total == "":
        return
    to_file = ""
    prev_time = 0
    for i in range(len(data[0])):
        line = ""
        if i == 0:
            for j in range(len(names)):
                if j == len(names)-1:
                    end = "\n"
                else:
                    end = ","
                line += names[j] + end
        else:
            if data[0][i] > T*0.999 + prev_time:
                for j in range(len(names)):
                    if j == len(names)-1:
                        end = "\n"
                    else:
                        end = ","
                    line += str(data[j][i]) + end
                prev_time = data[0][i]
        to_file += line
    try:
        with open(exports_path_total, "w", encoding="utf-8") as file:
            file.write(to_file)
        print("Data Exported Successfully")
    except EnvironmentError:
        print("Error Exporting Data")


class SaveFile:
    """
    Save file class. Handles the opening, reading and writing of the .txt's
    used to store the rocket and motor data.

    Methods
    -------
        update_path -- Update the path of the savefile instance.
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
                                "Mass Liftoff = ",
                                "Mass Burnout = ",
                                "Iy Liftoff = ",
                                "Iy Burnout = ",
                                "Xcg Liftoff = ",
                                "Xcg Burnout = ",
                                "Xt = ",
                                "Servo Resolution = ",
                                "Max Actuator Angle = ",
                                "Actuator Reduction = ",
                                "Initial Misalignment = ",
                                "Servo Compensation = ",
                                "Wind = ",
                                "Wind Gust = ",
                                "Launch Rod Length = ",
                                "Launch Rod Angle = ",
                                "Motor Misalignment = ",
                                "Rocket Roughness = ",
                                "Stabilization Fin Roughness = ",
                                "Control Fin Roughness = "]
        self.conf_3d_names = ["###=#",
                              "Toggle 3D = ",
                              "Camera Shake Toggle = ",
                              "Hide Forces = ",
                              "Variable FOV = ",
                              "Hide cg = ",
                              "Camera Type = ",
                              "Slow mo = ",
                              "Force Scale = ",
                              "FOV = "]
        self.conf_controller_names = ["###=#",
                                      "Torque Controller = ",
                                      "Anti Windup = ",
                                      "Input Type = ",
                                      "Kp = ",
                                      "Ki = ",
                                      "Kd = ",
                                      "K All = ",
                                      "K Damping = ",
                                      "Reference Thrust = ",
                                      "Input = ",
                                      "Input time = ",
                                      "Launch Time = ",
                                      "Servo Sample Time = ",
                                      "Controller Sample Time = ",
                                      "Maximum Sim Duration = ",
                                      "Sim Delta T = ",
                                      "Export T = ",
                                      "Launch Altitude = ",
                                      "Initial Altitude = ",
                                      "Initial Vertical Velocity = ",
                                      "Initial Horizontal Velocity = ",
                                      "Initial Pitch Angle = ",
                                      "Initial Pitch Rate = "]
        self.conf_sitl_names = ["###=#",
                                "Activate SITL = ",
                                "Use Sensor Noise = ",
                                "Python SITL=",
                                "File = ",
                                "Port = ",
                                "Baudrate = ",
                                "Gyroscope SD = ",
                                "Accelerometer SD = ",
                                "Altimeter SD = ",
                                "GNSS Pos SD = ",
                                "GNSS Vel SD = ",
                                "Gyroscope ST = ",
                                "Accelerometer ST = ",
                                "Altimeter ST = ",
                                "GNSS ST = "]
        self.conf_plot_names = ["###=#",
                                "First Plot = ",
                                "Second Plot = ",
                                "Third Plot = ",
                                "Fourth Plot = ",
                                "Fifth Plot = ",
                                "Sixth Plot = ",
                                "Seventh Plot = ",
                                "Eighth Plot = ",
                                "Ninth Plot = ",
                                "Tenth Plot = "]
        self.rocket_dim_names = ["###=#"]
        self.name = ""
        self.filepath = ""
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
        self.template_sitl = """
from src import python_sitl_functions as Sim
import importlib
import pathlib


def import_module(module):
    current_path = pathlib.Path(__file__).parent.resolve()
    module_temp = pathlib.Path(current_path / "Complementary Modules" / module)
    spec = importlib.util.spec_from_file_location(module, module_temp)
    module_temp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_temp)
    return module_temp


class SITLProgram:
    def __init__(self):
        pass

    ''' Available funtions, called with Sim.
    Sim.millis(), Sim.micros(),
    gyro, accx, accz, alt, pos_gnss, vel_gnss = Sim.getSimData()
    Sim.sendCommand(servo, parachute)
    Sim.plot_variable(variable, number) (from 1 to 5 for diferent plots)
    -->
    -->
    -->
    -->
    -->
    -->
    -->
    -->
    -->
    -->
    '''

    def everything_that_is_outside_functions(self):
        self.alt_prev = 0
        self.timer_all = 0
        self.sample_time_program = 0.1






    def void_setup(self):
        pass







    def void_loop(self):
        self.t = Sim.micros()/1000000
        if self.t >= self.timer_all + self.sample_time_program*0.999:
            self.gyro, self.accx, self.accz, self.alt, self.pos_gnss, self.vel_gnss = Sim.getSimData()
            parachute = self.parachute_deployment()
            servo = 0
            Sim.sendCommand(servo, parachute)






    '''########'''

    def parachute_deployment(self):
        if self.alt < self.alt_prev and self.alt > 10:
            return 1
        else:
            self.alt_prev = self.alt
            return 0

    """

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
        self.update_path(n)
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
                           "0",
                           "60",
                           "60",
                           "60"]
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
                                "0.003",
                                "0.1",
                                "0",
                                "0",
                                "0",
                                "0",
                                "0",
                                "0"]
        self.conf_sitl = ["False",
                          "False",
                          "False",
                          "",
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
                           "Off",
                           "Off",
                           "Off",
                           "Off",
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
            with open(self.filepath, "w", encoding="utf-8") as file:
                self.tofile = ""
                self.tofile = self._save_all(self.tofile)
                file.write(self.tofile)
            print("File Created Successfully")
        except EnvironmentError:
            print("Error Creating File")
        self._create_sitl_directories()
        self._create_sitl_template()

    def update_path(self, n):
        """Update the path of the savefile instance (not the actual file) to n."""
        self.filepath = n
        path_without_name = [e+"/" for e in n.split("/") if e != ""][:-1]
        self.filepath_without_name = "".join(path_without_name)
        self.name = n.split("/")[-1][:-4]

    def _create_sitl_directories(self):
        os.makedirs(self.filepath_without_name + 'SITL Modules/Complementary Modules',
                    exist_ok=True)

    def _create_sitl_template(self):
        filepath = Path(self.filepath_without_name + "/SITL Modules/template.py")
        try:
            with open(filepath, "w", encoding="utf-8") as file:
                file.write(self.template_sitl)
            print("SITL Template Created Successfully")
        except EnvironmentError:
            print("Error Creating SITL Template")

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
        self.old_filepath_without_name = self.filepath_without_name
        self.update_path(n)
        try:
            with open(self.filepath, "w", encoding="utf-8") as file:
                self.tofile = ""
                self.tofile = self._save_all(self.tofile)
                file.write(self.tofile)
            print("File Created Successfully")
        except EnvironmentError:
            print("Error Creating File")
        self._copy_sitl_files()

    def _copy_sitl_files(self):
        src = self.old_filepath_without_name + "SITL Modules"
        dst = self.filepath_without_name + "SITL Modules"
        if src != dst:
            shutil.copytree(src, dst, dirs_exist_ok=True)

    def save_all_configurations(self, saved_thing):
        """
        Save the configuration in the GUI to the active file.

        Returns
        -------
        None.
        """
        try:
            with open(self.filepath, "w", encoding="utf-8") as file:
                self.tofile = ""
                self.tofile = self._save_all(self.tofile)
                file.write(self.tofile)
                print(saved_thing + " Saved Successfully")
        except EnvironmentError:
            print("Error Saving File (EnvironmentError)")

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
        update_old_files_experimental = False
        if update_old_files_experimental is True:
            self.open_and_split_file()
            self.check_and_correct_v11_save()
            self.open_and_split_file()
            self.check_and_correct_v20_save()
        self.open_and_split_file()
        if self.error_opening_file_flag is False:
            print("File Opened Successfully")

    def open_and_split_file(self):
        try:
            with open(self.filepath, "r", encoding="utf-8") as file:
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
                with open(self.filepath, "w", encoding="utf-8") as file:
                    end = "\n"
                    for i, line in enumerate(self.raw_data):
                        file.write(line)
                        if line.startswith("Wind Gust"):
                            file.write("Launch Rod Length = 0" + end)
                            file.write("Launch Rod Angle = 0" + end)
                        elif line.startswith("Use Sensor Noise"):
                            file.write("Python SITL = True" + end)
                        elif line.startswith("Altimeter SD = "):
                            file.write("GNSS Pos SD = 0" + end)
                            file.write("GNSS Vel SD = 0" + end)
                            file.write("Gyroscope ST = 0.0025" + end)
                            file.write("Accelerometer ST = 0.0025" + end)
                            file.write("Altimeter ST = 0.005" + end)
                            file.write("GNSS ST = 1" + end)
                print("Save updated to v2.0")
            except EnvironmentError:
                print("Error updating to v2.0")

    def check_and_correct_v20_save(self):
        if self.check_file("Mass =", "Iy"):
            try:
                with open(self.filepath, "w", encoding="utf-8") as file:
                    counter = 4
                    end = "\n"
                    for i, line in enumerate(self.raw_data):
                        if line.startswith("Mass"):
                            mass = self.parameters[1]
                            file.write("Mass Liftoff = " + mass + end)
                            file.write("Mass Burnout = " + mass + end)
                        elif line.startswith("Iy"):
                            Iy = self.parameters[2]
                            file.write("Iy Liftoff = " + Iy + end)
                            file.write("Iy Burnout = " + Iy + end)
                        elif line.startswith("Xcg"):
                            xcg = self.parameters[3]
                            file.write("Xcg Liftoff = " + xcg + end)
                            file.write("Xcg Burnout = " + xcg + end)
                        elif line.startswith("Launch Rod Angle"):
                            file.write(line)
                            file.write("Motor Misalignment = 0" + end)
                            file.write("Roughness = 60" + end)
                            file.write("Roughness = 60" + end)
                            file.write("Roughness = 60" + end)
                        elif line.startswith("variable_fov"):
                            file.write(line)
                            file.write("Hide cg = True" + end)
                        elif line.startswith("Python SITL = "):
                            file.write(line)
                            file.write("File=example_python_sitl" + end)
                        elif line.startswith("Sim Delta"):
                            file.write(line)
                            file.write("Launch Altitude = 0" + end)
                            file.write("Initial Altitude = 0" + end)
                            file.write("Initial Vertical Velocity = 0" + end)
                            file.write("Initial Horizontal Velocity = 0" + end)
                            file.write("Initial Pitch Angle = 0" + end)
                            file.write("Initial Pitch Rate = 0" + end)
                            file.write("Export T = 0.003" + end)

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
                        elif (line.startswith("First Plot") or
                              line.startswith("Second Plot") or
                              line.startswith("Third Plot") or
                              line.startswith("Fourth Plot") or
                              line.startswith("Fifth Plot")):
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
                            if line.startswith("Fifth Plot"):
                                file.write("Sixth Plot = Off" + end)
                                file.write("Seventh Plot = Off" + end)
                                file.write("Eighth Plot = Off" + end)
                                file.write("Ninth Plot = Off" + end)
                                file.write("Tenth Plot = Off" + end)
                        else:
                            file.write(line)
                    print("Save updated to v2.1")
            except EnvironmentError:
                print("Error updating to v2.1")

    def check_file(self, parameter1, parameter2):
        flag = False
        found_param = False
        try:
            with open(self.filepath, "r", encoding="utf-8") as file:
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
            with open(motors_path / name, "r", encoding="utf-8") as file:
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

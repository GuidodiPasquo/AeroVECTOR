# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 16:21:38 2021

@author: Guido di Pasquo
"""

"""
Functions necessary to make the GUI.

    Functions:
        move_tk_object -- Move a tk object.

    Classes:
        ActiveFileLabel -- Active file label on the bottom left.
        Tab -- Basic tab.
        TabWithCanvas -- Tab with a canvas.
"""

import copy
import tkinter as tk
from tkinter import ttk
import numpy as np
import gui_setup
import rocket_functions


# Order in the savefile:
# Checkboxes
# combobox
# entry


def move_tk_object(obj, r=0, c=0, columnspan=1):
    """
    Moves a tk object to the selected position.

    Parameters
    ----------
    obj : tk object
        Object to move.
    r : int, optional
        move to row r. The default is 0.
    c : int, optional
        move to column c. The default is 0.
    columnspan : int, optional
        columnspan of the object. The default is 1.

    Returns
    -------
    None.
    """
    obj.grid(row=r, column=c, columnspan=columnspan)


class ActiveFileLabel:
    """
    Creates a label with the active file name.

    Methods:
        create_label -- Creates the label.
        update -- Updates the label.

    """

    def __init__(self):
        self.r = 0
        self.c = 0

    def create_label(self, root, r, c):
        """
        Creates the label in the canvas root, in the row = r and column = c

        Parameters
        ----------
        root : tk canvas
            canvas where to create a label.
        r : int
            row (position).
        c : int
            column (position).

        Returns
        -------
        None.
        """
        self.r = r
        self.c = c
        self.canvas = root
        self.text = tk.StringVar()
        self.active_file_label = tk.Label(self.canvas, textvariable=self.text)
        self.active_file_label.grid(row=r,column=c,sticky="W", columnspan=2)

    def update(self, active_file):
        """
        Update the label with the new file name

        Parameters
        ----------
        active_file : string
            File name.

        Returns
        -------
        None.
        """
        self.text.set("Active File is " + active_file)
        active_file_label = tk.Label(self.canvas, textvariable=self.text)
        active_file_label.grid(row=self.r, column=self.c, sticky="W", columnspan=2)


class Tab:
    """
    Basic tab template with checkboxes, combobox and entries.

    Methods:
        create_tab -- Creates the tab in the notebook.
        create_checkboxes -- Creates them in the tab.
        activate_all -- Activates everything in the tab.
        deactivate_all -- Deactivates everything in the tab.
        change_state -- Activates/Deactivates everything in the tab.
        create_combobox -- Creates comboboxes.
        create_entry -- Creates entries.
        populate -- fills the tab with the data.
        depopulate -- Deletes the data from the tab.
        get_configuration -- Returns the data in the tab (strings).
        get_configuration_destringed -- Returns the data in the tab.
        configure -- Sets the minimum size of the grid.
        create_active_file_label -- Creates the label.

    Class Methods:
        update_active_file_label -- Updates all labels.

    """

    objs = []

    def __init__(self, names_checkbox=None, names_combobox=None, names_entry=None):
        # List with tabs to update the active file label
        Tab.objs.append(self)
        if names_checkbox is not None:
            self.names_checkbox = names_checkbox
        else:
            self.names_checkbox = []
        self.checkbox_status = []
        self.checkbox = []
        if names_combobox is not None:
            self.names_combobox = names_combobox
        else:
            self.names_combobox = []
        self.combobox_options = []
        self.combobox = []
        self.combobox_label = []
        if names_entry is not None:
            self.names_entry = names_entry
        else:
            self.names_entry = []
        self.entry = []
        self.entry_label = []
        self.active_file_label = ActiveFileLabel()
        self.i = 0

    def create_tab(self, nb, name):
        """
        Create the new tab in the notebook nb and call it name

        Parameters
        ----------
        nb : ttk notebook
            main notebook.
        name : string
            name of the tab.

        Returns
        -------
        None.
        """
        self.tab = tk.Frame(nb, width=500, height=500, padx=10, pady=10)
        # Adds it to the notebook
        nb.add(self.tab, text=name)

    def create_checkboxes(self, names_checkbox, r, c, s="W", disables_all=False):
        """
        Create the checkboxes starting at the row = r, column = c,
        using the list names_checkboxes to determine their order and names

        Parameters
        ----------
        names_checkbox : list of strings
            labels for the checkboxes.
        r : int
            starting row of the list of checkboxes.
        c : int
            starting column of the list of checkboxes.
        s : tk position ("S", "W", etc), optional
            where the labels are aligned. The default is "W".
        disables_all : bool, optional
            The first item will enable or disable the tab. The default is False.

        Returns
        -------
        None.
        """
        self.names_checkbox = copy.deepcopy(names_checkbox)
        # Creates list with the status of the checkboxes
        self.checkbox_status = ["False"] * len(self.names_checkbox)
        self.checkbox = [0]*len(self.names_checkbox)
        for i in range(len(self.names_checkbox)):
            # If the first checkbox disables the tab:
            if i == 0:
                if disables_all is True:
                    self.checkbox_status[i] = tk.StringVar()
                    self.checkbox[i] = tk.Checkbutton(self.tab,
                                                       text=self.names_checkbox[i],
                                                       variable=self.checkbox_status[i],
                                                       onvalue="True", offvalue="False",
                                                       command=self.change_state)
                    self.checkbox[i].deselect()
                    self.checkbox[i].grid(row=r+i, column=c, sticky=s)
                else:
                    self.checkbox_status[i] = tk.StringVar()
                    self.checkbox[i] = tk.Checkbutton(self.tab,
                                                       text=self.names_checkbox[i],
                                                       variable=self.checkbox_status[i],
                                                       onvalue="True", offvalue="False")
                    self.checkbox[i].deselect()
                    self.checkbox[i].grid(row=r+i, column=c, sticky=s)
                continue
            self.checkbox_status[i] = tk.StringVar()
            self.checkbox[i] = tk.Checkbutton(self.tab,
                                               text=self.names_checkbox[i],
                                               variable=self.checkbox_status[i],
                                               onvalue="True", offvalue="False")
            self.checkbox[i].deselect()
            self.checkbox[i].grid(row=r+i, column=c, sticky=s)

    def activate_all(self):
        """
        Activates all widgets in the tab.

        Returns
        -------
        None.
        """
        for i in range(len(self.checkbox)-1):
            self.checkbox[i+1].config(state="normal")
        for i in range(len(self.names_combobox)):
            self.combobox[i].config(state="normal")
        for i in range(len(self.entry)):
            self.entry[i].config(state="normal")

    def deactivate_all(self):
        """
        Deactivates all widgets in the tab.

        Returns
        -------
        None.
        """
        for i in range(len(self.checkbox)-1):
            self.checkbox[i+1].config(state="disable")
        for i in range(len(self.names_combobox)):
            self.combobox[i].config(state="disable")
        for i in range(len(self.entry)):
            self.entry[i].config(state="disable")

    def change_state(self):
        """
        If the tab widgets are enable, it disables them, and vise versa.

        Returns
        -------
        None.
        """
        # If the first checkbox disables all
        if self.checkbox_status[0].get() == "True":
            self.activate_all()
        else:
            self.deactivate_all()

    def create_combobox(self, options, names_combobox, r, c, s="E"):
        """
        Create the comboboxes starting at the row = r, column = c,
        using the list names_combobox to determine their order and names
        and the nested list options as options of each combobox

        Parameters
        ----------
        options : nested list of strings
            options of each combobox.
        names_combobox : list of strings
            label of each combobox.
        r : int
            starting row of the list of checkboxes.
        c : int
            starting column of the list of checkboxes.
        s : tk position ("S", "W", etc), optional
            where the labels are aligned. The default is "E".

        Returns
        -------
        None.
        """
        self.names_combobox = copy.deepcopy(names_combobox)
        self.combobox_options = copy.deepcopy(options)
        self.combobox = [0]*len(self.names_combobox)
        self.combobox_label = [0]*len(self.names_combobox)
        for i in range(len(self.names_combobox)):
            self.combobox[i] = ttk.Combobox(self.tab, width=20, state='readonly')
            self.combobox[i].grid(row=r+i, column=c+1, sticky=s)
            self.combobox[i]["values"] = options[i]
            self.combobox_label[i] = tk.Label(self.tab, text=self.names_combobox[i])
            self.combobox_label[i].grid(row=r+i, column=c, sticky=s)
            self.combobox[i].set(self.combobox_options[i][0])

    def create_entry(self, names_entry, r, c, s="E", w=20):
        """
        Create the entries starting at the row = r, column = c,
        using the list names_entry to determine their order and names

        Parameters
        ----------
        names_entry : list of strings
            labels for the entries.
        r : int
            starting row of the list of entries.
        c : int
            starting column of the list of entries.
        s : tk position ("S", "W", etc), optional
            where the labels are aligned. The default is "E".
        w : int, optional
            width of the entry. The default is 20.

        Returns
        -------
        None.
        """
        self.names_entry = copy.deepcopy(names_entry)
        self.entry = [0]*len(self.names_entry)
        self.entry_label = [0]*len(self.names_entry)
        for i in range(len(self.entry)):
            self.entry_label[i] = tk.Label(self.tab, text=self.names_entry[i])
            self.entry_label[i].grid(row=r+i, column=c, sticky="E")
            self.entry[i] = tk.Entry(self.tab, width=w)
            self.entry[i].grid(row=r+i, column=c+1, sticky=s)

    def populate(self, l0):
        """
        Populate the tab with the information in l0 (checkboxes status,
        combobox selected option, entries, in that order).

        Parameters
        ----------
        l0 : list of strings
            information of all the tab's widgets.

        Returns
        -------
        None.
        """
        # Fills the widgets with the data of the save file
        l = copy.deepcopy(l0)
        # Can't write to a disable widget
        self.activate_all()
        # Checkbox, combobox, entry
        n_check = len(self.checkbox)
        n_comb = len(self.names_combobox)
        n_ent = len(self.entry)
        for i in range(n_check):
            if l[i] == "True":
                self.checkbox[i].select()
            elif l[i] == "False":
                self.checkbox[i].deselect()
        for i in range(n_comb):
            self.combobox[i].set(l[i+n_check])
        for i in range(n_ent):
            self.entry[i].insert(0,l[i+n_check+n_comb])

    def depopulate(self):
        """
        Clears all the widgets.

        Returns
        -------
        None.
        """
        self.activate_all()
        for i in range(len(self.checkbox)):
            self.checkbox[i].deselect()
        for i in range(len(self.names_combobox)):
            self.combobox[i].set(self.combobox_options[i][0])
        for i in range(len(self.entry)):
            self.entry[i].delete(0,15)

    def get_configuration(self):
        """
        Gets the status and information of all the tab's widgets.

        Returns
        -------
        list of strings
            information in the tab.
        """
        # Creates list with the data from the widgets
        # Order is checkbox, combobox, entry
        d = []
        for i in range(len(self.checkbox)):
            d.append(self.checkbox_status[i].get())
        for i in range(len(self.combobox)):
            d.append(self.combobox[i].get())
        for i in range(len(self.entry)):
            d.append(self.entry[i].get())
        return copy.deepcopy(d)

    def _destring_data(self, data):
        """
        Transform a list of strings into variables.

        Parameters
        ----------
        data : list of strings
            Data to convert.

        Returns
        -------
        list of variables
            Destringed data.
        """

        def is_number(s):
            """ Returns True is string is a number. """
            try:
                float(s)
                return True
            except ValueError:
                return False

        def string_or_bool(s):
            """Returns True if string == True"""
            if s == "True":
                return True
            if s == "False":
                return False
            return s

        def is_baudrate(f):
            """If f > 9000 almost certainly it's a baudrate"""
            return bool(f > 9000)
        for i, elem in enumerate(data):
            if is_number(data[i]):
                data[i] = float(data[i])
                if is_baudrate(data[i]):
                    data[i] = int(data[i])
            else:
                data[i] = string_or_bool(data[i])
        return data

    def get_configuration_destringed(self):
        """
        Gets the status and information of all the tab's widgets (in variable
        format).

        Returns
        -------
        list of variables
            information in the tab.
        """
        data = self.get_configuration()
        data = self._destring_data(data)
        return copy.deepcopy(data)

    def configure(self, n=10):
        """
        Configurates the tab with a minimum row and columns size of n.

        Parameters
        ----------
        n : int, optional
            Minimum size of the row/column. The default is 10.

        Returns
        -------
        None.
        """
        # creates the empty rows and columns
        # so as to have empty space between widgets
        col_count, row_count = self.tab.grid_size()
        for col in range(col_count):
            self.tab.grid_columnconfigure(col, minsize=n)
        for row in range(row_count):
            self.tab.grid_rowconfigure(row, minsize=n)

    def create_active_file_label(self, r, c):
        """
        Create a label with the active file name in the position row = r and
        column = c.

        Parameters
        ----------
        r : int
            position (row).
        c : int
            position (column).

        Returns
        -------
        None.
        """
        self.active_file_label.create_label(self.tab, r, c)

    @classmethod
    def update_active_file_label(cls, name):
        """
        Update the text in the active file label to match the new active file.

        Parameters
        ----------
        cls : Tab
            Tab.
        name : string
            Active file name.

        Returns
        -------
        None.
        """
        for obj in cls.objs:
            obj.active_file_label.update(name)


class TabWithCanvas(Tab):
    """
    For tabs with canvases.

    Methods:
        set_points -- Sets the rocket component points.
        add_point -- Adds a point to the rocket component.
        delete_point -- Delets a point of the rocket components.
        get_points -- Returns the points of a rocket component (string).
        get_points_float -- Returns the points of a rocket component.
        create_canvas -- Creates the canvas.
        draw_rocket -- Draws the rocket.
        populate -- Fills the tab.
        get_configuration -- Returns the rocket's configuration in the tab (string).
        get_configuration_destringed -- Returns the rocket's configuration in the tab.
        populate -- fills the tab with the data.
        depopulate -- Deletes the data from the tab.
        change_state_fins -- Activate/deactivate fins.
    """

    def __init__(self):
        super().__init__()
        self.canvas_height = 0
        self.canvas_width = 0
        # points[0] -> rocket, points[1] -> fin
        # Rocket points go from the tip down to the tail.
        # Fin[n][x position (longitudinal), z position (span)]
        #  [0]|\
        #     | \[1]
        #     | |
        #  [3]|_|[2]
        self.points = [["0,0","0,0"],
                       ["0.001,0.001","0.001,0.001","0.001,0.001","0.001,0.001"],
                       ["0.001,0.001","0.001,0.001","0.001,0.001","0.001,0.001"]]
        self.rocket = rocket_functions.Rocket()
        self.active_point = 0
        self.active_point_fins = 0
        self.flag_hollow_body = False
        self.aoa = 0.01
        self.aoa_ctrl_fin = 0
        self.tvc_angle = 0
        self.current_motor = ""
        self.velocity = 1
        self.rocket_length = 0
        self.max_fin_len = 0
        self.max_length = 0
        self.scale_y = 1
        self.centering = 0

    def _sort(self, l):
        """
        Sort the data l.

        Parameters
        ----------
        l : list of strings or floats.
            data to sort.

        Returns
        -------
        list of floats.
            data sorted.
        """
        def _l2j_is_greater_than(l2,j):
            if float(l2[j].split(",")[0]) > float(l2[j+1].split(",")[0]):
                return True
            return False
        l2 = copy.deepcopy(l)
        for _ in range(len(l2)):
            for j in range(len(l2)-1):
                if _l2j_is_greater_than(l2, j):
                    b = l2[j]
                    l2[j] = l2[j+1]
                    l2[j+1] = b
        return copy.deepcopy(l2)

    def set_points(self, n, l):
        """
        Recieves a list and sets the points of the rocket body to it.

        Parameters
        ----------
        n : int
            n = 0 = body, 1 = stabilization fin, 2 = control fin.
        l : list of strings
            points.

        Returns
        -------
        None.
        """

        #
        self.points[n] = copy.deepcopy(l)

    def add_point(self, n, s):
        """
        Adds the point "s" to the rocket part "n".

        Parameters
        ----------
        n : int
            n = 0 = body, 1 = stabilization fin, 2 = control fin.
        s : string
            format: "x,z".

        Returns
        -------
        None.
        """
        self.points[n].append(s)
        self.points[n] = self._sort(self.points[n])
        if n == 0:
            # If it is from the body it goes to the combobox
            self.combobox[n]["values"] = self.points[n]
            self.combobox[n].set(s)
        else:
            for i in range(len(self.entry)):
                # else you delete the entries and populates them
                # with the points, it is not used
                self.entry[i].delete(0, 15)
                self.entry[i].insert(0, s)

    def delete_point(self, n, s):
        """
        Deletes the point "s" form the rocket part "n".

        Parameters
        ----------
        n : int
            n = 0 = body, 1 = stabilization fin, 2 = control fin.
        s : string
            format: "x,z".

        Returns
        -------
        None.
        """
        for i in range(len(self.points[n])):
            if n == 0:
                if self.points[n][i] == s:
                    del self.points[n][i]
                    self.combobox[n]["values"] = self.points[n]
                    self.combobox[n].set(self.points[n][i-1])
                    break

    def get_points(self, n):
        """
        Get the points of the rocket part "n" as strings.

        Parameters
        ----------
        n : int
            n = 0 = body, 1 = stabilization fin, 2 = control fin..

        Returns
        -------
        Nested list of strings.
            points of the part "n".
        """
        return copy.deepcopy(self.points[n])

    def get_points_float(self, n):
        """
        Get the points of the rocket part "n" as floats.

        Parameters
        ----------
        n : int
            n = 0 = body, 1 = stabilization fin, 2 = control fin.

        Returns
        -------
        Nested list of floats.
            points of the part "n".
        """
        l = copy.deepcopy(self.points[n])
        l2 = []
        for element in l:
            a = element.split(",")
            l2.append([float(a[0]), float(a[1])])
        return copy.deepcopy(l2)

    def create_canvas(self, canvas_width, canvas_height):
        """
        Create a canvas of determined width and height

        Parameters
        ----------
        canvas_width : int
            Canvas width.
        canvas_height : int
            Canvas height.

        Returns
        -------
        None.
        """
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.canvas = (tk.Canvas(self.tab, width=self.canvas_width,
                                 heigh=self.canvas_height, bg="white"))
        self.canvas.grid(row=0, column=0, rowspan = 20)
        self._create_n_f_app_m_labels()

    def _create_n_f_app_m_labels(self):
        self.n_force_label = tk.Label(self.tab, text="")
        self.n_force_label.grid(row=14, column=3)
        self.f_point_label = tk.Label(self.tab, text="")
        self.f_point_label.grid(row=15, column=3)
        self.moment_label = tk.Label(self.tab, text="")
        self.moment_label.grid(row=16, column=3)
        self.thrust_label = tk.Label(self.tab, text="Thrust in N")
        self.thrust_label.grid(row=17, column=3)
        self.thrust_entry = tk.Entry(self.tab)
        move_tk_object(self.thrust_entry, 18, 3)

    def _re_draw_rocket(self, l2):
        # x in the canvas is y/z in the rocket
        # y in the canvas is x in the rocket
        for i in range(len(l2)-1):
            # checkbox_status[0] is the Ogive
            if i == 0 and self.checkbox_status[0].get() == "True":
                radius_nc = l2[1][1]
                len_nc = l2[1][0]
                rho_radius = (radius_nc**2 + len_nc**2) / (2*radius_nc)
                x_ogive_1 = 0
                # y = np.sqrt(rho_radius**2 - (len_nc-x_ogive_1)**2)+radius_nc-rho_radius
                # Draws an ogive with 10 points
                definition = 20
                for _ in range(definition):
                    x_ogive_2 = x_ogive_1 + len_nc/definition
                    y_ogive_1 = (np.sqrt(rho_radius**2 - (len_nc-x_ogive_1)**2)
                                 + radius_nc - rho_radius)
                    y_ogive_2 = (np.sqrt(rho_radius**2 - (len_nc-x_ogive_2)**2)
                                 + radius_nc - rho_radius)
                    x1 = (y_ogive_1*self.scale_y + self.canvas_width) / 2
                    y1 = x_ogive_1*self.scale_y + self.centering
                    x2 = (y_ogive_2*self.scale_y + self.canvas_width) / 2
                    y2 = x_ogive_2*self.scale_y + self.centering
                    x1_mirror = (-y_ogive_1*self.scale_y + self.canvas_width) / 2
                    x2_mirror = (-y_ogive_2*self.scale_y + self.canvas_width) / 2
                    self.canvas.create_line(x1, y1, x2, y2)
                    self.canvas.create_line(x1_mirror, y1, x2_mirror, y2)
                    x_ogive_1 += len_nc/definition
                point_diameter = 5
                self._create_point_cp(point_diameter)
                self._create_point_xcg(point_diameter)
                self._update_n_f_app_m_labels()
            else:
                # Conic nosecone / rest of the body
                x1 = (l2[i][1]*self.scale_y + self.canvas_width) / 2
                y1 = l2[i][0]*self.scale_y + self.centering
                x2 = (l2[i+1][1]*self.scale_y + self.canvas_width) / 2
                y2 = l2[i+1][0]*self.scale_y + self.centering
                x1_mirror = (-l2[i][1]*self.scale_y + self.canvas_width) / 2
                x2_mirror = (-l2[i+1][1]*self.scale_y + self.canvas_width) / 2
                self.canvas.create_line(x1, y1, x2, y2)
                self.canvas.create_line(x1_mirror, y1, x2_mirror, y2)
                if i == len(l2)-2:
                    self.canvas.create_line(x2_mirror, y2, x2, y2)
                point_diameter = 5
                self._create_point_cp(point_diameter)
                self._create_point_xcg(point_diameter)
                self._update_n_f_app_m_labels()
            self._draw_base_component(l2)

    def _draw_base_component(self,l2):
        # Draws the horizontal line that separates each component
        for element in l2:
            x1 = (element[1]*self.scale_y + self.canvas_width) / 2
            y1 = element[0]*self.scale_y + self.centering
            x2 = (element[1]*self.scale_y + self.canvas_width) / 2
            y2 = element[0]*self.scale_y + self.centering
            self.canvas.create_line(x1, y1, x2, y2)

    def _draw_fins(self, l2, s, attached, separate):
        # Stabilization fins are attached to a hollow
        # body, therefore they lose a lot of lift but
        # aren't physically separated from the body.
        # Control fins are attached  to a servo and they
        # are a distance apart from the body (or not,
        # it depends on the rocket)
        if separate == "True" and attached == "False":
            sep = 0.01
        else:
            sep = 0
        #Draws the fin
        for i in range(len(l2)-1):
            x1 = (l2[i][1]+sep) * self.scale_y + self.canvas_width/2
            y1 = l2[i][0]*self.scale_y + self.centering
            x2 = (l2[i+1][1]+sep) * self.scale_y + self.canvas_width/2
            y2 = l2[i+1][0]*self.scale_y + self.centering
            x1_mirror = -(l2[i][1]+sep) * self.scale_y + self.canvas_width/2
            x2_mirror = -(l2[i+1][1]+sep) * self.scale_y + self.canvas_width/2
            self.canvas.create_line(x1, y1, x2, y2, fill=s)
            self.canvas.create_line(x1_mirror, y1, x2_mirror, y2, fill=s)
        # Draws an horizontal line to "simulate" the cut body
        if attached == "False" and separate == "False":
            x1 = l2[0][1]*self.scale_y + self.canvas_width/2
            y1 = l2[0][0]*self.scale_y + self.centering
            x2 = -l2[0][1]*self.scale_y + self.canvas_width/2
            y2 = l2[0][0]*self.scale_y + self.centering
            self.canvas.create_line(x1, y1, x2, y2)
        # Draws the vertical line that connects the root chord of
        # the fin, (usually the body takes care of it, but in
        # this case, the fin is separated)
        if separate == "True":
            x1 = (l2[0][1]+sep) * self.scale_y + self.canvas_width/2
            y1 = l2[0][0]*self.scale_y + self.centering
            x2 = (l2[3][1]+sep) * self.scale_y + self.canvas_width/2
            y2 = l2[3][0]*self.scale_y + self.centering
            x1_m = -(l2[0][1]+sep) * self.scale_y + self.canvas_width/2
            y1_m = l2[0][0]*self.scale_y + self.centering
            x2_m =  -(l2[0][1]+sep) * self.scale_y + self.canvas_width/2
            y2_m = l2[3][0]*self.scale_y + self.centering
            self.canvas.create_line(x1, y1, x2, y2, fill=s)
            self.canvas.create_line(x1_m, y1_m, x2_m, y2_m, fill=s)

    def _create_point_cp(self, point_diameter):
        # Creates a point where the CP is located
        # the slider can move it by modifying the aoa
        f = point_diameter / 2
        xcg_point = float(gui_setup.savefile.get_parameters()[3])
        self.rocket.update_rocket(self.get_configuration_destringed(), xcg_point)
        v = [1, np.tan(self.aoa)]/np.sqrt(1 + np.tan(self.aoa)**2) * self.velocity
        cn, cm, ca, cp_point = self.rocket.calculate_aero_coef(v_loc_tot=v,
                                                               actuator_angle=self.aoa_ctrl_fin)
        self.normal_force, self.force_app_point = self._calculate_total_cn_cp(cn, cp_point)
        self._set_f_app_point_color(self.normal_force)
        self.canvas.create_oval(self.canvas_width/2-f, self.force_app_point*self.scale_y - f,
                                self.canvas_width/2+f, self.force_app_point*self.scale_y + f,
                                fill=self.f_app_colour, outline=self.f_app_colour)

    def _calculate_total_cn_cp(self, cn, cp_point):
        q = 0.5 * 1.225 * self.velocity**2
        aero_force = q * self.rocket.area_ref * cn
        thrust = self._get_motor_data()
        xt = float(gui_setup.param_file_tab.entry[3].get())
        normal_force = thrust*np.sin(self.tvc_angle) + aero_force
        force_app_point = (aero_force*cp_point + thrust*np.sin(self.tvc_angle) * xt) / normal_force
        return normal_force, force_app_point

    def _get_motor_data(self):
        if self.current_motor != gui_setup.param_file_tab.combobox[0].get():
            gui_setup.savefile.read_motor_data(gui_setup.param_file_tab.combobox[0].get())
            self.rocket.set_motor(gui_setup.savefile.get_motor_data())
            self.current_motor = gui_setup.param_file_tab.combobox[0].get()
            self.thrust_entry.delete(0,100)
            self.thrust_entry.insert(0,str(round(self.rocket.get_thrust(0.5, 0),3)))
            self.thrust = float(self.thrust_entry.get())
        else:
            self.thrust = float(self.thrust_entry.get())
        return self.thrust

    def _set_f_app_point_color(self, cn):
        if cn <= 0:
            self.f_app_colour = "red"
        else:
            self.f_app_colour = "green"

    def _update_n_f_app_m_labels(self):
        n_force = "N Force = " + str(round(self.normal_force,3)) + " N"
        f_point = "Force App point = " + str(round(self.force_app_point,3)) + " m"
        moment = self.normal_force * (self.force_app_point-self.xcg_point)
        moment_text = "Moment = " + str(round(moment,3)) + " N.m"
        self.n_force_label.config(text=n_force)
        self.f_point_label.config(text=f_point)
        self.moment_label.config(text=moment_text)

    def _create_point_xcg(self, point_diameter):
        f = point_diameter / 2
        self.xcg_point = float(gui_setup.savefile.get_parameters()[3])
        self.canvas.create_oval(self.canvas_width/2 - f, self.xcg_point*self.scale_y - f,
                                self.canvas_width/2 + f, self.xcg_point*self.scale_y + f,
                                fill="blue", outline="blue")

    def draw_rocket(self):
        """
        Draws the rocket using the set points.

        Returns
        -------
        None.
        """
        # x in the canvas is y/z in the rocket
        # y in the canvas is x in the rocket
        self.canvas.delete("all")
        l2 = self.get_points_float(0)
        self.rocket_length = l2[-1][0]
        self.max_fin_len = self.get_points_float(1)[2][0]
        if self.checkbox_status[1].get() == "True":
            if self.rocket_length > self.max_fin_len:
                self.max_length = self.rocket_length
            else:
                self.max_length = self.max_fin_len
        else:
            self.max_length = self.rocket_length
        if self.rocket_length != 0:
            self.scale_y = self.canvas_height / self.max_length
        else:
            self.scale_y = 1
        # Centers the rocket in the horizontal
        self.centering = (self.canvas_height-self.max_length*self.scale_y) / 2
        self._re_draw_rocket(l2)
        if self.checkbox_status[1].get() == "True":
            fin_stab_points = self.get_points_float(1)
            attached = self.checkbox_status[2].get()
            separate = "False"
            self._draw_fins(fin_stab_points,"black", attached, separate)
            if self.checkbox_status[3].get() == "True":
                fin_control_points = self.get_points_float(2)
                attached = self.checkbox_status[4].get()
                separate = "True"
                self._draw_fins(fin_control_points, "red", attached, separate)

    def populate(self,l0):
        # Populates the entries and more importantly, the points[i]
        # the savefile separates the body from the sabilization fins
        # with "Fins_s", and these from the control ones with "Fins_c"
        l = copy.deepcopy(l0)[:(len(self.checkbox)+len(self.combobox))]
        l1 = copy.deepcopy(l0)[(len(self.checkbox)):]
        l2 = []
        l3 = []
        l4 = []
        flag = "Body"
        for i, element in enumerate(l1):
            if element == "Fins_s" and i > 0:
                flag = "Fins_s"
                continue
            if element == "Fins_c":
                flag = "Fins_c"
                continue
            if flag == "Body":
                l2.append(element)
                continue
            if flag == "Fins_s":
                l3.append(element)
            if flag == "Fins_c":
                l4.append(element)
        # Populates the widgets, not the variables
        super().populate((l+l3))
        # Updates the points of the components
        self.combobox[0]["values"] = l2
        self.points[0] = copy.deepcopy(l2)
        self.points[1] = copy.deepcopy(l3)
        self.points[2] = copy.deepcopy(l4)
        self.draw_rocket()
        self.change_state_fins()

    def get_configuration(self):
        # Takes the whole configurations and dumps it
        # into a list, must include the separators
        # between fins
        d = []
        d0 = self.get_points(0)
        d1 = self.get_points(1)
        d2 = self.get_points(2)
        for i in range(len(self.checkbox)):
            d.append(self.checkbox_status[i].get())
        d += d0
        d.append("Fins_s")
        d += d1
        d.append("Fins_c")
        d += d2
        return copy.deepcopy(d)

    def get_configuration_destringed(self):
        # Format [che,ck,box, [body], [fin_s], [fin_c]]
        d = []
        d0 = self.get_points_float(0)
        d1 = self.get_points_float(1)
        d2 = self.get_points_float(2)
        for i in range(len(self.checkbox)):
            if self.checkbox_status[i].get() == "True":
                d.append(True)
            else:
                d.append(False)
        d.append(d0)
        d.append(d1)
        d.append(d2)
        return copy.deepcopy(d)

    def activate_all(self):
        for i in range(len(self.checkbox)-1):
            self.checkbox[i+1].config(state="normal")
        for i in range(len(self.names_combobox)):
            self.combobox[i].config(state="normal")
        for i in range(len(self.entry)):
            self.entry[i].config(state="normal")

    def deactivate_all(self):
        for i in range(len(self.checkbox)-2):
            self.checkbox[i+1+1].config(state="disable")
        for i in range(len(self.entry)):
            self.entry[i].config(state="disable")

    def change_state_fins(self):
        """
        Activate the fins if they were deactivated and vise versa.

        Returns
        -------
        None.
        """
        if self.checkbox_status[1].get() == "True":
            self.activate_all()
        else:
            self.deactivate_all()
        self.draw_rocket()

    def change_state_control_fins(self):
        a = self.scale_act_angle.get()
        if self.checkbox_status[3].get() == "True":
            self.aoa_ctrl_fin = float(a)/57.295
            self.tvc_angle = 0
        else:
            self.tvc_angle = float(a)/57.295
            self.aoa_ctrl_fin = 0
        self.scale_act_angle.set(float(a))
        self.draw_rocket()

    def create_sliders(self):
        def slider_aoa(a):
            # Changes the aoa and re draws the CP
            self.aoa = float(a)/57.295 + 0.01
            if self.aoa > 3.14159/2:
                self.aoa = 3.14159/2
            self.draw_rocket()

        self.aoa_scale = tk.Scale(self.tab, from_=0.01, to=90,
                             orient=tk.HORIZONTAL, command=slider_aoa, length=200)
        self.aoa_scale.grid(row=20, column=0)
        tk.Label(self.tab, text="Angle of Attack" + u' [\xb0]').grid(row=21, column=0)

        def slider_actuator_angle(a):
            if self.checkbox_status[3].get() == "True":
                self.aoa_ctrl_fin = float(a)/57.295
                self.tvc_angle = 0
            else:
                self.tvc_angle = float(a)/57.295
                self.aoa_ctrl_fin = 0
            self.draw_rocket()

        self.scale_act_angle = tk.Scale(self.tab, from_=-45, to=45,
                             orient=tk.HORIZONTAL, command=slider_actuator_angle, length=150)
        self.scale_act_angle.grid(row=20, column=1)
        tk.Label(self.tab, text="Actuator Deflection" + u' [\xb0]').grid(row=21, column=1)

        def slider_velocity(a):
            self.velocity = float(a)
            self.draw_rocket()

        self.scale_velocity = tk.Scale(self.tab, from_=1, to=200,
                             orient=tk.HORIZONTAL, command=slider_velocity, length=150)
        self.scale_velocity.set(10)
        self.scale_velocity.grid(row=20, column=3)
        tk.Label(self.tab, text="Speed [m/s]").grid(row=21, column=3)


# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 16:21:38 2021

@author: guido
"""
import tkinter as tk
from tkinter import ttk
import files
import copy
import GUI_setup as GUI_setup
import numpy as np

def move_tk_object(obj, r=0, c=0, columnspan = 1):
    obj.grid(row = r, column = c, columnspan = columnspan)

class active_file_label_class:
    def __init__(self):       
        pass    
    def create_label(self,root,r,c):
        self.r = r
        self.c = c
        self.canvas = root
        self.text = tk.StringVar()
        self.active_file_label = tk.Label(self.canvas, textvariable=self.text)
        self.active_file_label.grid(row=r,column=c,sticky="W", columnspan=2)        
    def update(self, active_file):
        self.text.set("Active File is " + active_file)
        active_file_label = tk.Label(self.canvas, textvariable=self.text)
        active_file_label.grid(row=self.r,column=self.c,sticky="W", columnspan=2)

"""
Order in the savefile:
Checkboxes
combobox
entry
"""

#################################################################################################################################
class tab_class:
    objs = []
    def __init__(self, names_checkbox = [], names_combobox = [], names_entry = []):
        # List with tabs to update the active file label
        tab_class.objs.append(self)
        self.names_checkbox = names_checkbox
        self.checkbox_status = []
        self.checkbox = []        
        self.names_combobox = names_combobox
        self.combobox_options = []
        self.combobox = []
        self.combobox_label = []        
        self.names_entry = names_entry
        self.entry = []
        self.entry_label = []        
        self.active_file_label = active_file_label_class()        
        self.i = 0        
        return
    
    def create_tab(self, nb, name):        
        self.tab = tk.Frame(nb, width = 500, height = 500,padx=10,pady=10)
        # Adds it to the notebook
        nb.add(self.tab, text = name)        
    
    def create_checkboxes(self,names_checkbox,r,c,s="W", disables_all = False):
        self.names_checkbox = copy.deepcopy(names_checkbox)
        # Creates list with the status of the checkboxes
        self.checkbox_status = ["False"]*len(self.names_checkbox)
        self.checkbox = [0]*len(self.names_checkbox)        
        for i in range(len(self.names_checkbox)):
            # If the first checkbox disables the tab:
            if i == 0:                
                if disables_all == True:
                    self.checkbox_status[i] = tk.StringVar()        
                    self.checkbox[i] = tk.Checkbutton(self.tab, text=self.names_checkbox[i], variable=self.checkbox_status[i], onvalue="True", offvalue="False", command = self.change_state)
                    self.checkbox[i].deselect()
                    self.checkbox[i].grid(row=r+i,column=c,sticky = s)
                else:
                    self.checkbox_status[i] = tk.StringVar()        
                    self.checkbox[i] = tk.Checkbutton(self.tab, text=self.names_checkbox[i], variable=self.checkbox_status[i], onvalue="True", offvalue="False")
                    self.checkbox[i].deselect()
                    self.checkbox[i].grid(row=r+i,column=c,sticky = s)
                continue
            self.checkbox_status[i] = tk.StringVar()        
            self.checkbox[i] = tk.Checkbutton(self.tab, text=self.names_checkbox[i], variable=self.checkbox_status[i], onvalue="True", offvalue="False")
            self.checkbox[i].deselect()
            self.checkbox[i].grid(row=r+i,column=c,sticky = s)
            
    def activate_all(self):
        for i in range(len(self.checkbox)-1):
            self.checkbox[i+1].config(state="normal")
        for i in range(len(self.names_combobox)):
            self.combobox[i].config(state="normal")
        for i in range(len(self.entry)):
            self.entry[i].config(state="normal")            
        return
    
    def deactivate_all(self):
        for i in range(len(self.checkbox)-1):
            self.checkbox[i+1].config(state="disable")
        for i in range(len(self.names_combobox)):
            self.combobox[i].config(state="disable")
        for i in range(len(self.entry)):
            self.entry[i].config(state="disable")
        return
    
    def change_state(self):
        # If the first checkbox disables all        
        if(self.checkbox_status[0].get()=="True"):
            self.activate_all()
        else:
            self.deactivate_all()            
        return 
    
    def create_combobox(self , options , names_combobox , r , c , s="E"):
        self.names_combobox = copy.deepcopy(names_combobox)
        self.combobox_options = copy.deepcopy(options)
        self.combobox = [0]*len(self.names_combobox)
        self.combobox_label = [0]*len(self.names_combobox)        
        for i in range(len(self.names_combobox)):
            self.combobox[i] = ttk.Combobox(self.tab,width=20,state='readonly')
            self.combobox[i].grid(row = r+i, column = c+1,sticky = s)
            self.combobox[i]["values"] = options[i]
            self.combobox_label[i] = tk.Label(self.tab, text=self.names_combobox[i])
            self.combobox_label[i].grid(row=r+i,column=c,sticky=s) 
            self.combobox[i].set(self.combobox_options[i][0])  
    
    def create_entry(self, names_entry , r , c , s="E", w=20):
        self.names_entry = copy.deepcopy(names_entry) 
        self.entry = [0]*len(self.names_entry)
        self.entry_label = [0]*len(self.names_entry)
        for i in range(len(self.entry)):
            self.entry_label[i] = tk.Label(self.tab, text=self.names_entry[i])
            self.entry_label[i].grid(row=r+i,column=c,sticky = "E")
            self.entry[i] = tk.Entry(self.tab, width = w)
            self.entry[i].grid(row=r+i,column=c+1,sticky = s)

    def populate(self, l0):
        # Fills the widgets with the data of the save file
        l = copy.deepcopy(l0)
        # Can't write to a disable widget
        self.activate_all()
        # Checkbox, combobox, entry
        n_check = len(self.checkbox)
        n_comb = len(self.names_combobox)
        n_ent = len(self.entry)
        for i in range(n_check):            
            if l[i]=="True":
                self.checkbox[i].select()
            elif l[i]=="False":
                self.checkbox[i].deselect()                
        for i in range(n_comb):            
            self.combobox[i].set(l[i+n_check])
        for i in range(n_ent):            
            self.entry[i].insert(0,l[i+n_check+n_comb])
    
    def depopulate(self):
        self.activate_all()
        for i in range(len(self.checkbox)):
            self.checkbox[i].deselect
        for i in range(len(self.names_combobox)):
            self.combobox[i].set(self.combobox_options[i][0]) 
        for i in range(len(self.entry)):
            self.entry[i].delete(0,15)
    
    def get_configuration(self):
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
    
    def destring_data(self, data):
        def is_number(s):
            """ Returns True is string is a number. """
            try:
                float(s)
                return True
            except ValueError:
                return False        
        def string_or_bool(s):
            "Returns True if string == True"
            if s == "True":
                return True
            elif s == "False":
                return False
            else:
                return s        
        def is_baudrate(f):
            if f > 9000:
                return True
            else:
                return False
            
        for i in range(len(data)):
            if is_number(data[i]):
                data[i] = float(data[i])
                if is_baudrate(data[i]):
                    data[i] = int(data[i])
            else:
                data[i] = string_or_bool(data[i])     
        return data
    
    def get_configuration_destringed(self):
        data = self.get_configuration()
        data = self.destring_data(data)
        return copy.deepcopy(data)
        
        
        
    
    def configure(self,n=10):
        # creates the empty rows and columns
        # so as to have empty space between widgets
        col_count, row_count = self.tab.grid_size()
        for col in range(col_count):
            self.tab.grid_columnconfigure(col, minsize=n)
        for row in range(row_count):
            self.tab.grid_rowconfigure(row, minsize=n)
            
    def create_active_file_label(self,r,c):
       self.active_file_label.create_label(self.tab,r,c)
    
    @classmethod 
    def update_active_file_label(cls,name):
        for obj in cls.objs: 
            obj.active_file_label.update(name)


####################################################################################################################    
class tab_with_canvas_class(tab_class):
    def __init__(self):
        super().__init__()
        import rocket_functions
        # points[0] -> rocket, points[1] -> fin
        """
        Rocket points go from the tip down to the tail
         
        Fin[n][x position (longitudinal), z position (span)]
        
         [0]|\
            | \[1]
            | |
         [3]|_|[2]
        """
        self.points = [["0,0","0,0"],
                       ["0.001,0.001","0.001,0.001","0.001,0.001","0.001,0.001"],
                       ["0.001,0.001","0.001,0.001","0.001,0.001","0.001,0.001"]]
        self.Rocket = rocket_functions.rocket_class()
        self.active_point = 0
        self.active_point_fins = 0
        self.flag_hollow_body = False
        self.AoA = 0.001
        pass
    
    def sort(self, l):
        def l2j_is_greater_than(l2,j):
            if float(l2[j].split(",")[0])>float(l2[j+1].split(",")[0]):
                return True
            else:
                return False
        l2 = copy.deepcopy(l)              
        for i in range(len(l2)):
            for j in range(len(l2)-1):
                if l2j_is_greater_than(l2,j):                    
                    b = l2[j]
                    l2[j] = l2[j+1]
                    l2[j+1] = b        
        return copy.deepcopy(l2)
    
    def set_points(self,n,l):
        # Recieves a list and sets the points to it
        # n = 0 = body, 1 = stabilization fin, 2 = control fin
        self.points[n] = copy.deepcopy(l)
    
    def add_point(self,n,s):        
        self.points[n].append(s)
        self.points[n] = self.sort(self.points[n])
        if n == 0:
            # If it is from the body it goes to the combobox
            self.combobox[n]["values"] = self.points[n]
            self.combobox[n].set(s)
        else:
            for i in range(len(self.entry)):
                # else you delete the entries and populates them
                # with the points, it is not used
                self.entry[i].delete(0,15)
                self.entry[i].insert(0,s)                
        
    def delete_point(self, n ,s):        
        for i in range(len(self.points[n])):
            if n == 0:
                if self.points[n][i] == s:
                    del self.points[n][i]
                    self.combobox[n]["values"] = self.points[n]
                    self.combobox[n].set(self.points[n][i-1])
                    break
    
    def get_points(self,n):
        return copy.deepcopy(self.points[n])
    
    def get_points_float(self,n):
        l = copy.deepcopy(self.points[n])
        l2 = []
        for i in range(len(l)):
            a = l[i].split(",")
            l2.append([float(a[0]),float(a[1])])        
        return copy.deepcopy(l2)
        
    def create_canvas(self, canvas_width, canvas_height):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.canvas = tk.Canvas(self.tab, width = self.canvas_width, height = self.canvas_height,bg="white")
        self.canvas.grid(row=0,column=0, rowspan = 20)
        
    def _re_draw_rocket(self,l2):
        # x in the canvas is y/z in the rocket
        # y in the canvas is x in the rocket
        for i in range(len(l2)-1):
            # checkbox_status[0] is the Ogive
            if i==0 and self.checkbox_status[0].get() == "True":
                R = l2[1][1]
                L = l2[1][0]
                rho_radius = (R**2 + L**2)/(2 * R)
                x_ogive_1 = 0
                # y = np.sqrt(rho_radius**2 - (L-x_ogive_1)**2)+R-rho_radius
                # Draws an ogive with 10 points
                for i in range(10):
                    x_ogive_2 = x_ogive_1 + L/10
                    y_ogive_1 = np.sqrt(rho_radius**2 - (L-x_ogive_1)**2) + R - rho_radius
                    y_ogive_2 = np.sqrt(rho_radius**2 - (L-x_ogive_2)**2) + R - rho_radius
                    x1 = ( y_ogive_1 * self.scaleY + self.canvas_width) / 2
                    y1 = x_ogive_1 * self.scaleY + self.centering
                    x2 = (y_ogive_2 * self.scaleY + self.canvas_width) / 2
                    y2 = x_ogive_2*self.scaleY+self.centering
                    x1_mirror = (-y_ogive_1*self.scaleY+self.canvas_width) / 2
                    x2_mirror = (-y_ogive_2*self.scaleY+self.canvas_width) / 2
                    self.canvas.create_line(x1,y1,x2,y2)
                    self.canvas.create_line(x1_mirror,y1,x2_mirror,y2)
                    x_ogive_1 += L/10                
                point_diameter = 5
                self.create_point_Cp(point_diameter)
                self.create_point_xcg(point_diameter)
            else:
                # Conic nosecone / rest of the body
                x1 = (l2[i][1]*self.scaleY+self.canvas_width)/2
                y1 = l2[i][0]*self.scaleY+self.centering
                x2 = (l2[i+1][1]*self.scaleY+self.canvas_width)/2
                y2 = l2[i+1][0]*self.scaleY+self.centering
                x1_mirror = (-l2[i][1]*self.scaleY+self.canvas_width)/2
                x2_mirror = (-l2[i+1][1]*self.scaleY+self.canvas_width)/2
                self.canvas.create_line(x1,y1,x2,y2)
                self.canvas.create_line(x1_mirror,y1,x2_mirror,y2)            
                if i == len(l2)-2:                
                    self.canvas.create_line(x2_mirror,y2,x2,y2)
                point_diameter = 5
                self.create_point_Cp(point_diameter)
                self.create_point_xcg(point_diameter)            
            self._draw_base_component(l2)
            
    def _draw_base_component(self,l2):
        # Draws the horizontal line that separates each component
        for i in range(len(l2)):
            x1 = (l2[i][1]*self.scaleY+self.canvas_width)/2
            y1 = l2[i][0]*self.scaleY+self.centering
            x2 = (-l2[i][1]*self.scaleY+self.canvas_width)/2
            y2 = l2[i][0]*self.scaleY+self.centering
            self.canvas.create_line(x1,y1,x2,y2)        
        
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
            x1 = (l2[i][1] + sep) * self.scaleY + self.canvas_width / 2
            y1 = l2[i][0] * self.scaleY + self.centering
            x2 = (l2[i+1][1]+sep) * self.scaleY + self.canvas_width / 2
            y2 = l2[i+1][0] * self.scaleY + self.centering
            x1_mirror = -(l2[i][1] + sep) * self.scaleY + self.canvas_width / 2
            x2_mirror = -(l2[i+1][1] + sep) * self.scaleY+self.canvas_width / 2
            self.canvas.create_line(x1,y1,x2,y2,fill=s)
            self.canvas.create_line(x1_mirror,y1,x2_mirror,y2,fill=s)
        # Draws an horizontal line to "simulate" the cut body            
        if attached == "False" and separate == "False":            
            x1 = (l2[0][1]*self.scaleY+self.canvas_width/2)
            y1 = l2[0][0]*self.scaleY+self.centering
            x2 = -l2[0][1]*self.scaleY+self.canvas_width/2
            y2 = l2[0][0]*self.scaleY+self.centering
            self.canvas.create_line(x1,y1,x2,y2) 
        # Draws the vertical line that connects the root chord of 
        # the fin, (usually the body takes care of it, but in 
        # this case, the fin is separated)
        if separate == "True":
            x1 = (l2[0][1] + sep) * self.scaleY + self.canvas_width / 2
            y1 = l2[0][0] * self.scaleY + self.centering
            x2 = (l2[3][1] + sep) * self.scaleY + self.canvas_width / 2
            y2 = l2[3][0] * self.scaleY + self.centering            
            x1_m = -(l2[0][1] + sep) * self.scaleY + self.canvas_width / 2
            y1_m = l2[0][0] * self.scaleY + self.centering
            x2_m =  -(l2[0][1] + sep) * self.scaleY + self.canvas_width / 2
            y2_m = l2[3][0] * self.scaleY + self.centering
            self.canvas.create_line(x1,y1,x2,y2, fill = s)
            self.canvas.create_line(x1_m,y1_m,x2_m,y2_m, fill = s)
        
    def create_point_Cp(self,point_diameter):
        # Creates a point where the CP is located
        # the slider can move it by modifying the AoA
        f = point_diameter/2
        self.Rocket.Update_Rocket(self.get_configuration_destringed(),self.rocket_length/2)
        Cn , Cp_point, ca = self.Rocket.Calculate_Aero_coef(self.AoA)
        self.canvas.create_oval(self.canvas_width/2-f, Cp_point * self.scaleY-f, self.canvas_width/2+f, Cp_point * self.scaleY+f, fill="red", outline = "red")
    
    def create_point_xcg(self,point_diameter):
        f = point_diameter/2
        xcg_point = float(GUI_setup.savefile.get_parameters()[3])
        self.canvas.create_oval(self.canvas_width/2-f, xcg_point * self.scaleY-f, self.canvas_width/2+f, xcg_point * self.scaleY+f, fill="blue", outline = "blue")
        
    def draw_rocket(self):
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
            self.scaleY = self.canvas_height/self.max_length
        else:
            self.scaleY = 1
        # Centers the rocket in the horizontal
        self.centering = (self.canvas_height-self.max_length*self.scaleY)/2
        self._re_draw_rocket(l2)
        if self.checkbox_status[1].get() == "True":
            fin_stab_points = self.get_points_float(1)
            attached = self.checkbox_status[2].get()
            separate = "False"
            self._draw_fins(fin_stab_points,"black",attached, separate)
            if self.checkbox_status[3].get() == "True":
                fin_control_points = self.get_points_float(2)
                attached = self.checkbox_status[4].get()
                separate = "True"
                self._draw_fins(fin_control_points,"red",attached, separate)
        
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
        for i in range(len(l1)):            
            if l1[i] == "Fins_s" and i > 0:
                flag = "Fins_s"
                continue
                
            if l1[i] == "Fins_c":
                flag = "Fins_c"
                continue
            
            if flag == "Body":
                l2.append(l1[i])                
                continue
            
            if flag == "Fins_s":
                l3.append(l1[i])
            
            if flag == "Fins_c":
                l4.append(l1[i])
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
        d+=d0
        d.append("Fins_s")
        d+=d1
        d.append("Fins_c")
        d+=d2
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
        return
    
    def deactivate_all(self):
        for i in range(len(self.checkbox)-2):
            self.checkbox[i+1+1].config(state="disable")
        for i in range(len(self.entry)):
            self.entry[i].config(state="disable")
        return
    
    def change_state_fins(self):        
        if(self.checkbox_status[1].get()=="True"):
            self.activate_all()
        else:
            self.deactivate_all()            
        self.draw_rocket()
        return 
    
    

   
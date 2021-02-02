# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 21:20:41 2021

@author: guido
"""
import tkinter as tk
import files
import GUI_functions as fun
import ZPC_PID_SIMULATOR_V05 as sim

deg2rad =  3.14159265/180
rad2deg = 1/deg2rad


## Global Variables
active_file_name = ""
savefile = files.savefile()
## File
file_tab = fun.tab_class()
## Parameters
param_file_tab = fun.tab_class()
## Draw Rocket
draw_rocket_tab = fun.tab_with_canvas_class()
## Conf 3D
conf_3D_tab = fun.tab_class()
## Sim Setup
sim_setup_tab = fun.tab_class()
## SITL
conf_SITL_tab = fun.tab_class()
## Sim Run
run_sim_tab = fun.tab_class()

## Global Functions
def configure_root(root,notebook):
    rows = 0
    while rows < 50:
        root.rowconfigure(rows, weight=1)
        root.columnconfigure(rows, weight=1)
        rows += 1    
    notebook.grid(row=1, column=0, columnspan=50, rowspan=49, sticky='NESW')
    return notebook

def update_file_labels(name):
    fun.tab_class.update_active_file_label(name)

######## Tabs
## CREATE FILE TAB - CREATE FILE TAB - CREATE FILE TAB - CREATE FILE TAB - CREATE FILE TAB - CREATE FILE TAB - CREATE FILE TAB 
def create_file_tab(notebook):    
    file_tab.create_tab(notebook, "File")        
    combobox_options = [files.get_save_names()]
    names_combobox = [""]
    file_tab.create_combobox(combobox_options,names_combobox,1,0,s="EW")    
    names_entry = ["",""]
    file_tab.create_entry(names_entry, 0, 0, s="EW", w = 30)    
    x = 45
    for i in range(len(file_tab.entry)):
        file_tab.entry[i].grid(padx=x, pady=20)
    for i in range(len(file_tab.combobox)):   
         file_tab.combobox[i].grid(padx=x, pady=5)

    def create_new_file_b():
        global  active_file_name
        name = file_tab.entry[0].get()
        if savefile.check_if_file_exists2overwrite(name) == True:
            savefile.create_file(name)
            file_tab.combobox[0]["values"] = files.get_save_names()
            file_tab.combobox[0].set(savefile.name)
            update_file_labels(savefile.name)
            draw_rocket_tab.populate(savefile.get_rocket_dim())
            conf_3D_tab.depopulate()
            conf_3D_tab.change_state()
            sim_setup_tab.depopulate()
        else:
            print("File was not created")
        return    
    new_file_button = tk.Button(file_tab.tab, text="Create New File", command = create_new_file_b, width=30)
    new_file_button.grid(row = 0, column = 2, padx=10)    
    
    def read_file():        
        global  active_file_name
        if file_tab.combobox[0].get() != "":
            savefile.update_name(file_tab.combobox[0].get())        
            savefile.read_file()
            param_file_tab.depopulate()
            param_file_tab.populate(savefile.get_parameters())      
            draw_rocket_tab.depopulate()
            draw_rocket_tab.populate(savefile.get_rocket_dim())       
            update_file_labels(savefile.name)        
            conf_3D_tab.depopulate()
            conf_3D_tab.populate(savefile.get_conf_3D())
            conf_3D_tab.change_state()
            conf_SITL_tab.depopulate()
            conf_SITL_tab.populate(savefile.get_conf_SITL())
            conf_SITL_tab.change_state()
            sim_setup_tab.depopulate()
            sim_setup_tab.populate(savefile.get_conf_controller())        
            run_sim_tab.depopulate()
            run_sim_tab.populate(savefile.get_conf_plots())      
            savefile.read_motor_data(param_file_tab.combobox[0].get())
        else:
            print("Select valid file")
        return    
    open_file_button = tk.Button(file_tab.tab, text="Open Selected File", command = read_file, width=30)
    open_file_button.grid(row = 1, column = 2, pady=50)  
    
    def save_as():        
        global active_file_name
        if file_tab.entry[1].get() != "":
            name = file_tab.entry[1].get()            
            if savefile.check_if_file_exists2overwrite(name) == True:
                savefile.create_file_as(name)
                
                d = param_file_tab.get_configuration()
                savefile.set_parameters(d)
                d = draw_rocket_tab.get_configuration()
                savefile.set_rocket_dim(d)
                d = conf_3D_tab.get_configuration()
                savefile.set_conf_3D(d)
                d = conf_SITL_tab.get_configuration()        
                savefile.set_conf_SITL(d)
                d = sim_setup_tab.get_configuration()
                savefile.set_conf_controller(d)
                d = run_sim_tab.get_configuration()
                savefile.set_conf_plots(d)
                
                file_tab.combobox[0]["values"] = files.get_save_names()
                file_tab.combobox[0].set(savefile.name)
                update_file_labels(savefile.name)
                savefile.save_all_configurations()
            else:
                print("File was not created")
        else:
            print("Select file")
        return    
    save_as_button = tk.Button(file_tab.tab, text="Save As", command = save_as, width=30)
    save_as_button.grid(row = 2, column = 2)    
    fun.move_tk_object(file_tab.entry[1],2,1)
    
    file_tab.create_active_file_label(32,0)
    file_tab.configure()  
    return


## PARAMETERS TAB - PARAMETERS TAB - PARAMETERS TAB - PARAMETERS TAB - PARAMETERS TAB - PARAMETERS TAB - PARAMETERS TAB - PARAMETERS TAB - PARAMETERS TAB - PARAMETERS TAB - 
def create_parameters_tab(notebook):     
    param_file_tab.create_tab(notebook, "Parameters")
    tk.Label(param_file_tab.tab, text="Parameters",fg="black",bg="#CCFFCC",padx=0).grid(row=0,column=0, sticky="NESW", columnspan = 3)
    combobox_options = [files.get_motor_names()]
    names_combobox = ["Motor[N] = "]
    param_file_tab.create_combobox(combobox_options,names_combobox,1,1)    
    names_entry = ["Mass [kg] = ","Iy [kg*m**2] = ", "Xcg [m] = ","Xt [m] = ", "Servo Definition [º] = ",
             "Max Actuator Angle [º] = ", "Actuator Reduction = ","Initial Misalignment [º] = ", "Servo Velocity Compensation",
             "Wind [m/s] = ","Wind Gust = "]
    param_file_tab.create_entry(names_entry, 2, 1, "W")     
    h = 39
    def button_save_parameters():
        d = param_file_tab.get_configuration()
        savefile.set_parameters(d)
        savefile.save_all_configurations()
        savefile.read_motor_data(param_file_tab.combobox[0].get())
        return     
    save_file_button = tk.Button(param_file_tab.tab, text="Save", command = button_save_parameters, width = 20)
    save_file_button.grid(row = h, column = 14)    
    param_file_tab.create_active_file_label(h,0)
    param_file_tab.configure()    
    return

## DRAW ROCKET TAB - DRAW ROCKET TAB - DRAW ROCKET TAB - DRAW ROCKET TAB - DRAW ROCKET TAB - DRAW ROCKET TAB - DRAW ROCKET TAB - DRAW ROCKET TAB - DRAW ROCKET TAB -
def create_draw_rocket_tab(notebook):
    """
        Rocket points go from the tip down to the tail
         
        Fin[n][x position (longitudinal), z position (span)]
        
         [0]|\
            | \[1]
            | |
         [3]|_|[2]
    """
    draw_rocket_tab.create_tab(notebook, "Set Rocket Body")    
    draw_rocket_tab.create_canvas(250,450) 
    # Create checboxes
    checkboxes = ["Ogival Nosecone", "Use fins", "Fins Attached to Body", "Control Fins", "Control Fins Attached"]
    draw_rocket_tab.create_checkboxes(checkboxes,0,1,"W")
    draw_rocket_tab.checkbox[0].grid(columnspan=10)
    draw_rocket_tab.checkbox[0].config(command = draw_rocket_tab.draw_rocket)
    # Uses fins checkbox    
    draw_rocket_tab.checkbox[1].config(command = draw_rocket_tab.change_state_fins)
    draw_rocket_tab.checkbox[3].config(command = draw_rocket_tab.draw_rocket)
    draw_rocket_tab.checkbox[4].config(command = draw_rocket_tab.draw_rocket)    
    def hollow_fin_body():
        # Draws a straight line to simulate
        # the cut in the body
        draw_rocket_tab.draw_rocket()    
    draw_rocket_tab.checkbox[2].config(command = hollow_fin_body)
    # Moves the checkbox from where they were created to it's final 
    # position a little bit lower    
    for i in range(2):
        fun.move_tk_object(draw_rocket_tab.checkbox[i+1],r=8+i,c=1,columnspan=1)
    for i in range(2):
        fun.move_tk_object(draw_rocket_tab.checkbox[i+3],r=8+i,c=3,columnspan=1)
    # create combobox
    combobox_options = [["0,0"]]
    names_combobox = [""]
    draw_rocket_tab.create_combobox(combobox_options,names_combobox,3,0,s="EW")
    draw_rocket_tab.combobox[0].config(state="readonly")
    draw_rocket_tab.combobox_label[0].destroy()   
    tk.Label(draw_rocket_tab.tab, text="Insert Point").grid(row=1,column=1,sticky="WE")
    # Creates the entry for the rocket body points
    entry_rocket = tk.Entry(draw_rocket_tab.tab,width=20)
    entry_rocket.grid(row=2,column=1,sticky="EW") 
    # Entries for the fins
    names_entry = ["1", "2", "3", "4"]
    draw_rocket_tab.create_entry(names_entry, 14, 0,s="EW")
    for i in range(4):
        fun.move_tk_object(draw_rocket_tab.entry_label[i], 14+i, 2)
        
    def button_add_point():
        draw_rocket_tab.add_point(0, entry_rocket.get())
        draw_rocket_tab.draw_rocket()          
        return     
    add_point_button = tk.Button(draw_rocket_tab.tab, text="Add Point", command = button_add_point, width = 10, fg = "white", bg = "green")
    add_point_button.grid(row = 2, column = 3, sticky = "N")
    
    def button_delete_point():
        p = draw_rocket_tab.combobox[0].get()
        if p != "0,0":
            draw_rocket_tab.delete_point(0,p)
        else:
            print("Cannot Delete the Nosecone Tip")            
        draw_rocket_tab.draw_rocket()            
        return     
    delete_point_button = tk.Button(draw_rocket_tab.tab, text="Delete Point", command = button_delete_point, width = 10, fg = "white", bg = "red")
    delete_point_button.grid(row = 3, column = 3, sticky = "N")
    
    draw_rocket_tab.active_point = "0,0"
    def button_select_point():
        draw_rocket_tab.active_point = draw_rocket_tab.combobox[0].get()
        if draw_rocket_tab.active_point == "0,0":
            print("Cannot Modify the Nosecone Tip")
        select_point_button.config(bg="yellow")
        return     
    select_point_button = tk.Button(draw_rocket_tab.tab, text="Select Point", command = button_select_point, width = 10, fg = "white", bg = "blue")
    select_point_button.grid(row = 4, column = 1, sticky = "N")
    
    def button_modify_point():
        p = entry_rocket.get()        
        if draw_rocket_tab.active_point != "0,0":
            draw_rocket_tab.delete_point(0 , draw_rocket_tab.active_point)        
            draw_rocket_tab.combobox[0].set(p)
            button_add_point()
            select_point_button.config(bg="blue")
            draw_rocket_tab.active_point = "0,0"
        else:
            print("Please Select Point")
        return     
    modify_point_button = tk.Button(draw_rocket_tab.tab, text="Modify Point", command = button_modify_point, width = 10, fg = "white", bg = "blue")
    modify_point_button.grid(row = 4, column = 3, sticky = "N")    
    
    def button_save():        
        d = draw_rocket_tab.get_configuration()
        savefile.set_rocket_dim(d)
        savefile.save_all_configurations()
        return     
    save_file_button = tk.Button(draw_rocket_tab.tab, text="Save", command = button_save, width = 20)
    save_file_button.grid(row = 24, column = 3)
    
    ## Fins
    def load_fins_stabi():
        # populates the entries with the fin data
        for i in range(len(draw_rocket_tab.entry)):
            draw_rocket_tab.entry[i].delete(0,15)
            draw_rocket_tab.entry[i].insert(0, draw_rocket_tab.points[1][i])
        draw_rocket_tab.draw_rocket()
        return     
    update_fins_stabi_button = tk.Button(draw_rocket_tab.tab, text="Load Stabilization Fins", command = load_fins_stabi, width = 20, fg = "white", bg = "#12B200")
    update_fins_stabi_button.grid(row = 10, column = 1, sticky = "N", columnspan = 1)
    
    def update_fins_stabi():
        # updates the fin data with the entries
        for i in range(len(draw_rocket_tab.entry)):
            draw_rocket_tab.points[1][i] = draw_rocket_tab.entry[i].get()
        draw_rocket_tab.draw_rocket()
        return     
    update_fins_stabi_button = tk.Button(draw_rocket_tab.tab, text="Update Stabilization Fins", command = update_fins_stabi, width = 20, fg = "black", bg = "#C8FFC4")
    update_fins_stabi_button.grid(row = 12, column = 1, sticky = "N", columnspan = 1)
    
    def load_fins_control():
        # Same as prev
        for i in range(len(draw_rocket_tab.entry)):
            draw_rocket_tab.entry[i].delete(0,15)
            draw_rocket_tab.entry[i].insert(0, draw_rocket_tab.points[2][i])
        draw_rocket_tab.draw_rocket()
        return     
    update_fins_stabi_button = tk.Button(draw_rocket_tab.tab, text="Load Control Fins", command = load_fins_control, width = 20, fg = "white", bg = "#B20000")
    update_fins_stabi_button.grid(row = 10, column = 3, sticky = "N")
    
    def update_fins_control():
        # Same as prev
        for i in range(len(draw_rocket_tab.entry)):
            draw_rocket_tab.points[2][i] = draw_rocket_tab.entry[i].get()
        draw_rocket_tab.draw_rocket()
        return     
    update_fins_control_button = tk.Button(draw_rocket_tab.tab, text="Update Control Fins", command = update_fins_control, width = 20, fg = "black", bg = "#FFACAC")
    update_fins_control_button.grid(row = 12, column = 3, sticky = "N")
    
    def reset_fin():
        # Populates the entries with a "0" area fin
        for i in range(len(draw_rocket_tab.entry)):
            draw_rocket_tab.entry[i].delete(0,15)
            zero_fin = ["0.0001,0.0000","0.0001,0.0001","0.0002,0.0001","0.0002,0.0000"]
            draw_rocket_tab.entry[i].insert(0, zero_fin[i])
        draw_rocket_tab.draw_rocket()
        return     
    update_fins_control_button = tk.Button(draw_rocket_tab.tab, text="Reset Fin", command = reset_fin, width = 12, fg = "black", bg = "yellow")
    update_fins_control_button.grid(row = 19, column = 1, sticky = "N")
    
    def slider_AoA(a):
        # Changes the AoA and re draws the CP
        draw_rocket_tab.AoA = float(a)*deg2rad+0.01
        draw_rocket_tab.draw_rocket()
    AoA_scale = tk.Scale(draw_rocket_tab.tab, from_=0.01, to=90, orient=tk.HORIZONTAL, command = slider_AoA)
    AoA_scale.grid(row = 20, column = 0)
    tk.Label(draw_rocket_tab.tab, text="Angle of Attack").grid(row=21,column=0)
    
    draw_rocket_tab.create_active_file_label(24,0)
    draw_rocket_tab.configure()  
    return

     

## 3D TAB - 3D TAB - 3D TAB - 3D TAB - 3D TAB - 3D TAB - 3D TAB - 3D TAB - 3D TAB - 3D TAB - 3D TAB - 3D TAB - 3D TAB - 3D TAB - 3D TAB - 3D TAB - 3D TAB - 3D TAB -
def create_conf_3D_tab(notebook):    
    conf_3D_tab.create_tab(notebook, "3D Configuration")    
    checkboxes = ["Activate 3D Graphics","Camera Shake","Hide Forces","Variable Fov"]
    conf_3D_tab.create_checkboxes(checkboxes,0,0,"W",True)    
    combobox_options = [["Follow", "Fixed", "Follow Far"]]
    names_combobox = ["Camera Type"]
    conf_3D_tab.create_combobox(combobox_options,names_combobox,1,1)    
    names_entry = ["Slow Motion", "Force Scale Factor", "Fov"]
    conf_3D_tab.create_entry(names_entry, 2, 1, "W")    
    def button_save():              
        d = conf_3D_tab.get_configuration()
        savefile.set_conf_3D(d)
        savefile.save_all_configurations()      
        return         
    save_conf_3D_button = tk.Button(conf_3D_tab.tab, text="Save", command = button_save, width = 20)
    save_conf_3D_button.grid(row = 87, column = 4) 
    conf_3D_tab.change_state()    
    conf_3D_tab.create_active_file_label(87,0)
    conf_3D_tab.configure(5)
    return


## SITL TAB - SITL TAB - SITL TAB - SITL TAB - SITL TAB - SITL TAB - SITL TAB - SITL TAB - SITL TAB - SITL TAB - SITL TAB - SITL TAB - SITL TAB - SITL TAB 
def create_SITL_tab(notebook):
    conf_SITL_tab.create_tab(notebook, "SITL")
    checkboxes = ["Activate Software in the Loop", "Use Simulated Sensor Noise"]
    conf_SITL_tab.create_checkboxes(checkboxes,0,0,"W",True)    
    fun.move_tk_object(conf_SITL_tab.checkbox[0], columnspan = 2)
    fun.move_tk_object(conf_SITL_tab.checkbox[1], 5, 0, 2)
    names_entry = ["Port", "Baudrate", 
                     "Gyroscope SD", "Accelerometer SD","Altimeter SD"]
    conf_SITL_tab.create_entry(names_entry, 1, 0, "W")
    for i in range(3):
        fun.move_tk_object(conf_SITL_tab.entry_label[i+2], 6+i)
        fun.move_tk_object(conf_SITL_tab.entry[i+2], 6+i, 1)
    def change_state_SITL():
        if conf_SITL_tab.checkbox_status[1].get() == "True":
            for i in range(3):
                conf_SITL_tab.entry[i+2].config(state="normal")
        else:
            for i in range(3):
                conf_SITL_tab.entry[i+2].config(state="disable")
    conf_SITL_tab.checkbox[1].config(command = change_state_SITL)
    conf_SITL_tab.change_state()
    def button_save_conf_SITL():              
        d = conf_SITL_tab.get_configuration()        
        savefile.set_conf_SITL(d) 
        savefile.save_all_configurations()
        return         
    save_conf_SITL_button = tk.Button(conf_SITL_tab.tab, text="Save", command = button_save_conf_SITL, width = 20)
    save_conf_SITL_button.grid(row = 45, column = 23) 
    conf_SITL_tab.create_active_file_label(45,0)
    conf_SITL_tab.configure(10)
    return

## SIM SETUP TAB - SIM SETUP TAB - SIM SETUP TAB - SIM SETUP TAB - SIM SETUP TAB - SIM SETUP TAB - SIM SETUP TAB - SIM SETUP TAB - SIM SETUP TAB - SIM SETUP TAB -
def create_simulation_setup_tab(notebook):    
    sim_setup_tab.create_tab(notebook, "Sim Setup")    
    checkboxes = ["Torque Controller","Anti Windup"]
    sim_setup_tab.create_checkboxes(checkboxes,0,2,"W")    
    combobox_options = [["Step [º]", "Ramp [º/s]", "Up"]]
    names_combobox = [""]
    sim_setup_tab.create_combobox(combobox_options,names_combobox,6,1)    
    names_entry = ["Kp =", "Ki =", "Kd =", "K All =", "K Damping =", "Reference Thrust [N] =",
                       "Input =","Input Time =","Launch Time =", "Servo Sample Time [s] =",
                       "Controller Sample Time [s] =", "Maximum Sim Duration [s] =", "Sim Delta T [s] ="]
    sim_setup_tab.create_entry(names_entry, 0, 0, "W")    
    def button_save():
        d = sim_setup_tab.get_configuration()
        savefile.set_conf_controller(d)
        savefile.save_all_configurations()
        return 
    h = 38
    save_conf_controller_button = tk.Button(sim_setup_tab.tab, text="Save", command = button_save, width = 20)
    save_conf_controller_button.grid(row = h, column = 3, columnspan=2, sticky="E")    
    sim_setup_tab.create_active_file_label(h,0)
    sim_setup_tab.configure(10)
    return


## RUN SIM TAB - RUN SIM TAB - RUN SIM TAB - RUN SIM TAB - RUN SIM TAB - RUN SIM TAB - RUN SIM TAB - RUN SIM TAB - RUN SIM TAB - RUN SIM TAB - RUN SIM TAB
def create_run_sim_tab(notebook):    
    run_sim_tab.create_tab(notebook, "Run Simulation")    
    combobox_options = []    
    for i in range(5):
        combobox_options.append(["Off","Setpoint", "Pitch Angle", "Pitch Rate", "Actuator deflection",
                                 "Local Velocity X",  "Local Velocity Z", "Global Velocity X","Global Velocity Z", "Total Velocity",
                                 "Local Acc X",  "Local Acc Z", "Global Acc X","Global Acc Z",
                                 "Angle of Atack","Cp Position", "Altitude","Distance Downrange",
                                 "Proportional Contribution", "Integral Contribution", "Derivative Contribution","Total Error",
                                 "Simulated Gyro", "Simulated Acc X", "Simulated Acc Z", "Simulated Altimeter"])        
    names_combobox = ["First Plot","Second Plot", "Third Plot", "Fourth Plot", "Fifth Plot"]
    run_sim_tab.create_combobox(combobox_options,names_combobox,0,0)    
    def button_run_sim():
        sim.run_simulation()
        sim.run_3D()
        return         
    save_conf_controller_button = tk.Button(run_sim_tab.tab, text="Run Simulation", command = button_run_sim, width = 20, bg="red", fg = "white")
    save_conf_controller_button.grid(row = 2, column = 5)
    def button_save():
        d = run_sim_tab.get_configuration()
        savefile.set_conf_plots(d)
        savefile.save_all_configurations()      
        return     
    save_file_button = tk.Button(run_sim_tab.tab, text="Save", command = button_save, width = 20)
    save_file_button.grid(row = 47, column = 10)    
    run_sim_tab.create_active_file_label(47,0)  
    run_sim_tab.configure()
    return
        
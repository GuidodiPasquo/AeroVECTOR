# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 17:09:52 2021

@author: guido
"""
import copy
import numpy as np

deg2rad=np.pi/180
rad2deg=1/deg2rad

class airfoil:
    def __init__(self):
        """
        There are three ways of calculating the CL of the fin:
            Using a 2 Pi slope, this is what Open Rocket uses
            It is a good approximation for small AoA (less than
            5º for large AR, less than 15º for small AoA)
            
            Using wind tunnel data, this might seem like the 
            best idea, but the stall occurs always at 5º, 
            thing that doesn't happen in real life
            
            Using modified wind tunnel data: I just removed
            the stall from the data, so small AR fins have
            a more realistic behavior. They don't stall at
            5º nor increase they CL indefinitely.
        """        
        
        self.use_2pi = False
        self.use_windtunel = False
        self.use_windtunnel_modified = True
        
        """
        NACA 0009 from:
        Aerodynamic Characteristics of Seven
        Symmetrical Airfoil Sections Through
        180-Degree Angle of Attack for Use in
        Aerodynamic Analysis of Vertical Axis
        Wind Turbines
        """
        self.AoA_CL = [0.0, 0.08386924860853417, 0.1620549165120595, 0.25889438775510176, 0.4498044063079776, 0.62461094619666,
                  0.7766831632653057, 0.9458061224489793, 1.1321255565862705, 1.3189550556586267, 1.595045918367347, 
                  1.846872263450835, 2.0253221243042674, 2.2514994897959184, 2.4605532467532463, 2.6914669294990725,
                  2.83479517625232, 2.98810612244898, 3.105202458256029, 3.150015306122449]
        self.CL_list = [0.0, 0.5363636363636364, 0.781818181818182, 0.7, 0.8818181818181818, 1.0727272727272728, 1.1, 1.0, 
              0.7545454545454546, 0.44545454545454555, 0.0, -0.418181818181818, -0.6818181818181817, -0.9000000000000004,
              -0.9818181818181819, -0.790909090909091, -0.6727272727272728, -0.8000000000000003, -0.40909090909090917, 0.0]        
        
        self.AoA_CL_modified = [0.0, 0.13658937500000003, 0.7249800000000002, 0.9137768750000002, 1.0950218750000003, 1.57079,
                                2.0239025, 2.4166, 2.937679375, 3.103820625]        
        self.CL_list_modified = [0.0, 0.7739130434782606, 1.1043478260869564, 1.026086956521739, 0.8086956521739128, 0.0,
                                 -0.7217391304347824, -0.991304347826087, -0.8086956521739128, -0.008695652173913215]        
        
        self.AoA_CD = [0, 0.18521494914149783, 0.3982734933252401, 0.7643841635684416, 1.10988734808282, 1.3509577689207464,
                  1.5468165300174515, 1.7918205335344766, 1.960067086805354, 2.1704261134642255, 2.359869138105717,
                  2.4862486286726804, 2.668785095490799, 2.819614501925884, 2.9527883625749265, 3.040406988525037,
                  3.1349532075000113, 3.1415972095888027]
        self.CD_list = [0.001, 0.11315179890472304, 0.3266637804448056, 1.053982763006967, 1.5607557140498007, 1.7623865781756995, 
              1.8074741244259172, 1.7606280656808881, 1.5891418548289915, 1.3457262344177494, 1.0581157136927424,
              0.8182796034866748, 0.47852963361347545, 0.262866375366543, 0.13929739838341826, 0.05557613600353495,
              0.011970382007828295, 0.0009187156610269724]
        
    def _Calculate_CN_CA(self,AoA,Beta):
        self.Beta = Beta
        self._sign = self.__sign_correction(AoA)
        # Data goes from 0º-180º, later it is corrected with
        # the self._sign variable in case the AoA is negative
        self.x = abs(AoA)
        # Obtain current CL, CD and use them to obtain the normal and
        # axial coefficients RELATED TO THE FIN
        
        if self.use_windtunel == True:
            AoA_list_interp = self.AoA_CL
            CL_list_interp = self.CL_list
        elif self.use_windtunnel_modified == True:
            AoA_list_interp = self.AoA_CL_modified
            CL_list_interp = self.CL_list_modified
            
        
        self.CL = np.interp(self.x, AoA_list_interp , CL_list_interp)
        self.CD = np.interp(self.x, self.AoA_CD , self.CD_list)
        # Sing correction
        self.CN = self._sign * (self.CL * np.cos(self.x) + self.CD * np.sin(self.x))
        # CA always agains the fin, independent of AoA   
        self.CA = -self.CL * np.sin(self.x) + self.CD * np.cos(self.x)     
        # Compressibility correction
        self.CN = self.CN/self.Beta        
        return self.CN
        
    def _Calculate_CN_Alpha(self, AoA, Beta):
        # Prandtl-Glauert applied in the CN 
        self.CN_Alpha = self.CN/AoA
        if self.use_2pi == True:
            self.CN_Alpha = 2*np.pi/self.Beta
        
    def __sign_correction(self, AoA):    
        # Saves the sign of the AoA to apply it after computing the CN
        if AoA >= 0:
            x = 1
        else:
            x = -1
        return x
    
    def get_Aero_coef(self, AoA, Beta):
        self._Calculate_CN_CA(AoA, Beta)
        self._Calculate_CN_Alpha(AoA, Beta)
        return self.CN, self.CA, self.CN_Alpha

class fin_class:
    def __init__(self):
        self.dim = 0
        self.flat_plate = airfoil()
        
    def update(self,l):
        self.dim = copy.deepcopy(l)
        self._Calculate_real_dimension()
        self._Calculate_area()
        self._Calculate_MAC_Xf_AR()
        self._Calculate_force_application_point()
        self._Calculate_sweepback_angle()
        
    def _Calculate_real_dimension(self):
        """
        Rocket points go from the tip down to the tail
         
        Fin[n][x position (longitudinal), z position (span)]
        
         [0]|\
            | \[1]
            | |
         [3]|_|[2]
         """
        self.c_root = self.dim[3][0] - self.dim[0][0]
        self.c_tip = self.dim[2][0] - self.dim[1][0]
        # sweep length
        self.Xt = self.dim[1][0] - self.dim[0][0]
        # b = wingspan
        self.b = self.dim[1][1] - self.dim[0][1]         
        
    def _Calculate_area(self):
        self.area = (self.c_root + self.c_tip) * self.b / 2        
    
    def Area(self):
        return self.area
    
    def Dim(self):
        return copy.deepcopy(self.dim)
    
    def Total_pos(self):
        return self.dim[0][0] + self.c_root / 2
    
    def _Calculate_MAC_Xf_AR(self):
        k1 = self.c_root + 2 * self.c_tip
        k2 = self.c_root + self.c_tip
        k3 = self.c_root**2 + self.c_tip**2 + self.c_root*self.c_tip
        if k2 != 0 :
            # Position of the MAC along the wingspan (from c_root)
            self.yMAC = (self.b / 3) * (k1/k2)
            # Position of the CP in relation to the LE of c_root
            self.Xf = (self.Xt/3) * (k1/k2) + (1/6) * (k3/k2)
        else:
            self.Xf = 0
        if self.area != 0:
            # Because the fin is attached to the body, there are
            # no wingtip vortices in the root. For this reason 
            # the aspect ratio is that of a wing composed of two
            # fins attached together at the root of each other
            self.AR = 2 * self.b**2 / self.area            
    
    def _Calculate_force_application_point(self):
        self.CP_total = self.dim[0][0] + self.Xf
    
    def CP(self):
        return self.CP_total
        
    def _Calculate_sweepback_angle(self):
        x_tip = self.dim[1][0] + 0.25 * self.c_tip
        x_root = self.dim[0][0] + 0.25 * self.c_root
        self.sb_angle = np.arctan((x_tip - x_root) / self.b)  
        
    def Calculate_CN_Alpha(self,AoA, Beta, fins_attached):
        # 2D coefficients
        self.CN0, self.CA, self.CN_Alpha_0 = self.flat_plate.get_Aero_coef(AoA, Beta)        
        k1 = 1/(2*np.pi)        
        if fins_attached == True:
            AR = self.AR
        else:
            # Fins aren't attached to the body, so the lift distribution is closer to the one of the fin alone.
            # End plate theory in Airplane Performance, Stability and Control, Perkings and Hage.
            # Is assumed that the piece of body where the fin is attached acts as an end plate of 0.2 h/b
            # Rounding down, the aspect ratio is increased by 1-(r/2) being r the correction factor (r = 0.75)
            # One must remember that this end plate acts not to increase the AR/Lift Slope, but to mitigate 
            # its reduction. It also affects only the root, leaving the tip unaltered (unlike the ones in the
            # Perkins).
            # The formula checks at the extremes: r=1 (no end plate, AR = AR of one fin alone (0.5 calculated 
            # AR)), r=0 (fin attached to the body, AR doubles (i.e. stays the same as the one calculated prior))
            # It also compensates in the case that the fin is separated from the body by a servo or rod, 
            # accounting for the increase in lift the body produces, but not going to the extreme of calculating 
            # it as if it was attached.
            AR = 0.625 * self.AR
        # Diederich's semi-empirical method
        F_D = AR / (k1 * self.CN_Alpha_0 * np.cos(self.sb_angle))        
        self.CN_Alpha = (self.CN_Alpha_0 * F_D * np.cos(self.sb_angle)) / (2 + F_D * np.sqrt(1+(4/F_D**2)))        
        return self.CN_Alpha
        
fin = [0]*2
fin[0] = fin_class()
fin[1] = fin_class()

class rocket_class:
    def __init__(self):
        self.Thrust = 0
        self.CN = 0
        self.xCp = 0
        self.CA = 0
        self.Q_damp = 0
        self.motor=[[],[]]
        self.is_in_the_pad_flag = True
        self.component_Cn = []
        self.component_Cn_Alpha = []
        self.component_Cm = []        
        self.station_cross_area = [0]
        self.component_plan_area = []
        self.component_volume = []
        self.component_centroid = []
        self.ogive = False        
        self.R_crit = 1
        self.Rel_rough = 150e-6 
        # Empirical method to calculate the CA from the CD, it should use a 
        # fitted third order polinomial but linear interpolations are easier
        self.AoA_list_Ca = [-180*deg2rad,-(180-17)*deg2rad,-(180-70)*deg2rad, -90*deg2rad,-70*deg2rad, -17*deg2rad, 
                            0, 17*deg2rad,70*deg2rad, 90*deg2rad,(180-70)*deg2rad,(180-17)*deg2rad,180*deg2rad]
        self.Ca_scale = [-1, -1.3 , -0.097777 , 0 , 0.097777, 1.3, 
                         1 , 1.3, 0.097777 , 0 , -0.097777 , -1.3, -1]
        
    def __initialize(self, n):
        self.is_in_the_pad_flag = True
        self.component_Cn = [0]*(n-1)
        self.component_Cn_Alpha = [0]*(n-1)
        self.component_Cm = [0]*(n-1)
        self.station_cross_area = [0]*n
        self.component_plan_area = [0]*(n-1)
        self.component_volume = [0]*(n-1)
        self.component_centroid = [0]*(n-1) 
        self.rocket_dim_Gothert = []
    
    def Update_Rocket(self, l0, xcg):
        l = copy.deepcopy(l0)        
        self.ogive = l[0]
        self.use_fins = l[1]
        self.fins_attached = [l[2],l[4]]
        self.use_fins_control = l[3] 
        self.update_rocket_dim(l[5])
        # In case one fin is not set up
        zero_fin = [[0.00001,0.0000],[0.0001,0.0001],[0.0002,0.0001],[0.0002,0.0000]]
        fin[1].update(zero_fin)         
        if self.use_fins == True:            
            fin[0].update(l[6]) 
            if self.use_fins_control == True:
                fin[1].update(l[7])                        
        self.Calculate_pitch_damping(xcg)
        self.Calculate_R_crit() 
    
    ## GEOMETRY - GEOMETRY - GEOMETRY - GEOMETRY - GEOMETRY - GEOMETRY - GEOMETRY -
    def update_rocket_dim(self,l):        
        self.rocket_dim = copy.deepcopy(l)
        self.__initialize(len(self.rocket_dim))
        self.max_diam = self._maximum_diameter()
        self.length = self.rocket_dim[-1][0]
        self.diameter = self.rocket_dim[1][1]
        self.fineness = self.length / self.max_diam 
        self._compute_total_rocket_plan_area() 
        self._Calculate_wet_area_body()        
        # Looks like the Reference Area is the maximum one,
        # not the base of the nosecone
        self.A_ref = np.pi * (self.max_diam / 2)**2
        # In case I used Gothert to calculate the compressible
        # coefficients of the body, it's not implemented
        self.A_ref_incompressible = self.A_ref
        for i in range(len(self.rocket_dim)-1):
            # Area of the top and base of each component             
            l = self.rocket_dim[i+1][0] - self.rocket_dim[i][0]
            r1 = self.rocket_dim[i][1] / 2
            r2 = self.rocket_dim[i+1][1] / 2            
            """
            r2 or else the list goes out of range, because the tip has 
            Area = 0, the list is initialized with that value
            """
            area = np.pi * r2**2
            self.station_cross_area[i+1] = area            
            plan_area = l * (r1 + r2)
            self.component_plan_area[i] = plan_area            
            volume = (1/3) * np.pi * l * (r1**2 + r2**2 + r1 * r2)
            self.component_volume[i] = volume            
            centroid = (l*(2*r2+r1))/(3*(r1+r2))
            self.component_centroid[i] = centroid            
            if self.ogive == True:
                # Ogive instead of cone
                l = self.rocket_dim[1][0] - self.rocket_dim[0][0]
                self.component_centroid[0] = 0.4625 * l
                d = self.rocket_dim[1][1]
                self.component_plan_area[0] = 2/3 * l * d
    
    def _compute_total_rocket_plan_area(self):
        self.plan_area = 0
        for i in range(len(self.component_plan_area)):
            self.plan_area += self.component_plan_area[i]
    
    def _Calculate_wet_area_body(self):
        self.A_wet_component = [0]*(len(self.rocket_dim)-1)
        for i in range(len(self.A_wet_component)):
            l = self.rocket_dim[i+1][0] - self.rocket_dim[i][0]
            r1 = self.rocket_dim[i][1] / 2
            r2 = self.rocket_dim[i+1][1] / 2  
            self.A_wet_component[i] = np.pi * (r1+r2) * np.sqrt((r2-r1)**2 + l**2)
        self.A_wet_body = 0
        for i in range(len(self.A_wet_component)):
            self.A_wet_body += self.A_wet_component[i]
            
    def _maximum_diameter(self):
            d = 0
            for i in range(len(self.rocket_dim)):                
                if self.rocket_dim[i][1] > d:
                    d = self.rocket_dim[i][1]
            return d
    
    def _Calculate_avg_radius_fore(self):
        i = 0
        tot_radius = 0
        while self.rocket_dim[i][0] < self.xcg:
            tot_radius += self.rocket_dim[i][0]
            i += 1
        self.avg_rad_fore = tot_radius/self.xcg
        return
    
    def _Calculate_avg_radius_aft(self):
        i = len(self.rocket_dim)-1
        tot_radius = 0
        while self.rocket_dim[i][0] > self.xcg:
            tot_radius += self.rocket_dim[i][0]
            i -= 1
        self.avg_rad_aft = tot_radius/(self.rocket_dim[-1][0] - self.xcg)
        return 
    
    def get_A_ref(self):
        return self.A_ref_incompressible
                
    
    ## AERODYNAMICS - AERODYNAMICS - AERODYNAMICS - AERODYNAMICS - AERODYNAMICS -    
    def Calculate_Aero_coef(self,AoA,V0 = 10,rho = 1.225,mu = 1.784e-5, M = 0.0, Actuator_angle = 0):
        self.M = M
        self.Beta = np.sqrt(1-M**2)
        self.Actuator_angle = Actuator_angle
        self.AoA_ctrl_fin = AoA - Actuator_angle
        self._Calculate_total_Cn(AoA, Actuator_angle)
        self._Calculate_Cp_position(AoA)
        self._Calculate_total_Ca(AoA, V0, rho, mu)
        return self.Cn, self.Cp, self.Ca
    
    ## CN - CN - CN - CN - CN - CN - CN - CN
    def _Barrowman_Cn(self, AoA):        
        for i in range(len(self.station_cross_area)-1):
            k1 = ((2*np.sin(AoA))/self.A_ref)
            Cn = k1*(self.station_cross_area[i+1]-self.station_cross_area[i])            
            self.component_Cn[i] = Cn
        return
            
    def _Body_Cn(self, AoA):        
        K = 1.1
        Cn = [0]*len(self.component_plan_area)
        for i in range(len(self.component_plan_area)):
            k1 = (self.component_plan_area[i]/self.A_ref)
            Cn[i] = K * k1 * np.sin(AoA)**2
        for i in range(len(self.component_Cn)):
            self.component_Cn[i] += Cn[i]
        return
    
    def _Fin_CN(self, AoA, Actuator_angle):
        # og = original 3D non corrected for body
        # interference or separation
        self.CN_Alpha_og= [0]*len(fin) 
        self.CN_Alpha_fin_control = [0]*len(fin) 
        self.CN_Alpha_og[0] = fin[0].Calculate_CN_Alpha(AoA, self.Beta, self.fins_attached[0])
        """
        Because the Cn slope is linearized for Alpha
        Cn(AoA - Actuator_angle) = Cn(AoA) - Cn(Actuator_angle)
        As long as one computes the total CN for Cn(AoA - Actuator_angle)/AoA_ctrl_fin
        Minus because of how the actuator angle is measured, it rotates in the same direction
        as the TVC mount, wherever it is located.
        
        Since the fin rotated, and its normal force is no longer normal to the rocket,
        two slopes are calculated
        self.CN_Alpha_control_N = slope of the fin, normal to itself
        self.CN_Alpha_fin[1] -> normal of the fin, corrected to be normal to the rocket
        self.CN_Alpha_Axial -> Axial to the rocket
        
        self.CN_Alpha_fin[1] = self.CN_Alpha_control_N * np.cos(Actuator_angle)
        Cn = -self.CN_Alpha_fin[1] * (AoA - Actuator_angle)
        Induces extra drag that is added in self.Ca due to the normal force 
        beign rotated backwards
        
        The plotted Rocket CP does not contemplate the actuator influence, therefore, 
        to obtain the "real" cp one would need to include the extra force in there.
        It is that way because seeing the struggle of forces between the actuator's 
        and body's is the fun part.
        Example, imagine the rocket at AoA=30º and the fin at 30º, the real AoA of
        the fin is 0º, but it would be weird to see the fin deflected not generating
        any force. For this reason, the arrow would represent those 30º of AoA (fin),
        and the arrow of the body would also include a fictional fin at 30º (although
        parallel to the body), and, since a positive AoA (body) produces a negative CN,
        and a positive deflection angle (control fin) produces a positive CN, the end 
        result is the expected zero.        
        The opposite happens with the TVC, the misalignment is shown with a force arrow, 
        and when it disappears, it means that the motor is not producing any torque, even
        though the TVC mount has an angle.
        
        """                
        self.CN_Alpha_og[1] = fin[1].Calculate_CN_Alpha(self.AoA_ctrl_fin, self.Beta, self.fins_attached[1])        
        n_fin = 2
        self.CN_Alpha_fin = [0]*len(fin)
        # adimensionalise the CN for the rocket's area
        for i in range(len(fin)):
            self.CN_Alpha_fin[i] = n_fin * (fin[i].Area() / self.A_ref) * self.CN_Alpha_og[i]
        # Compensation for body interference            
        for i in range(len(fin)):
            if self.fins_attached[i] == True:
                r_body_at_fin = fin[i].Dim()[0][1]
                KT = 1 + (r_body_at_fin / (fin[i].b + r_body_at_fin))
                self.CN_Alpha_fin[i] = KT * self.CN_Alpha_fin[i]
                self.CN_Alpha_og[i] = KT * self.CN_Alpha_og[i]        
        # Still normal to the fin       
        self.CN_Alpha_control_N = self.CN_Alpha_fin[1]
        # now normal to the rocket:
        self.CN_Alpha_fin[1] = self.CN_Alpha_control_N * np.cos(Actuator_angle)        
        # Absolute because CA is always positive, i.e., pointing backwards, 
        # independently of the actuator angle being positive or negative
        self.CN_Alpha_Axial = self.CN_Alpha_control_N * abs(np.sin(Actuator_angle))
    
    def _Calculate_Cn_Alpha(self, AoA):
        for i in range(len(self.component_Cn)):
            if AoA != 0:
                self.component_Cn_Alpha[i] = self.component_Cn[i]/AoA
            else:
                self.component_Cn_Alpha[i] = 0
        return
   
    def _Calculate_total_Cn(self,AoA, Actuator_angle):
        self.Cn = 0
        self._sign_corr = self._sign_correction(AoA)
        if AoA == 0:
            AoA = 0.0001            
        else:
            AoA = abs(AoA)                
        self._Barrowman_Cn(AoA)
        self._Body_Cn(AoA)
        self._Calculate_Cn_Alpha(AoA)
        if self.use_fins == True:
            self._Fin_CN(AoA, Actuator_angle)                   
        for i in range(len(self.component_Cn)):
            self.Cn += self.component_Cn[i]            
        if self.use_fins == True:
            for i in range(len(fin)):           
                self.Cn += self.CN_Alpha_fin[i] * AoA             
        self.Cn = self._sign_corr * self.Cn
        return self.Cn
    
    def _sign_correction(self, AoA):        
        # Corrects the Cn since positive AoA produces a negative Cn in Z        
        if AoA >= 0:
            x = -1
        else:
            x = 1
        return x
    
    ## CP - CP - CP - CP - CP - CP - CP - CP
    def _Calculate_Cp_position(self,AoA):
        a = 0
        b = 0
        self.Cp = 0
        # Cp position = moment/force
        # It is supposed that the force is applied
        # in the centroid of the body component
        # Since the AoA is the same for all components,
        # the CN slopes are used
        for i in range(len(self.component_Cn_Alpha)):
           a += (self.component_centroid[i] + self.rocket_dim[i][0]) * self.component_Cn_Alpha[i]
           b += self.component_Cn_Alpha[i]        
        if self.use_fins == True:
            for i in range(len(fin)):
                a += fin[i].CP() * self.CN_Alpha_fin[i]                
                b += self.CN_Alpha_fin[i]
        if b != 0:
            self.Cp = a/b
        else:
            self.Cp = self.component_centroid[0]
        return self.Cp
    
    ## CM - CM - CM - CM - CM - CM - CM - CM
    def _Calculate_Cm(self,AoA):
        # Not used nor verified.
        k1 = (2*np.sin(AoA))/(self.A_ref*self.diameter)
        for i in range(len(self.station_cross_area)):
            l_component = self.rocket_dim[i+1][0]-self.rocket_dim[i][0]
            k2 = l_component*self.station_cross_area[i+1]-self.component_volume[i]
            Cm = k1*k2
            self.component_Cm[i] = Cm
        return
    
    ## PITCH DAMPING - PITCH DAMPING
    def Calculate_pitch_damping(self, xcg):
        # Average drag moment of a cilinder rotating, 
        # i.e., facing a 90º AoA
        self.xcg = xcg
        self._Calculate_avg_radius_fore()
        self._Calculate_avg_radius_aft()
        l_fore = self.xcg
        l_aft = (self.rocket_dim[-1][0] - self.xcg)        
        Q_damp_fore = 0.275*self.avg_rad_fore*l_fore**4
        Q_damp_aft = 0.275*self.avg_rad_aft*l_aft**4        
        if self.use_fins == True:
            Q_damp_fin = self._Calculate_pitch_damp_fin()
        else:
            Q_damp_fin = 0        
        self.Q_damp =  Q_damp_fore + Q_damp_aft + Q_damp_fin
        # Has to be multiplied by rho * Q**2 to obtain a moment
    
    def _Calculate_pitch_damp_fin(self):
        # Average drag moment of a flat plate
        # rotating, i.e., facing a 90º AoA
        CD_flat_plate = 1.28
        m=0
        for i in range(len(fin)):
            F = 0.5 * (fin[i].CP()-self.xcg)**2 * CD_flat_plate * fin[i].Area()*2 # * rho * Q**2 for the force 
            m += abs(F * (fin[i].CP() - self.xcg))        
        return m
    
    def get_Q_damp(self):
        return self.Q_damp
    
    ## DRAG - DRAG - DRAG - DRAG - DRAG
    def _Calculate_total_Ca(self,AoA = 0, V0 = 10, rho = 1.225, mu = 1.784e-5):
        """
        Boundry layer always turbulent
        There are no boattails, if they are, they are treated as shoulders
        Base drag is not reduced by motor exhaust
        There is no interpolation of the pressure drag for compressibility
        
        More detailed explanations of the methods applied here are in the
        Open Rocket's documentation.
        """
        self._Calculate_Reynolds(V0, rho, mu)
        self._Calculate_Cf()
        self._Calculate_pressure_drag(AoA)
        self._Calculate_base_drag()
        self._Calculate_CD(AoA)
        self._Calculate_Ca(AoA)
        return
    
    def _Calculate_Reynolds(self, V0, rho, mu):
        self.Reynolds = (rho * V0 * self.length)/(mu)
        
    def Calculate_R_crit(self):
        self.Rel_rough = 60e-6
        self.R_crit = 51 * (self.Rel_rough/self.length)**-1.039
    
    def _Calculate_Cf(self):
        if self.Reynolds < 10e4:
            self.Cf = 1.48e-2
        elif self.Reynolds < self.R_crit:
            self.Cf = 1/((1.5 * np.log(self.Reynolds) - 5.6)**2)
        else:
            self.Cf = 0.032 * np.power((self.Rel_rough/self.length),0.2)
        self.Cf = self.Cf * (1 - 0.1 * self.M**2)
        
    def _Calculate_pressure_drag(self,AoA):
        n = (len(self.rocket_dim)-1)
        self.CD_pressure_component = [0] * n
        for i in range(n):
            l = self.rocket_dim[i+1][0] - self.rocket_dim[i][0]
            r1 = self.rocket_dim[i][1] / 2
            r2 = self.rocket_dim[i+1][1] / 2
            try:
                phi = np.arctan((r2-r1)/l)
            except ZeroDivisionError:
                print("Component length is zero")
            self.CD_pressure_component[i] = 0.8 * np.sin(phi)**2
        if self.ogive == True:
            self.CD_pressure_component[0] = 0
            
    def _Calculate_base_drag(self):
        self.Base_drag = 0.12 + 0.13 * self.M**2
        
    def _Calculate_CD(self,AoA):        
        self.CD0_friction = (self.Cf * ((1 + 1/(2*self.fineness)) * self.A_wet_body))/self.A_ref
        n = (len(self.station_cross_area)-1)
        self.pressure_CD = 0
        for i in range(n):            
            if i == n-1 and abs(AoA) > np.pi/2:
                # Rocket flying backwards, A_ref_component would be 0 so it has to be tricked
                A_ref_component = self.station_cross_area[i+1]
                # 1 is the pressure cd for a hollow cilinder
                self.CD_pressure_component[i] = 1
                self.pressure_CD += (A_ref_component/self.A_ref)*self.CD_pressure_component[i]
            else:
                A_ref_component = abs(self.station_cross_area[i+1] - self.station_cross_area[i])
                self.pressure_CD += (A_ref_component/self.A_ref)*self.CD_pressure_component[i]            
        self.base_drag_CD = (self.station_cross_area[-1]/self.A_ref) * self.Base_drag
        self.CD0 = self.pressure_CD + self.base_drag_CD + self.CD0_friction
        
    def _Fin_Ca(self):
        for i in range(len(fin)):
            # Asumes control fin parallel to the body
            self.Ca += fin[i].CA * (2 * fin[i].Area()/self.A_ref)
            
    def _control_fin_drag(self,AoA):
        # CN_Alpha_Axial = CN_Alpha * sin(actuator angle)
        # real force axial to the rocket = CN_Alpha_Axial * real AoA of the fin
        self.Ca += self.CN_Alpha_Axial * (AoA - self.Actuator_angle)
        
    def _Calculate_Ca(self, AoA):        
        # First order interpolation
        CD2Ca = np.interp(AoA,self.AoA_list_Ca, self.Ca_scale)        
        self.Ca = self.CD0 * CD2Ca
        if self.use_fins == True:
            self._Fin_Ca()
            self._control_fin_drag(AoA)  
    
    ## MOTOR - MOTOR - MOTOR - MOTOR - MOTOR - MOTOR - MOTOR - MOTOR - MOTOR         
    def set_motor(self, data):
        # Motor data from text files
        # t, Thrust        
        self.motor[0] = copy.deepcopy(data[0])
        self.motor[1] = copy.deepcopy(data[1])
        
    def get_Thrust(self, t, t_launch):
        x = t-t_launch
        self.Thrust = np.interp(x,self.motor[0], self.motor[1])
        if self.Thrust < 0.001:
            self.Thrust = 0.001
        return float(self.Thrust)
    
    def is_in_the_pad(self, alt):
        if alt>0.2 and self.is_in_the_pad_flag == True:
            self.is_in_the_pad_flag = False
        return self.is_in_the_pad_flag
    
    def burnout_time(self):
        return self.motor[0][-1]
    
    def reset_variables(self):
        self.Thrust = 0
        self.CN = 0
        self.xCp = 0
        self.CA = 0
        self.motor=[[],[]]
        self.is_in_the_pad_flag = True

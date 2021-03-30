# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 17:09:52 2021

@author: Guido di Pasquo
"""
"""
Handles the airfoil, fin and rocket body aerodynamics.

Classes:
    Airfoil -- 2D fin aerodynamics.
    Fin -- 3D fin aerodynamics.
    Rocket -- Rocket aerodynamics.
"""

import copy
import numpy as np

DEG2RAD = np.pi / 180
RAD2DEG = 1 / DEG2RAD


class Airfoil:
    """
    Airfoil class with wind tunnel data

    Methods:
        get_aero_coef -- Returns the CN, CA, and CN slope.
    """

    def __init__(self):
        """
        There are three ways of calculating the cl of the fin:
        Using a 2 Pi slope, this is what Open Rocket uses
        It is a good approximation for small aoa (less than
        5º for large AR, less than 15º for small aoa)

        Using wind tunnel data, this might seem like the
        best idea, but the stall occurs always at 5º,
        thing that doesn't happen in real life

        Using modified wind tunnel data: I just removed
        the stall from the data, so small AR fins have
        a more realistic behavior. They don't stall at
        5º nor increase they cl indefinitely.
        """
        self.use_2pi = False
        self.use_windtunel = False
        self.use_windtunnel_modified = True
        """
        NAca 0009 from:
        Aerodynamic Characteristics of Seven
        Symmetrical Airfoil Sections Through
        180-Degree Angle of Attack for Use in
        Aerodynamic Analysis of Vertical Axis
        Wind Turbines
        """
        self.aoa_cl = [0.0, 0.08386924860853417, 0.1620549165120595,
                0.25889438775510176, 0.4498044063079776, 0.62461094619666,
                0.7766831632653057, 0.9458061224489793, 1.1321255565862705,
                1.3189550556586267, 1.595045918367347, 1.846872263450835,
                2.0253221243042674, 2.2514994897959184, 2.4605532467532463,
                2.6914669294990725, 2.83479517625232, 2.98810612244898,
                3.105202458256029, 3.150015306122449]
        self.cl_list = [0.0, 0.5363636363636364, 0.781818181818182, 0.7,
                0.8818181818181818, 1.0727272727272728, 1.1, 1.0,
                0.7545454545454546, 0.44545454545454555, 0.0,
                -0.418181818181818, -0.6818181818181817, -0.9000000000000004,
                -0.9818181818181819, -0.790909090909091, -0.6727272727272728,
                -0.8000000000000003, -0.40909090909090917, 0.0]

        self.aoa_cl_modified = [0.0, 0.13658937500000003, 0.7249800000000002,
                0.9137768750000002, 1.0950218750000003, 1.57079,
                2.0239025, 2.4166, 2.937679375, 3.103820625]
        self.cl_list_modified = [0.0, 0.7739130434782606, 1.1043478260869564,
                1.026086956521739, 0.8086956521739128, 0.0,
                -0.7217391304347824, -0.991304347826087,
                -0.8086956521739128, -0.008695652173913215]

        self.aoa_cd = [0, 0.18521494914149783, 0.3982734933252401,
                0.7643841635684416, 1.10988734808282, 1.3509577689207464,
                1.5468165300174515, 1.7918205335344766, 1.960067086805354,
                2.1704261134642255, 2.359869138105717, 2.4862486286726804,
                2.668785095490799, 2.819614501925884, 2.9527883625749265,
                3.040406988525037, 3.1349532075000113, 3.1415972095888027]
        self.cd_list = [0.001, 0.11315179890472304, 0.3266637804448056,
                1.053982763006967, 1.5607557140498007, 1.7623865781756995,
                1.8074741244259172, 1.7606280656808881, 1.5891418548289915,
                1.3457262344177494, 1.0581157136927424, 0.8182796034866748,
                0.47852963361347545, 0.262866375366543, 0.13929739838341826,
                0.05557613600353495, 0.011970382007828295,
                0.0009187156610269724]
        self._sign = 1
        self.x = 0
        self.cl = 0
        self.cd = 0
        self.cn = 0
        self.ca = 0
        self.cn_alpha = 0

    def _calculate_cn_ca(self, aoa):
        self._sign = self.__sign_correction(aoa)
        # Data goes from 0º-180º, later it is corrected with
        # the self._sign variable in case the aoa is negative
        self.x = abs(aoa)
        # Obtain current cl, cd and use them to obtain the normal and
        # axial coefficients RELATED TO THE FIN
        if self.use_windtunel is True:
            aoa_list_interp = self.aoa_cl
            cl_list_interp = self.cl_list
        elif self.use_windtunnel_modified is True:
            aoa_list_interp = self.aoa_cl_modified
            cl_list_interp = self.cl_list_modified
        self.cl = np.interp(self.x, aoa_list_interp, cl_list_interp)
        self.cd = np.interp(self.x, self.aoa_cd, self.cd_list)
        # Sing correction
        self.cn = (self._sign *
                  (self.cl*np.cos(self.x) + self.cd*np.sin(self.x)))
        # ca always agains the fin, independent of aoa
        self.ca = -self.cl*np.sin(self.x) + self.cd*np.cos(self.x)
        return self.cn

    def _calculate_cn_alpha(self, aoa):
        # Prandtl-Glauert applied in the cn
        self.cn_alpha = self.cn / aoa
        if self.use_2pi is True:
            self.cn_alpha = 2 * np.pi

    def __sign_correction(self, aoa):
        # Saves the sign of the aoa to apply it after computing the cn
        if aoa >= 0:
            x = 1
        else:
            x = -1
        return x

    def get_aero_coef(self, aoa):
        """
        Calculate the CN, CA, and CN slope of the fin at a certain AoA

        Parameters
        ----------
        aoa : double
            Angle of attack (rad).

        Returns
        -------
        cn : double
            Normal force coefficient.
        ca : double
            Axial force coefficient.
        cn_alpha : double
            Normal force coefficient slope.
        """
        self._calculate_cn_ca(aoa)
        self._calculate_cn_alpha(aoa)
        return self.cn, self.ca, self.cn_alpha


class Fin:
    """Class that handles the fin geometry and its individual parameters
    (i.e. no body interference)

    Methods:
        update -- Update the fin characteristics
        calculate_cn_alpha -- Calculate the real CN_alpha of the fin.
    """

    def __init__(self):
        self._dim = 0
        self.flat_plate = Airfoil()
        self.fin_attached = True
        self.c_root = 1
        self.c_tip = 1
        self.x_tail = 1
        self.wingspan = 1
        self.area = 1
        self.y_mac = 1
        self.x_force_fin = 1
        self.aspect_ratio = 1
        self.cp_total = 1
        self.sb_angle = 1
        self.beta = 1
        self.mach = 0
        self.cn0 = 0
        self.ca = 0
        self.cn_alpha_0 = 0
        self.cn_alpha = 0

    def update(self, l, fin_attached=True):
        """
        Update the rocket with the data in l

        Parameters
        ----------
        l : list (variables)
            Data of the rocket.
        fin_attached : bool, optional
            Are fins attached?. The default is True.

        Returns
        -------
        None.
        """
        self.fin_attached = fin_attached
        self._dim = copy.deepcopy(l)
        self._calculate_real_dimension()
        self._calculate_area()
        self._calculate_mac_xf_ar()
        self._calculate_force_application_point()
        self._calculate_sweepback_angle()

    def _calculate_real_dimension(self):
        r"""
        Rocket points go from the tip down to the tail

        Fin[n][x position (longitudinal), z position (span)]

         [0]|\
            | \[1]
            | |
         [3]|_|[2]
         """
        self.c_root = self._dim[3][0] - self._dim[0][0]
        self.c_tip = self._dim[2][0] - self._dim[1][0]
        # sweep length
        self.x_tail = self._dim[1][0] - self._dim[0][0]
        self.wingspan = self._dim[1][1] - self._dim[0][1]

    def _calculate_area(self):
        self.area = (self.c_root+self.c_tip) * self.wingspan / 2

    @property
    def dim(self):
        return copy.deepcopy(self._dim)

    def _total_pos(self):
        return self._dim[0][0] + self.c_root/2

    def _calculate_mac_xf_ar(self):
        k1 = self.c_root + 2*self.c_tip
        k2 = self.c_root + self.c_tip
        k3 = self.c_root**2 + self.c_tip**2 + self.c_root*self.c_tip
        if k2 != 0 :
            # Position of the MAC along the wingspan (from c_root)
            self.y_mac = (self.wingspan/3) * (k1/k2)
            # Position of the CP in relation to the LE of c_root
            self.x_force_fin = (self.x_tail/3) * (k1/k2) + (1/6) * (k3/k2)
        else:
            self.x_force_fin = 0
        if self.area != 0:
            # Because the fin is attached to the body, there are
            # no wingtip vortices in the root. For this reason
            # the aspect ratio is that of a wing composed of two
            # fins attached together at the root of each other
            self.aspect_ratio = 2 * self.wingspan**2 / self.area
        if self.fin_attached is True:
            # aspect_ratio = aspect_ratio calculated before (2*aspect_ratio fin alone)
            pass
        else:
            # Fins aren't attached to the body, so the lift distribution is
            # closer to the one of the fin alone. End plate theory in Airplane
            # Performance, Stability and Control, Perkings and Hage. Is assumed
            # that the piece of body where the fin is attached acts as an end
            # plate of 0.2 h/wingspan. Rounding down, the aspect ratio is increased
            # by 1-(r/2) being r the correction factor (r = 0.75). One must
            # remember that this end plate acts not to increase the aspect_ratio/Lift
            # Slope, but to mitigate its reduction. It also affects only the
            # root, leaving the tip unaltered (unlike the ones in the Perkins).
            # The formula checks at the extremes: r=1 (no end plate, aspect_ratio = aspect_ratio
            # of one fin alone (0.5 calculated aspect_ratio)), r=0 (fin attached to the
            # body, aspect_ratio doubles (i.e. stays the same as the one calculated
            # prior)). It also compensates in the case that the fin is separated
            # from the body by a servo or rod, accounting for the increase in
            # lift the body produces, but not going to the extreme of calculating
            # it as if it was attached.
            self.aspect_ratio *= 0.625

    def _calculate_force_application_point(self):
        self.cp_total = self._dim[0][0] + self.x_force_fin

    @property
    def cp(self):
        return self.cp_total

    def _calculate_sweepback_angle(self):
        # 25% of the chord because
        x_tip = self._dim[1][0] + 0.25*self.c_tip
        x_root = self._dim[0][0] + 0.25*self.c_root
        self.sb_angle = np.arctan((x_tip-x_root) / self.wingspan)

    def calculate_cn_alpha(self, aoa, beta, mach):
        self.beta = beta
        self.mach = mach
        # 2D coefficients
        self.cn0, self.ca, self.cn_alpha_0 = self.flat_plate.get_aero_coef(aoa)
        # Diederich's semi-empirical method
        self._correct_cna_diederich()
        return self.cn_alpha

    def _correct_cna_diederich(self):
        # Plantform modification as in the original paper.
        eff_factor = self.cn_alpha_0 / (2*np.pi)
        cos_sbe = (((self.beta)
                    / np.sqrt((1 - self.mach**2 * np.cos(self.sb_angle)**2)))
                   * np.cos(self.sb_angle))
        k1 = 1 / self.beta
        k2 = (self.aspect_ratio * self.beta) / (eff_factor * cos_sbe)
        k3 = (4 * eff_factor**2 * cos_sbe**2) / (self.aspect_ratio**2 * self.beta**2)
        k4 = np.sqrt(1 + k3)
        self.cn_alpha = k1 * ((k2)/(k2*k4 + 2)) * self.cn_alpha_0 * cos_sbe



fin = [Fin(), Fin()]

class Rocket:
    """
    Handles the rocket's body and aerodynamics

    Methods:
        update_rocket -- Update the rocket characteristics.
        calculate_aero_coef -- Compute the aerodynamics of the rocket.
        get_q_damp -- Returns the damping coefficient.
        set_motor -- Set the rocket's motor.
        get_thrust -- Returns the motor's thrust.
        is_in_the_pad -- Check if the rocket is in the pad.
        burnout_time -- Returns burnout time of the motor.
        reset_variables -- Resets some variables of the rocket.
    """

    def __init__(self):
        self.thrust = 0
        self.cn = 0
        self._sign_corr = -1
        self.xcp = 0
        self.ca = 0
        self.cp = 0
        self.cm_xcg = 0
        self.xcg = 1
        self.q_damp = 0
        self.motor=[[],[]]
        self.is_in_the_pad_flag = True
        self.component_cn = []
        self.component_cn_alpha = []
        self.component_cm = []
        self.station_cross_area = [0]
        self.component_plan_area = []
        self.component_volume = []
        self.component_centroid = []
        self.ogive = False
        self.reynolds_crit = 1
        self.relative_rough = 150e-6
        # Empirical method to calculate the ca from the cd, it should use a
        # fitted third order polinomial but linear interpolations are easier
        self.aoa_list_ca = [-180 * DEG2RAD,-(180 - 17) * DEG2RAD,
                            -(180 - 70) * DEG2RAD, -90 * DEG2RAD,
                            -70 * DEG2RAD, -17 * DEG2RAD, 0,
                            17 * DEG2RAD, 70 * DEG2RAD, 90 * DEG2RAD,
                            (180 - 70) * DEG2RAD, (180 - 17) * DEG2RAD,
                            180 * DEG2RAD]
        self.ca_scale = [-1, -1.3 , -0.097777 , 0 , 0.097777, 1.3,
                         1 , 1.3, 0.097777 , 0 , -0.097777 , -1.3, -1]
        self.use_fins = False
        self.fins_attached = True
        self.use_fins_control = False
        self.rocket_dim = []
        self.max_diam = 1
        self.length = 1
        self.fineness = 1
        self.area_ref = 1
        self.plan_area = 1
        self.area_wet_component = 1
        self.area_wet_body = 1
        self.avg_rad_aft = 1
        self.avg_rad_fore = 1
        self.v_loc_tot = [10,0]
        self.mach = 0.001
        self.beta = 1
        self.q = 0
        self.actuator_angle = 0
        self.aoa_ctrl_fin = 0
        self.cn_alpha_og = 6.28
        self.cn_alpha_fin_control = 2
        self.cn_alpha_control_normal = 2
        self.cn_alpha_fin = 2
        self.ctrl_fin_ca = 0
        self.q_damp_body = 0
        self.reynolds = 100
        self.Cf = 1
        self.cd_pressure_component = 1
        self.base_drag = 1
        self.cd0_friction = 1
        self.total_pressure_drag = 1
        self.total_base_drag = 1
        self.cd0 = 1
        self.is_in_the_pad_flag = True
        self.component_cn = [0]*(1)
        self.component_cn_alpha = [0]*(1)
        self.component_cm = [0]*(1)
        self.station_cross_area = [0]*0
        self.component_plan_area = [0]*(1)
        self.component_volume = [0]*(1)
        self.component_centroid = [0]*(1)
        self.component_centroid_pos = [0]*(1)
        self.component_aoa = [0]
        self.component_tan_vel = [0]
        self.fin_aoa = [0,0]
        self.fin_tan_vel = [0,0]
        self.aoa_total = 0
        self._sign_correction = [0]
        self._sign_correction_fin = [0,0]
        self.fin_cn = [0,0]
        self.cn_alpha_ctrl_fin_3d_arrow = 0
        self.cp_w_o_ctrl_fin = 0

    def update_rocket(self, l0, xcg):
        """
        Update the Rocket instance with the data in l0 and the cg position.

        Parameters
        ----------
        l0 : list of variables
            Rocket data.
        xcg : float
            cg position.

        Returns
        -------
        None.
        """
        l = copy.deepcopy(l0)
        self.ogive = l[0]
        self.use_fins = l[1]
        self.fins_attached = [l[2], l[4]]
        self.use_fins_control = l[3]
        self.xcg = xcg
        self._update_rocket_dim(l[5])
        # In case one fin is not set up
        zero_fin = [[0.00001,0.0000],[0.0001,0.0001],[0.0002,0.0001],[0.0002,0.0000]]
        fin[1].update(zero_fin) # In case there are no control fins
        if self.use_fins is True:
            fin[0].update(l[6], self.fins_attached[0])
            if self.use_fins_control is True:
                fin[1].update(l[7], self.fins_attached[1])
        self._calculate_pitch_damping_body()
        self._calculate_reynolds_crit()

    def _update_rocket_dim(self, l):
        # Looks like the Reference area is the maximum one,
        # not the base of the nosecone.
        self.rocket_dim = copy.deepcopy(l)
        self._separate_xcg_component()
        self.__initialize(len(self.rocket_dim))
        self.max_diam = self._maximum_diameter()
        self.length = self.rocket_dim[-1][0]
        self.fineness = self.length / self.max_diam
        self.area_ref = np.pi * (self.max_diam/2)**2
        for i in range(len(self.rocket_dim)-1):
            l = self.rocket_dim[i+1][0] - self.rocket_dim[i][0]
            r1 = self.rocket_dim[i][1] / 2
            r2 = self.rocket_dim[i+1][1] / 2
            # r2 or else the list goes out of range, because the tip has
            # area = 0, the list is initialized with that value
            area = np.pi * r2**2
            self.station_cross_area[i+1] = area
                # area of the top and base of each component
            plan_area = l * (r1+r2)
            self.component_plan_area[i] = plan_area
            volume = (1/3) * np.pi * l * (r1**2 + r2**2 + r1*r2)
            self.component_volume[i] = volume
            centroid = (l * (2*r2 + r1)) / (3 * (r1+r2))
            self.component_centroid[i] = centroid
            self.component_centroid_pos[i] = centroid + self.rocket_dim[i][0]
            if self.ogive is True:
                # Ogive instead of cone
                l = self.rocket_dim[1][0] - self.rocket_dim[0][0]
                self.component_centroid[0] = 0.4625 * l
                d = self.rocket_dim[1][1]
                self.component_plan_area[0] = 2/3 * l * d
        self._compute_total_rocket_plan_area()
        self._calculate_wet_area_body()

    def _separate_xcg_component(self):
        for i,elem in enumerate(self.rocket_dim):
            if elem[0] > self.xcg:
                index = i
                break
            if elem[0] == self.xcg:
                return None
        x = [self.rocket_dim[index-1][0], self.rocket_dim[index][0]]
        diam = [self.rocket_dim[index-1][1], self.rocket_dim[index][1]]
        diameter_at_xcg = np.interp(self.xcg, x, diam)
        self.rocket_dim.insert(index, [self.xcg, diameter_at_xcg])
        return None

    def __initialize(self, n):
        self.is_in_the_pad_flag = True
        self.component_cn = [0]*(n-1)
        self.component_cn_alpha = [0]*(n-1)
        self.component_cm = [0]*(n-1)
        self.station_cross_area = [0]*n
        self.component_plan_area = [0]*(n-1)
        self.component_volume = [0]*(n-1)
        self.component_centroid = [0]*(n-1)
        self.component_centroid_pos = [0]*(n-1)

    def _maximum_diameter(self):
        d = 0
        for i in range(len(self.rocket_dim)):
            if self.rocket_dim[i][1] > d:
                d = self.rocket_dim[i][1]
        return d

    def _compute_total_rocket_plan_area(self):
        self.plan_area = 0
        for i in range(len(self.component_plan_area)):
            self.plan_area += self.component_plan_area[i]

    def _calculate_wet_area_body(self):
        self.area_wet_component = [0]*(len(self.rocket_dim)-1)
        for i in range(len(self.area_wet_component)):
            l = self.rocket_dim[i+1][0] - self.rocket_dim[i][0]
            r1 = self.rocket_dim[i][1] / 2
            r2 = self.rocket_dim[i+1][1] / 2
            self.area_wet_component[i] = np.pi * (r1+r2) * np.sqrt((r2-r1)**2 + l**2)
        self.area_wet_body = 0
        for i in range(len(self.area_wet_component)):
            self.area_wet_body += self.area_wet_component[i]

    @property
    def reference_area(self):
        return self.area_ref

    def _calculate_pitch_damping_body(self):
        # Average drag moment of a cilinder rotating, i.e., facing a 90º
        # aoa. It's how Open Rocket does it, it gives good looking
        # results (idk how accurate), and does not affect much when Q is
        # low, so it stays and its added in the simulation (maybe multiplied
        # by 0.5 or 0.1, since it would increase the stability of the rocket).
        # Has to be multiplied by rho * Q**2 to obtain a moment.
        self._calculate_avg_radius_fore()
        self._calculate_avg_radius_aft()
        l_fore = self.xcg
        l_aft = (self.rocket_dim[-1][0] - self.xcg)
        q_damp_fore = 0.275 * self.avg_rad_fore * l_fore**4
        q_damp_aft = 0.275 * self.avg_rad_aft * l_aft**4
        self.q_damp_body =  q_damp_fore + q_damp_aft

    def _calculate_avg_radius_fore(self):
        i = 0
        tot_radius = 0
        while self.rocket_dim[i][0] < self.xcg:
            tot_radius += self.rocket_dim[i][0]
            i += 1
        self.avg_rad_fore = tot_radius / self.xcg

    def _calculate_avg_radius_aft(self):
        i = len(self.rocket_dim)-1
        tot_radius = 0
        while self.rocket_dim[i][0] > self.xcg:
            tot_radius += self.rocket_dim[i][0]
            i -= 1
        self.avg_rad_aft = tot_radius / (self.rocket_dim[-1][0]-self.xcg)

    def get_q_damp_body(self):
        return self.q_damp_body

    def _calculate_reynolds_crit(self):
        self.relative_rough = 60e-6
        self.reynolds_crit = 51 * (self.relative_rough/self.length)**-1.039

    ## AERODYNAMICS - AERODYNAMICS - AERODYNAMICS - AERODYNAMICS - AERODYNAMICS -
    def calculate_aero_coef(self, v_loc_tot=[10,0], q=0, rho=1.225, mu=1.784e-5,
                            mach=0.001, actuator_angle=0):
        """
        Calculate the CN, CP position and the CA for a certain AoA, velocity
        density, viscosity, mach, and actuator angle.

        Parameters
        ----------
        v_loc_tot : list, optional
            Total velocity. The default is [10,0].
        q : float, optional
            Pitching velocity. The default is 0.
        rho : float, optional
            Density. The default is 1.225.
        mu : float, optional
            Dynamic viscosity. The default is 1.784e-5.
        mach : float, optional
            Mach number. The default is 0.001.
        actuator_angle : float, optional
            Angle of the control fins. The default is 0.

        Returns
        -------
        cn : float
            Normal force coefficient.
        cp : float
            position of the Center of Pressure.
        ca
            Axial force coefficient.
        """
        self.q = q
        self.v_loc_tot = v_loc_tot
        v_modulus = np.sqrt(v_loc_tot[0]**2 + v_loc_tot[1]**2)
        if mach < 0.001:
            self.mach = 0.001
        else:
            self.mach = mach
        self.beta = np.sqrt(1 - mach**2)
        self.actuator_angle = actuator_angle
        self._calculate_aoa_components()
        self._calculate_total_cn()
        self._calculate_cp_position()
        self._calculate_cm()
        self._calculate_total_ca(self.aoa_total, v_modulus, rho, mu)
        return self.cn, self.cm_xcg, self.ca, self.cp

    def _calculate_aoa_components(self):
        self.component_aoa = [0] * len(self.component_centroid)
        self.component_tan_vel = [0] * len(self.component_centroid)
        self.fin_aoa = [0,0]
        self.fin_tan_vel = [0,0]
        for i, elem in enumerate(self.component_tan_vel):
            r = self.component_centroid_pos[i] - self.xcg
            self.component_tan_vel[i] = self.q * r
        for i, elem in enumerate(self.component_aoa):
            self.component_aoa[i] = self._calculate_aoa(self.component_tan_vel[i])
        for i in range(2):
            r = fin[i].cp - self.xcg
            self.fin_tan_vel[i] = self.q * r
        for i in range(2):
            if i == 0:
                self.fin_aoa[i] = self._calculate_aoa(self.fin_tan_vel[i])
            else:
                self.fin_aoa[i] = self._calculate_aoa(self.fin_tan_vel[i])
                self.fin_aoa[i] -= self.actuator_angle # + delta gives -aoa
        self.aoa_total = self._calculate_aoa(0)

    def _calculate_aoa(self, v_tan):
        if self.v_loc_tot[0] != 0:
            aoa = np.arctan2(self.v_loc_tot[1]+v_tan, self.v_loc_tot[0])
        else:
            aoa = np.pi/2
        return aoa

    def _calculate_total_cn(self):
        self.cn = 0
        self.__sign_correction()
        self._barrowman_cn()
        self._body_cn()
        if self.use_fins is True:
            self._fin_cn()
        for i in range(len(self.component_cn)):
            self.cn += self.component_cn[i]
        if self.use_fins is True:
            for i in range(len(fin)):
                self.cn += self.fin_cn[i]
        return self.cn

    def __sign_correction(self):
        # Corrects the cn since positive aoa produces a negative cn in Z
        self._sign_correction = [0] * len(self.component_aoa)
        for i in range(len(self.component_aoa)):
            if self.component_aoa[i] >= 0:
                self._sign_correction[i] = -1
            else:
                self._sign_correction[i] = 1
        self._sign_correction_fin = [0,0]
        for i in range(2):
            if self.fin_aoa[i] >= 0:
                self._sign_correction_fin[i] = -1
            else:
                self._sign_correction_fin[i] = 1

    def _barrowman_cn(self):
        for i in range(len(self.station_cross_area)-1):
            aoa = abs(self.component_aoa[i])
            k1 = (2*np.sin(aoa) / self.area_ref)
            cn = k1 * (self.station_cross_area[i+1]-self.station_cross_area[i])
            cn *= self._sign_correction[i]
            self.component_cn[i] = cn

    def _body_cn(self):
        K = 1.1
        cn = [0]*len(self.component_plan_area)
        for i in range(len(self.component_plan_area)):
            aoa = abs(self.component_aoa[i])
            k1 = self.component_plan_area[i] / self.area_ref
            cn[i] = K * k1 * np.sin(aoa)**2
            cn[i] *= self._sign_correction[i]
        for i in range(len(self.component_cn)):
            self.component_cn[i] += cn[i]

    def _fin_cn(self):
        # Since the control fin rotates, its normal force is no
		# longer normal to the rocket, the cn is corrected with the
        # cos(actuator_angle) to compensate.
        # Induces extra drag that is added in self.ca due to the
        # normal force beign rotated backwards

        # The cn 3D arrow does not contemplate the actuator influence
        # (whole control fin, reason why it has its own cn slope
        # (cn_alpha_ctrl_fin_3d_arrow)). It is that way because seeing
        # the struggle of forces between the actuator's and body's is
        # the fun part. The plotted cp is the total one, including all
        # the aerodynamic forces.

        # The cp can move beyond the limits show with the slider in
        # the GUI due to damping, which can produce a moment while the
        # cn is zero.

        # og = original 3D non corrected for body interference or separation.
        self.cn_alpha_og = [0]*len(fin)
        self.fin_cn = [0]*len(fin)
        self.cn_alpha_fin_control = [0]*len(fin)
        self.cn_alpha_fin = [0]*len(fin)
        self._obtain_fin_cn()
        self._nondimensionalize_fin_cn()
        self._compute_body_interference()
        self._correct_sign_cn_fin()
        self._correct_ctrl_fin_coeff_for_actuator_angle()

    def _obtain_fin_cn(self):
        for i in range(2):
            self.cn_alpha_og[i] = fin[i].calculate_cn_alpha(abs(self.fin_aoa[i]), self.beta, self.mach)

    def _nondimensionalize_fin_cn(self):
        n_fin = 2
        for i in range(len(fin)):
            self.cn_alpha_fin[i] = n_fin * (fin[i].area/self.area_ref) * self.cn_alpha_og[i]

    def _compute_body_interference(self):
        for i in range(len(fin)):
            if self.fins_attached[i] is True:
                r_body_at_fin = fin[i].dim[0][1]
                KT = 1 + (r_body_at_fin / (fin[i].wingspan+r_body_at_fin))
                self.cn_alpha_fin[i] = KT * self.cn_alpha_fin[i]

    def _correct_sign_cn_fin(self):
        for i in range (len(fin)):
            self.fin_cn[i] = self.cn_alpha_fin[i] * abs(self.fin_aoa[i])
            self.fin_cn[i] *= self._sign_correction_fin[i]

    def _correct_ctrl_fin_coeff_for_actuator_angle(self):
        ctrl_fin_cn =  self.fin_cn[1]# Still normal to the fin
        self.fin_cn[1] = ctrl_fin_cn * np.cos(self.actuator_angle) # now normal to the rocket
        self.ctrl_fin_ca = ctrl_fin_cn * abs(np.sin(self.actuator_angle))
        # Absolute because ca is always positive, i.e., pointing backwards,
        # independently of the actuator angle being positive or negative
        self.cn_alpha_ctrl_fin_3d_arrow = self.cn_alpha_fin[1] * np.cos(self.actuator_angle)

    def _calculate_cp_position(self):
        a = 0
        b = 0
        self.cp = 0
        # cp position = moment/force
        # It is supposed that the force is applied
        # in the centroid of the body component
        for i in range(len(self.component_cn_alpha)):
            a += self.component_centroid_pos[i] * self.component_cn[i]
            b += self.component_cn[i]
        if self.use_fins is True:
            a += fin[0].cp * self.fin_cn[0]
            b += self.fin_cn[0]
        if b != 0:
            self.cp_w_o_ctrl_fin = a / b # For the 3D Cn Arrow
        else:
            self.cp_w_o_ctrl_fin = self.component_centroid[0]
        if self.use_fins is True:
            a += fin[1].cp * self.fin_cn[1]
            b += self.fin_cn[1]
        if b != 0:
            self.cp = a / b
        else:
            self.cp = self.component_centroid[0]
        return self.cp

    def _calculate_cm(self):
        self.cm_xcg = self.cn * (self.cp-self.xcg) / self.max_diam

    ## DRAG - DRAG - DRAG - DRAG - DRAG
    def _calculate_total_ca(self, aoa=0, v_modulus=10, rho=1.225, mu=1.784e-5):
        """
        Boundry layer always turbulent
        There are no boattails, if they are, they are treated as shoulders
        Base drag is not reduced by motor exhaust
        There is no interpolation of the pressure drag for compressibility

        More detailed explanations of the methods applied here are in the
        Open Rocket's documentation.
        """
        self._calculate_reynolds(v_modulus, rho, mu)
        self._calculate_cf()
        self._calculate_pressure_drag()
        self._calculate_base_drag()
        self._calculate_cd(aoa)
        self._calculate_ca(aoa)

    def _calculate_reynolds(self, v_modulus, rho, mu):
        self.reynolds = (rho * v_modulus * self.length) / mu

    def _calculate_cf(self):
        if self.reynolds < 10e4:
            self.Cf = 1.48e-2
        elif self.reynolds < self.reynolds_crit:
            self.Cf = 1 / ((1.5*np.log(self.reynolds)-5.6)**2)
        else:
            self.Cf = 0.032 * np.power((self.relative_rough/self.length), 0.2)
        self.Cf = self.Cf * (1 - 0.1*self.mach**2)

    def _calculate_pressure_drag(self):
        n = (len(self.rocket_dim)-1)
        self.cd_pressure_component = [0]*n
        for i in range(n):
            l = self.rocket_dim[i+1][0] - self.rocket_dim[i][0]
            r1 = self.rocket_dim[i][1] / 2
            r2 = self.rocket_dim[i+1][1] / 2
            try:
                phi = np.arctan((r2-r1) / l)
            except ZeroDivisionError:
                print("Component length is zero")
            self.cd_pressure_component[i] = 0.8 * np.sin(phi)**2
        if self.ogive is True:
            self.cd_pressure_component[0] = 0

    def _calculate_base_drag(self):
        self.base_drag = 0.12 + 0.13*self.mach**2

    def _calculate_cd(self, aoa):
        self._calculate_cd0_friction()
        self._calculate_total_pressure_drag(aoa)
        self._calculate_total_base_drag()
        self.cd0 = self.cd0_friction + self.total_pressure_drag + self.total_base_drag

    def _calculate_cd0_friction(self):
        self.cd0_friction = ((self.Cf * ((1+1/(2*self.fineness))*self.area_wet_body))
                             /self.area_ref)

    def _calculate_total_pressure_drag(self, aoa):
        n = (len(self.station_cross_area)-1)
        self.total_pressure_drag = 0
        for i in range(n):
            if i == n-1 and abs(aoa) > np.pi/2:
                # Rocket flying backwards, area_ref_component would be 0 so it has to be tricked
                area_ref_component = self.station_cross_area[i+1]
                # 1 is the pressure cd for a hollow cilinder
                self.cd_pressure_component[i] = 1
                self.total_pressure_drag += ((area_ref_component/self.area_ref)
                                             * self.cd_pressure_component[i])
            else:
                area_ref_component = abs(self.station_cross_area[i+1] - self.station_cross_area[i])
                self.total_pressure_drag += ((area_ref_component/self.area_ref)
                                             * self.cd_pressure_component[i])

    def _calculate_total_base_drag(self):
        self.total_base_drag = (self.station_cross_area[-1]/self.area_ref) * self.base_drag

    def _calculate_ca(self, aoa):
        # First order interpolation
        cd2ca = np.interp(aoa, self.aoa_list_ca, self.ca_scale)
        self.ca = self.cd0 * cd2ca
        if self.use_fins is True:
            self._add_fin_ca()
            self._add_control_fin_drag()

    def _add_fin_ca(self):
        for i in range(len(fin)):
            # Asumes control fin parallel to the body
            self.ca += fin[i].ca * (2*fin[i].area/self.area_ref)

    def _add_control_fin_drag(self):
        self.ca += self.ctrl_fin_ca

    ## MOTOR - MOTOR - MOTOR - MOTOR - MOTOR - MOTOR - MOTOR - MOTOR - MOTOR
    def set_motor(self, data):
        """
        Set the rocket motor with the corresponding data.

        Parameters
        ----------
        data : nested list
            [time, thrust].

        Returns
        -------
        None.
        """
        # Motor data from text files
        # t, thrust
        self.motor[0] = copy.deepcopy(data[0])
        self.motor[1] = copy.deepcopy(data[1])

    def get_thrust(self, t, t_launch):
        """
        Input the current time and the time at which the motor ignited
        to get its current thrust.

        Parameters
        ----------
        t : float
            current time.
        t_launch : float
            ignition time.

        Returns
        -------
        thrust : float
            thrust.
        """
        x = t - t_launch
        self.thrust = np.interp(x, self.motor[0], self.motor[1])
        if self.thrust < 0.001:
            self.thrust = 0.001
        return float(self.thrust)

    def is_in_the_pad(self, alt):
        """
        Check if the rocket is in the pad

        Parameters
        ----------
        alt : float
            Current altitude.

        Returns
        -------
        is_in_the_pad_flag : bool
            Is the rocket in the pad?.
        """
        if alt > 0.2 and self.is_in_the_pad_flag is True:
            self.is_in_the_pad_flag = False
        return self.is_in_the_pad_flag

    def burnout_time(self):
        """
        Burn time of the motor.

        Returns
        -------
        float
            Burn time.
        """
        return self.motor[0][-1]

    def reset_variables(self):
        """Resets some variables of the rocket."""
        self.thrust = 0
        self.cn = 0
        self.xcp = 0
        self.ca = 0
        self.motor=[[],[]]
        self.is_in_the_pad_flag = True

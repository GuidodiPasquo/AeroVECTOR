# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 17:09:52 2021

@author: Guido di Pasquo
"""

import copy
import numpy as np
from scipy.interpolate import interp1d
from src import ISA_calculator as atm
from src.aerodynamics import fin_aerodynamics as fin_aero
from src import warnings_and_cautions


"""
Handles the rocket body aerodynamics.

Classes:
    Rocket -- Rocket aerodynamics.
"""


DEG2RAD = np.pi / 180
RAD2DEG = 1 / DEG2RAD
atmosphere = atm.get_atmosphere()


fin = [fin_aero.Fin(), fin_aero.Fin()]


def loc2glob(u0, v0, theta):
    # Rotational matrix 2x2
    # Axes are rotated, there is more info in the Technical documentation.
    A = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    u = np.array([[u0], [v0]])
    x = np.dot(A, u)
    a = [x[0, 0], x[1, 0]]
    return a


class Rocket:
    """
    Handles the rocket's body and aerodynamics.

    Methods
    -------
        update_rocket -- Update the rocket characteristics.
        calculate_aero_coef -- Compute the aerodynamics of the rocket.
        set_motor -- Set the rocket's motor.
        get_thrust -- Returns the motor's thrust.
        is_in_the_pad -- Check if the rocket is in the pad.
        burnout_time -- Returns burnout time of the motor.
        reset_variables -- Resets some variables of the rocket.
    """

    def __init__(self):
        self.xcg = 1
        self.ogive_flag = False
        self.motor = [[], []]
        self.t_burnout = 1
        self.reynolds = 1
        # Empirical method to calculate the ca from the cd, it should use a
        # fitted third order polinomial but interpolations are easier
        self.aoa_list_ca = [-180 * DEG2RAD,
                            -(180 - 17) * DEG2RAD,
                            -(180 - 70) * DEG2RAD,
                            -90 * DEG2RAD,
                            -70 * DEG2RAD,
                            -18 * DEG2RAD,
                            -16 * DEG2RAD,
                            0,
                            16 * DEG2RAD,
                            18 * DEG2RAD,
                            70 * DEG2RAD,
                            90 * DEG2RAD,
                            (180 - 70) * DEG2RAD,
                            (180 - 17) * DEG2RAD,
                            180 * DEG2RAD]
        self.ca_scale = [-1,
                         -1.3,
                         -0.097777,
                         0,
                         0.097777,
                         1.295,
                         1.295,
                         1,
                         1.3,
                         1.3,
                         0.097777,
                         0,
                         -0.097777,
                         -1.3,
                         -1]
        self.f_ca_interp = interp1d(self.aoa_list_ca, self.ca_scale,
                                    kind='quadratic')





    """ BODY ==============================================================#"""

    def update_rocket(self, l0, mass_param, roughness=[60e-6]*3):
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
        if len(l[5]) <= 1:
            print("WARNING: incorrect rocket dimensions.")
            warnings_and_cautions.w_and_c.warnings.incorrect_rocket_dimensions = True
        else:
            warnings_and_cautions.w_and_c.warnings.incorrect_rocket_dimensions = False
            self.relative_rough = roughness[0]
            self.reset_variables()
            self._set_variables(l)
            self.update_mass_parameters(mass_param)
            self._update_rocket_dim(l[5])
            self._update_fins(l, roughness)
            self._calculate_reynolds_crit()
            self._check_if_cg_falls_inside_nose_cone()
            self._calculate_cd_pressure_compresibility_correction()
            self.calculate_aero_coef()

    def reset_variables(self):
        self.is_in_the_pad_flag = True
        self.ogive_flag = False
        self.use_fins = False
        self.fins_attached = True
        self.use_fins_control = False
        self.is_in_the_pad_flag = True
        self.is_supersonic = False

    def _set_variables(self, l):
        self.ogive_flag = l[0]
        self.use_fins = l[1]
        self.fins_attached = [l[2], l[4]]
        self.use_fins_control = l[3]

    def update_mass_parameters(self, mass_param):
        self.m_liftoff = mass_param[0]
        self.m_burnout = mass_param[1]
        self.Iy_liftoff = mass_param[2]
        self.Iy_burnout = mass_param[3]
        self.xcg_liftoff = mass_param[4]
        self.xcg_burnout = mass_param[5]
        self.xcg = self.xcg_liftoff

    def _update_rocket_dim(self, l):
        # Looks like the Reference area is the maximum one,
        # not the base of the nosecone.
        self.rocket_dim = copy.deepcopy(l)
        self.length = self.rocket_dim[-1][0]
        self._separate_xcg_component()
        self.__initialize(len(self.rocket_dim))
        self.max_diam = self._maximum_diameter()
        self.fineness = self.length / self.max_diam
        self.area_ref = np.pi * (self.max_diam/2)**2
        for i in range(len(self.rocket_dim)-1):
            l = self.rocket_dim[i+1][0] - self.rocket_dim[i][0]
            self.component_length[i] = l
            r1 = self.rocket_dim[i][1] / 2
            r2 = self.rocket_dim[i+1][1] / 2
            if r1 > r2:
                self.component_fineness[i] = l/r1
            else:
                self.component_fineness[i] = l/r2
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
            if self.ogive_flag is True and i == 0:
                plan_area, centroid = self._integrate_ogive()
                self.component_centroid[0] = centroid
                self.component_centroid_pos[0] = centroid
                self.component_plan_area[0] = plan_area
        self._compute_total_rocket_plan_area()
        self._calculate_wet_area_body()
        self._calculate_barrowman_constants()
        self._calculate_body_cn_constants()
        self._update_rocket_diam_interpolation()

    def _separate_xcg_component(self):
        index = 1
        if self.xcg <= self.length:
            for i, elem in enumerate(self.rocket_dim):
                if elem[0] > self.xcg:
                    index = i
                    break
                if elem[0] == self.xcg:
                    return None
            x = [self.rocket_dim[index-1][0], self.rocket_dim[index][0]]
            diam = [self.rocket_dim[index-1][1], self.rocket_dim[index][1]]
            diameter_at_xcg = np.interp(self.xcg, x, diam)
            self.rocket_dim.insert(index, [self.xcg, diameter_at_xcg])
        discretize_rocket_experimental = False
        if discretize_rocket_experimental is True:
            x_list = []
            d_list = []
            for i in range(len(self.rocket_dim)):
                x_list.append(self.rocket_dim[i][0])
                d_list.append(self.rocket_dim[i][1])
            x = self.rocket_dim[1][0]
            l = self.rocket_dim[-1][0]
            n = 30
            step = (l-x) / n
            d_interp_list = [self.rocket_dim[0]]
            for i in range(n+1):
                d = np.interp(x, x_list, d_list)
                d_interp_list.append([x, d])
                x += step
            self.rocket_dim = copy.deepcopy(d_interp_list)

    def __initialize(self, n):
        self.component_cn = [0]*(n-1)
        self.component_cn_alpha = [0]*(n-1)
        self.component_cm = [0]*(n-1)
        self.station_cross_area = [0]*n
        self.component_plan_area = [0]*(n-1)
        self.component_volume = [0]*(n-1)
        self.component_centroid = [0]*(n-1)
        self.component_centroid_pos = [0]*(n-1)
        self.component_length = [0]*(n-1)
        self.component_fineness = [0]*(n-1)
        self.fin_cn, self.fin_ca = [0]*2, [0]*2
        self.v_sq_over_v_tot_sq_body = [1] * (n-1)
        self.v_sq_over_v_tot_sq_fin = [1] * 2

    def _maximum_diameter(self):
        d = 0
        for i in range(len(self.rocket_dim)):
            if self.rocket_dim[i][1] > d:
                d = self.rocket_dim[i][1]
        if d == 0:
            d = 0.0000001
        return d

    def _integrate_ogive(self):
        len_nc = self.rocket_dim[1][0]
        radius_nc = self.rocket_dim[1][1] / 2
        rho_radius = (radius_nc**2 + len_nc**2) / (2*radius_nc)
        self.ogive = Ogive(radius_nc, len_nc, rho_radius)
        return self.ogive.calculate_area_cp_and_volume()

    def _compute_total_rocket_plan_area(self):
        self.plan_area = 0
        for i in range(len(self.component_plan_area)):
            self.plan_area += self.component_plan_area[i]

    def _calculate_wet_area_body(self):
        self.component_wet_area = [0]*(len(self.rocket_dim)-1)
        for i in range(len(self.component_wet_area)):
            l = self.rocket_dim[i+1][0] - self.rocket_dim[i][0]
            r1 = self.rocket_dim[i][1] / 2
            r2 = self.rocket_dim[i+1][1] / 2
            self.component_wet_area[i] = np.pi * (r1+r2) * np.sqrt((r2-r1)**2 + l**2)
        if self.ogive_flag is True:
            self.component_wet_area[0] = self.ogive.wet_area
        self.wet_area_body = 0
        for i in range(len(self.component_wet_area)):
            self.wet_area_body += self.component_wet_area[i]

    def _calculate_barrowman_constants(self):
        self._barrowman_const = [0] * (len(self.station_cross_area)-1)
        for i in range(len(self._barrowman_const)):
            k1 = (2 / self.area_ref)
            self._barrowman_const[i] = k1 * (self.station_cross_area[i+1]
                                             - self.station_cross_area[i])

    def _calculate_body_cn_constants(self):
        K = 1.1
        self._body_cn_const = [0]*len(self.component_plan_area)
        for i in range(len(self.component_plan_area)):
            k1 = self.component_plan_area[i] / self.area_ref
            self._body_cn_const[i] = K * k1

    def _update_rocket_diam_interpolation(self):
        pos = []
        diam = []
        for i in range(len(self.rocket_dim)):
            pos.append(self.rocket_dim[i][0])
            diam.append(self.rocket_dim[i][1])
        self.diam_interp = interp1d(pos, diam, bounds_error=False,
                                    fill_value=(0, diam[-1]))

    def _update_fins(self, l, roughness):
        # In case one fin is not set up
        zero_fin = [[0, 0.0000000001], [0, 0], 0.0000000002, 0]
        if self.use_fins is True:
            fin[0].update(l[6],
                          self.fins_attached[0],
                          which_fin=0,
                          roughness=roughness[1])
            if self.use_fins_control is True:
                fin[1].update(l[7],
                              self.fins_attached[1],
                              which_fin=1,
                              roughness=roughness[2])
            else:
                # In case there are no control fins
                fin[1].update(zero_fin, which_fin=1)
        else:
            for i in range(2):
                fin[i].update(zero_fin, which_fin=i)

    def _calculate_reynolds_crit(self):
        self.reynolds_crit = 51 * (self.relative_rough/self.length)**-1.039
        self.component_re_crit = [0] * len(self.component_length)
        for i in range(len(self.component_length)):
            self.component_re_crit[i] = 51 * \
                (self.relative_rough/self.component_centroid_pos[i])**-1.039

    def _check_if_cg_falls_inside_nose_cone(self):
        if self.rocket_dim[1][0] > self.xcg_burnout and self.ogive_flag is True:
            print("WARNING: CG lays inside the nose cone, ogive will not be " +
                  "properly calculated.")
            warnings_and_cautions.w_and_c.warnings.wrong_cg = True
        else:
            warnings_and_cautions.w_and_c.warnings.wrong_cg = False

    def _calculate_cd_pressure_compresibility_correction(self):
        self._calculate_cd_nosecone_mach_1()
        self._calculate_a_and_b_cd_pressure()

    def _calculate_cd_nosecone_mach_1(self):
        d = self.rocket_dim[1][1]
        leng = self.rocket_dim[1][0]
        e = np.arctan(d/2 / leng)
        sin_e = np.sin(e)
        self.cd_mach_1_cone = sin_e
        self.cd_mach_1_cone_derivative = 4 / (1.4+1) * (1 - 0.5*self.cd_mach_1_cone)
        self.cd_pressure_nosecone = 0.8 * sin_e**2

    def _calculate_a_and_b_cd_pressure(self):
        self.a_cd_pressure = self.cd_mach_1_cone - self.cd_pressure_nosecone
        self.b_cd_pressure = self.cd_mach_1_cone_derivative / self.a_cd_pressure





    """AERODYNAMICS =======================================================#"""

    def calculate_aero_coef(self, v_loc_tot=[10, 0], Q=0, h=0, actuator_angle=0):
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
        self.Q = Q
        self.v_loc_tot = v_loc_tot
        self.actuator_angle = actuator_angle
        self.v_modulus_sq = v_loc_tot[0]**2 + v_loc_tot[1]**2
        self.v_modulus = np.sqrt(self.v_modulus_sq)
        self.T, self.P, self.rho, self.spd_sound, self.mu = atm.calculate_at_h(h, atmosphere)
        self._calculate_mach()
        self._calculate_aoa_components()
        fin[0].update_conditions(h, self.v_modulus, self.fin_aoa[0], 0)
        fin[1].update_conditions(h, self.v_modulus, self.fin_aoa[1], actuator_angle)
        self._calculate_total_cn()
        self._calculate_cp_position()
        self._calculate_cm()
        self._calculate_total_ca(self.aoa_total)
        return self.cn, self.cm_xcg, self.ca, self.cp

    def _calculate_mach(self):
        mach = self.v_modulus / self.spd_sound
        if mach < 0.001:
            self.mach = 0.001
        elif mach >= 0.9:
            self.is_supersonic = True
        else:
            self.mach = mach
        self.beta = np.sqrt(1 - self.mach**2)

    def _calculate_aoa_components(self):
        """Using Q and the radius."""
        self.aoa_total = self._calculate_aoa(0)
        self.component_aoa = [0] * len(self.component_centroid)
        self.component_tan_vel = [0] * len(self.component_centroid)
        self.point_tan_vel = [0] * (len(self.rocket_dim))
        self.fin_aoa = [0, 0]
        self.fin_tan_vel = [0, 0]
        for i in range(len(self.component_tan_vel)):
            r = self.component_centroid_pos[i] - self.xcg
            self.component_tan_vel[i] = self.Q * r
        for i in range(len(self.point_tan_vel)):
            r = self.rocket_dim[i][0] - self.xcg
            self.point_tan_vel[i] = self.Q * r
        for i in range(len(self.component_aoa)):
            self.component_aoa[i] = self._calculate_aoa(self.component_tan_vel[i])
        for i in range(2):
            r = fin[i].pp.cp - self.xcg
            self.fin_tan_vel[i] = self.Q * r
        for i in range(2):
            if i == 1:
                self.fin_aoa[i] = self._calculate_aoa(self.fin_tan_vel[i])
                self.fin_aoa[i] -= self.actuator_angle  # + delta gives -aoa
                if self.fin_aoa[i] > np.pi:
                    self.fin_aoa[i] -= 2*np.pi
                elif self.fin_aoa[i] < -np.pi:
                    self.fin_aoa[i] += 2*np.pi
            else:
                self.fin_aoa[i] = self._calculate_aoa(self.fin_tan_vel[i])

    def _calculate_aoa(self, v_tan):
        if self.v_loc_tot[0] != 0:
            aoa = np.arctan2(self.v_loc_tot[1]+v_tan, self.v_loc_tot[0])
        else:
            aoa = np.pi/2
        if aoa == 0:
            aoa = 0.000001
        return aoa

    def _calculate_total_cn(self):
        self.cn = 0
        self.__sign_correction()
        compute_dynamic_pressure_damping_experimental = False
        if compute_dynamic_pressure_damping_experimental is True:
            self._compute_dynamic_pressure_scale()
        self._barrowman_cn()
        self._body_cn()
        if self.use_fins is True:
            self._fin_cn()
        for i in range(len(self.component_cn)):
            self.cn += self.component_cn[i]
        if self.use_fins is True:
            for i in range(2):
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

    def _compute_dynamic_pressure_scale(self):
        self.v_sq_over_v_tot_sq_body = [0] * len(self.component_tan_vel)
        for i in range(len(self.v_sq_over_v_tot_sq_body)):
            v_component_sq_1 = (self.v_loc_tot[0]**2
                                + (self.v_loc_tot[1] + self.point_tan_vel[i])**2)
            v_component_sq_2 = (self.v_loc_tot[0]**2
                                + (self.v_loc_tot[1] + self.point_tan_vel[i+1])**2)
            v_component_sq_avg = (v_component_sq_1 + v_component_sq_2)/2
            self.v_sq_over_v_tot_sq_body[i] = ((v_component_sq_avg+0.001)
                                               / (self.v_modulus_sq+0.001))
        self.v_sq_over_v_tot_sq_fin = [0] * 2
        for i in range(len(self.v_sq_over_v_tot_sq_fin)):
            v_fin_sq = (self.v_loc_tot[0]**2 +
                        (self.v_loc_tot[1] + self.fin_tan_vel[i])**2)
            self.v_sq_over_v_tot_sq_fin[i] = (v_fin_sq+0.001) / (self.v_modulus_sq+0.001)

    def _barrowman_cn(self):
        for i in range(len(self.component_aoa)):
            aoa = abs(self.component_aoa[i])
            cn = np.sin(aoa) * self._barrowman_const[i] * np.cos(aoa)
            cn *= self._sign_correction[i]
            self.component_cn[i] = cn

    def _body_cn(self):
        cn = [0]*len(self.component_plan_area)
        for i in range(len(self.component_plan_area)):
            aoa = abs(self.component_aoa[i])
            cn[i] = self._body_cn_const[i] * np.sin(aoa)**2
            cn[i] *= self._sign_correction[i]
        self._calculate_ogive_cp(self.ogive_flag, cn[0])
        for i in range(len(self.component_cn)):
            self.component_cn[i] += cn[i]
            self.component_cn[i] *= self.v_sq_over_v_tot_sq_body[i]

    def _calculate_ogive_cp(self, ogive_flag, cn):
        if ogive_flag is True:
            barrowman_cn = self.component_cn[0]
            plan_area_cn = cn
            total_cn = barrowman_cn + plan_area_cn
            moment = (barrowman_cn * self.ogive.center_of_pressure +
                      plan_area_cn * self.ogive.center_of_area)
            self.component_centroid[0] = moment / total_cn
            self.component_centroid_pos[0] = self.component_centroid[0]

    def _fin_cn(self):
        # The cp can move beyond the limits show with the slider in
        # the GUI due to damping, which can produce a moment while the
        # cn is zero.
        self.fin_cn = [0]*2
        self.fin_ca = [0]*2
        self.fac = [fin_aero.AerodynamicCoefficients(), fin_aero.AerodynamicCoefficients()]
        self._obtain_fin_coeff()
        self._nondimensionalize_fin_coeff()
        self._compute_body_interference()
        self._compute_fin_cn()

    def _obtain_fin_coeff(self):
        self._calculate_reynolds()
        Re, Re_crit = self.reynolds, self.reynolds_crit
        use_fin_reynolds_experimental = False
        use_rocket_re = not use_fin_reynolds_experimental
        self.fac[0] = fin[0].get_aero_coeff(Re, Re_crit, use_rocket_re=use_rocket_re)
        if self.use_fins_control is True:
            self.fac[1] = fin[1].get_aero_coeff(Re, Re_crit, use_rocket_re=use_rocket_re)

    def _calculate_reynolds(self):
        self.reynolds = (self.rho * self.v_modulus * self.length) / self.mu
        # Now calculate the Re in the centroid of the component
        self.re_component = [0] * len(self.component_centroid_pos)
        for i in range(len(self.re_component)):
            self.re_component[i] = ((self.rho
                                     * self.v_modulus
                                     * self.component_centroid_pos[i])
                                    / self.mu)

    def _nondimensionalize_fin_coeff(self):
        angle = [0, self.actuator_angle]
        for i in range(2):
            adim_cte = 2 * (fin[i].pp.area/self.area_ref)
            cn_ca_rocket_coordinates = loc2glob(self.fac[i].cn, self.fac[i].ca, angle[i])
            cn, ca = cn_ca_rocket_coordinates[0], cn_ca_rocket_coordinates[1]
            self.fin_cn[i] = cn * adim_cte  # only one pair of fins generates normal force
            self.fin_ca[i] = ca * adim_cte + adim_cte * self.fac[i].cd0  # Two fins that move + two that don't, all generate drag

    def _compute_body_interference(self):
        for i in range(2):
            if fin[i].pp.is_attached is True:
                r_body_at_fin = self.diam_interp(fin[i].cp) / 2
                KT = 1 + (r_body_at_fin / (fin[i].pp.wingspan+r_body_at_fin))
                self.fin_cn[i] *= KT

    def _compute_fin_cn(self):
        for i in range(2):
            self.fin_cn[i] *= -1  # Rocket's reference system.
            self.fin_cn[i] *= self.v_sq_over_v_tot_sq_fin[i]

    def _calculate_cp_position(self):
        a = 0
        b = 0
        self.cp = 0
        # cp position = moment/force
        for i in range(len(self.component_cn_alpha)):
            a += self.component_centroid_pos[i] * self.component_cn[i]
            b += self.component_cn[i]
        if self.use_fins is True:
            a += fin[0].cp * self.fin_cn[0]
            b += self.fin_cn[0]
        self.cp_w_o_ctrl_fin = a / b  # For the 3D Cn Arrow
        self.passive_cn = b  # For the 3D Cn Arrow
        if self.use_fins is True:
            a += fin[1].cp * self.fin_cn[1]
            b += self.fin_cn[1]
        self.cp = a / b
        return self.cp

    def _calculate_cm(self):
        # just to be consistent with the books (M = cm * Sqd)
        self.cm_xcg = self.cn * (self.cp-self.xcg) / self.max_diam





    """ DRAG =============================================================#"""

    def _calculate_total_ca(self, aoa=0):
        """
        Base drag is not reduced by motor exhaust
        There is no interpolation of the pressure drag for compressibility

        More detailed explanations of the methods applied here are in the
        Open Rocket's documentation.
        """
        self._calculate_drag_with_individual_re_experimental = False
        self._calculate_cf()
        self._calculate_base_drag()
        self._calculate_pressure_drag()
        self._calculate_cd(aoa)
        self._calculate_ca(aoa)

    def _calculate_cf(self):
        if self._calculate_drag_with_individual_re_experimental is False:
            # Cf of the whole rocket.
            if self.reynolds < 10e4:
                self.Cf = 1.48e-2
            elif self.reynolds < self.reynolds_crit:
                self.Cf = 1 / ((1.5*np.log(self.reynolds)-5.6)**2)
            else:
                self.Cf = 0.032 * np.power((self.relative_rough/self.length), 0.2)
            self.Cf = self.Cf * (1 - 0.1*self.mach**2)
        else:
            # Cf of each component, remember that the nose cone might be laminar
            # but not the tail.
            self.Cf_component = [0] * len(self.re_component)
            for i in range(len(self.re_component)):
                if self.re_component[i] < 10e4:
                    self.Cf_component[i] = 1.48e-2
                elif self.re_component[i] < self.component_re_crit[i]:
                    self.Cf_component[i] = 1 / ((1.5*np.log(self.re_component[i])-5.6)**2)
                else:
                    self.Cf_component[i] = 0.032 * np.power((self.relative_rough
                                                             / self.component_length[i]),
                                                            0.2)
                self.Cf_component[i] = 1 / ((1.5*np.log(self.re_component[i])-5.6)**2)
                self.Cf_component[i] *= (1 - 0.1*self.mach**2)

    def _calculate_base_drag(self):
        self.base_drag = 0.12 + 0.13*self.mach**2

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
            if self.component_is_boattail(phi):
                gamma = l / ((r1-r2) * 2)
                if gamma > 3:
                    self.cd_pressure_component[i] = 0
                elif gamma < 1:
                    self.cd_pressure_component[i] = self.base_drag
                else:
                    self.cd_pressure_component[i] = (3-gamma) / 2 * self.base_drag
            else:
                self.cd_pressure_component[i] = 0.8 * np.sin(phi)**2
            if self.ogive_flag is True:
                self.cd_pressure_component[0] = self.cd_p_comp_correction(self.mach)
            else:
                self.cd_pressure_component[0] += self.cd_p_comp_correction(self.mach)

    def component_is_boattail(self, phi):
        if phi >= 0:
            return False
        else:
            return True

    def cd_p_comp_correction(self, mach):
        cd_compensation = self.a_cd_pressure * mach ** self.b_cd_pressure
        return cd_compensation

    def _calculate_cd(self, aoa):
        self.cd0 = 0
        self._calculate_cd0_friction()
        self._calculate_total_pressure_drag(aoa)
        self._calculate_total_base_drag()
        self.cd0 += self.cd0_friction + self.total_pressure_drag + self.total_base_drag

    def _calculate_cd0_friction(self):
        if self._calculate_drag_with_individual_re_experimental is False:
            # Like OR calculates the skin friction, it never gives the same result tho.
            cd0_body = (1 + 1/(2*self.fineness)) * self.wet_area_body
            self.cd0_friction = self.Cf * cd0_body / self.area_ref
        else:
            # Taking into account that the flow has different Re in each component.
            self.cd0_friction = 0
            for i in range(len(self.component_wet_area)):
                cd0_comp = self.Cf_component[i] * self.component_wet_area[i]
                self.cd0_friction += (1 + 1/(2*self.fineness)) * cd0_comp / self.area_ref

    def _calculate_total_pressure_drag(self, aoa):
        n = (len(self.station_cross_area)-1)
        self.total_pressure_drag = 0
        for i in range(n):
            if i == n-1 and abs(aoa) > np.pi/2:
                """Rocket flying backwards,
                area_ref_component would be 0 so it has to be tricked"""
                area_ref_component = self.station_cross_area[i+1]
                # 1 is the pressure cd for a hollow cilinder
                self.cd_pressure_component[i] = 1
                self.total_pressure_drag += ((area_ref_component / self.area_ref)
                                             * self.cd_pressure_component[i])
            else:
                area_ref_component = abs(self.station_cross_area[i+1]
                                         - self.station_cross_area[i])
                self.total_pressure_drag += ((area_ref_component / self.area_ref)
                                             * self.cd_pressure_component[i])

    def _calculate_total_base_drag(self):
        # * 0.75 to account for the exhaust plume.
        self.total_base_drag = ((self.station_cross_area[-1] / self.area_ref)
                                * self.base_drag) * 0.75

    def _calculate_ca(self, aoa):
        self.cd2ca = self.f_ca_interp(aoa)
        self.ca = self.cd0 * self.cd2ca
        for i in range(2):
            self.ca += self.fin_ca[i]

    # MOTOR - MOTOR - MOTOR - MOTOR - MOTOR - MOTOR - MOTOR - MOTOR - MOTOR
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
        self.t_burnout = self.burnout_time()

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
        Check if the rocket is in the pad.

        Parameters
        ----------
        alt : float
            Current altitude.

        Returns
        -------
        is_in_the_pad_flag : bool
            Is the rocket in the pad?.
        """
        if alt > 0.001 and self.is_in_the_pad_flag is True:
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

    def get_mass(self, t, t_launch):
        x = t - t_launch
        yp = [self.m_liftoff, self.m_burnout]
        xp = [0, self.t_burnout]
        self.m = np.interp(x, xp, yp)
        return float(self.m)

    def get_Iy(self, t, t_launch):
        x = t - t_launch
        yp = [self.Iy_liftoff, self.Iy_burnout]
        xp = [0, self.t_burnout]
        self.Iy = np.interp(x, xp, yp)
        return float(self.Iy)

    def get_xcg(self, t, t_launch):
        """
        Input the current time and the time at which the motor ignited
        to get its current xcg.

        Parameters
        ----------
        t : float
            current time.
        t_launch : float
            ignition time.

        Returns
        -------
        thrust : float
            xcg.
        """
        x = t - t_launch
        yp = [self.xcg_liftoff, self.xcg_burnout]
        xp = [0, self.t_burnout]
        self.xcg = np.interp(x, xp, yp)
        return self.xcg

    def get_mass_parameters(self, t, t_launch):
        """Get mass, Iy and xcg and updates the internal ones."""
        m = self.get_mass(t, t_launch)
        Iy = self.get_Iy(t, t_launch)
        xcg = self.get_xcg(t, t_launch)
        return m, Iy, xcg

    """ END ROCKET ========================================================#"""


class Ogive():
    def __init__(self, radius_nc, len_nc, rho_radius):
        self.radius_nc = radius_nc
        self.len_nc = len_nc
        self.rho_radius = rho_radius

    def radius(self, x):
        y = (np.sqrt(self.rho_radius**2 - (self.len_nc - x)**2)
             + self.radius_nc - self.rho_radius)
        return y

    def calculate_area_cp_and_volume(self):
        self.area = 0
        moment = 0
        self.volume = 0
        self.wet_area = 0
        x1 = 0
        definition = 100
        step = self.len_nc/definition
        for j in range(definition):
            x2 = x1 + step
            y1 = self.radius(x1)
            ym = self.radius((x1+x2)/2)
            y2 = self.radius(x2)
            integral_delta = 2 * ((x2-x1)/6) * (y1+4*ym+y2)
            self.area += integral_delta
            moment += integral_delta * (x1 + x2)/2
            volume_1 = (1/3) * np.pi * step/2 * (y1**2 + ym**2 + y1*ym)
            volume_2 = (1/3) * np.pi * step/2 * (ym**2 + y2**2 + ym*y2)
            self.volume += volume_1 + volume_2
            wet_area_step = np.pi * (y1+y2) * np.sqrt((y2-y1)**2 + step**2)
            self.wet_area += wet_area_step
            x1 += step
        self.center_of_area = moment / self.area
        cross_area = np.pi * self.radius_nc**2
        self.center_of_pressure = (self.len_nc * cross_area - self.volume) / (cross_area)
        return self.area, self.center_of_pressure

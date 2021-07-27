# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 19:07:07 2021

@author: Guido di Pasquo
"""

import numpy as np
from scipy.interpolate import interp1d
from src.aerodynamics import flight_conditions
from src.aerodynamics import wind_tunnel_data
from src.aerodynamics import fin_physical_properties
from src import warnings_and_cautions

"""
Handles the airfoil and fin aerodynamics.

Classes:
    Airfoil -- 2D fin aerodynamics.
    Fin -- 3D fin aerodynamics.
"""

DEG2RAD = np.pi / 180
RAD2DEG = 1 / DEG2RAD
wtd = wind_tunnel_data.FinWindTunnelData()


def convert_aoa_to_something_usable(aoa):
    """Keeps the AoA in the 0º - 90º range, the sign works only for the Cl"""
    if aoa > np.pi/2:
        aoa = np.pi - aoa
        sign = -1
    elif aoa < -np.pi/2:
        aoa = np.pi - abs(aoa)
        sign = 1
    elif aoa < 0:
        aoa = abs(aoa)
        sign = -1
    else:
        sign = 1
    return aoa, sign


class Fin:
    """Class that handles the fin geometry and its individual parameters
    (i.e. no body interference)

    Methods
    -------
        update -- Update the fin physical characteristics.
        update_conditions -- Update the speed, mach, etc.
        get_aero_coeff -- Returns all the aerodynamic coefficients.
    """

    def __init__(self):
        self.pp = fin_physical_properties.PhysicalProperties()
        self.aero_properties = AerodynamicProperties()
        self.ffc = flight_conditions.FinFlightCondition()
        self.cp = 0

    def update(self, li, fin_attached=True, which_fin=0, roughness=60e-6):
        """
        Update the rocket with the data in l.

        Parameters
        ----------
        li : list (variables)
            Data of the rocket.
        fin_attached : bool, optional
            Are fins attached?. The default is True.

        Returns
        -------
        None.
        """
        self.fin_attached = fin_attached
        self._check_which_fin(which_fin)
        self.pp.update(li, fin_attached, roughness, self.type_of_fin)
        self._check_if_fins_are_correct()
        self.aero_properties.update(self.pp)

    def _check_which_fin(self, which_fin):
        self.which_fin = which_fin
        if which_fin == 0:
            self.type_of_fin = ["Stabilization", 0]
        elif which_fin == 1:
            self.type_of_fin = ["Control", 1]
        else:
            self.type_of_fin = ["", 3]

    def _check_if_fins_are_correct(self):
        if self.pp.wingspan > 0.00001:
            if self.pp.c_root < 0.0001:
                print("WARNING: " + self.type_of_fin[0] + " fin has incorrect dimensions.")
                warnings_and_cautions.w_and_c.warnings.fin_incorrect_dim[self.type_of_fin[1]] = True
            else:
                warnings_and_cautions.w_and_c.warnings.fin_incorrect_dim[self.type_of_fin[1]] = False
            if self.pp.thickness < 0.00001:
                print("CAUTION: " + self.type_of_fin[0] + " fin has zero thickness," +
                      " drag calculations will be inaccurate.")
                warnings_and_cautions.w_and_c.cautions.fin_zero_thickness[self.type_of_fin[1]] = True
            else:
                warnings_and_cautions.w_and_c.cautions.fin_zero_thickness[self.type_of_fin[1]] = False
        else:
            warnings_and_cautions.w_and_c.cautions.fin_zero_thickness[self.type_of_fin[1]] = False
            warnings_and_cautions.w_and_c.warnings.fin_incorrect_dim[self.type_of_fin[1]] = False
            warnings_and_cautions.w_and_c.stalled_fins.stalled_fin[self.type_of_fin[1]] = False

    def update_conditions(self, h, v, aoa, angle):
        self.ffc.update(h, v, aoa, angle)
        self._calculate_reynolds()

    def _calculate_reynolds(self):
        self.reynolds = (self.ffc.atm.rho * self.ffc.speed * self.pp.mac) / self.ffc.atm.mu

    def get_aero_coeff(self, Re, Re_crit, use_rocket_re=True):
        self.acoeff = self.aero_properties.get_aero_coeff(self.ffc, Re, Re_crit, use_rocket_re)
        self.cp = self.acoeff.hac * self.pp.mac + self.pp._dim[0][0]
        return self.acoeff


class AerodynamicProperties:

    def __init__(self):
        self.acoeff = AerodynamicCoefficients()
        self.cd0_obj = CD0()

    def update(self, physic_prop):
        self.pp = physic_prop
        self.cd0_obj.update(physic_prop)
        self._update_cut_and_max_coefficients()

    def _update_cut_and_max_coefficients(self):
        self._update_max_cd()
        if self.pp.is_ultra_low_AR is True:
            self._calculate_coefficients_ular()
        else:
            self._calculate_coefficients_normal_ar()

    def _update_max_cd(self):
        a = self.pp.c_root
        if self.pp.is_attached is True:
            b = 2 * self.pp.wingspan
        else:
            b = self.pp.wingspan
        if self.pp.is_sweptback or self.pp.is_sweptforward:
            cos_sb = np.cos(self.pp.sb_angle_50)
            a *= cos_sb
            b /= cos_sb
        self.cd_max = 1.11 + 0.02 * (a/b + b/a)  # Valid for reasonable ARs

    def _calculate_coefficients_ular(self):
        self.alpha_cut_lineal = 5 * DEG2RAD
        self.cl_cut_0 = self._cla_diederich(mach=0) * self.alpha_cut_lineal
        self._calculate_cl_max_and_alpha_cl_max()
        self._update_linear_interpolation_ular()
        self._update_quad_interpolation_ular()

    def _calculate_cl_max_and_alpha_cl_max(self):
        self.cl_max = np.interp(self.pp.aspect_ratio,
                                self.pp.transition_ar,
                                [1.25, 0.77])
        alpha_cut_list = [45 * DEG2RAD, 35*DEG2RAD]  # aoa of max cl
        ar_cut_list = [0.5, self.pp.transition_ar[0]]
        self.alpha_cl_max = np.interp(self.pp.aspect_ratio,
                                      ar_cut_list,
                                      alpha_cut_list)

    def _update_linear_interpolation_ular(self):
        """
        Interpolates the non linear Cl curve by finding the AoA that produces
        certain Cl for multiple ARs, and based on the current AR of the fin
        obtains the points to make the picewise CL(AoA) function.
        """
        n = 21
        alphas_list = [0]*n
        cl_end_interp_linear = 1.1
        step_cl = cl_end_interp_linear / (n-1)  # so as to conserve the [0,0]
        cl_list = [0]*n
        cl = 0
        for i in range(n):
            alphas_list[i] = [wtd.alpha_f_of_cl["0.5"](cl),
                              wtd.alpha_f_of_cl["0.75"](cl),
                              wtd.alpha_f_of_cl["1"](cl),
                              wtd.alpha_f_of_cl["1.25"](cl)]
            cl_list[i] = cl
            cl += step_cl

        AR_list = [0.5,
                   0.75,
                   1,
                   1.25]
        alpha_interp_list = [0]*n
        for i in range(n):
            alpha_interp_list[i] = np.interp(self.pp.aspect_ratio,
                                             AR_list,
                                             alphas_list[i])
        self.cl_interp_linear = interp1d(alpha_interp_list,
                                         cl_list,
                                         bounds_error=False,
                                         fill_value=(cl_list[0], cl_list[-1]))
        self.alpha_interp_linear = interp1d(cl_list,
                                            alpha_interp_list,
                                            bounds_error=True,
                                            fill_value=(alpha_interp_list[0],
                                                        alpha_interp_list[-1]))

    def _update_quad_interpolation_ular(self):
        """For the top part, can't use the same interpolation as before"""
        aoa_list = [self.alpha_interp_linear(0.8),
                    self.alpha_cl_max,
                    2*self.alpha_cl_max - self.alpha_interp_linear(0.8)]
        cl_list = [0.8,
                   self.cl_max,
                   0.8]
        self.cl_interp_quad = interp1d(aoa_list, cl_list, "quadratic",
                                       fill_value="extrapolate",
                                       bounds_error=False)

    def _calculate_coefficients_normal_ar(self):
        self.cl_cut = 0.77
        cla_lineal = self._cla_diederich(mach=0)
        self.alpha_cut_lineal = self.cl_cut / cla_lineal
        self.alpha_cut_cos = np.arccos(self.cl_cut / self.cd_max)

    def _cla_diederich(self, mach):
        """Diederich in his paper stablish that the lifting line approximation
        is not as good as the adapted equation for low aspect ratios, I found
        the opposite, that as long as the AR is not really low (AR<0.5), the
        lifting line approximation was better, that's why both slopes are
        interpolated for ULARs.
        """
        sb_angle = self.pp.sb_angle
        AR = self.pp.aspect_ratio
        cos_sb_angle = np.cos(sb_angle)
        # Plantform modification as in the original paper.
        cla_comp = (2 * np.pi) / (np.sqrt(1 - mach**2*cos_sb_angle**2))
        cla_comp_swept = cla_comp * cos_sb_angle
        eff_factor = cla_comp / (2*np.pi)
        F = AR / (eff_factor * cos_sb_angle)
        if self.pp.aspect_ratio <= self.pp.transition_ar[1]:
            cla_diederich = (F / (F * np.sqrt(1+(4/F**2)) + 2)
                             * cla_comp_swept)
            cla_lineal_lifting_line = F / (F + 2) * cla_comp_swept
            cla_avg = (cla_diederich + cla_lineal_lifting_line) / 2  # Best Result for transition_ar[0]
            cla_list = [cla_diederich, cla_avg]
            ar_list = [0, self.pp.transition_ar[0]]
            cla_lineal = np.interp(self.pp.aspect_ratio,
                                   ar_list,
                                   cla_list)
        else:
            cla_lineal = F / (F + 2) * cla_comp_swept
        return cla_lineal

    def get_aero_coeff(self, fin_flight_cond, Re, Re_crit, use_rocket_re):
        self.calculate_drag(fin_flight_cond, Re, Re_crit, use_rocket_re)
        self.acoeff.set_cl_cd_cn_ca_hac_cd0(self._get_cl_cd_cn_ca_hac(fin_flight_cond), self.cd0)
        return self.acoeff

    def calculate_drag(self, fin_flight_cond, Re, Re_crit, use_rocket_re):
        self.cd0 = self.cd0_obj.get(fin_flight_cond, Re, Re_crit, use_rocket_re)

    def _get_cl_cd_cn_ca_hac(self, fin_flight_cond):
        self.ffc = fin_flight_cond
        self.cl, self.cd = self._calculate_cl_cd(self.ffc)
        self.cn, self.ca = self._calculate_cn_ca()
        self.cm = self._calculate_cm()
        self.hac = self._calculate_hac()
        return self.cl, self.cd, self.cn, self.ca, self.cm, self.hac

    def _calculate_cl_cd(self, fin_flight_cond):
        self.ffc = fin_flight_cond
        self.cl = self._calculate_cl()  # cl_cut gets updated inside cl()
        self.alpha_cut_cos = np.arccos(self.cl_cut / self.cd_max)
        self.cd = self._get_cd()
        return self.cl, self.cd

    def _calculate_cl(self):
        self._update_cl_max_reynolds()
        if self.pp.is_ultra_low_AR is True:
            cl = self._calculate_cl_ular()
        else:
            cl = self._get_cl_normal_ar()
        return cl

    def _update_cl_max_reynolds(self):
        if self.pp.is_ultra_low_AR is True:
            cl_max_re = np.interp(self.ffc.Re,
                                  [1e4, 1e5],
                                  [1.13, 1.25])
            self.cl_max = np.interp(self.pp.aspect_ratio,
                                    self.pp.transition_ar,
                                    [cl_max_re, 0.77])
            self._update_quad_interpolation_ular()
        else:
            self.cl_cut = np.interp(self.ffc.Re,
                                    [1e4, 1e5],
                                    [0.7, 0.77])
            cla_lineal = self._cla_diederich(mach=0)
            self.alpha_cut_lineal = self.cl_cut / cla_lineal

    def _calculate_cl_ular(self):
        aoa = self.ffc.aoa
        cla = self._cla_diederich(self.ffc.mach)
        self.cl_cut = cla * self.alpha_cut_lineal
        self.delta_cl_cut = self.cl_cut - self.cl_interp_linear(self.alpha_cut_lineal)
        delta_stall_angle = 2*DEG2RAD * (self.pp.transition_ar[0] / self.pp.aspect_ratio)
        if delta_stall_angle > 5*DEG2RAD:
            delta_stall_angle = 5*DEG2RAD
        self.alpha_stall = self.alpha_cl_max + delta_stall_angle
        aoa, sign = convert_aoa_to_something_usable(aoa)
        if aoa <= self.alpha_cut_lineal:
            cl = cla * aoa
        elif aoa <= self.alpha_interp_linear(0.8):
            cl = self.cl_interp_linear(aoa) + self.delta_cl_cut
        elif self._check_joint_cos_with_quad(aoa, self.alpha_stall):
            cl = self.cl_interp_quad(aoa) + self.delta_cl_cut
            cl = self._correct_cl_for_delta_cl_cut(cl, aoa)
        elif aoa <= np.pi/2:
            cl = self.cd_max * np.cos(aoa)
        else:
            cl = self.cd_max * np.cos(aoa)
        self._check_stalled_fin_for_label(aoa)
        return cl * sign

    def _check_joint_cos_with_quad(self, aoa, delta_stall_angle):
        if aoa <= self.alpha_stall:
            return True
        elif self.cl_max < self.cd_max * np.cos(self.alpha_stall):
            if self.cl_interp_quad(aoa) + self.delta_cl_cut < self.cd_max * np.cos(aoa):
                return True
            else:
                return False

    def _correct_cl_for_delta_cl_cut(self, cl, aoa):
        alpha_list = [self.alpha_interp_linear(0.8),
                      self.alpha_cl_max,
                      2*self.alpha_cl_max - self.alpha_interp_linear(0.8)]
        beta_list = [1, (self.cl_max / (self.cl_max + self.delta_cl_cut)), 1]
        beta_interp = interp1d(alpha_list, beta_list, "quadratic",
                               bounds_error=False,
                               fill_value="extrapolate")
        beta = beta_interp(aoa)
        return cl*beta

    def _check_stalled_fin_for_label(self, aoa):
        if self.pp.wingspan > 0.0001:
            if aoa > self.alpha_stall:
                fin_number = self.pp.type_of_fin[1]
                warnings_and_cautions.w_and_c.stalled_fins.stalled_fin[fin_number] = True
            else:
                fin_number = self.pp.type_of_fin[1]
                warnings_and_cautions.w_and_c.stalled_fins.stalled_fin[fin_number] = False

    def _get_cl_normal_ar(self):
        aoa = self.ffc.aoa
        cla = self._cla_diederich(self.ffc.mach)
        self.cl_cut = cla * self.alpha_cut_lineal
        self.alpha_stall = self.alpha_cut_lineal
        self.alpha_cut_cos = np.arccos(self.cl_cut / self.cd_max)
        aoa, sign = convert_aoa_to_something_usable(aoa)
        if aoa <= self.alpha_cut_lineal:
            cl = cla * aoa
        elif aoa <= self.alpha_cut_cos:
            cl = self.cl_cut
        elif aoa <= np.pi/2:
            cl = self.cd_max * np.cos(aoa)
        else:
            cl = self.cd_max * np.cos(aoa)
        self._check_stalled_fin_for_label(aoa)
        return cl * sign

    def _get_cd(self):
        if self.pp.is_ultra_low_AR is True:
            if self.ffc.aoa <= -np.pi + self.alpha_stall:
                cd = self.cl * np.tan(self.ffc.aoa) + self.cd0
            elif self.ffc.aoa <= -self.alpha_stall:
                cd = -self.cd_max * np.sin(self.ffc.aoa)
            elif self.ffc.aoa <= 0:
                cd = self.cl * np.tan(self.ffc.aoa) + self.cd0
            elif self.ffc.aoa <= self.alpha_stall:
                cd = self.cl * np.tan(self.ffc.aoa) + self.cd0
            elif self.ffc.aoa <= np.pi - self.alpha_stall:
                cd = self.cd_max * np.sin(self.ffc.aoa)
            else:
                cd = self.cl * np.tan(self.ffc.aoa) + self.cd0
        else:
            """The continuity correction ensures that there are no jumps when the Cl
            changes the way it's calculated. However, it distorts the curve in
            the Sin(AoA) part, but not much.
            """
            if self.ffc.aoa <= -np.pi + self.alpha_cut_cos:
                cd = self.cl * np.tan(self.ffc.aoa) + self.cd0
            elif self.ffc.aoa <= -self.alpha_cut_cos:
                continuity_correction = np.interp(self.ffc.aoa,
                                                  [-np.pi + self.alpha_cut_cos,
                                                   -np.pi/2,
                                                   -self.alpha_cut_cos],
                                                  [self.cd0, 0, self.cd0])
                cd = -self.cd_max * np.sin(self.ffc.aoa) + continuity_correction
            elif self.ffc.aoa <= 0:
                cd = self.cl * np.tan(self.ffc.aoa) + self.cd0
            elif self.ffc.aoa <= self.alpha_cut_cos:
                cd = self.cl * np.tan(self.ffc.aoa) + self.cd0
            elif self.ffc.aoa <= np.pi - self.alpha_cut_cos:
                continuity_correction = np.interp(self.ffc.aoa,
                                                  [self.alpha_cut_cos,
                                                   np.pi/2,
                                                   np.pi - self.alpha_cut_cos],
                                                  [self.cd0, 0, self.cd0])
                cd = self.cd_max * np.sin(self.ffc.aoa) + continuity_correction
            else:
                cd = self.cl * np.tan(self.ffc.aoa) + self.cd0
        return cd

    def _calculate_cn_ca(self):
        x = self.ffc.aoa
        cn = self.cl*np.cos(x) + self.cd*np.sin(x)
        ca = -self.cl*np.sin(x) + self.cd*np.cos(x)
        return cn, ca

    def _calculate_cm(self):
        aoa, sign = convert_aoa_to_something_usable(self.ffc.aoa)
        if self.ffc.aoa < 0:
            sign = -1
        else:
            sign = 1
        if self.pp.is_ultra_low_AR:
            if aoa < 7*DEG2RAD:
                cm = 0
            else:
                cm = np.interp(aoa,
                               [7*DEG2RAD, np.pi/2],
                               [0, -0.25 * self.cd_max])
        else:
            if aoa < 7*DEG2RAD:
                cm = 0
            elif aoa < 17*DEG2RAD:
                cm = np.interp(aoa,
                               [7*DEG2RAD, 17*DEG2RAD],
                               [0, -0.11])
            elif aoa < 40*DEG2RAD:
                cm = np.interp(aoa,
                               [17*DEG2RAD, 40*DEG2RAD],
                               [-0.11, -0.14])
            else:
                cm = np.interp(aoa,
                               [40*DEG2RAD, 90*DEG2RAD],
                               [-0.14, -0.25 * self.cd_max])
        return cm * sign

    def _calculate_hac(self):
        aoa, sign = convert_aoa_to_something_usable(self.ffc.aoa)
        if self.pp.aspect_ratio <= 1.25:
            ar_list = [1.5, 1.75, 2]
            aoa_list = [30*DEG2RAD, 25*DEG2RAD, 20*DEG2RAD]
            cut_aoa = np.interp(self.pp.aspect_ratio, ar_list, aoa_list)
            if aoa <= 38*DEG2RAD:
                hac = np.interp(aoa,
                                [0, 38*DEG2RAD],
                                [0.18, 0.38])
            else:
                hac = np.interp(aoa,
                                [38*DEG2RAD, np.pi/2],
                                [0.38, 0.5])
        elif self.pp.aspect_ratio < 4:
            ar_list = [1.5, 1.75, 2]
            aoa_list = [30*DEG2RAD, 25*DEG2RAD, 20*DEG2RAD]
            cut_aoa = np.interp(self.pp.aspect_ratio, ar_list, aoa_list)
            if aoa < 7*DEG2RAD:
                hac = np.interp(aoa,
                                [0, 7*DEG2RAD],
                                [0.2, 0.22])
            elif aoa < cut_aoa:
                hac = np.interp(aoa,
                                [7*DEG2RAD, cut_aoa],
                                [0.22, 0.36])
            elif aoa <= 60*DEG2RAD:
                hac = np.interp(aoa,
                                [cut_aoa, 60*DEG2RAD],
                                [0.36, 0.38])
            elif aoa <= 90*DEG2RAD:
                hac = np.interp(aoa,
                                [60*DEG2RAD, 90*DEG2RAD],
                                [0.38, 0.5])
        else:
            ar_list = [1.5, 1.75, 2]
            aoa_list = [30*DEG2RAD, 25*DEG2RAD, 20*DEG2RAD]
            cut_aoa = np.interp(self.pp.aspect_ratio, ar_list, aoa_list)
            if aoa < 7*DEG2RAD:
                hac = 0.25
            elif aoa < cut_aoa:
                hac = np.interp(aoa,
                                [7*DEG2RAD, cut_aoa],
                                [0.25, 0.36])
            elif aoa <= 60*DEG2RAD:
                hac = np.interp(aoa,
                                [cut_aoa, 60*DEG2RAD],
                                [0.36, 0.38])
            elif aoa <= 90*DEG2RAD:
                hac = np.interp(aoa,
                                [60*DEG2RAD, 90*DEG2RAD],
                                [0.38, 0.5])
        if abs(self.ffc.aoa) > np.pi/2:
            hac = 1 - hac  # Correct for aoa coming from behind.
        return hac


class AerodynamicCoefficients:

    def __init__(self):
        self.cl, self.cd, self.cn, self.ca, self.cm, self.hac, self.cd0 = 0, 0, 0, 0, 0, 0, 0

    def set_cl_cd_cn_ca_hac_cd0(self, a, b):
        self.cl, self.cd, self.cn, self.ca, self.cm, self.hac = a
        self.cd0 = b


class CD0:

    def __init__(self):
        self.lift_force = 0
        pass

    def update(self, physic_prop):
        # CD0 nondimensionalized for each fin.
        self.pp = physic_prop
        self.frontal_area_fin = self.pp.wingspan * self.pp.thickness
        self.cos_le_angle_sq = np.cos(self.pp.le_angle)**2
        self.roug_limited_cte = np.power((self.pp.relative_rough/self.pp.mac), 0.2)
        self.wet_area = 2 * self.pp.area  # top and bottom, both are wet.

    def get(self, fin_flight_cond, Re, Re_crit, use_rocket_re):
        self.ffc = fin_flight_cond
        self._calculate_base_drag()
        self._calculate_pressure_drag()
        self._calculate_cf(Re, Re_crit, use_rocket_re)
        self._calculate_friction_drag()
        self._calculate_total_drag()
        return self.cd0

    def _calculate_base_drag(self):
        self.base_drag = 0.12 + 0.13*self.ffc.mach**2

    def _calculate_pressure_drag(self):
        cd_le = ((1-self.ffc.mach**2)**-0.417 - 1) * self.cos_le_angle_sq
        cd_te = self.base_drag / 2
        self.pressure_drag = (cd_le + cd_te) * self.frontal_area_fin

    def _calculate_cf(self, Re, Re_crit, use_rocket_re):
        if use_rocket_re is False:  # the rocket ones are the argument
            Re = self.ffc.Re
            Re_crit = self.pp.reynolds_crit
        if Re < 10e4:
            self.Cf = 1.48e-2
        elif Re < Re_crit:
            self.Cf = 1 / ((1.5*np.log(Re)-5.6)**2)
        else:
            self.Cf = 0.032 * self.roug_limited_cte
        self.Cf = self.Cf * (1 - 0.1*self.ffc.mach**2)

    def _calculate_friction_drag(self):
        # / 1.5 To better fit Open Rocket's results
        # !!!
        self.cd_friction = self.Cf * (1 + 2*self.pp.thickness/self.pp.mac) * self.wet_area / 1.5

    def _calculate_total_drag(self):
        self.drag = self.pressure_drag + self.cd_friction
        self.cd0 = self.drag / self.pp.area  # nondimentionalized for each fin.

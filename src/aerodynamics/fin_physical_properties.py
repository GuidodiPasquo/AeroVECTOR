# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 18:39:42 2021

@author: Guido di Pasquo
"""

import numpy as np
import copy
from src import warnings_and_cautions


class PhysicalProperties:

    def __init__(self):
        pass

    def update(self, dim, fin_attached, roughness, type_of_fin):
        self.is_attached = fin_attached
        self.relative_rough = roughness
        self.type_of_fin = type_of_fin
        self._dim = copy.deepcopy(dim)
        self._calculate_real_dimension()
        self._calculate_area()
        self._calculate_mac_xf_ar(fin_attached)
        self._calculate_force_application_point()
        self._calculate_sweepback_angle()
        self._calculate_reynolds_crit()
        self._check_if_fin_has_ultra_low_ar()

    def _calculate_real_dimension(self):
        r"""Rocket points go from the tip down to the tail."""
        self.pos_root = self._dim[0][0]
        self.c_root = self._dim[0][1]
        self.c_tip = self._dim[1][1]
        # sweep length
        self.x_tail = self._dim[1][0]
        self.wingspan = self._dim[2]
        self.thickness = self._dim[3]

    def _calculate_area(self):
        self.area = (self.c_root+self.c_tip) * self.wingspan / 2

    def _total_pos(self):
        return self._dim[0][0] + self.c_root/2

    def _calculate_mac_xf_ar(self, fin_attached):
        try:
            self.mac = 2/3 * (self.c_root + self.c_tip -
                              ((self.c_root * self.c_tip) /
                               (self.c_root + self.c_tip)))
            k1 = self.c_root + 2*self.c_tip
            k2 = self.c_root + self.c_tip
            k3 = self.c_root**2 + self.c_tip**2 + self.c_root*self.c_tip
            if k2 != 0:
                # Position of the MAC along the wingspan (from c_root)
                self.y_mac = (self.wingspan/3) * (k1/k2)
                # Position of the CP in relation to the LE of c_root
                self.x_force_fin = (self.x_tail/3) * (k1/k2) + (1/6) * (k3/k2)
            else:
                self.x_force_fin = 0
            if self.area != 0:
                self.aspect_ratio = 2 * self.wingspan**2 / self.area
            if fin_attached is False:
                self.aspect_ratio *= 0.625
        except ZeroDivisionError:
            pass

    def _calculate_force_application_point(self):
        self.cp = self._dim[0][0] + self.x_force_fin

    def _calculate_sweepback_angle(self):
        try:
            # 25% of the chord because
            x_root = self._dim[0][0]
            x_tip = self._dim[1][0] + x_root
            x_tip_25 = x_tip + 0.25*self.c_tip
            x_root_25 = x_root + 0.25*self.c_root
            x_tip_50 = x_tip + 0.50*self.c_tip
            x_root_50 = x_root + 0.50*self.c_root
            x_tip_100 = x_tip + 1*self.c_tip
            x_root_100 = x_root + 1*self.c_root
            self.sb_angle = np.arctan((x_tip_25-x_root_25) / self.wingspan)
            self.sb_angle_50 = np.arctan((x_tip_50-x_root_50) / self.wingspan)
            self.le_angle = np.arctan((x_tip-x_root) / self.wingspan)
            self.te_angle = np.arctan((x_tip_100-x_root_100) / self.wingspan)
            if self.le_angle >= 0 and self.te_angle <= 0:
                self.is_delta = True
                self.is_sweptback = False
                self.is_sweptforward = False
            elif self.le_angle > 0 and self.te_angle > 0:
                self.is_delta = False
                self.is_sweptback = True
                self.is_sweptforward = False
            else:
                self.is_delta = False
                self.is_sweptback = False
                self.is_sweptforward = True
        except ZeroDivisionError:
            pass

    def _calculate_reynolds_crit(self):
        self.reynolds_crit = 51 * (self.relative_rough/self.mac)**-1.039

    def _check_if_fin_has_ultra_low_ar(self):
        self.transition_ar = [0]*2
        ar_transition_list = [[2, 1.25, 1.5, 1], [2.4, 1.7, 1.7, 1.2]]
        transition_taper_ratio = [0, 0.25, 0.5, 1]
        self.taper_ratio = self.c_tip / self.c_root
        self.transition_ar[0] = np.interp(self.taper_ratio,  # Lower bound, it behaves as a ULAR wing
                                          transition_taper_ratio,
                                          ar_transition_list[0])
        self.transition_ar[1] = np.interp(self.taper_ratio,  # Upper bound, it behaves as a LAR wing
                                          transition_taper_ratio,
                                          ar_transition_list[1])
        if self.aspect_ratio > self.transition_ar[1]:
            self.is_ultra_low_AR = False
        else:
            self.is_ultra_low_AR = True
        if self.wingspan > 0.000001:
            if self.transition_ar[1] > self.aspect_ratio:
                if self.aspect_ratio > 1:
                    avg_t_ar = (self.transition_ar[1] + 1) / 2
                    if self.aspect_ratio > avg_t_ar:
                        todo = " enlarging or shrinking"
                    else:
                        todo = " shrinking or enlarging"
                    print("CAUTION: " + self.type_of_fin[0]
                          + " fin is in the transition Aspect Ratio (AR = " + str(round(self.aspect_ratio, 2))
                          + "), consider"
                          + todo + " the wingspan.")
                    warnings_and_cautions.w_and_c.cautions.fin_transition_ar[self.type_of_fin[1]] = True
                else:
                    warnings_and_cautions.w_and_c.cautions.fin_transition_ar[self.type_of_fin[1]] = False
            else:
                warnings_and_cautions.w_and_c.cautions.fin_transition_ar[self.type_of_fin[1]] = False
        else:
            warnings_and_cautions.w_and_c.cautions.fin_transition_ar[self.type_of_fin[1]] = False

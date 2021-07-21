# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 21:02:53 2021

@author: Guido di Pasquo
"""


class Warnings:

    def __init__(self):
        self.incorrect_rocket_dimensions = False
        self.wrong_cg = False
        self.fin_incorrect_dim = [False, False]


class Cautions:

    def __init__(self):
        self.fin_zero_thickness = [False, False]
        self.fin_transition_ar = [False, False]


class StalledFins:

    def __init__(self):
        self.stalled_fin = [False, False]


class WarningsAndCautions:

    def __init__(self):
        self.warnings = Warnings()
        self.cautions = Cautions()
        self.stalled_fins = StalledFins()
        self.prev_counter_cautions, self.prev_counter_warnings = False, False

    def check_warnings_and_cautions(self):
        new_events = False
        counter_warnings = 0
        for warn in vars(self.warnings):
            state = vars(self.warnings)[warn]
            if type(state) is bool:
                if state is True:
                    counter_warnings += 1
            else:
                for elem in state:
                    if elem is True:
                        counter_warnings += 1
        if self.prev_counter_warnings != counter_warnings:
            new_events = True
        self.prev_counter_warnings = counter_warnings

        counter_cautions = 0
        for caution in vars(self.cautions):
            state = vars(self.cautions)[caution]
            if type(state) is bool:
                if state is True:
                    counter_cautions += 1
            else:
                for elem in state:
                    if elem is True:
                        counter_cautions += 1
        if self.prev_counter_cautions != counter_cautions:
            new_events = True
        self.prev_counter_cautions = counter_cautions

        return counter_warnings, counter_cautions, new_events

    def check_stalled_fins(self):
        stalled_fins_list = []
        for caution in vars(self.stalled_fins):
            state = vars(self.stalled_fins)[caution]
            for elem in state:
                if elem is True:
                    stalled_fins_list.append(True)
                else:
                    stalled_fins_list.append(False)
        return stalled_fins_list


w_and_c = WarningsAndCautions()

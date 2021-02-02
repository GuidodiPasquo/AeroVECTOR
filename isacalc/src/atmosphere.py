import numpy as np
from .layers import NormalLayer, IsothermalLayer


class Atmosphere(object):

    def __init__(self, *args, **kwargs):

        if kwargs:
            # User Defined Atmosphere Model
            self.__p0 = kwargs['p0']
            self.__d0 = kwargs['d0']

            self.__Hn = kwargs['heights']
            self.__Tn = kwargs['temp']

            try:
                self.__Nn = kwargs['names']

            except KeyError:
                self.__Nn = ['Noname']*len(self.__Hn)

        else:
            # Standard Values
            self.__p0 = 101325.0
            self.__d0 = 1.225

            self.__Nn = ["Troposphere", "Tropopause", "Stratosphere", "Stratosphere", "Stratopause", "Mesosphere", "Mesosphere", "Mesopause", "Thermosphere", "Thermosphere", "Thermosphere"]
            self.__Tn = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.95, 186.95, 201.95, 251.95]
            # Guido here, __Hn[0] was 0, modified to -2 or __Hn would go out of range for -0.00... meters
            self.__Hn = [-2, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0, 90000.0, 100000.0, 110000.0]

        self.__Lt = self.__get_lapse(self.__Hn, self.__Tn)

        self.__layers = []
        self.__build()

    def get_height_boundaries(self):
        """
        Method to calculate for which range the atmosphere model can be used
        :return: Min, Max Height
        """
        return self.__Hn[0], self.__Hn[-1]

    @staticmethod
    def __get_lapse(Hn, Tn) -> list:
        """
        Static Method to calculate the layer types of all layers
        :param Hn: Heights
        :param Tn: Temperatures
        :return: Layer Types
        """

        types = []

        for i in range(len(Hn)-1):
            delta_T = Tn[i+1] - Tn[i]

            lapse = delta_T/(Hn[i+1] - Hn[i])

            if lapse != 0:
                if abs(delta_T) > 0.5:
                    types.append(1)

                else:
                    types.append(0)

            elif lapse == 0:
                types.append(0)

        return types

    def __build(self) -> None:

        p0, d0 = self.__p0, self.__d0

        for name, h0, h_i, T0, T_i, layer_type in zip(self.__Nn, self.__Hn[:-1], self.__Hn[1:], self.__Tn[:-1], self.__Tn[1:], self.__Lt):

            if layer_type == 1:

                Layer = NormalLayer(base_height=h0,
                                    base_temperature=T0,
                                    base_pressure=p0,
                                    base_density=d0,
                                    max_height=h_i,
                                    top_temperature=T_i,
                                    name=name)

                T0, p0, d0, a0, mu0 = Layer.get_ceiling_values()

            elif layer_type == 0:

                Layer = IsothermalLayer(base_height=h0,
                                        base_temperature=T0,
                                        base_pressure=p0,
                                        base_density=d0,
                                        max_height=h_i,
                                        name=name)

                T0, p0, d0, a0, mu0 = Layer.get_ceiling_values()

            else:
                raise ValueError

            self.__layers.append(Layer)

    def calculate(self, h) -> list:

        if h > self.__Hn[-1] or h < self.__Hn[0]:
            raise ValueError("Height is out of bounds")

        for idx in range(len(self.__layers)):

            if h == self.__Hn[idx+1]:
                return self.__layers[idx].get_ceiling_values()

            elif h == self.__Hn[idx]:
                return self.__layers[idx].get_base_values()

            elif self.__Hn[idx] < h < self.__Hn[idx + 1]:
                return self.__layers[idx].get_intermediate_values(h)

            elif h > self.__Hn[idx + 1]:
                continue


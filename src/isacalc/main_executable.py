from src.isacalc.src import Atmosphere

def get_atmosphere(*args, **kwargs) -> Atmosphere:
    """
    Function to obtain the atmosphere model
    :param kwargs: To define a custom atmosphere model
    :return: Atmospheric Model
    """
    return Atmosphere(*args, **kwargs)


def calculate_at_h(h: float, atmosphere_model: Atmosphere = get_atmosphere()) -> list:
    """
    Function to calculate Temperature, Pressure and Density at h
    :param h:                   Height in [m]
    :param atmosphere_model:    Atmosphere Object
    :return:                    [h, T, P, D]
    """
    return atmosphere_model.calculate(h)


if __name__ == "__main__":

    atmosphere = get_atmosphere()
    h = 80000

    T, P, d, a, mu = calculate_at_h(h, atmosphere)

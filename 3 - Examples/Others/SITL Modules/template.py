
from src import python_sitl_functions as Sim
import importlib
import pathlib


def import_module(module):
    current_path = pathlib.Path(__file__).parent.resolve()
    module_temp = pathlib.Path(current_path / "Complementary Modules" / module)
    spec = importlib.util.spec_from_file_location(module, module_temp)
    module_temp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_temp)
    return module_temp


class SITLProgram:
    def __init__(self):
        pass

    ''' Available funtions, called with Sim.
    Sim.millis(), Sim.micros(),
    gyro, accx, accz, alt, pos_gnss, vel_gnss = Sim.getSimData()
    Sim.sendCommand(servo, parachute)
    Sim.plot_variable(variable, number) (from 1 to 5 for diferent plots)
    -->
    -->
    -->
    -->
    -->
    -->
    -->
    -->
    -->
    -->
    '''

    def everything_that_is_outside_functions(self):
        self.alt_prev = 0
        self.timer_all = 0
        self.sample_time_program = 0.1






    def void_setup(self):
        pass







    def void_loop(self):
        self.t = Sim.micros()/1000000
        if self.t >= self.timer_all + self.sample_time_program*0.999:
            self.gyro, self.accx, self.accz, self.alt, self.pos_gnss, self.vel_gnss = Sim.getSimData()
            parachute = self.parachute_deployment()
            servo = 0
            Sim.sendCommand(servo, parachute)






    '''########'''

    def parachute_deployment(self):
        if self.alt < self.alt_prev and self.alt > 10:
            return 1
        else:
            self.alt_prev = self.alt
            return 0

    
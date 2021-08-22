def test_complementary_module_import():
    print("Example: This could be a PID module")
    
    
class PID():
    
    def __init__(self, kp, ki, kd, anti_windup, saturation):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.anti_windup = anti_windup
        self.fin_max = saturation        
        self.delta_t = 0
        self.t_prev = 0
        self.last_error = 0
        self.cum_error = 0
    
    def compute_output(self, inp, t):
        self.delta_t = t - self.t_prev
        # Compute derivative
        rate_error = (inp-self.last_error) / self.delta_t
        if self.anti_windup is True:
            # PID output
            out_pid = self.kp*inp + self.ki*self.cum_error + self.kd*rate_error
            # Anti windup by clamping
            if -self.fin_max < out_pid < self.fin_max:
                # Compute integral (trapezoidal) only if the TVC is not staurated
                self.cum_error = (((((self.last_error) + ((inp-self.last_error)/2)))
                                  * self.delta_t) + self.cum_error)
                # Recalculate the output
                out_pid = self.kp*inp + self.ki*self.cum_error + self.kd*rate_error
            # Saturation, prevents the TVC from deflecting more that it can
            if out_pid > self.fin_max:
                out_pid = self.fin_max
            elif out_pid < -self.fin_max:
                out_pid = -self.fin_max
        else:
            self.cum_error = ((((self.last_error + ((inp-self.last_error)/2)))
                             * self.delta_t) + self.cum_error)
            out_pid = self.kp*inp + self.ki*self.cum_error + self.kd*rate_error
        # Remember current error
        self.last_error = inp
        self.t_prev = t
        # Have function return the PID output
        return out_pid
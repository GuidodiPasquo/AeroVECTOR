import sys
sys.path.append("..")
from src.simulation import servo_lib

servo = servo_lib.Servo()
servo.setup(actuator_weight_compensation=1.45, definition=1, servo_s_t=0.02)
servo.test(u_deg=30)

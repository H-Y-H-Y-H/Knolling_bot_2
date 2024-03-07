import numpy as np


obj_width_range = np.array([0.021, 0.026, 0.032, 0.039, 0.045, 0.052, 0.057])
# motor_pos_range = np.array([2100, 2200, 2250, 2350, 2450, 2500, 2550])
motor_pos_range = np.array([2100, 2200, 2250, 2350, 2450, 2550, 2650])


formula_parameters = np.polyfit(obj_width_range, motor_pos_range, 3)
motor_pos = np.poly1d(formula_parameters)

input_data = np.arange(0.021, 0.057, 0.001)
output_data = motor_pos(input_data)
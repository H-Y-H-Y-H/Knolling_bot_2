import numpy as np
import matplotlib.pyplot as plt
import time

# Parameters for the motors and control
num_motors = 5
setpoint = 100  # Desired speed (arbitrary units)
kp = 0.5  # Proportional gain
kd = 0.1  # Derivative gain
max_speed = 200  # Maximum motor speed

# Initialize motor speeds
motor_speeds = np.zeros(num_motors)
last_errors = np.zeros(num_motors)

# Simulation settings
num_steps = 100
time_interval = 0.1

# Simulate the control loop
for step in range(num_steps):
    errors = setpoint - motor_speeds
    d_errors = errors - last_errors

    for i in range(num_motors):
        control_signal = kp * errors[i] + kd * d_errors[i]

        # Limit the control signal to the maximum speed
        if control_signal > max_speed:
            control_signal = max_speed
        elif control_signal < -max_speed:
            control_signal = -max_speed

        motor_speeds[i] += control_signal

        # Simulate some noise and disturbances
        # motor_speeds[i] += np.random.uniform(-5, 5)

        # Ensure motor speed stays within limits
        motor_speeds[i] = np.clip(motor_speeds[i], 0, max_speed)

        last_errors[i] = errors[i]

    # Plot the motor speeds
    plt.clf()
    plt.bar(range(num_motors), motor_speeds)
    plt.ylim(0, max_speed)
    plt.xlabel('Motor')
    plt.ylabel('Speed')
    plt.title(f'Step {step + 1}')
    plt.pause(time_interval)

# Keep the plot window open
plt.show()

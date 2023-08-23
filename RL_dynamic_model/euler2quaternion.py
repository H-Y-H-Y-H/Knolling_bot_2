import numpy as np

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles to a quaternion.

    :param roll: Rotation around the X-axis (in radians)
    :param pitch: Rotation around the Y-axis (in radians)
    :param yaw: Rotation around the Z-axis (in radians)
    :return: Quaternion as a numpy array [w, x, y, z]
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])

# Example Euler angles in radians
roll = np.pi / 4
pitch = np.pi / 6
yaw = np.pi / 3

quaternion = euler_to_quaternion(roll, pitch, yaw)
print("Quaternion:", quaternion)

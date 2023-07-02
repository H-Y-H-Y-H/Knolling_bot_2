import numpy as np

def apply_rotation(roll, pitch, yaw, point):
    # Convert angles to radians
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)

    # Calculate rotation matrix components
    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # Create rotation matrix
    rotation_matrix = np.array([[cos_yaw * cos_pitch, cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll, cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll],
                                [sin_yaw * cos_pitch, sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll, sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll],
                                [-sin_pitch, cos_pitch * sin_roll, cos_pitch * cos_roll]])
    rotated_point = np.dot(rotation_matrix, point)[:3]

    return rotated_point

# Example usage
point = np.array([1, 1, 1])  # Point in 3D environment
roll = 0  # Roll angle in degrees
pitch = 0  # Pitch angle in degrees
yaw = 45  # Yaw angle in degrees

# Apply rotation to the point
rotated_point = apply_rotation(roll, pitch, yaw, point)

print("Rotated point:", rotated_point)


import pybullet as p
import numpy as np

# Define a specific quaternion
specific_quaternion = [-0.13322883194441312, 0.6944464405318476, 0.1341095894378673, 0.6942685630159133]  # Replace with your specific quaternion

# quaternion: -0.13322883194441312, 0.6944464405318476, 0.1341095894378673, 0.6942685630159133
# angle: (0.0, 1.5707963267948966, 0.3790917685036834)

# quaternion: -0.13863952372569088, 0.6950841589742535, 0.15061950586903328, 0.6891667859494801
# angle: (1.8235828622107468, 1.5518997351235828, 2.235582928171257)

# quaternion: -0.13532912964128, 0.6974615529538299, 0.13729004232574993, 0.6902063844242237
# angle: (2.6836337239374046, 1.5601677633292024, 3.07160966609525)

# Convert the quaternion to Euler angles
euler_angles = p.getEulerFromQuaternion(specific_quaternion)

# Convert Euler angles to degrees for convenience
euler_angles_deg = np.degrees(euler_angles)

print("Euler angles (in degrees):", euler_angles)
